import os
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy
from mpi4py import MPI
from dolfinx.io import gmshio
# import gmsh
# import dolfinx
from dolfinx import fem
import basix
import tqdm
from petsc4py import PETSc
import ufl
# import pyvista
from  dolfinx.fem import petsc
# import sys
# from dolfinx import mesh
from ufl import (
    TestFunction,
    TrialFunction,
    dx,
    inner,
)
import dolfinx
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx import mesh as dolfinx_mesh

import numpy.typing as npt


from library.fvm.solver import Settings
from library.model.models.shallow_water import ShallowWaterEquations
from library.mesh.mesh import Mesh
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.misc.misc import Zstruct
import library.transformation.to_ufl as trafo

def compute_facet_distances(mesh: mesh.Mesh) -> fem.Function:
    """
    Compute distance from each facet center to its adjacent cell center.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The DOLFINx mesh.

    Returns
    -------
    distance : dolfinx.fem.Function
        A DG0 function defined on facets (fdim) with the distances.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    coords = mesh.geometry.x

    # --- Cell centers ---
    cell2v = mesh.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    cell_coords = coords[cell2v, :]            # (num_cells, tdim+1, tdim)
    cell_centers = cell_coords.mean(axis=1)    # (num_cells, tdim)

    # --- Facet centers ---
    facet2v = mesh.topology.connectivity(fdim, 0).array.reshape(-1, tdim)
    facet_coords = coords[facet2v, :]          # (num_facets, tdim, tdim)
    facet_centers = facet_coords.mean(axis=1)  # (num_facets, tdim)

    # --- Map each facet to a neighboring cell ---
    facet2cell_conn = mesh.topology.connectivity(fdim, tdim)
    facet_to_cell = np.array([c[0] if len(c) > 0 else -1 for c in facet2cell_conn.links()])
    valid_facets = facet_to_cell >= 0

    # --- Compute distances ---
    facet_distances = np.zeros(facet_centers.shape[0], dtype=np.float64)
    facet_distances[valid_facets] = np.linalg.norm(
        facet_centers[valid_facets] - cell_centers[facet_to_cell[valid_facets]], axis=1
    )

    # --- Store in DG0 function ---
    V_facets = fem.FunctionSpace(mesh, ("DG", 0))
    distance = fem.Function(V_facets, name="distance")
    distance.x.array[:] = facet_distances
    distance.x.scatter_forward()

    return distance


def load_mesh(path_to_mesh):
    mesh = Mesh.from_gmsh(path_to_mesh)
    min_inradius = np.min(mesh.cell_inradius)
    tags = [int(v) for v in mesh.boundary_conditions_sorted_physical_tags]
    tags.sort()
    map_tag_to_id = {v: i for i, v in enumerate(tags)}
    
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        path_to_mesh, MPI.COMM_WORLD, 0, gdim=2
    )
    unique_facet_tags = np.unique(facet_tags.values)
    facet_boundary_function_id = np.array([map_tag_to_id[tag] for tag in facet_tags.values[:]])
    return mesh, cell_tags, facet_tags, unique_facet_tags, facet_boundary_function_id, min_inradius


def create_function_space(domain):
    elem_Q = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(3,))
    space_Q = fem.functionspace(domain, elem_Q)
    elem_Qaux= basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(2,))
    space_Qaux = fem.functionspace(domain, elem_Qaux)
    return space_Q, space_Qaux


def evaluate_on_all_interior_facets_midpoint(
        expr : ufl.core.expr.Expr,
        domain : dolfinx_mesh.Mesh,
        side : str = "+"
) -> tuple[npt.NDArray[np.floating],  # X_mid  (N, gdim)
           npt.NDArray[np.inexact]]:   # values (N, …)
    """
    Mid-point evaluation of `expr` on *all* interior facets owned by this
    MPI rank.

    Parameters
    ----------
    expr   : UFL expression (may contain u("+"), u("-"), jump(u), …)
    domain : dolfinx mesh
    side   : "+"  – take value from the first adjacent cell  (default)
             "-"  – take value from the second adjacent cell
             "avg" – arithmetic mean of + and -

    Returns
    -------
    X_mid  : coordinates of facet mid-points
    values : expression evaluated according to `side`
    """

    assert side in {"+", "-", "avg"}

    tdim, fdim = domain.topology.dim, domain.topology.dim - 1

    # --- identify interior facets -------------------------------------
    domain.topology.create_connectivity(fdim, tdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)
    n_loc_facets = domain.topology.index_map(fdim).size_local

    interior_facets = np.array(
        [f for f in range(n_loc_facets) if len(f_to_c.links(f)) == 2],
        dtype=np.int32)

    if interior_facets.size == 0:
        # nothing on this rank
        shape_val = expr.ufl_shape or ()
        return (np.empty((0, domain.geometry.dim)),
                np.empty((0, *shape_val)))

    # --- reference mid-point of one facet -----------------------------
    facet_type = basix.cell.subentity_types(domain.basix_cell())[fdim][0]
    xi_facet_mid = basix.cell.geometry(facet_type).mean(axis=0)

    xi_cell = np.zeros((1, tdim))
    xi_cell[0, :xi_facet_mid.shape[0]] = xi_facet_mid

    expr_comp  = fem.Expression(expr, xi_cell)
    coord_comp = fem.Expression(ufl.SpatialCoordinate(domain), xi_cell)

    # --- (cell, local_facet) pairs for interior facets ----------------
    cell_facet = fem.compute_integration_domains(
        fem.IntegralType.interior_facet, domain.topology, interior_facets, fdim)

    # --- evaluate (+ and - in one call) --------------------------------
    values_2 = expr_comp.eval(domain, cell_facet)     # shape (2N, …)
    X_2      = coord_comp.eval(domain, cell_facet)    # shape (2N, gdim)

    # UFL/FFC order: first cell → “+”, second cell → “-”
    val_plus  = values_2[0::2]
    val_minus = values_2[1::2]
    X_mid     = X_2[0::2]        # same coordinates twice

    if side == "+":
        values = val_plus
    elif side == "-":
        values = val_minus
    else:                         # "avg"
        values = 0.5*(val_plus + val_minus)

    return X_mid, values


# ---------------------------------------------------------------------
# 2. exterior facets – one value per facet
# ---------------------------------------------------------------------
def evaluate_on_all_exterior_facets_midpoint(
        expr   : ufl.core.expr.Expr,
        domain : dolfinx_mesh.Mesh
) -> tuple[npt.NDArray[np.floating],  # X_mid  (N, gdim)
           npt.NDArray[np.inexact]]:   # values (N, …)
    """
    Mid-point evaluation of `expr` on all boundary (exterior) facets
    owned by this rank.
    """

    tdim, fdim = domain.topology.dim, domain.topology.dim - 1

    domain.topology.create_connectivity(fdim, tdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)
    n_loc_facets = domain.topology.index_map(fdim).size_local

    exterior_facets = np.array(
        [f for f in range(n_loc_facets) if len(f_to_c.links(f)) == 1],
        dtype=np.int32)

    if exterior_facets.size == 0:
        shape_val = expr.ufl_shape or ()
        return (np.empty((0, domain.geometry.dim)),
                np.empty((0, *shape_val)))

    facet_type = basix.cell.subentity_types(domain.basix_cell())[fdim][0]
    xi_facet_mid = basix.cell.geometry(facet_type).mean(axis=0)

    xi_cell = np.zeros((1, tdim))
    xi_cell[0, :xi_facet_mid.shape[0]] = xi_facet_mid

    expr_comp  = fem.Expression(expr, xi_cell)
    coord_comp = fem.Expression(ufl.SpatialCoordinate(domain), xi_cell)

    cell_facet = fem.compute_integration_domains(
        fem.IntegralType.exterior_facet, domain.topology, exterior_facets, fdim)

    values = expr_comp.eval(domain, cell_facet)   # shape (N, …)
    X_mid  = coord_comp.eval(domain, cell_facet)  # shape (N, gdim)

    return X_mid, values


# ---------------------------------------------------------------------
# 3. interior + exterior combined  (two outputs only)
# ---------------------------------------------------------------------
def evaluate_on_all_facets_midpoint(
        expr   : ufl.core.expr.Expr,
        domain : dolfinx_mesh.Mesh,
        side   : str = "+"
) -> tuple[npt.NDArray[np.floating],  # X_mid_all  (N_tot, gdim)
           npt.NDArray[np.inexact]]:   # values_all (N_tot, …)
    """
    Mid-point evaluation on *all* facets owned by this rank.

    • interior facets – take value according to `side` ("+", "-", "avg")
    • exterior facets – the only available value

    Returns concatenated arrays of coordinates and values.
    """

    X_int,  val_int  = evaluate_on_all_interior_facets_midpoint(expr, domain, side)
    X_ext,  val_ext  = evaluate_on_all_exterior_facets_midpoint(expr, domain)

    X_all   = np.vstack((X_int, X_ext))
    values  = np.concatenate((val_int, val_ext), axis=0)

    return X_all, values


def numerical_flux(model, Ql, Qr, Qauxl, Qauxr, parameters, n, domain):

    I = ufl.as_tensor([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    return ufl.dot(0.5 *(model.flux(Ql, Qauxl, parameters)+ model.flux(Qr, Qauxr, parameters)), n)- 0.5 * 0.5*(max_abs_eigenvalue(model, Ql, Qauxl, n, domain) + max_abs_eigenvalue(model, Qr, Qauxr, n, domain) )* I * (Qr- Ql)

def extract_scalar_fields(Q):
    n_dofs = Q.function_space.num_sub_spaces
    out = []
    for i in range(n_dofs):
        qi = Q.sub(i).collapse()
        # qi.x.array[qi.x.array < 1e-12] = 0.
        qi.name = f"q_{i}"
        out.append(qi)
    return out

def _max_abs_eigenvalue(model, Q, Qaux, n, domain):
    
    eigenvalues = model.eigenvalues(Q, Qaux, model.parameters, n)
    evs = evaluate_on_all_facets_midpoint(eigenvalues, domain)[1]
    return np.max(abs(evs))

def max_abs_eigenvalue(model, Q, Qaux, n, domain):
    ev = model.eigenvalues(Q, Qaux, model.parameters, n)
    max_ev = abs(ev[0, 0])
    for i in range(1, model.n_variables):
        max_ev = ufl.conditional(ev[1, 0] > max_ev, ev[1, 0], max_ev)
    return max_ev

def compute_time_step_size(model, Q, Qaux, n, reference_cell_diameter, domain, CFL=0.45):
    
    n1 = ufl.as_vector((1, 0))
    n2 = ufl.as_vector((0, 1))
    evs_m = _max_abs_eigenvalue(model, Q, Qaux, n1, domain)
    evs_p = _max_abs_eigenvalue(model, Q, Qaux, n2, domain)

    local_max_eigenvalue = max(evs_m, evs_p)

    # Global maximum reduction across all ranks
    global_max_eigenvalue = MPI.COMM_WORLD.allreduce( local_max_eigenvalue, op=MPI.MAX)
    
    dt = CFL * reference_cell_diameter / global_max_eigenvalue
    
    if np.isnan(dt) or np.isinf(dt) or dt < 10**(-6):
        dt = 10**(-6)

    return dt


# %%
def update_qaux(Q, Qaux, Qold, Qauxold, model, parameters, time, dt):
    return Qaux


# %%
def weak_form_swe(model, functionspace, q_n, q_np, qaux_n, qaux_np, parameters, t, x, dt, domain, cell_tags, facet_tags, unique_facet_tags, facet_boundary_function_id):
    # facet normals
    # domain = functionspace.mesh
    n = ufl.FacetNormal(domain)

    # our integration measures over the inner boundaries, the domain boundaries and the whole domain. 
    # Note that we separate the domain boundaries in order to potentially apply different boundary conditions
    # on each side
    dS = ufl.Measure("dS", domain=domain)
    # facet_tags = generate_facets_tags(domain, P0, P1)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.dx


    # implicit/explicit switch
    q = q_n
    qaux = qaux_n
    # q_ghost = ufl.Function(functionspace)


    # We would like to have gradients of the bottom topography. However, DG0 is constant in each cell, resulting in zero gradients.
    # we help ourselves by projecting DG0 to a CG1 (linear continuous functions) space, where the gradients do exist.
    # note that this is a 'cheap trick'. In reality, the computation of the bottom topography gradient is critical and deserves
    # more attention.
    elem_CG1 = basix.ufl.element("CG", domain.topology.cell_name(), 1)
    space_CG1 = fem.functionspace(domain, elem_CG1)



    test_q = ufl.TestFunction(functionspace)
    trial_q = ufl.TrialFunction(functionspace)
    
    # weak formulation
    weak_form =  ufl.dot(test_q, (trial_q-q)/dt) * dx
    weak_form += ufl.dot((test_q("+") - test_q("-")), 
                         numerical_flux(model, q("+"), q("-"), qaux("+"), qaux("-"), parameters, n("+"), domain)) * dS
    # weak_form += ufl.dot((test_q), numerical_flux(q, q_extrapolation, n)) * (ds(1) + ds(2) + ds(3) + ds(4))
    for i, tag in enumerate(unique_facet_tags):
        # q_ghost.interpolate(boundary_functions[i](q))
        #TODO dX is wrong
        dX = x[0]
        weak_form += ufl.dot((test_q), numerical_flux(model, q, model.bcs(t, x, dX, q, qaux, parameters, n)[i,:], qaux, qaux,parameters, n, domain)) * ds(tag)
        # weak_form += ufl.dot((test_q), numerical_flux(q, q, n)) * ds(tag)

    #################TODO#################################
    weak_form += 0
    ######################################################


    weak_form_lhs = fem.form(ufl.lhs(weak_form))
    weak_form_rhs = fem.form(ufl.rhs(weak_form))

    return weak_form_lhs, weak_form_rhs


# %%
def prepare_solver(weak_form_lhs, weak_form_rhs):
    A = petsc.create_matrix(weak_form_lhs)
    b = petsc.create_vector(weak_form_rhs)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.BCGS)
    preconditioner = solver.getPC()
    preconditioner.setType(PETSc.PC.Type.JACOBI)

    return solver, A, b


# %%
def solve_time_loop(name: str, path_to_mesh: str, model, weak_form_function,  initial_condition, end_time, output_path, CFL):
    
    domain, cell_tags, facet_tags, unique_facet_tags, facet_boundary_function_id, min_inradius = load_mesh(path_to_mesh)
    
    model = trafo.FenicsXRuntimeModel.from_model(domain, model)
    
    ### Parameters
    gx = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0))
    gy = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0))
    gz = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(9.81))
    friction_coeff = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.))
    
    reference_edge_length = min_inradius
    
    functionspace, functionspace_qaux = create_function_space(domain)
    parameters = model.parameters
    
    
    
    t = fem.Constant(domain, dolfinx.default_scalar_type(0.0))    
    
    # dt will be overwritten
    dt = fem.Constant(domain, dolfinx.default_scalar_type(0.1))    

    x = ufl.SpatialCoordinate(domain)
    x = ufl.as_vector((x[0], x[1], 0)) 
        
    q_n = fem.Function(functionspace, name=r'$q^n$')
    q_np1 = fem.Function(functionspace, name=r'$q^{n+1}$')
    
    qaux_n = fem.Function(functionspace_qaux, name=r'$q_{aux}$')
    qaux_np1 = fem.Function(functionspace_qaux, name=r'$q_{aux}^{n+1}$')
    
    q_n.interpolate(initial_condition)
    q_np1.interpolate(initial_condition)
    
    normals = ufl.FacetNormal(domain)


    update_qaux(q_n, qaux_n, q_np1, qaux_np1, model, model.parameters, t, dt)
    
    weak_form_lhs, weak_form_rhs = weak_form_function(model, functionspace, q_n, q_np1, qaux_n, qaux_np1, parameters, t, x, dt, domain, cell_tags, facet_tags, unique_facet_tags, facet_boundary_function_id)
    
    solver, A, b = prepare_solver(weak_form_lhs, weak_form_rhs)    
    A = petsc.create_matrix(weak_form_lhs)
    b = petsc.create_vector(weak_form_rhs)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.BCGS)
    preconditioner = solver.getPC()
    preconditioner.setType(PETSc.PC.Type.JACOBI)

    num_timesteps = int(end_time/dt.value)
    
    # VTK writer
    os.makedirs(output_path, exist_ok=True)
    vtk_file_abs_path_name = os.path.join(output_path, f"{name}.pvd")
    vtk_writer = dolfinx.io.VTKFile(
        domain.comm, vtk_file_abs_path_name, "w+"
    )
    

    vtk_writer.write_function(extract_scalar_fields(q_n), t=0.0)
    n_snapshots = 50
    dt_snapshot = end_time / n_snapshots
    next_snapshot_time = dt_snapshot
    

    progress = tqdm.tqdm(desc="Setup " + name + ", solving PDE", total=end_time)

    max_timesteps = 10000
    i=0
    while t.value < end_time and i < max_timesteps:
        q_n.interpolate(q_np1)
        
        # time step size estimation
        dt.value  = compute_time_step_size(model, q_np1, qaux_np1, normals, reference_edge_length, domain, CFL=CFL)
        progress.update(dt.value)


        A.zeroEntries()
        petsc.assemble_matrix(A,weak_form_lhs)
        A.assemble()
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b,weak_form_rhs)

        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver.solve(b, q_np1.x.petsc_vec)

        q_np1.x.scatter_forward()
        t.value += dt.value
        i += 1
        
        
        if t.value > next_snapshot_time:
            vtk_writer.write_function(extract_scalar_fields(q_np1), t=t.value)
            next_snapshot_time += dt_snapshot

    progress.close()
    return q_np1
