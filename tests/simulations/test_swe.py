import numpy as np
import pytest
from types import SimpleNamespace

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io


@pytest.mark.critical
@pytest.mark.unfinished
def test_swe_1d():
    settings = Settings(
        name="ShallowWater",
        parameters={"g": 1.0},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=1.0,
        output_snapshots=100,
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP(
        left=lambda n_field: np.array([2.0, 0.0]),
        right=lambda n_field: np.array([1.0, 0.0]),
    )
    model = ShallowWater(
        dimension=1,
        fields=2,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )
    mesh = Mesh.create_1d((-1, 1), 100)

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


@pytest.mark.critical
@pytest.mark.unfinished
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_swe_2d(mesh_type):
    settings = Settings(
        name="ShallowWater2d",
        parameters={"g": 1.0, "C": 1.0},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.zero(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=1.0,
        output_snapshots=100,
    )

    bc_tags = ["left", "right", "top", "bottom"]
    bc_tags_periodic_to = ["right", "left", "bottom", "top"]

    bcs = BC.BoundaryConditions(
        [
            BC.Periodic(physical_tag=tag, periodic_to_physical_tag=tag_periodic_to)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP(
        left=lambda n_field: np.array([2.0, 0.0, 0.0]),
        right=lambda n_field: np.array([1.0, 0.0, 0.0]),
    )
    model = ShallowWater2d(
        dimension=2,
        fields=3,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": ["chezy"], "eigenvalue_mode": "symbolic"},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)),
        mesh_type,
    )

    fvm_unsteady_semidiscrete(mesh, model, settings, RK1)
    io.generate_vtk(settings.output_dir)


if __name__ == "__main__":
    # test_swe_1d()
    test_swe_2d("quad")
    # test_swe_2d("triangle")
