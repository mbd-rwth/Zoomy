import os
from library.model.models import *
from library.fvm.solver import *
import library.fvm.flux
import library.fvm.reconstruction
import library.fvm.nonconservative_flux
import library.fvm.timestepping
import library.mesh.mesh as petscMesh

import library.misc.io


def run(model_class, ic, bcs, mesh, solver, settings, **param):
    level = param['level']
    model = model_class(
        dimension=2,
        fields=3 + 2 * level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings=settings,
    )

    solver(
        mesh, model, settings)

    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))
