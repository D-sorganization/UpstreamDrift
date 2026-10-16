"""One class per mesh-generator backend, whatever the import path (#12706).

``PrimitiveMeshGenerator`` and ``SMPLXMeshGenerator`` each used to exist twice
(``_primitive_generator`` vs ``_mesh_primitives``, ``_smplx_generator`` vs
``_mesh_smplx``). The copies drifted apart: measured SMPL-X betas (#8404) were
added to the copy the public ``mesh_generator`` facade never imported. Every
path must now resolve to the same class object, so a fix to one is a fix to all.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit

_PACKAGES = (
    "src.shared.python.humanoid_character_builder.generators",
    "humanoid_character_builder.generators",
)


def _class_from(package: str, module: str, name: str) -> type:
    return getattr(importlib.import_module(f"{package}.{module}"), name)


@pytest.mark.parametrize(
    ("name", "modules", "home"),
    [
        (
            "PrimitiveMeshGenerator",
            ("_primitive_generator", "_mesh_primitives", "mesh_generator"),
            "_primitive_generator",
        ),
        (
            "SMPLXMeshGenerator",
            ("_mesh_smplx", "_smplx_generator", "mesh_generator"),
            "_mesh_smplx",
        ),
    ],
)
def test_every_import_path_resolves_to_one_class(
    name: str, modules: tuple[str, ...], home: str
) -> None:
    classes = {
        id(_class_from(package, module, name))
        for package in _PACKAGES
        for module in modules
    }
    assert len(classes) == 1, f"{name} is defined more than once: {modules}"

    cls = _class_from(_PACKAGES[0], modules[0], name)
    assert cls.__module__.rsplit(".", 1)[-1] == home


def test_facade_factory_creates_the_single_classes() -> None:
    mesh_generator = importlib.import_module(f"{_PACKAGES[0]}.mesh_generator")
    backend = mesh_generator.MeshGeneratorBackend
    smplx = _class_from(_PACKAGES[0], "_smplx_generator", "SMPLXMeshGenerator")
    primitive = _class_from(_PACKAGES[0], "_mesh_primitives", "PrimitiveMeshGenerator")

    assert type(mesh_generator.MeshGenerator.create(backend.SMPLX)) is smplx
    assert type(mesh_generator.MeshGenerator.create(backend.PRIMITIVE)) is primitive
