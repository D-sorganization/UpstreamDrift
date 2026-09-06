# ARCHITECTURE_DEBT — tracked as GitHub issue #1937
# This file is 1,173 lines and contains 4 distinct mesh generator backends
# (Primitive, MakeHuman, SMPLX) plus the public MeshGenerator facade.
# Recommended split:
#   mesh_generator_primitive.py  — PrimitiveMeshGenerator
#   mesh_generator_makehuman.py  — MakeHumanMeshGenerator
#   mesh_generator_smplx.py      — SMPLXMeshGenerator
#   mesh_generator.py            — MeshGenerator facade + MeshGeneratorBackend enum
# Risk: low-medium — backends are independent; facade is the only public API.
# Prerequisite: parametrize existing tests over all backends before splitting.

"""
Mesh generation interfaces for humanoid character builder.

This module was refactored from a single 1155-line file into focused submodules
to comply with the line budget:

    _mesh_types          — MeshGeneratorBackend, GeneratedMeshResult,
                           MeshGeneratorInterface
    _primitive_generator — PrimitiveMeshGenerator
    _makehuman_generator — MakeHumanMeshGenerator
    _smplx_generator     — SMPLXMeshGenerator

All public symbols remain importable from this module.
"""

from __future__ import annotations  # noqa: E402, F404

import logging
from typing import Any, cast

from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,  # noqa: F401
)

# Re-export sub-module symbols (public API unchanged)
from ._mesh_makehuman import MakeHumanMeshGenerator  # noqa: F401
from ._mesh_primitives import PrimitiveMeshGenerator  # noqa: F401
from ._mesh_smplx import (  # noqa: F401
    SMPLX_AVAILABLE,
    TRIMESH_AVAILABLE,
    SMPLXMeshGenerator,
    _smplx_module,
    _trimesh_module,
)
from ._mesh_types import (  # noqa: F401
    GeneratedMeshResult,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
)

logger = logging.getLogger(__name__)


class MeshGenerator:
    """
    Factory class for creating mesh generators.

    Provides a unified interface to multiple mesh generation backends.
    """

    _generators: dict[MeshGeneratorBackend, type[Any]] = {
        MeshGeneratorBackend.PRIMITIVE: PrimitiveMeshGenerator,
        MeshGeneratorBackend.MAKEHUMAN: MakeHumanMeshGenerator,
        MeshGeneratorBackend.SMPLX: SMPLXMeshGenerator,
    }

    @classmethod
    def create(
        cls,
        backend: MeshGeneratorBackend | str = MeshGeneratorBackend.PRIMITIVE,
        **kwargs: Any,
    ) -> MeshGeneratorInterface:
        """
        Create a mesh generator for the specified backend.

        Args:
            backend: Backend to use
            **kwargs: Backend-specific initialization options

        Returns:
            MeshGeneratorInterface instance
        """
        if isinstance(backend, str):
            backend = MeshGeneratorBackend(backend.lower())

        generator_class = cls._generators.get(backend)
        if generator_class is None:
            raise ValueError(f"Unknown backend: {backend}")

        return cast(MeshGeneratorInterface, generator_class(**kwargs))

    @classmethod
    def get_available_backends(cls) -> list[MeshGeneratorBackend]:
        """Return list of available backends."""
        available = []
        for backend, generator_class in cls._generators.items():
            try:
                generator = generator_class()
                if generator.is_available:
                    available.append(backend)
            except (ImportError, RuntimeError, OSError) as e:
                logger.debug("Backend %s not available: %s", backend.value, e)
        return available

    @classmethod
    def get_best_available(cls) -> MeshGeneratorInterface:
        """
        Get the best available mesh generator.

        Preference order: MakeHuman > SMPL-X > Primitive
        """
        preference = [
            MeshGeneratorBackend.MAKEHUMAN,
            MeshGeneratorBackend.SMPLX,
            MeshGeneratorBackend.PRIMITIVE,
        ]

        for backend in preference:
            try:
                generator = cls.create(backend)
                if generator.is_available:
                    return generator
            except (ImportError, RuntimeError, OSError) as e:
                logger.debug("Backend %s not available: %s", backend.value, e)
                continue

        return PrimitiveMeshGenerator()
