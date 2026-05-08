"""Thin facade for humanoid mesh-generator backends."""

from __future__ import annotations

import logging
from typing import Any

from ._mesh_makehuman import MakeHumanMeshGenerator
from ._mesh_primitives import PrimitiveMeshGenerator
from ._mesh_smplx import (
    SMPLXMeshGenerator,
    SMPLX_AVAILABLE,
    TRIMESH_AVAILABLE,
    _smplx_module,  # type: ignore[attr-defined]
    _trimesh_module,  # type: ignore[attr-defined]
)
from ._mesh_types import (
    GeneratedMeshResult,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
)

__all__ = [
    "GeneratedMeshResult",
    "MakeHumanMeshGenerator",
    "MeshGenerator",
    "MeshGeneratorBackend",
    "MeshGeneratorInterface",
    "PrimitiveMeshGenerator",
    "SMPLXMeshGenerator",
    "SMPLX_AVAILABLE",
    "TRIMESH_AVAILABLE",
    "_smplx_module",
    "_trimesh_module",
]

logger = logging.getLogger(__name__)


class MeshGenerator:
    """
    Factory class for creating mesh generators.

    Provides a unified interface to multiple mesh generation backends.
    """

    _generators: dict[MeshGeneratorBackend, type[MeshGeneratorInterface]] = {
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

        return generator_class(**kwargs)

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

        # Final fallback
        return PrimitiveMeshGenerator()
