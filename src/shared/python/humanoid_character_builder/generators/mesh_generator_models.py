"""Shared mesh-generator data models and interfaces.

This module is now a thin re-export of :mod:`._mesh_types`, the canonical
home for mesh-generator types in this subsystem. Kept for backward
compatibility with the legacy ``mesh_generator_models`` import path used
by ``mesh_generator_factory``, ``mesh_generator_makehuman``,
``mesh_generator_primitive``, and ``mesh_generator_smplx``.

See issue #4602 (URDF Hardening Campaign / arch-B/3): the two duplicate
``GeneratedMeshResult`` / ``MeshGeneratorBackend`` / ``MeshGeneratorInterface``
definitions in this directory were collapsed onto ``_mesh_types`` so
identity comparisons (``MeshGeneratorBackend.SMPLX is X``) work across
both legacy import paths.
"""

from __future__ import annotations

from humanoid_character_builder.generators._mesh_types import (
    GeneratedMeshResult,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
    segment_mesh_by_range,
)

__all__ = [
    "GeneratedMeshResult",
    "MeshGeneratorBackend",
    "MeshGeneratorInterface",
    "segment_mesh_by_range",
]
