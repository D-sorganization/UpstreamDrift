from __future__ import annotations

import logging
from typing import Any

from ._cg_primitive_fitting import generate_primitives
from ._cg_types import CollisionGeometryResult, SimplificationMethod

logger = logging.getLogger(__name__)


def generate_decimated(mesh: Any, max_triangles: int) -> CollisionGeometryResult:
    if not (max_triangles is not None):
        raise ValueError("max_triangles must be provided")
    if not (max_triangles is not None):
        raise ValueError("max_triangles must be provided")
    if len(mesh.faces) <= max_triangles:
        return CollisionGeometryResult(
            success=True,
            method_used=SimplificationMethod.DECIMATION,
            components=[mesh.copy()],
            original_triangles=len(mesh.faces),
            final_triangles=len(mesh.faces),
            reduction_ratio=0.0,
            volume_preservation=1.0,
            hausdorff_distance=0.0,
        )

    try:
        simplified = mesh.simplify_quadric_decimation(max_triangles)
    except (ValueError, RuntimeError, IndexError):
        try:
            reduction = max_triangles / len(mesh.faces)
            pitch = mesh.extents.max() * (1 - reduction) / 10
            voxelized = mesh.voxelized(pitch)
            simplified = voxelized.marching_cubes
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            simplified = mesh.copy()

    return CollisionGeometryResult(
        success=True,
        method_used=SimplificationMethod.DECIMATION,
        components=[simplified],
        original_triangles=len(mesh.faces),
        final_triangles=len(simplified.faces),
        reduction_ratio=1.0 - len(simplified.faces) / len(mesh.faces),
        volume_preservation=(
            simplified.volume / mesh.volume if mesh.volume > 0 else 1.0
        ),
        hausdorff_distance=0.0,
    )


def generate_hybrid(
    mesh: Any,
    max_primitives: int,
    max_triangles: int,
) -> CollisionGeometryResult:
    if not (max_primitives is not None):
        raise ValueError("max_primitives must be provided")
    if not (max_primitives is not None):
        raise ValueError("max_primitives must be provided")
    prim_result = generate_primitives(mesh, max_primitives)

    if prim_result.primitive_fits and prim_result.primitive_fits[0].volume_ratio > 0.8:
        return prim_result

    return generate_decimated(mesh, max_triangles)


__all__ = [
    "generate_decimated",
    "generate_hybrid",
]
