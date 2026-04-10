from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from ._cg_types import (
    CollisionGeometryResult,
    SimplificationMethod,
    VHACDParameters,
)

logger = logging.getLogger(__name__)


def is_roughly_convex(mesh: Any, threshold: float = 0.95) -> bool:
    try:
        convex = mesh.convex_hull
        volume_ratio = mesh.volume / convex.volume
        return bool(volume_ratio > threshold)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return False


def generate_convex_hull(mesh: Any) -> CollisionGeometryResult:
    hull = mesh.convex_hull

    return CollisionGeometryResult(
        success=True,
        method_used=SimplificationMethod.CONVEX_HULL,
        components=[hull],
        original_triangles=len(mesh.faces),
        final_triangles=len(hull.faces),
        reduction_ratio=1.0 - len(hull.faces) / len(mesh.faces),
        volume_preservation=mesh.volume / hull.volume if hull.volume > 0 else 1.0,
        hausdorff_distance=0.0,
    )


def _vhacd_pybullet(mesh: Any, params: VHACDParameters) -> list[Any]:
    if not (params is not None):
        raise ValueError("params must be provided")
    if not (params is not None):
        raise ValueError("params must be provided")
    import pybullet as p
    import trimesh

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.obj")
        output_path = os.path.join(tmpdir, "output.obj")

        mesh.export(input_path)

        p.vhacd(
            input_path,
            output_path,
            os.path.join(tmpdir, "log.txt"),
            maxNumVerticesPerCH=params.max_vertices_per_hull,
            resolution=params.resolution,
            concavity=params.concavity,
        )

        result = trimesh.load(output_path)
        if isinstance(result, trimesh.Scene):
            return list(result.geometry.values())
        return [result]


def generate_vhacd(
    mesh: Any,
    max_hulls: int,
    vhacd_params: VHACDParameters | None,
) -> CollisionGeometryResult:
    if not (max_hulls is not None):
        raise ValueError("max_hulls must be provided")
    if not (max_hulls is not None):
        raise ValueError("max_hulls must be provided")
    import trimesh

    params = vhacd_params or VHACDParameters(max_hulls=max_hulls)

    try:
        if hasattr(trimesh.interfaces, "vhacd"):
            convex_hulls = trimesh.interfaces.vhacd.convex_decomposition(
                mesh,
                maxhulls=params.max_hulls,
                resolution=params.resolution,
            )
        else:
            convex_hulls = _vhacd_pybullet(mesh, params)

        if not convex_hulls:
            raise ValueError("VHACD produced no output")

        return CollisionGeometryResult(
            success=True,
            method_used=SimplificationMethod.VHACD,
            components=list(convex_hulls),
            original_triangles=len(mesh.faces),
            final_triangles=sum(len(h.faces) for h in convex_hulls),
            reduction_ratio=0.0,
            volume_preservation=1.0,
            hausdorff_distance=0.0,
        )

    except (ValueError, TypeError, RuntimeError, OSError) as e:
        logger.warning(f"VHACD failed, falling back to convex hull: {e}")
        return generate_convex_hull(mesh)


__all__ = [
    "generate_convex_hull",
    "generate_vhacd",
    "is_roughly_convex",
]
