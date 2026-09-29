"""
Collision geometry generation for humanoid character builder.

This module provides a simplified interface for collision geometry generation.
It re-exports and adapts functionality from collision_generator.py for backward
compatibility.

.. deprecated::
    This module is maintained for backward compatibility. New code should use
    collision_generator.py directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .collision_generator import CollisionGeometryGenerator as _CoreGenerator
from .collision_generator import (
    CollisionGeometryResult,
    ComplexityLevel,
    PrimitiveFit,
    SimplificationMethod,
    VHACDParameters,
)

if TYPE_CHECKING:
    import trimesh

__all__ = [
    "CollisionGeometry",
    "CollisionGeometryGenerator",
    # Re-export core types for convenience
    "CollisionGeometryResult",
    "SimplificationMethod",
    "ComplexityLevel",
    "PrimitiveFit",
    "VHACDParameters",
]

logger = logging.getLogger(__name__)


@dataclass
class CollisionGeometry:
    """Result of collision geometry generation.

    This is a backward-compatible wrapper around CollisionGeometryResult.
    New code should use CollisionGeometryResult directly.
    """

    meshes: list[trimesh.Trimesh]
    method: str
    quality_score: float = 0.0
    volume_preservation: float = 0.0
    vertex_count: int = 0
    face_count: int = 0
    processing_time: float = 0.0

    def export(self, path: str) -> None:
        """Export collision geometry to file."""
        if path is None:
            raise ValueError("path must be provided")
        if not self.meshes:
            return

        import trimesh

        if len(self.meshes) == 1:
            self.meshes[0].export(path)
        else:
            combined = trimesh.util.concatenate(self.meshes)
            combined.export(path)

    @classmethod
    def from_result(
        cls, result: CollisionGeometryResult, processing_time: float = 0.0
    ) -> CollisionGeometry:
        """Create CollisionGeometry from CollisionGeometryResult.

        Args:
            result: Core generator result
            processing_time: Processing time in seconds

        Returns:
            CollisionGeometry instance
        """
        if result is None:
            raise ValueError("result must be provided")
        total_verts = sum(
            len(m.vertices) if hasattr(m, "vertices") else 0 for m in result.components
        )
        total_faces = sum(
            len(m.faces) if hasattr(m, "faces") else 0 for m in result.components
        )

        return cls(
            meshes=result.components,
            method=result.method_used.name.lower(),
            quality_score=result.volume_preservation,  # Use volume as quality proxy
            volume_preservation=result.volume_preservation,
            vertex_count=total_verts,
            face_count=total_faces,
            processing_time=processing_time,
        )


class CollisionGeometryGenerator:
    """
    Generate optimized collision geometry from visual meshes.

    This class provides a simplified interface to the core CollisionGeometryGenerator.
    It wraps the results in the backward-compatible CollisionGeometry dataclass.

    Supports multiple strategies:
    - Convex Hull Decomposition (VHACD)
    - Primitive Fitting (Box, Sphere, Cylinder, Capsule)
    - Mesh Decimation
    - Hybrid/Auto selection
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self._core = _CoreGenerator()

    def generate(
        self,
        visual_mesh: Any,
        method: str = "auto",  # "primitives", "vhacd", "decimation", "auto"
        target_complexity: str = "balanced",  # "minimal", "balanced", "accurate"
        max_primitives: int = 16,
        max_triangles: int = 500,
    ) -> CollisionGeometry:
        """
        Generate optimized collision geometry from visual mesh.

        Args:
            visual_mesh: Input trimesh object
            method: Simplification method to use
            target_complexity: Quality/Performance tradeoff preset
            max_primitives: Maximum number of hulls/primitives
            max_triangles: Target triangle count for decimation

        Returns:
            CollisionGeometry object containing simplified meshes
        """
        if method is None:
            raise ValueError("method must be provided")
        import time

        start_time = time.time()

        # Delegate to core generator
        result = self._core.generate(
            visual_mesh=visual_mesh,
            method=method,
            target_complexity=target_complexity,
            max_primitives=max_primitives,
            max_triangles=max_triangles,
            max_hulls=max_primitives,  # Map to VHACD parameter
        )

        processing_time = time.time() - start_time

        # Wrap in backward-compatible result
        return CollisionGeometry.from_result(result, processing_time)

    def compute_quality_metrics(
        self, original: Any, generated: list[Any]
    ) -> dict[str, float]:
        """Compute quality metrics comparing generated collision geometry to original.

        Args:
            original: Original mesh
            generated: List of generated meshes

        Returns:
            Dictionary with quality_score and volume_preservation
        """
        if generated is None:
            raise ValueError("generated must be provided")
        import trimesh

        if not generated:
            return {"quality_score": 0.0, "volume_preservation": 0.0}

        combined_gen = trimesh.util.concatenate(generated)

        # Volume preservation
        try:
            vol_orig = original.volume
            vol_gen = combined_gen.volume
            if vol_orig > 1e-6:
                vol_preservation = (
                    min(vol_gen / vol_orig, vol_orig / vol_gen) if vol_gen > 0 else 0
                )
            else:
                vol_preservation = 1.0
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            vol_preservation = 0.0

        # Hausdorff distance (approximate)
        try:
            samples_orig, _ = trimesh.sample.sample_surface(original, 1000)
            samples_gen, _ = trimesh.sample.sample_surface(combined_gen, 1000)

            _, dist_o_g, _ = trimesh.proximity.closest_point(combined_gen, samples_orig)
            _, dist_g_o, _ = trimesh.proximity.closest_point(original, samples_gen)

            import numpy as np

            hausdorff = max(np.max(dist_o_g), np.max(dist_g_o))

            scale = original.scale
            normalized_hausdorff = hausdorff / scale if scale > 0 else hausdorff

            quality_score = max(0.0, 1.0 - normalized_hausdorff * 10)
            quality_score = (quality_score + vol_preservation) / 2.0

        except ImportError as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            quality_score = 0.0

        return {"quality_score": quality_score, "volume_preservation": vol_preservation}
