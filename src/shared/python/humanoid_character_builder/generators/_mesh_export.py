"""Shared mesh export and validation helpers for character builder backends.

Internal submodule extracted from mesh generator backends to maintain DRY
and keep file sizes within architecture budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from ._mesh_types import segment_mesh_by_range

_MAKEHUMAN_MODIFIER_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def validate_output_path_within_base(output_path: Path, base: Path) -> None:
    """Raise ValueError if output_path is not contained within base."""
    try:
        output_path.resolve().relative_to(base.resolve())
    except ValueError:
        raise ValueError(
            f"Output path {output_path!r} escapes expected base {base!r}"
        ) from None


def validate_makehuman_script_inputs(
    modifiers: dict[str, float],
    output_dir: Path,
    base_output_dir: Path | None = None,
) -> None:
    """Validate generated-script inputs before invoking MakeHuman."""
    output_path = output_dir.resolve()
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output_dir must resolve to a directory")
    if base_output_dir is not None:
        validate_output_path_within_base(output_dir, base_output_dir)
    for key, value in modifiers.items():
        if not isinstance(key, str) or not _MAKEHUMAN_MODIFIER_RE.fullmatch(key):
            raise ValueError(f"Invalid MakeHuman modifier key: {key!r}")
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"Invalid MakeHuman modifier value for {key!r}")


@dataclass(frozen=True)
class MeshExportTarget:
    """Target directories and resources for exporting segmented meshes."""

    visual_dir: Path
    collision_dir: Path
    trimesh_module: Any
    logger: Any


def export_segmented_mesh(
    all_vertices: Any,
    all_faces: Any,
    indices: list[int],
    segment_name: str,
    target: MeshExportTarget,
) -> tuple[Path, Path] | None:
    """Segment vertices/faces by range and export visual and collision STL meshes.

    Returns (visual_path, collision_path) if successful, or None if skipped/failed.
    """
    if not indices:
        return None
    try:
        seg_verts, seg_faces = segment_mesh_by_range(
            all_vertices,
            all_faces,
            min(indices),
            max(indices) + 1,
        )
        if len(seg_verts) == 0:
            return None
        submesh = target.trimesh_module.Trimesh(vertices=seg_verts, faces=seg_faces)
        vpath = target.visual_dir / f"{segment_name}.stl"
        submesh.export(str(vpath))
        cpath = target.collision_dir / f"{segment_name}.stl"
        try:
            submesh.convex_hull.export(str(cpath))
        except (RuntimeError, ValueError, OSError) as hull_exc:
            target.logger.warning(
                "Convex hull failed for %s (%s); using segment mesh as collision",
                segment_name,
                hull_exc,
            )
            submesh.export(str(cpath))
        return vpath, cpath
    except (
        AttributeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        TypeError,
    ) as exc:
        target.logger.warning("Failed to segment %s: %s", segment_name, exc)
        return None


def export_all_segments(
    vertex_groups: dict[str, list[int]],
    allowed_segments: Any,
    all_vertices: Any,
    all_faces: Any,
    target: MeshExportTarget,
    *,
    min_indices: int = 1,
    name_map: dict[str, str] | None = None,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Export segmented visual and collision meshes for each segment."""
    mesh_paths: dict[str, Path] = {}
    collision_paths: dict[str, Path] = {}

    def _iterate_groups() -> list[tuple[str, list[int]]]:
        if name_map is not None:
            return [
                (seg_name, vertex_groups.get(group_key, []))
                for group_key, seg_name in name_map.items()
            ]
        return list(vertex_groups.items())

    for segment_name, indices in _iterate_groups():
        if allowed_segments is not None and segment_name not in allowed_segments:
            continue
        if len(indices) < min_indices:
            continue

        exported = export_segmented_mesh(
            all_vertices=all_vertices,
            all_faces=all_faces,
            indices=indices,
            segment_name=segment_name,
            target=target,
        )
        if exported is not None:
            mesh_paths[segment_name], collision_paths[segment_name] = exported

    return mesh_paths, collision_paths
