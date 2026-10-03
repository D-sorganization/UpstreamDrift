"""Project shared body-part segment meshes into the calibrated comparison view."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.comparison import ComparisonLayer
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.bindings import BindingKind, MarkerBinding
from src.shared.python.body_part_viz.fitters import BetweenTwoMarkersFitter
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.shapes import EllipsoidShape
from src.shared.python.force_overlay.conversions import SegmentAxis
from src.shared.python.force_overlay.renderers.opencv_glyphs import PinholeProjector
from src.shared.python.force_overlay.renderers.opencv_segments import (
    SegmentPose,
    SegmentShading,
    draw_segment_meshes_on_frame,
    segment_poses_from_axes,
)
from src.shared.python.pose_estimation.observations import CameraCalibration

Camera = PinholeCamera | CameraCalibration


def segment_mesh(
    a: np.ndarray, b: np.ndarray, radius_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """An illustrative ellipsoid with endpoints a/b and radius ratio * length.

    Reuses shared mesh and attachment math. These are display volumes, not
    estimated anatomy, inertial geometry or uncertainty confidence regions.
    """
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
        raise ValueError("Segment endpoints must be finite 3-vectors")
    diff = b - a
    length = float(np.sqrt(diff.dot(diff)))
    if length <= 1e-9 or not np.isfinite(radius_ratio) or not 0 < radius_ratio <= 0.5:
        raise ValueError("Segment needs positive length and radius ratio in (0, 0.5]")
    shape = EllipsoidShape(length / 2, length * radius_ratio, length * radius_ratio)
    binding = MarkerBinding(BindingKind.BETWEEN_TWO, ("a", "b"), (length,))
    fitted = BetweenTwoMarkersFitter().fit(shape, binding, {"a": a[None], "b": b[None]})
    return shape.transform(fitted)[0], shape.faces()


def draw_segment_volumes(
    frame: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    camera: Camera,
    layer: ComparisonLayer,
    *,
    loads: AxialLoadFrame | None = None,
    color_scale: ForceColorScale | None = None,
) -> np.ndarray:
    """Shade depth-sorted segment meshes, then composite volume alpha once (FTO-26).

    Replaces separate volume renderers with the shared OpenCV segment renderer.
    """
    draw_vol = layer.draw_ellipsoids or getattr(layer, "draw_model_volumes", False)
    if not draw_vol or layer.ellipsoid_opacity <= 0 or camera is None:
        return frame

    axes: list[SegmentAxis] = []
    for a, b in edges:
        if not (valid[a] and valid[b]):
            continue
        p = points[a]
        d = points[b]
        diff = d - p
        if np.vdot(diff, diff) <= 1e-12:
            continue
        axes.append(
            SegmentAxis(
                segment=f"segment_{a}_{b}",
                joint_label=f"joint_{a}_{b}",
                proximal_m=(float(p[0]), float(p[1]), float(p[2])),
                distal_m=(float(d[0]), float(d[1]), float(d[2])),
            )
        )

    if not axes:
        return frame

    poses: list[SegmentPose] = []
    for axis in axes:
        p = np.asarray(axis.proximal_m, dtype=float)
        d = np.asarray(axis.distal_m, dtype=float)
        diff = d - p
        length = float(np.linalg.norm(diff))
        radius = length * layer.segment_radius_ratio
        sub_poses = segment_poses_from_axes([axis], radius_m=radius)
        for sp in sub_poses:
            poses.append(
                SegmentPose(
                    name=sp.name,
                    mesh_id=sp.mesh_id,
                    T_world_segment=sp.T_world_segment,
                    scale=sp.scale,
                    base_color=layer.colour,
                )
            )

    projector = PinholeProjector(camera)
    shading = SegmentShading(ambient=0.35, opacity=layer.ellipsoid_opacity)
    receipt = draw_segment_meshes_on_frame(
        frame,
        poses,
        projector,
        shading=shading,
        loads=loads,
        color_scale=color_scale,
    )
    return receipt.frame if receipt.frame is not None else frame
