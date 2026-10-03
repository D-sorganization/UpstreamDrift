"""Project shared body-part ellipsoid meshes into the calibrated comparison view."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.comparison import ComparisonLayer
from src.motion_capture.reference.registration import project_reference_to_camera
from src.shared.python.body_part_viz.bindings import BindingKind, MarkerBinding
from src.shared.python.body_part_viz.fitters import BetweenTwoMarkersFitter
from src.shared.python.body_part_viz.shapes import EllipsoidShape
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
    length = float(
        np.sqrt((b - a).dot(b - a))
    )  # ⚡ Bolt: ndarray.dot + sqrt is ~2x faster than np.linalg.norm for small 1D arrays
    if length <= 1e-9 or not np.isfinite(radius_ratio) or not 0 < radius_ratio <= 0.5:
        raise ValueError("Segment needs positive length and radius ratio in (0, 0.5]")
    shape = EllipsoidShape(length / 2, length * radius_ratio, length * radius_ratio)
    binding = MarkerBinding(BindingKind.BETWEEN_TWO, ("a", "b"), (length,))
    fitted = BetweenTwoMarkersFitter().fit(shape, binding, {"a": a[None], "b": b[None]})
    return shape.transform(fitted)[0], shape.faces()


def _camera_pose(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(camera, PinholeCamera):
        return camera.rotation_world_from_camera, camera.translation_world_from_camera_m
    extrinsics = camera.extrinsics
    return (
        extrinsics.rotation_world_from_camera,
        extrinsics.translation_world_from_camera_m,
    )


def draw_segment_volumes(
    frame: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    camera: Camera,
    layer: ComparisonLayer,
    loads: Any = None,
    color_scale: Any = None,
) -> np.ndarray:
    """Shade depth-sorted mesh triangles, then composite volume alpha once.

    Missing and zero-length skeleton links cannot define a volume. Near-plane
    crossing triangles are omitted; OpenCV clips polygons at image borders.
    """
    if getattr(layer, "draw_model_volumes", False) and layer.model_volume_opacity > 0:
        from src.shared.python.force_overlay.conversions import SegmentAxis
        from src.shared.python.force_overlay.renderers.opencv_glyphs import (
            PinholeProjector,
        )
        from src.shared.python.force_overlay.renderers.opencv_segments import (
            SegmentShading,
            draw_segment_meshes_on_frame,
            segment_poses_from_axes,
        )

        axes: list[SegmentAxis] = []
        for a, b in edges:
            diff = points[b] - points[a]
            if not (valid[a] and valid[b]) or np.vdot(diff, diff) <= 1e-18:
                continue
            axes.append(
                SegmentAxis(
                    segment=f"edge_{a}_{b}",
                    joint_label=f"joint_{a}",
                    proximal_m=(
                        float(points[a, 0]),
                        float(points[a, 1]),
                        float(points[a, 2]),
                    ),
                    distal_m=(
                        float(points[b, 0]),
                        float(points[b, 1]),
                        float(points[b, 2]),
                    ),
                )
            )
        if axes:
            lengths = [
                float(
                    np.linalg.norm(np.asarray(ax.distal_m) - np.asarray(ax.proximal_m))
                )
                for ax in axes
            ]
            radius_m = max(
                0.02, float(float(np.mean(lengths)) * layer.segment_radius_ratio)
            )
            poses = segment_poses_from_axes(axes, radius_m=radius_m)
            projector = PinholeProjector(camera)
            shading = SegmentShading(opacity=layer.model_volume_opacity)
            result_frame, _ = draw_segment_meshes_on_frame(
                frame,
                poses,
                projector,
                shading=shading,
                loads=loads,
                color_scale=color_scale,
            )
            return result_frame
        return frame

    if not layer.draw_ellipsoids or layer.ellipsoid_opacity <= 0:
        return frame
    rotation, position = _camera_pose(camera)
    triangles = []
    for a, b in edges:
        diff = points[b] - points[a]
        if not (valid[a] and valid[b]) or np.vdot(diff, diff) <= 1e-18:
            continue
        vertices, faces = segment_mesh(points[a], points[b], layer.segment_radius_ratio)
        pixels, visible = project_reference_to_camera(
            vertices, np.ones(len(vertices), dtype=bool), camera, clip_image=False
        )
        camera_vertices = (vertices - position) @ rotation
        for face in faces:
            if not visible[face].all() or np.abs(pixels[face]).max() > 1e7:
                continue
            xyz = camera_vertices[face]
            normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            norm = float(math.sqrt(np.vdot(normal, normal)))
            if norm <= 1e-12:
                continue
            shade = 0.35 + 0.65 * abs(float(normal[2])) / norm
            colour = (
                int(round(layer.colour_bgr[0] * shade)),
                int(round(layer.colour_bgr[1] * shade)),
                int(round(layer.colour_bgr[2] * shade)),
            )
            triangles.append((float(xyz[:, 2].mean()), pixels[face], colour))
    drawn = frame.copy()
    for _, triangle, colour in sorted(
        triangles, key=lambda item: item[0], reverse=True
    ):
        cv2.fillConvexPoly(
            drawn, np.rint(triangle).astype(np.int32), colour, cv2.LINE_AA
        )
    return np.asarray(
        cv2.addWeighted(
            drawn, layer.ellipsoid_opacity, frame, 1 - layer.ellipsoid_opacity, 0
        ),
        dtype=np.uint8,
    )
