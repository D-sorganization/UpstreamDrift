"""Headless 2D renderer (y-z plane); overlays via native_viewer_export compositor."""

from __future__ import annotations

import numpy as np

from src.shared.python.grip_build.club_sdf import ClubSDF
from src.shared.python.grip_build.hand_chain import HandChain
from src.shared.python.grip_contact.grip_geometry import GripGeometry, get_grip_preset
from src.shared.python.grip_spec.types import GripSpec
from src.tools.grip_build_preview.frame_plan import PreviewFramePlan
from src.tools.grip_build_preview.settings import PreviewSettings
from src.tools.grip_build_preview.stage_timeline import hand_chain_for_spec
from src.tools.native_viewer_export.compositor import draw_label


def _apply_plan_to_chain(chain: HandChain, plan: PreviewFramePlan) -> None:
    for finger, joints in plan.joint_angles_rad.items():
        if finger not in chain.joint_angles:
            continue
        angles = chain.joint_angles[finger]
        angles.mcp_rad = joints["mcp"]
        angles.pip_rad = joints["pip"]
        angles.dip_rad = joints["dip"]
        chain.apply_dip_coupling(finger)


def _world_to_pixel(
    y_m: float,
    z_m: float,
    settings: PreviewSettings,
    *,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> tuple[int, int]:
    width = settings.width_px
    height = settings.height_px
    margin = settings.margin_m
    span_y = (y_max - y_min) + 2 * margin
    span_z = (z_max - z_min) + 2 * margin
    scale = min(width / span_y, height / span_z)
    py = int((y_m - y_min + margin) * scale)
    px = int(height - (z_m - z_min + margin) * scale)
    return (max(0, min(width - 1, py)), max(0, min(height - 1, px)))


def render_preview_frame(
    spec: GripSpec,
    plan: PreviewFramePlan,
    settings: PreviewSettings,
    *,
    geometry: GripGeometry | None = None,
) -> np.ndarray:
    if not isinstance(plan, PreviewFramePlan):
        raise TypeError("plan must be PreviewFramePlan")
    geometry = geometry if geometry is not None else get_grip_preset()
    chain = hand_chain_for_spec(spec, geometry=geometry)
    _apply_plan_to_chain(chain, plan)
    club = ClubSDF(geometry=geometry, axial_offset_m=spec.lead.index_contact_station)
    axis0, axis1 = club.axis_segment()

    capsules = chain.all_capsules()
    y_vals = [float(c[2][1]) for c in capsules] + [float(c[3][1]) for c in capsules]
    z_vals = [float(c[2][2]) for c in capsules] + [float(c[3][2]) for c in capsules]
    y_vals.extend([float(axis0[1]), float(axis1[1])])
    z_vals.extend([float(axis0[2]), float(axis1[2])])
    y_min, y_max = min(y_vals), max(y_vals)
    z_min, z_max = min(z_vals), max(z_vals)

    img = np.zeros((settings.height_px, settings.width_px, 3), dtype=np.uint8)
    img[:, :] = np.array(settings.background_rgb, dtype=np.uint8)

    def draw_capsule(p0: np.ndarray, p1: np.ndarray, rgb: tuple[int, int, int]) -> None:
        x0, y0 = _world_to_pixel(
            float(p0[1]),
            float(p0[2]),
            settings,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        x1, y1 = _world_to_pixel(
            float(p1[1]),
            float(p1[2]),
            settings,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            img[y, x] = rgb

    draw_capsule(axis0, axis1, settings.club_line_rgb)

    for _finger, _seg, p0, p1, _r in capsules:
        draw_capsule(p0, p1, settings.bone_capsule_rgb)

    if capsules:
        _finger, _seg, _p0, tip, _r = capsules[-1]
        tx, ty = _world_to_pixel(
            float(tip[1]),
            float(tip[2]),
            settings,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        img[ty, tx] = settings.contact_marker_rgb

    status_bar = np.array(plan.status_color_rgb, dtype=np.uint8)
    img[0, :8] = status_bar
    img[:8, 0] = status_bar

    label = plan.stage_label
    if plan.clearance_m is not None and plan.tolerance_m is not None:
        gap = plan.clearance_m
        tol = plan.tolerance_m
        label = f"{label}  gap={gap * 1e3:.2f}mm tol={tol * 1e3:.2f}mm"
    return draw_label(
        img,
        label,
        (12, settings.height_px - 28),
        size=14,
        fill=(
            int(plan.status_color_rgb[0]),
            int(plan.status_color_rgb[1]),
            int(plan.status_color_rgb[2]),
        ),
    )
