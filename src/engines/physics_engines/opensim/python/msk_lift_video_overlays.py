"""Lift video HUD overlays and wrench extraction (OSL-8, #13045)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.force_overlay.contracts import OverlayWrench, WrenchKind


def muscle_color_rgb(activation: float) -> tuple[float, float, float]:
    """Color a muscle by activation from blue (0.0) to red (1.0)."""
    val = float(np.clip(activation, 0.0, 1.0))
    return (val, 0.0, 1.0 - val)


def _vec3(arr: Any) -> tuple[float, float, float]:
    """Convert a 3-sequence to a typed float 3-tuple."""
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def extract_lift_wrenches(targets: Any, frame_idx: int) -> list[OverlayWrench]:
    """Extract 4 OverlayWrenches (2 foot GRF, 2 hand bar wrenches) for frame_idx."""
    lms = targets.landmarks(frame_idx)
    mid_foot_l = _vec3((lms["heel_l"] + lms["forefoot_l"]) / 2.0)
    mid_foot_r = _vec3((lms["heel_r"] + lms["forefoot_r"]) / 2.0)

    grf_l_f = _vec3(targets.grf_force_n["left"][frame_idx])
    grf_r_f = _vec3(targets.grf_force_n["right"][frame_idx])

    grip_l_p = _vec3(targets.grip_m["L"][frame_idx])
    grip_r_p = _vec3(targets.grip_m["R"][frame_idx])

    grip_l_f = _vec3(targets.hand_force_n["L"][frame_idx])
    grip_r_f = _vec3(targets.hand_force_n["R"][frame_idx])

    grip_l_tau = _vec3(targets.hand_torque_nm["L"][frame_idx])
    grip_r_tau = _vec3(targets.hand_torque_nm["R"][frame_idx])

    return [
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:grf_left",
            body="calcn_l",
            point_m=mid_foot_l,
            force_n=grf_l_f,
            source="opensim_lift",
        ),
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:grf_right",
            body="calcn_r",
            point_m=mid_foot_r,
            force_n=grf_r_f,
            source="opensim_lift",
        ),
        OverlayWrench(
            kind=WrenchKind.GRIP,
            label="grip:hand_l",
            body="hand_l",
            point_m=grip_l_p,
            force_n=grip_l_f,
            torque_nm=grip_l_tau,
            source="opensim_lift",
        ),
        OverlayWrench(
            kind=WrenchKind.GRIP,
            label="grip:hand_r",
            body="hand_r",
            point_m=grip_r_p,
            force_n=grip_r_f,
            torque_nm=grip_r_tau,
            source="opensim_lift",
        ),
    ]


def _draw_legend_arrows_and_scale(
    out: np.ndarray,
    bx: int,
    by: int,
    font: int,
    scale: float,
    color_text: tuple[int, int, int],
    force_scale_n: float,
    force_scale_px: int,
) -> None:
    import cv2

    # GRF (Green)
    cv2.circle(out, (bx + 18, by + 40), 5, (60, 220, 60), -1, cv2.LINE_AA)
    cv2.putText(
        out,
        "GRF contact force",
        (bx + 32, by + 44),
        font,
        scale,
        color_text,
        1,
        cv2.LINE_AA,
    )

    # Grip (Blue)
    cv2.circle(out, (bx + 18, by + 60), 5, (60, 160, 255), -1, cv2.LINE_AA)
    cv2.putText(
        out,
        "Hand grip wrench",
        (bx + 32, by + 64),
        font,
        scale,
        color_text,
        1,
        cv2.LINE_AA,
    )

    # Force scale bar
    sx0, sy0 = bx + 18, by + 82
    sx1 = sx0 + int(force_scale_px)
    cv2.line(out, (sx0, sy0), (sx1, sy0), (220, 220, 220), 2, cv2.LINE_AA)
    cv2.line(out, (sx0, sy0 - 3), (sx0, sy0 + 3), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(out, (sx1, sy0 - 3), (sx1, sy0 + 3), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"{int(force_scale_n)} N scale",
        (sx1 + 10, sy0 + 4),
        font,
        scale,
        color_text,
        1,
        cv2.LINE_AA,
    )


def _draw_legend_activation_bar(
    out: np.ndarray,
    bx: int,
    by: int,
    font: int,
    color_dim: tuple[int, int, int],
) -> None:
    import cv2

    ax0, ay0, aw, ah = bx + 18, by + 102, 130, 10
    grad = np.zeros((ah, aw, 3), dtype=np.uint8)
    for c in range(aw):
        val = c / max(1.0, float(aw - 1))
        grad[:, c, 0] = int(val * 255)
        grad[:, c, 1] = 0
        grad[:, c, 2] = int((1.0 - val) * 255)
    out[ay0 : ay0 + ah, ax0 : ax0 + aw] = grad
    cv2.rectangle(
        out, (ax0, ay0), (ax0 + aw, ay0 + ah), (150, 150, 150), 1, cv2.LINE_AA
    )
    cv2.putText(out, "0.0", (ax0, ay0 + ah + 12), font, 0.35, color_dim, 1, cv2.LINE_AA)
    cv2.putText(
        out,
        "Activation",
        (ax0 + 40, ay0 + ah + 12),
        font,
        0.35,
        color_dim,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "1.0",
        (ax0 + aw - 16, ay0 + ah + 12),
        font,
        0.35,
        color_dim,
        1,
        cv2.LINE_AA,
    )


def draw_lift_legend(
    frame_rgb: np.ndarray,
    lift: str,
    time_s: float,
    in_activation_window: bool,
    force_scale_n: float = 500.0,
    force_scale_px: int = 50,
) -> np.ndarray:
    """Draw 2D HUD legend overlay on an RGB video frame.

    Displays:
    1. Title with lift name and current playback timestamp.
    2. Arrow indicators for per-foot GRF contact force (green) and hand grip wrench (blue).
    3. Scaled force reference bar in Newtons (N).
    4. Muscle activation color gradient bar from 0.0 (blue) to 1.0 (red).
    5. OSL-4 activation window status indicator.
    """
    import cv2

    out = frame_rgb.copy()
    h, w = out.shape[:2]
    bx, by, bw, bh = 15, 15, 270, 160
    if bx + bw > w or by + bh > h:
        return out

    sub = out[by : by + bh, bx : bx + bw]
    overlay = np.zeros_like(sub)
    out[by : by + bh, bx : bx + bw] = cv2.addWeighted(overlay, 0.7, sub, 0.3, 0)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    color_text = (230, 230, 230)
    color_dim = (170, 170, 170)

    # 1. Title
    lift_title = lift.replace("_", " ").title()
    title = f"{lift_title} | t = {time_s:.2f} s"
    cv2.putText(
        out, title, (bx + 12, by + 20), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA
    )

    # 2. Arrows & force scale
    _draw_legend_arrows_and_scale(
        out, bx, by, font, scale, color_text, force_scale_n, force_scale_px
    )

    # 3. Activation bar
    _draw_legend_activation_bar(out, bx, by, font, color_dim)

    # 4. Status line
    if in_activation_window:
        status_txt = "activation: OSL-4 active"
        status_col = (100, 255, 100)
    else:
        status_txt = "activation: OSL-4 window only"
        status_col = (200, 200, 120)
    cv2.putText(
        out,
        status_txt,
        (bx + 12, by + bh - 10),
        font,
        0.38,
        status_col,
        1,
        cv2.LINE_AA,
    )

    return out
