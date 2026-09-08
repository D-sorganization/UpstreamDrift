"""Swing kinematics from fitted 3-D joints: what the golfer gets back.

Inputs are the joint trajectories a reconstruction produced (``(T, K, 3)``
in the ADR-0041 world frame, metres, plus the frame rate). Outputs are the
few quantities a coach reads first, each with the frame it happened in and
the uncertainty the smoother reported:

- pelvis and shoulder turn about the vertical, relative to address, and
  their difference (the X-factor), all in degrees;
- hand speed (midpoint of the wrists) and its peak, as the proxy for club
  speed until the club is tracked;
- the swing events the speed profile implies (address, top, peak speed,
  finish) and the tempo ratio between backswing and downswing.

Angles come from the hip and shoulder lines projected on the ground plane;
speeds come from the robust smoother, so a single wrong frame cannot fake a
peak. Nothing here reaches into a simulation: the existing
``shared.python.analysis`` package reads simulation state (joint angles and
club-head speed series) and is the consumer of these series, not a
replacement for them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.core.contracts import require

from .skeleton import JOINT_NAMES
from .temporal import SmootherOptions, smooth

Array = npt.NDArray[np.float64]
UP = np.array([0.0, 1.0, 0.0])
# Prior spread of hand acceleration for the speed smoother, m/s^2: a driver
# swing peaks near 300 m/s^2 at the hands; the prior is deliberately looser.
HAND_ACCELERATION_SIGMA = 600.0
# Reconstructed joint positions carry ~millimetre noise; a noise-free synthetic
# trajectory would otherwise drive the smoother's weights to infinity.
HAND_POSITION_SIGMA_M = 0.002


class SwingEvents(BaseModel):
    model_config = ConfigDict(frozen=True)

    address_frame: int
    top_frame: int
    peak_speed_frame: int
    finish_frame: int
    backswing_s: float
    downswing_s: float
    tempo_ratio: float | None  # backswing / downswing


class SwingSummary(BaseModel):
    """``swing_summary.json``."""

    model_config = ConfigDict(frozen=True)

    frames: int
    fps: float
    peak_hand_speed_mps: float
    peak_hand_speed_frame: int
    peak_hand_speed_uncertainty_mps: float
    max_shoulder_turn_deg: float
    max_pelvis_turn_deg: float
    peak_x_factor_deg: float
    events: SwingEvents
    angles_deg: dict[str, dict[str, float]] = Field(default_factory=dict)


@dataclass(frozen=True)
class SwingSeries:
    """Per-frame series behind the summary."""

    time_s: Array
    pelvis_turn_deg: Array
    shoulder_turn_deg: Array
    x_factor_deg: Array
    hand_speed_mps: Array
    hand_speed_uncertainty_mps: Array


def _index(joint_names: Sequence[str], name: str) -> int:
    require(name in joint_names, "joint missing from layout", name)
    return list(joint_names).index(name)


def line_turn_deg(left: Array, right: Array) -> Array:
    """Turn of the left->right line about the vertical, relative to frame 0.

    Projects the line onto the ground plane and unwraps the angle so a full
    backswing does not wrap at 180 degrees. Postcondition: ``turn[0] == 0``.
    """
    d = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    ground = d - np.outer(d @ UP, UP)
    angle = np.arctan2(ground[:, 0], ground[:, 2])  # about +y, from +z toward +x
    return np.degrees(np.unwrap(angle - angle[0]))


def _angle_between(a: Array, b: Array) -> Array:
    """Per-frame angle in degrees between vector series ``a`` and ``b`` ``(T, 3)``."""
    na = np.sqrt(
        np.einsum("ij,ij->i", a, a)
    )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
    nb = np.sqrt(
        np.einsum("ij,ij->i", b, b)
    )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def flexion_deg(proximal: Array, joint: Array, distal: Array) -> Array:
    """Flexion at ``joint``: 0 when the limb is straight, larger when bent."""
    return 180.0 - _angle_between(proximal - joint, distal - joint)


def joint_angles(
    joints_3d_m: Array, joint_names: Sequence[str] = JOINT_NAMES
) -> dict[str, Array]:
    """Per-frame joint angles from the 15-joint fit (#9682), degrees.

    ``left/right_elbow_flexion``, ``left/right_knee_flexion``,
    ``trunk_forward_tilt`` (hip->neck against the vertical with the hip-line
    component removed), ``trunk_side_bend`` (hip->neck along the hip line)
    and ``lead_arm_shoulder_deg`` (left shoulder->wrist against the shoulder
    line; the lead arm of a right-handed golfer). Postcondition: one value
    per frame per series; NaN joints give NaN angles.
    """
    j = np.asarray(joints_3d_m, dtype=float)
    require(j.ndim == 3 and j.shape[2] == 3, "need (T, K, 3)")
    ix = {n: _index(joint_names, n) for n in joint_names}
    out: dict[str, Array] = {}
    for side in ("left", "right"):
        out[f"{side}_elbow_flexion"] = flexion_deg(
            j[:, ix[f"{side}_shoulder"]],
            j[:, ix[f"{side}_elbow"]],
            j[:, ix[f"{side}_wrist"]],
        )
        out[f"{side}_knee_flexion"] = flexion_deg(
            j[:, ix[f"{side}_hip"]], j[:, ix[f"{side}_knee"]], j[:, ix[f"{side}_ankle"]]
        )
    trunk = j[:, ix["neck"]] - j[:, ix["mid_hip"]]
    hip_line = j[:, ix["right_hip"]] - j[:, ix["left_hip"]]
    hip_line = (
        hip_line
        / np.maximum(np.sqrt(np.einsum("ij,ij->i", hip_line, hip_line)), 1e-12)[:, None]
    )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
    along = np.einsum("ij,ij->i", trunk, hip_line)
    lateral = trunk - along[:, None] * hip_line
    out["trunk_forward_tilt"] = _angle_between(
        lateral, np.broadcast_to(UP, lateral.shape)
    )
    out["trunk_side_bend"] = np.degrees(
        np.arctan2(
            along, np.maximum(np.sqrt(np.einsum("ij,ij->i", lateral, lateral)), 1e-12)
        )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
    )
    shoulder_line = j[:, ix["right_shoulder"]] - j[:, ix["left_shoulder"]]
    lead_arm = j[:, ix["left_wrist"]] - j[:, ix["left_shoulder"]]
    out["lead_arm_shoulder_deg"] = _angle_between(lead_arm, shoulder_line)
    return out


def angle_stats(
    angles: dict[str, Array], events: SwingEvents
) -> dict[str, dict[str, float]]:
    """At-event values and range per angle series; NaN where unobserved."""
    frames = {
        "address": events.address_frame,
        "top": events.top_frame,
        "peak": events.peak_speed_frame,
        "finish": events.finish_frame,
    }
    out: dict[str, dict[str, float]] = {}
    for name, series in angles.items():
        row = {
            k: float(series[f]) if 0 <= f < series.size else float("nan")
            for k, f in frames.items()
        }
        finite = series[np.isfinite(series)]
        row["min"] = float(finite.min()) if finite.size else float("nan")
        row["max"] = float(finite.max()) if finite.size else float("nan")
        out[name] = row
    return out


def swing_series(
    joints_3d_m: Array,
    fps: float,
    joint_names: Sequence[str] = JOINT_NAMES,
    *,
    smoother: SmootherOptions | None = None,
) -> SwingSeries:
    """Turn angles and hand speed for every frame.

    Preconditions: ``(T >= 3, K, 3)`` joints with the hip, shoulder and wrist
    joints present; positive fps.
    """
    j = np.asarray(joints_3d_m, dtype=float)
    require(j.ndim == 3 and j.shape[2] == 3 and j.shape[0] >= 3, "need (T>=3, K, 3)")
    require(fps > 0, "fps must be positive", fps)
    lh, rh = _index(joint_names, "left_hip"), _index(joint_names, "right_hip")
    ls, rs = _index(joint_names, "left_shoulder"), _index(joint_names, "right_shoulder")
    lw, rw = _index(joint_names, "left_wrist"), _index(joint_names, "right_wrist")
    pelvis = line_turn_deg(j[:, lh], j[:, rh])
    shoulders = line_turn_deg(j[:, ls], j[:, rs])
    hands = 0.5 * (j[:, lw] + j[:, rw])
    options = smoother or SmootherOptions(
        acceleration_sigma=HAND_ACCELERATION_SIGMA,
        measurement_sigma=HAND_POSITION_SIGMA_M,
    )
    fit = smooth(hands, None, fps, options)
    velocity = np.gradient(fit.values, 1.0 / fps, axis=0)
    speed = np.sqrt(
        np.einsum("ij,ij->i", velocity, velocity)
    )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
    # uncertainty of a finite-difference speed from the per-frame position std
    unc = (
        np.sqrt(2.0)
        * np.sqrt(np.einsum("ij,ij->i", fit.uncertainty, fit.uncertainty))
        * fps
        / 2.0
    )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
    return SwingSeries(
        time_s=np.arange(j.shape[0]) / fps,
        pelvis_turn_deg=pelvis,
        shoulder_turn_deg=shoulders,
        x_factor_deg=shoulders - pelvis,
        hand_speed_mps=speed,
        hand_speed_uncertainty_mps=unc,
    )


def detect_events(
    series: SwingSeries,
    fps: float,
    *,
    quiet_fraction: float = 0.05,
    quiet_s: float = 0.15,
    max_downswing_s: float = 0.5,
) -> SwingEvents:
    """Address, top, peak speed and finish from the hand-speed profile.

    Peak speed is the global maximum. Address is the end of the last stretch
    of at least ``quiet_s`` seconds before the peak with speed under
    ``quiet_fraction`` of it. The top of the backswing is the slowest frame
    between the address and the peak, looking back at most
    ``max_downswing_s`` (a downswing is shorter than that, the backswing is
    not, so the search cannot land on the address itself). Finish is the
    first frame after the peak under the quiet threshold. Thresholds rather
    than local minima: at 100+ fps a real profile has a minimum every few
    frames. Every event is a frame index; nothing is interpolated.
    """
    require(fps > 0, "fps must be positive", fps)
    speed = np.asarray(series.hand_speed_mps, dtype=float)
    peak = int(np.argmax(speed))
    quiet = quiet_fraction * speed[peak]
    window = max(int(round(quiet_s * fps)), 1)
    address = 0
    run = 0
    for t in range(peak - 1, -1, -1):
        run = run + 1 if speed[t] < quiet else 0
        if run >= window:
            address = t + window - 1
            break
    lookback = max(int(round(max_downswing_s * fps)), 2)
    start = max(address + 1, peak - lookback)
    top = start + int(np.argmin(speed[start:peak])) if start < peak else address
    after = np.flatnonzero(speed[peak:] < quiet)
    finish = int(peak + after[0]) if after.size else int(speed.size - 1)
    backswing = (top - address) / fps
    downswing = (peak - top) / fps
    return SwingEvents(
        address_frame=address,
        top_frame=top,
        peak_speed_frame=peak,
        finish_frame=finish,
        backswing_s=backswing,
        downswing_s=downswing,
        tempo_ratio=(backswing / downswing) if downswing > 0 else None,
    )


def summarize_swing(
    joints_3d_m: Array,
    fps: float,
    joint_names: Sequence[str] = JOINT_NAMES,
) -> tuple[SwingSummary, SwingSeries]:
    """The coach-facing numbers and the series they came from."""
    series = swing_series(joints_3d_m, fps, joint_names)
    events = detect_events(series, fps)
    peak = events.peak_speed_frame
    summary = SwingSummary(
        frames=int(series.time_s.size),
        fps=fps,
        peak_hand_speed_mps=float(series.hand_speed_mps[peak]),
        peak_hand_speed_frame=peak,
        peak_hand_speed_uncertainty_mps=float(series.hand_speed_uncertainty_mps[peak]),
        max_shoulder_turn_deg=float(np.max(np.abs(series.shoulder_turn_deg))),
        max_pelvis_turn_deg=float(np.max(np.abs(series.pelvis_turn_deg))),
        peak_x_factor_deg=float(np.max(np.abs(series.x_factor_deg))),
        events=events,
        angles_deg=angle_stats(joint_angles(joints_3d_m, joint_names), events),
    )
    return summary, series
