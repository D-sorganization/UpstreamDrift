"""Interpolate keyframes into a deterministic frame plan."""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.grip_build.receipt import BuildStatus
from src.tools.grip_build_preview.settings import PreviewSettings
from src.tools.grip_build_preview.stage_timeline import StageKeyframe


@dataclass(frozen=True)
class PreviewFramePlan:
    joint_angles_rad: dict[str, dict[str, float]]
    stage_id: str
    stage_label: str
    status: BuildStatus
    keyframe_index: int
    clearance_m: float | None
    tolerance_m: float | None
    status_color_rgb: tuple[int, int, int]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interpolate_angles(
    start: dict[str, dict[str, float]], end: dict[str, dict[str, float]], t: float
) -> dict[str, dict[str, float]]:
    fingers = sorted(set(start) | set(end))
    out: dict[str, dict[str, float]] = {}
    for finger in fingers:
        s = start.get(finger, {"mcp": 0.0, "pip": 0.0, "dip": 0.0})
        e = end.get(finger, s)
        out[finger] = {
            joint: _lerp(s[joint], e[joint], t) for joint in ("mcp", "pip", "dip")
        }
    return out


def sample_preview_frames(
    keyframes: tuple[StageKeyframe, ...],
    settings: PreviewSettings,
) -> tuple[PreviewFramePlan, ...]:
    if len(keyframes) < 1:
        raise ValueError("keyframes must not be empty")
    steps = settings.frames_per_segment
    if steps < 1:
        raise ValueError("frames_per_segment must be >= 1")
    plans: list[PreviewFramePlan] = []
    for segment in range(len(keyframes) - 1):
        left = keyframes[segment]
        right = keyframes[segment + 1]
        for step in range(steps):
            t = step / steps
            angles = _interpolate_angles(
                left.joint_angles_rad, right.joint_angles_rad, t
            )
            meta = right if t >= 0.5 else left
            plans.append(
                PreviewFramePlan(
                    joint_angles_rad=angles,
                    stage_id=meta.stage_id,
                    stage_label=meta.stage_id.replace("_", " "),
                    status=meta.status,
                    keyframe_index=segment if t < 0.5 else segment + 1,
                    clearance_m=meta.clearance_m,
                    tolerance_m=meta.tolerance_m,
                    status_color_rgb=settings.color_for_status(meta.status),
                )
            )
    last = keyframes[-1]
    plans.append(
        PreviewFramePlan(
            joint_angles_rad=last.joint_angles_rad,
            stage_id=last.stage_id,
            stage_label=last.stage_id.replace("_", " "),
            status=last.status,
            keyframe_index=len(keyframes) - 1,
            clearance_m=last.clearance_m,
            tolerance_m=last.tolerance_m,
            status_color_rgb=settings.color_for_status(last.status),
        )
    )
    return tuple(plans)
