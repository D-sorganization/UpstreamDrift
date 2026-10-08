"""Extract launch-monitor-style impact parameters (GCV-15).

See the package docstring for the binding definitions.  Every field is either
computed or listed in ``ImpactParameters.unavailable`` with a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .clubhead_series import ClubheadSeries
from .target_frame import TargetFrame
from .tools_gateway import (
    ToolsDeliveryEstimate,
    ToolsDeliveryGateway,
    ToolsDeliveryUnavailableError,
    load_tools_delivery_gateway,
)

MPS_TO_MPH = 2.2369362920544
PLANE_WINDOW_S = 0.020
_EPS = 1e-12
_COLLINEAR_RATIO = 1e-3
TOOLS_AGREEMENT_TOL_DEG = 0.01


@dataclass(frozen=True)
class ImpactParameters:
    """Impact parameters at the last pre-contact state.

    Angles are degrees; ``None`` means unavailable and ``unavailable[name]``
    carries the reason.  ``frame`` is the recorded target frame.
    """

    frame: dict[str, object]
    impact_index: int
    impact_time_s: float
    impact_time_source: str
    clubhead_speed_mps: float
    clubhead_speed_mph: float
    attack_angle_deg: float
    club_path_deg: float | None = None
    face_angle_deg: float | None = None
    face_to_path_deg: float | None = None
    dynamic_loft_deg: float | None = None
    spin_loft_deg: float | None = None
    swing_direction_deg: float | None = None
    swing_plane_angle_deg: float | None = None
    low_point_ahead_of_ball_m: float | None = None
    low_point_height_m: float | None = None
    toe_mm: float | None = None
    high_mm: float | None = None
    smash_factor: float | None = None
    smash_factor_label: str | None = None
    tools: ToolsDeliveryEstimate | None = None
    tools_max_deviation_deg: float | None = None
    unavailable: dict[str, str] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        """JSON-friendly record; unavailable fields are ``None`` + reason."""
        out = {k: v for k, v in self.__dict__.items() if k != "tools"}
        out["tools"] = None if self.tools is None else dict(self.tools.__dict__)
        return out


def _heading(frame: TargetFrame, vec: np.ndarray) -> float | None:
    x, y, _ = frame.components(vec)
    if math.hypot(x, y) <= _EPS:
        return None
    return math.degrees(math.atan2(-frame.lateral_sign * y, x))


def _elevation(frame: TargetFrame, vec: np.ndarray) -> float:
    x, y, z = frame.components(vec)
    return math.degrees(math.atan2(z, math.hypot(x, y)))


def _wrap(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _resolve_impact_index(
    series: ClubheadSeries, impact_index: int | None, contact_index: int | None
) -> tuple[int, str]:
    n = len(series)
    if impact_index is not None and contact_index is not None:
        raise ValueError("give impact_index or contact_index, not both")
    if impact_index is not None:
        if not 0 <= impact_index < n:
            raise ValueError(f"impact_index {impact_index} out of range [0, {n})")
        return int(impact_index), "explicit"
    if contact_index is not None:
        if not 1 <= contact_index < n:
            raise ValueError(
                "contact_index must be in [1, N): the impact state is the last "
                "pre-contact sample, contact_index - 1"
            )
        return int(contact_index) - 1, "last_pre_contact"
    from src.shared.python.motion_matching.loaders._align import (
        detect_impact_index,
    )

    return int(detect_impact_index(series.times_s, series.face_center_m)), (
        "peak_clubhead_speed"
    )


def _plane_fit(
    series: ClubheadSeries, frame: TargetFrame, idx: int
) -> tuple[float, float] | str:
    """Swing direction and vertical plane angle, or a reason string."""
    t_imp = series.times_s[idx]
    mask = np.abs(series.times_s - t_imp) <= PLANE_WINDOW_S
    pts = series.face_center_m[mask]
    if pts.shape[0] < 3:
        return "fewer than 3 samples within +/-20 ms of impact"
    _, sv, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    if sv[0] <= _EPS or sv[1] / sv[0] < _COLLINEAR_RATIO:
        return "face-centre path is collinear within +/-20 ms; plane undefined"
    normal = vt[2]
    vel = series.velocity_mps[idx]
    tangent = vel - float(vel @ normal) * normal
    heading = _heading(frame, tangent)
    if heading is None:
        return "in-plane travel direction has no horizontal component"
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal @ frame.z_t)))))
    return heading, tilt


def _low_point(
    series: ClubheadSeries, frame: TargetFrame, idx: int
) -> tuple[float, float]:
    heights = series.face_center_m @ frame.z_t
    top = int(np.argmax(heights[: idx + 1]))
    k = top + int(np.argmin(heights[top:]))
    ahead = float((series.face_center_m[k] - np.asarray(frame.ball_m)) @ frame.x_t)
    return ahead, float(heights[k] - frame.ground_height_m)


def _face_location(
    series: ClubheadSeries, idx: int, contact: object
) -> tuple[float, float] | str:
    if series.toe_axis is None or series.face_normal is None:
        return "toe axis / face normal unobservable"
    point = np.asarray(contact, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("ball_contact_m must be a finite 3-vector")
    n = series.face_normal[idx] / np.linalg.norm(series.face_normal[idx])
    toe = series.toe_axis[idx] - float(series.toe_axis[idx] @ n) * n
    if np.linalg.norm(toe) <= _EPS:
        return "toe axis is parallel to the face normal"
    toe = toe / np.linalg.norm(toe)
    high = np.cross(toe, n)
    rel = point - series.face_center_m[idx]
    return float(rel @ toe) * 1e3, float(rel @ high) * 1e3


def _face_angles(
    frame: TargetFrame, v: np.ndarray, n: np.ndarray, path: float | None
) -> dict[str, float | None]:
    face = _heading(frame, n)
    to_path = None if face is None or path is None else _wrap(face - path)
    cosine = float(np.clip(v @ n / (np.linalg.norm(v) * np.linalg.norm(n)), -1, 1))
    return {
        "face_angle_deg": face,
        "face_to_path_deg": to_path,
        "dynamic_loft_deg": _elevation(frame, n),
        "spin_loft_deg": math.degrees(math.acos(cosine)),
    }


def _tools_estimate(
    gateway: ToolsDeliveryGateway | None, v: np.ndarray, n: np.ndarray, f: TargetFrame
) -> tuple[ToolsDeliveryEstimate | None, str | None]:
    try:
        gw = gateway if gateway is not None else load_tools_delivery_gateway()
    except ToolsDeliveryUnavailableError as exc:
        return None, f"Tools unavailable (fail closed): {exc}"
    return gw.estimate(v, n, f), None


def _max_deviation(p: dict[str, Any], tools: ToolsDeliveryEstimate) -> float:
    pairs = (
        (p.get("club_path_deg"), tools.club_path_deg),
        (p["attack_angle_deg"], tools.attack_angle_deg),
        (p.get("face_angle_deg"), tools.face_angle_deg),
        (p.get("dynamic_loft_deg"), tools.dynamic_loft_deg),
        (p.get("face_to_path_deg"), tools.face_to_path_deg),
        (p.get("spin_loft_deg"), tools.spin_loft_3d_deg),
    )
    diffs = [abs(a - b) for a, b in pairs if a is not None and b is not None]
    return max(diffs) if diffs else 0.0


def extract_impact_parameters(  # noqa: PLR0913 - keyword-only options
    series: ClubheadSeries,
    frame: TargetFrame,
    *,
    impact_index: int | None = None,
    contact_index: int | None = None,
    min_speed_mps: float = 1.0,
    use_tools: bool = True,
    tools_gateway: ToolsDeliveryGateway | None = None,
    ball_contact_m: object | None = None,
    ball_speed_mps: float | None = None,
    impact_model_status: str | None = None,
) -> ImpactParameters:
    """Compute impact parameters relative to ``frame``.

    Preconditions: ``min_speed_mps > 0``; speed at impact >= ``min_speed_mps``.
    Postconditions: each field is a finite float or ``None`` with an entry in
    ``unavailable``; the impact state is the last pre-contact sample.

    Raises:
        ValueError: speed below ``min_speed_mps`` or invalid indices/inputs.
    """
    if not isinstance(series, ClubheadSeries) or not isinstance(frame, TargetFrame):
        raise TypeError("series must be ClubheadSeries and frame a TargetFrame")
    if not min_speed_mps > 0:
        raise ValueError("min_speed_mps must be positive")
    idx, source = _resolve_impact_index(series, impact_index, contact_index)
    v = series.velocity_mps[idx]
    speed = float(np.linalg.norm(v))
    if speed < min_speed_mps:
        raise ValueError(
            f"clubhead speed {speed:.3g} m/s at impact is below the "
            f"{min_speed_mps:g} m/s minimum; impact parameters are undefined"
        )
    unavailable: dict[str, str] = {}
    path = _heading(frame, v)
    if path is None:
        unavailable["club_path_deg"] = "velocity has no horizontal component"
    values: dict[str, Any] = {
        "attack_angle_deg": _elevation(frame, v),
        "club_path_deg": path,
    }
    n = None
    if series.face_normal is None:
        reason = str(series.face_unobservable_reason)
        for name in (
            "face_angle_deg",
            "face_to_path_deg",
            "dynamic_loft_deg",
            "spin_loft_deg",
            "tools_estimates",
        ):
            unavailable[name] = f"face unobservable: {reason}"
    else:
        n = series.face_normal[idx]
        values.update(_face_angles(frame, v, n, path))
        if values["face_angle_deg"] is None:
            unavailable["face_angle_deg"] = "face normal has no horizontal part"
            unavailable["face_to_path_deg"] = "face angle undefined"
    tools = None
    if n is not None and use_tools:
        tools, why = _tools_estimate(tools_gateway, v, n, frame)
        if why:
            unavailable["tools_estimates"] = why
    _fill_geometry(values, unavailable, series, frame, idx, ball_contact_m)
    _fill_smash(values, unavailable, speed, ball_speed_mps, impact_model_status)
    deviation = None if tools is None else _max_deviation(values, tools)
    return ImpactParameters(
        frame=frame.to_record(),
        impact_index=idx,
        impact_time_s=float(series.times_s[idx]),
        impact_time_source=source,
        clubhead_speed_mps=speed,
        clubhead_speed_mph=speed * MPS_TO_MPH,
        tools=tools,
        tools_max_deviation_deg=deviation,
        unavailable=unavailable,
        **values,
    )


def _fill_geometry(
    values: dict[str, Any],
    unavailable: dict[str, str],
    series: ClubheadSeries,
    frame: TargetFrame,
    idx: int,
    ball_contact_m: object | None,
) -> None:
    plane = _plane_fit(series, frame, idx)
    if isinstance(plane, str):
        for name in ("swing_direction_deg", "swing_plane_angle_deg"):
            unavailable[name] = plane
    else:
        values["swing_direction_deg"], values["swing_plane_angle_deg"] = plane
    ahead, height = _low_point(series, frame, idx)
    values["low_point_ahead_of_ball_m"] = ahead
    values["low_point_height_m"] = height
    if ball_contact_m is None:
        for name in ("toe_mm", "high_mm"):
            unavailable[name] = "needs face geometry (GCV-11) and ball (GCV-13)"
        return
    loc = _face_location(series, idx, ball_contact_m)
    if isinstance(loc, str):
        unavailable["toe_mm"] = unavailable["high_mm"] = loc
    else:
        values["toe_mm"], values["high_mm"] = loc


def _fill_smash(
    values: dict[str, Any],
    unavailable: dict[str, str],
    speed: float,
    ball_speed: float | None,
    status: str | None,
) -> None:
    if ball_speed is None or not status:
        unavailable["smash_factor"] = (
            "needs an impact model ball speed and its calibration status"
        )
        return
    if not math.isfinite(ball_speed) or ball_speed < 0:
        raise ValueError("ball_speed_mps must be finite and non-negative")
    values["smash_factor"] = ball_speed / speed
    values["smash_factor_label"] = f"impact model calibration: {status}"
