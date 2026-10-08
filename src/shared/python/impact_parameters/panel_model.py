"""Display model for the impact-parameters panel (GCV-17, #11723).

One card model feeds the API route, the PyQt widget and the web panel so the
rows, units and "unavailable" reasons are defined once.  Values come straight
from :class:`ImpactParameters`; an unavailable quantity has ``value=None`` and
a reason, never zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .extract import MPS_TO_MPH, ImpactParameters
from .target_frame import UD_DEFAULT_TARGET_DIR, TargetFrame

UNITS = ("mph", "m/s")


@dataclass(frozen=True)
class CardRow:
    """One launch-monitor row; ``value is None`` means unavailable."""

    key: str
    label: str
    unit: str
    value: float | None
    reason: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ImpactCard:
    """Rows plus the D-plane / path-view angles for the diagrams."""

    units: str
    impact_time_s: float
    impact_time_source: str
    frame: dict[str, object]
    rows: tuple[CardRow, ...]
    d_plane: dict[str, float | None]
    available: bool = True
    reason: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "rows"}
        out["rows"] = [r.to_dict() for r in self.rows]
        return out


def _row(
    params: ImpactParameters,
    key: str,
    label: str,
    unit: str,
    field_name: str | None = None,
    scale: float = 1.0,
) -> CardRow:
    name = field_name or key
    value = getattr(params, name)
    reason = None if value is not None else params.unavailable.get(name)
    if value is None and reason is None:
        reason = "not computed"
    return CardRow(
        key=key,
        label=label,
        unit=unit,
        value=None if value is None else float(value) * scale,
        reason=reason,
    )


def _impact_location(params: ImpactParameters) -> CardRow:
    if params.toe_mm is None or params.high_mm is None:
        reason = params.unavailable.get("toe_mm", "not computed")
        return CardRow(
            "impact_location", "Impact Location (toe, high)", "mm", None, reason
        )
    return CardRow(
        "impact_location",
        "Impact Location (toe, high)",
        "mm",
        float(params.toe_mm),
        note=f"high {params.high_mm:.1f} mm",
    )


def build_impact_card(params: ImpactParameters, units: str = "mph") -> ImpactCard:
    """Build the card for ``params``.  Raises ``ValueError`` for unknown units."""
    if units not in UNITS:
        raise ValueError(f"units must be one of {UNITS}, got {units!r}")
    speed = params.clubhead_speed_mph if units == "mph" else params.clubhead_speed_mps
    smash = _row(params, "smash_factor", "Smash Factor", "")
    if params.smash_factor_label and smash.value is not None:
        smash = CardRow(
            smash.key, smash.label, "", smash.value, note=params.smash_factor_label
        )
    rows = (
        CardRow("clubhead_speed", "Clubhead Speed", units, float(speed)),
        _row(params, "attack_angle_deg", "Attack Angle", "deg"),
        _row(params, "club_path_deg", "Club Path", "deg"),
        _row(params, "face_angle_deg", "Face Angle", "deg"),
        _row(params, "face_to_path_deg", "Face to Path", "deg"),
        _row(params, "dynamic_loft_deg", "Dynamic Loft", "deg"),
        _row(params, "spin_loft_deg", "Spin Loft", "deg"),
        _row(params, "low_point_ahead_of_ball_m", "Low Point Ahead of Ball", "m"),
        _impact_location(params),
        smash,
    )
    d_plane = {
        name: getattr(params, name)
        for name in (
            "attack_angle_deg",
            "club_path_deg",
            "face_angle_deg",
            "face_to_path_deg",
            "dynamic_loft_deg",
            "spin_loft_deg",
            "swing_direction_deg",
            "swing_plane_angle_deg",
        )
    }
    return ImpactCard(
        units=units,
        impact_time_s=params.impact_time_s,
        impact_time_source=params.impact_time_source,
        frame=dict(params.frame),
        rows=rows,
        d_plane=d_plane,
    )


def parse_target_dir(text: str | None) -> tuple[float, float, float]:
    """Parse ``"x,y"`` or ``"x,y,z"`` into a horizontal unit target direction."""
    if text is None or not text.strip():
        return UD_DEFAULT_TARGET_DIR
    try:
        parts = [float(p) for p in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"target_dir must be numeric 'x,y[,z]', got {text!r}") from exc
    if len(parts) not in (2, 3):
        raise ValueError("target_dir needs 2 or 3 comma-separated components")
    if not all(math.isfinite(p) for p in parts):
        raise ValueError("target_dir components must be finite")
    x, y = parts[0], parts[1]
    norm = math.hypot(x, y)
    if norm < 1e-9:
        raise ValueError("target_dir needs a nonzero horizontal component")
    return (x / norm, y / norm, 0.0)


def target_frame_from_heading(
    heading_deg: float, handedness: str = "right"
) -> TargetFrame:
    """Default target line rotated by ``heading_deg`` counter-clockwise about up."""
    if not math.isfinite(heading_deg):
        raise ValueError("heading_deg must be finite")
    base_x, base_y, _ = UD_DEFAULT_TARGET_DIR
    c, s = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    return TargetFrame(
        target_dir=(c * base_x - s * base_y, s * base_x + c * base_y, 0.0),
        handedness=handedness,
    )


__all__ = [
    "UNITS",
    "CardRow",
    "ImpactCard",
    "MPS_TO_MPH",
    "build_impact_card",
    "parse_target_dir",
    "target_frame_from_heading",
]
