"""JSON-safe presentation of the existing iron divot heuristic (#15523).

This adapter exports diagnostics, without recomputing turf work or applying
launch penalties. It does not establish physical qualification.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from numbers import Real

from src.shared.python.physics.divot_model import DivotResult

__all__ = ["DIVOT_REPORT_FORMAT", "export_divot_report"]

DIVOT_REPORT_FORMAT = "upstreamdrift/iron-divot-report/1.0"

_OFFSETS = ("entry_offset_m", "exit_offset_m")
_NONNEGATIVE = (
    "length_m",
    "max_depth_m",
    "volume_m3",
    "pre_ball_length_m",
    "energy_loss_j",
)


def export_divot_report(result: DivotResult) -> dict[str, object]:
    """Return a fresh, strict-JSON-compatible report in SI units.

    Preconditions: a DivotResult with boolean flags, finite non-negative
    magnitudes, ordered contact offsets and 0 <= spin <= speed <= 1.
    No-contact results must have zero magnitudes, unit retentions and no
    fat-shot flag. Their undefined offsets are exported as None (JSON null).
    Invalid types raise TypeError; invalid fields raise ValueError.
    Postconditions: no NaN/Infinity; input is unchanged; model status remains
    explicitly uncalibrated. Contact offsets are relative to the ball along
    the target direction, not world positions or timestamps.
    """
    if not isinstance(result, DivotResult):
        raise TypeError("result must be a DivotResult")
    for name in ("has_divot", "is_fat_shot"):
        if type(getattr(result, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")

    fields = asdict(result)
    for name, value in fields.items():
        if name in ("has_divot", "is_fat_shot"):
            continue
        # Only the model's undefined no-contact offsets may be non-finite.
        if name in _OFFSETS and not result.has_divot:
            fields[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if name in _NONNEGATIVE and number < 0:
            raise ValueError(f"{name} must be non-negative")
        if name.endswith("_retention") and not 0 <= number <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
        fields[name] = number

    if result.spin_retention > result.speed_retention:
        raise ValueError("spin_retention must not exceed speed_retention")
    if result.has_divot:
        if result.entry_offset_m >= result.exit_offset_m:
            raise ValueError("contact offsets must be strictly increasing")
    elif (
        result.is_fat_shot
        or any(getattr(result, name) != 0 for name in _NONNEGATIVE)
        or result.speed_retention != 1
        or result.spin_retention != 1
    ):
        raise ValueError(
            "no-contact results must have zero geometry/loss and no penalty"
        )

    return {
        "format": DIVOT_REPORT_FORMAT,
        "model_status": "uncalibrated_heuristic",
        **fields,
    }
