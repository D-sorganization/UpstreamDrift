"""Configurable physics inputs for the offline grip-loss evaluator (#15128)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GripBreakawayThresholds:
    """Hysteretic margin thresholds and dwell times (SI seconds, unitless margins)."""

    slip_enter_margin: float
    slip_exit_margin: float
    release_enter_margin: float
    min_slip_dwell_s: float
    min_release_dwell_s: float

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.slip_enter_margin)
            and math.isfinite(self.slip_exit_margin)
            and math.isfinite(self.release_enter_margin)
        ):
            raise ValueError("threshold margins must be finite")
        if self.slip_exit_margin <= self.slip_enter_margin:
            raise ValueError(
                "slip_exit_margin must exceed slip_enter_margin for hysteresis"
            )
        if self.release_enter_margin >= self.slip_enter_margin:
            raise ValueError("release_enter_margin must sit below slip_enter_margin")
        for name in ("min_slip_dwell_s", "min_release_dwell_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class GripLossPhysicsConfig:
    """Coulomb margin parameters for offline grip-loss prediction."""

    friction: float
    normal_floor_n: float
    thresholds: GripBreakawayThresholds
    squeeze_scale: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.friction) and self.friction > 0.0):
            raise ValueError("friction must be positive and finite")
        if not (math.isfinite(self.normal_floor_n) and self.normal_floor_n > 0.0):
            raise ValueError("normal_floor_n must be positive and finite")
        if not (math.isfinite(self.squeeze_scale) and self.squeeze_scale > 0.0):
            raise ValueError("squeeze_scale must be positive and finite")


__all__ = ["GripBreakawayThresholds", "GripLossPhysicsConfig"]
