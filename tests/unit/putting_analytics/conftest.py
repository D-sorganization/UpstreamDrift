"""Toy outcome models with known structure for the putting analytics tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.shared.python.putting_analytics.expected_putts import PuttAction, PuttOutcomes

CUP_RADIUS_FT = 0.177  # 4.25 in cup
CAPTURE_PACE_FT = 16.0  # roll-past distance at which the effective cup closes


@dataclass(frozen=True)
class ConstantMakeModel:
    """Holes everything inside 1 ft; elsewhere holes with p and leaves 0.5 ft."""

    make_probability: float

    def simulate(
        self, distance_ft: float, action: PuttAction, n: int, seed: int
    ) -> PuttOutcomes:
        draw = np.random.default_rng(seed).random(n)
        holed = (distance_ft <= 1.0) | (draw < self.make_probability)
        return PuttOutcomes(holed, np.where(holed, 0.0, 0.5))


@dataclass(frozen=True)
class ToyBreakModel:
    """Breaking putt: the line must start ``break_deg`` high; pace narrows the cup.

    ``line_sd_deg`` and ``speed_cv`` stand in for a skill profile (#14169).
    """

    line_sd_deg: float = 1.0
    speed_cv: float = 0.08
    break_deg: float = 2.0

    def simulate(
        self, distance_ft: float, action: PuttAction, n: int, seed: int
    ) -> PuttOutcomes:
        line_noise, speed_noise = np.random.default_rng(seed).standard_normal((2, n))
        line_deg = action.aim_offset_deg + self.line_sd_deg * line_noise
        lateral_ft = distance_ft * np.radians(line_deg - self.break_deg)
        speed_ratio = np.exp(self.speed_cv * speed_noise)
        roll_past_ft = (distance_ft + action.pace_ft_past) * speed_ratio - distance_ft
        half_width = CUP_RADIUS_FT * np.clip(1.0 - roll_past_ft / CAPTURE_PACE_FT, 0, 1)
        holed = (roll_past_ft >= 0.0) & (np.abs(lateral_ft) <= half_width)
        leave = np.where(holed, 0.0, np.hypot(lateral_ft, roll_past_ft))
        return PuttOutcomes(holed, leave)


@pytest.fixture
def toy_model() -> ToyBreakModel:
    return ToyBreakModel()


@pytest.fixture
def actions() -> list[PuttAction]:
    return [PuttAction(a, p) for a in range(5) for p in (0.5, 1, 1.5, 2, 3, 4)]


@pytest.fixture
def grid() -> np.ndarray:
    return np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30])
