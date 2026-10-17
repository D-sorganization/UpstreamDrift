"""Hand-built vertical plate trials for the force-plate contract tests (OF-01).

Every value is written out so a reader can check it by hand: two feet, three
samples, loads in N and centres of pressure in the stance frame in metres.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.force_plates import (
    PlateProvenance,
    PlateUncertainty,
    VerticalPlateTrial,
)

TIMES_S = (0.0, 0.005, 0.010)
LEFT_N = (400.0, 300.0, 5.0)
RIGHT_N = (400.0, 500.0, 795.0)
LEFT_COP_M = ((-0.185, 0.02), (-0.18, 0.0), (np.nan, np.nan))
RIGHT_COP_M = ((0.185, -0.02), (0.19, 0.01), (0.20, 0.03))


def _net_cop() -> np.ndarray:
    """Load-weighted CoP of the feet above the 20 N threshold (source rule)."""
    out = np.full((3, 2), np.nan)
    for i in range(3):
        loads, cops = [], []
        for load, cop in ((LEFT_N[i], LEFT_COP_M[i]), (RIGHT_N[i], RIGHT_COP_M[i])):
            if not np.isnan(cop[0]):
                loads.append(load)
                cops.append(cop)
        if loads:
            weights = np.asarray(loads)
            out[i] = weights @ np.asarray(cops) / weights.sum()
    return out


def uncertainty() -> PlateUncertainty:
    return PlateUncertainty(
        force_floor_n=2.0,
        force_fraction=0.01,
        cop_m=0.010,
        sensing_depth_sigma_m=0.002,
        source="declared hypothesis for tests",
    )


def provenance() -> PlateProvenance:
    return PlateProvenance(
        device="test-plates",
        source_format="hand-built",
        source_revision="none",
        file_sha256={},
    )


def trial_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "times_s": np.array(TIMES_S),
        "load_n": {"left": np.array(LEFT_N), "right": np.array(RIGHT_N)},
        "cop_m": {
            "left": np.array(LEFT_COP_M),
            "right": np.array(RIGHT_COP_M),
            "net": _net_cop(),
        },
        "body_weight_n": 800.0,
        "handedness": "RH",
        "events_s": {"max_trail": 0.005, "impact": None},
        "cop_min_load_n": 20.0,
        "capacity_n": 3923.0,
        "sensing_depth_m": 0.015,
        "provenance": provenance(),
        "uncertainty": uncertainty(),
    }
    kwargs.update(overrides)
    return kwargs


def make_trial(**overrides: Any) -> VerticalPlateTrial:
    return VerticalPlateTrial(**trial_kwargs(**overrides))
