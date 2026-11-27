"""The neutral address bounds name v1 trunk coordinates; a model without them
(the regional ``full-body-v2`` trunk) is solved with the ones it has (#15296)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.address import best_address
from src.shared.python.motion_matching.pipeline.constants import NEUTRAL_BOUNDS_DEG

pytestmark = [pytest.mark.unit]
BASE = ("x", "y", "z", "rx", "ry", "rz")


def _bounds_seen(order: tuple[str, ...]) -> set[str]:
    seen: dict[str, Any] = {}

    def solve_pose(points: Any, valid: Any, q0: np.ndarray, **options: Any) -> Any:
        seen.update(options["bounds"])
        return SimpleNamespace(marker_rms_m=0.01, q=q0)

    kin = SimpleNamespace(
        coordinate_order=order, arm_mirrored=False, solve_pose=solve_pose
    )
    lane = SimpleNamespace(
        bounds={},
        anthropometric=False,
        labels=(),
        points=np.zeros((1, 0, 3)),
        valid=np.zeros((1, 0), bool),
        ground=None,
        stance=[()],
        marker_weights={},
        prior_weights={},
        feet=None,
        knees=None,
    )
    best_address(lane, kin, np.zeros(len(order)), neutral=True)
    return set(seen)


def test_neutral_bounds_apply_only_to_coordinates_the_model_has() -> None:
    v1 = (*BASE, *NEUTRAL_BOUNDS_DEG)
    assert _bounds_seen(v1) == set(NEUTRAL_BOUNDS_DEG)
    regional = (*BASE, "lumbar_flexion", "thoracic_flexion", "LScapInputY")
    assert _bounds_seen(regional) == {"LScapInputY"}
