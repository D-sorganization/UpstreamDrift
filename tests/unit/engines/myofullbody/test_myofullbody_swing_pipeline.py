"""Pipeline helpers of the MyoFullBody swing analysis (issue #11689)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.myofullbody import neck, redundancy, swing_pipeline

pytestmark = pytest.mark.unit


def test_config_rejects_unknown_rom_policy_and_bad_workers() -> None:
    with pytest.raises(ValueError, match="rom_policy"):
        swing_pipeline.SwingConfig(Path("x"), rom_policy="other")
    with pytest.raises(ValueError, match="workers"):
        swing_pipeline.SwingConfig(Path("x"), workers=0)


def test_solve_with_neck_splits_muscle_and_neck_activation() -> None:
    order = ("NeckInputX", "Other")
    base = redundancy.FrameBasis(
        np.array([100.0]), np.array([0.0]), np.array([[1.0, 1.0]]), np.eye(2)
    )
    aug = neck.augment(base, order, [0, 1])
    tau = np.array([[10.0, 0.0]])
    muscle, neck_act, res, ok = swing_pipeline.solve_with_neck([aug], tau, 100.0, 1)
    assert ok
    assert muscle.shape == (1, 1)
    assert neck_act.shape == (1, 2)
    # the neck pair carries the neck torque, the signed difference is non-negative
    assert neck_act[0, 0] >= neck_act[0, 1]
    assert np.abs(res).max() < 1.0
