"""Tests for the constant hand-to-club rotation fit (MM-2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import grip_fit as module

pytestmark = pytest.mark.unit

# The anthropometric wrist base: forearm z -> -y of the base frame.
BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
RANGES = [(-90.0, 90.0), (-40.0, 25.0), (-70.0, 70.0)]


def _chain(pronation: float, cock: float, flexion: float) -> np.ndarray:
    rz = Rotation.from_euler("z", pronation).as_matrix()
    rx = Rotation.from_euler("x", cock).as_matrix()
    rz2 = Rotation.from_euler("z", flexion).as_matrix()
    return rz @ BASE @ rx @ rz2


def _angles(seed: int, frames: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo, hi = np.radians(np.asarray(RANGES)).T
    return rng.uniform(lo * 0.8, hi * 0.8, size=(frames, 3))


def test_wrist_angles_decompose_the_chain_exactly() -> None:
    angles = _angles(1)
    relative = np.stack([_chain(*row) for row in angles])
    recovered = module.wrist_angles(relative, BASE)
    assert np.allclose(recovered, angles, atol=1e-10)
    assert np.allclose(module.wrist_rotation(angles, BASE), relative, atol=1e-12)


def test_preconditions() -> None:
    with pytest.raises(ValueError):
        module.wrist_angles(np.eye(3), BASE)
    with pytest.raises(ValueError):
        module.wrist_angles(np.eye(3)[None], np.eye(3))  # wrong base convention
    with pytest.raises(ValueError):
        module.grip_rotation([0.0, 1.0])
    with pytest.raises(ValueError):
        module.excess_deg(np.zeros((2, 3)), [(1.0, 0.0)] * 3)
    with pytest.raises(ValueError):
        module.fit_grip_rotation(np.eye(3)[None], BASE, RANGES, grid_step_deg=0)


def test_excess_is_zero_inside_and_linear_outside() -> None:
    angles = np.array([[0.0, 30.0, -80.0], [-100.0, -40.0, 70.0]])
    ex = module.excess_deg(angles, RANGES)
    assert np.allclose(ex, [[0.0, 5.0, 10.0], [10.0, 0.0, 0.0]])


def test_fit_recovers_a_hidden_hand_rotation() -> None:
    angles = _angles(2, frames=80)
    good = np.stack([_chain(*row) for row in angles])
    hidden = module.grip_rotation([20.0, -35.0, 50.0])
    observed = good @ hidden.T  # the document's hand is rotated by hidden^-1
    before = np.degrees(module.wrist_angles(observed, BASE))
    assert module.excess_deg(before, RANGES).max() > 5.0
    fit = module.fit_grip_rotation(observed, BASE, RANGES)
    assert fit.cost_after <= fit.cost_before
    assert fit.cost_after < 0.5
    assert max(fit.max_excess_after_deg) < 2.0
    doc = fit.as_document()
    assert set(doc["angle_range_after_deg"]) == set(module.ANGLE_NAMES)
    assert doc["cost_after_rms_excess_deg"] <= doc["cost_before_rms_excess_deg"]


def test_fit_keeps_identity_when_nothing_is_out_of_range() -> None:
    angles = _angles(3)
    relative = np.stack([_chain(*row) for row in angles])
    fit = module.fit_grip_rotation(relative, BASE, RANGES, grid_step_deg=45.0)
    assert fit.cost_before == 0.0 and fit.cost_after == 0.0
    assert fit.rotation_deg == (0.0, 0.0, 0.0)
