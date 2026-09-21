"""Tests for fitting the dual-grip closure weld from a pose."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import closure_fit as module

pytestmark = pytest.mark.unit


def _pose(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return Rotation.random(random_state=seed).as_matrix(), rng.normal(size=3)


def test_fitted_placement_makes_the_weld_hold_exactly() -> None:
    document = {
        "closure": {
            "body_a": "hand",
            "body_b": "club",
            "name": "grip",
            "placement_a": np.eye(4).tolist(),
            "placement_b": np.eye(4).tolist(),
        }
    }
    hand, club = _pose(1), _pose(2)
    before = module.closure_residual(hand, club, document["closure"]["placement_b"])
    assert before[0] > 0.1
    fitted = module.fit_closure_placement(document, hand, club)
    after = module.closure_residual(hand, club, fitted["closure"]["placement_b"])
    assert after[0] < 1e-12 and after[1] < 1e-9
    assert fitted["closure"]["placement_a"] == document["closure"]["placement_a"]
    assert fitted["closure_fit"]["translation_change_m"] > 0
    assert document["closure"]["placement_b"] == np.eye(4).tolist()  # input untouched
    with pytest.raises(ValueError):
        module.fit_closure_placement({}, hand, club)
    with pytest.raises(ValueError):
        module.fit_closure_placement(document, (np.ones((3, 3)), np.zeros(3)), club)
