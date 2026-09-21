"""Tests for the marker-derived elbow pit direction."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching import posture_metrics as module

pytestmark = pytest.mark.unit


def test_elbow_pit_direction_is_the_fold_side_or_none_when_straight() -> None:
    shoulder = np.array([0.0, 0.0, 1.5])
    elbow = np.array([0.0, 0.0, 1.2])  # upper arm straight down
    wrist = np.array([0.1, 0.0, 0.95])  # forearm folds toward +x
    pit = module.elbow_pit_direction(shoulder, elbow, wrist)
    assert pit is not None
    np.testing.assert_allclose(pit, [1.0, 0.0, 0.0], atol=1e-12)
    assert (
        module.elbow_pit_direction(shoulder, elbow, np.array([0.0, 0.0, 0.9])) is None
    )
    with pytest.raises(ValueError):
        module.elbow_pit_direction(shoulder, shoulder, wrist)
