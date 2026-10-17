"""Branch continuity of the shoulder orientation solve (#13056).

When the target orientation is out of range, the closest in-range solution
can move between two local minima of nearly equal residual. Choosing the
lowest residual per frame then flips the coordinates by radians in one frame,
and inverse dynamics charges the implied arm acceleration to the pelvis.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fullbody_native import (
    select_continuous,
)

pytestmark = pytest.mark.unit

_NEAR = np.array([0.0, 0.1, 0.0])
_FAR = np.array([3.0, -0.4, 0.2])


def test_converged_candidates_keep_the_branch_nearest_the_previous_frame() -> None:
    picked, _ = select_continuous(
        [(_FAR, 1e-12), (_NEAR, 1e-9)], np.zeros(3), tolerance_rad=1e-6, slack_rad=0.1
    )
    np.testing.assert_array_equal(picked, _NEAR)


def test_converged_beats_a_nearer_unconverged_candidate() -> None:
    picked, angle = select_continuous(
        [(_NEAR, 0.05), (_FAR, 1e-9)], np.zeros(3), tolerance_rad=1e-6, slack_rad=0.1
    )
    np.testing.assert_array_equal(picked, _FAR)
    assert angle == 1e-9


def test_unreachable_frame_stays_on_branch_within_the_slack() -> None:
    picked, angle = select_continuous(
        [(_FAR, 0.90), (_NEAR, 0.95)], np.zeros(3), tolerance_rad=1e-6, slack_rad=0.1
    )
    np.testing.assert_array_equal(picked, _NEAR)
    assert angle == 0.95


def test_unreachable_frame_switches_branch_beyond_the_slack() -> None:
    picked, _ = select_continuous(
        [(_FAR, 0.70), (_NEAR, 0.95)], np.zeros(3), tolerance_rad=1e-6, slack_rad=0.1
    )
    np.testing.assert_array_equal(picked, _FAR)


def test_first_frame_takes_the_lowest_residual() -> None:
    picked, _ = select_continuous(
        [(_NEAR, 0.5), (_FAR, 0.4)], None, tolerance_rad=1e-6, slack_rad=0.1
    )
    np.testing.assert_array_equal(picked, _FAR)


def test_selection_preconditions() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        select_continuous([], None, tolerance_rad=1e-6, slack_rad=0.1)
    with pytest.raises(ValueError, match="slack_rad"):
        select_continuous([(_NEAR, 0.1)], None, tolerance_rad=1e-6, slack_rad=-1.0)
