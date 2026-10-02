"""Bounded native pose solving contracts using deterministic synthetic kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK, PoseFit

pytestmark = pytest.mark.unit


class _LinearPose(BaseFullBodyIK):
    supports_trf = True

    def __init__(self) -> None:
        super().__init__(
            coordinate_order=("bounded", "unbounded"), labels=("a", "b", "c")
        )
        self.q = np.zeros(2)

    def _set(self, q: np.ndarray) -> None:
        self.q = q.copy()

    def _positions(self) -> np.ndarray:
        return np.tile([self.q.sum(), 0.0, 0.0], (3, 1))

    def _marker_jacobian(self, positions: np.ndarray) -> np.ndarray:
        jac = np.zeros((3, 3, 2))
        jac[:, 0, :] = 1.0
        return jac

    def _sphere_heights(self, ground: GroundPlane) -> dict[str, float]:
        return {"synthetic": 0.0}


GROUND = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)


def _solve(model: _LinearPose, **kwargs: object) -> PoseFit:
    return model.solve_pose(
        np.tile([3.0, 0.0, 0.0], (3, 1)),
        np.ones(3, dtype=bool),
        np.zeros(2),
        ground=GROUND,
        solver="trf",
        closure_weight=0.0,
        ground_weight=0.0,
        prior_weight=0.0,
        **kwargs,
    )


def test_trf_partial_bounds_and_locked_coordinate() -> None:
    model = _LinearPose()
    fit = _solve(model, bounds={"bounded": (0.0, 1.0)}, locked={"unbounded": 0.0})
    assert fit.q[0] == pytest.approx(1.0, abs=1e-7)
    assert fit.q[1] == 0.0
    free = _solve(model, bounds={"bounded": (0.0, 1.0)})
    assert 0.0 <= free.q[0] <= 1.0
    assert free.q.sum() == pytest.approx(3.0, abs=1e-7)


def test_trf_all_locked_coordinates_stay_exact() -> None:
    fit = _solve(_LinearPose(), locked={"bounded": 0.5, "unbounded": 0.25})
    np.testing.assert_array_equal(fit.q, [0.5, 0.25])


def test_trf_rejects_locked_bound_conflict_and_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="locked.*bound|Locked.*bound"):
        _solve(_LinearPose(), bounds={"bounded": (0.0, 1.0)}, locked={"bounded": 2.0})
    model = _LinearPose()
    model.supports_trf = False
    with pytest.raises(ValueError, match="TRF|trf"):
        _solve(model)


def test_pose_rejects_unknown_solver() -> None:
    with pytest.raises(ValueError, match="solver|Solver"):
        _LinearPose().solve_pose(
            np.zeros((3, 3)),
            np.ones(3, dtype=bool),
            np.zeros(2),
            ground=GROUND,
            solver="bad",
        )
