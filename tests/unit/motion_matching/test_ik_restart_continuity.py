"""Continuity-preserving trajectory IK restarts (#12042).

A synthetic one-joint model has two branches that fit the well-observed marker
equally (``+theta`` and ``-theta``). A weakly observed marker carries noise
whose sign flips from frame to frame, so the mirror branch fits some frames
better by more than the restart margin. Legacy free restarts then hop between
the branches; continuity-preserving restarts stay on the anatomical branch the
trajectory starts on, whatever the perturbation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK

pytestmark = pytest.mark.unit

GROUND = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=-10.0)
RADIUS_M = 1.0
WEAK_M = 0.15
ANGLE_RAD = 1.0
FRAMES = 40
RESTARTS = {
    "restarts": 4,
    "restart_threshold_m": 0.03,
    "restart_margin_m": 0.003,
    "restart_spread_rad": 1.5,
}


class _TwoBranchJoint(BaseFullBodyIK):
    """Six root coordinates (unobserved) and one axial joint ``theta``.

    Marker ``A`` = (R cos theta, 0, 0) cannot tell ``+theta`` from ``-theta``;
    marker ``B`` = (0, w sin theta, 0) can, but only weakly; ``C`` is fixed.
    """

    def __init__(self) -> None:
        super().__init__(
            coordinate_order=("x", "y", "z", "rx", "ry", "rz", "axial"),
            labels=("A", "B", "C"),
        )
        self._q = np.zeros(7)

    def _set(self, q: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=float).copy()

    def _positions(self) -> np.ndarray:
        theta = self._q[6]
        return np.array(
            [
                [RADIUS_M * np.cos(theta), 0.0, 0.0],
                [0.0, WEAK_M * np.sin(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def _marker_jacobian(self, positions: np.ndarray) -> np.ndarray:
        theta = self._q[6]
        jac = np.zeros((3, 3, 7))
        jac[0, 0, 6] = -RADIUS_M * np.sin(theta)
        jac[1, 1, 6] = WEAK_M * np.cos(theta)
        return jac

    def _sphere_heights(self, ground: object) -> dict[str, float]:
        return {"foot": 1.0}


def _targets(sign_seed: int, offset_m: float = 0.0) -> np.ndarray:
    signs = np.where(np.random.default_rng(sign_seed).random(FRAMES) < 0.5, 1.0, -1.0)
    targets = np.zeros((FRAMES, 3, 3))
    targets[:, 0, 0] = RADIUS_M * np.cos(ANGLE_RAD) + offset_m
    targets[:, 1, 1] = signs * WEAK_M * np.sin(ANGLE_RAD)
    targets[:, 2, 2] = 1.0
    return targets


def _solve(
    targets: np.ndarray, start_rad: float, max_step_rad: float | None
) -> np.ndarray:
    q0 = np.zeros(7)
    q0[6] = start_rad
    q, _ = _TwoBranchJoint().solve_trajectory(
        targets,
        np.ones(targets.shape[:2], dtype=bool),
        q0,
        ground=GROUND,
        restart_max_step_rad=max_step_rad,
        **RESTARTS,
    )
    return q[:, 6]


def _branch_hops(theta: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(np.sign(theta))))


def test_the_two_branches_fit_the_strong_marker_equally() -> None:
    kin = _TwoBranchJoint()
    for theta in (ANGLE_RAD, -ANGLE_RAD):
        q = np.zeros(7)
        q[6] = theta
        kin._set(q)
        assert kin._positions()[0, 0] == pytest.approx(RADIUS_M * np.cos(ANGLE_RAD))


def test_free_restarts_hop_between_equally_fitting_branches() -> None:
    theta = _solve(_targets(3), ANGLE_RAD, None)
    assert _branch_hops(theta) > 5


@pytest.mark.parametrize(
    ("sign_seed", "offset_m", "start_rad"),
    [(3, 0.0, ANGLE_RAD), (3, 1e-4, ANGLE_RAD), (3, 0.0, 0.95), (11, 0.0, 1.05)],
)
def test_continuous_restarts_keep_the_anatomical_branch_under_perturbation(
    sign_seed: int, offset_m: float, start_rad: float
) -> None:
    theta = _solve(_targets(sign_seed, offset_m), start_rad, 0.1)
    assert _branch_hops(theta) == 0
    assert np.all(theta > 0.9)


def test_continuous_restarts_are_repeatable_under_a_near_null_perturbation() -> None:
    base = _solve(_targets(3), ANGLE_RAD, 0.1)
    nudged = _solve(_targets(3, 1e-4), ANGLE_RAD, 0.1)
    assert np.max(np.abs(np.degrees(base - nudged))) < 0.5


def test_restart_step_bound_must_be_finite_and_positive() -> None:
    for bad in (0.0, -0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="restart_max_step_rad"):
            _solve(_targets(3), ANGLE_RAD, bad)


def test_prior_anchor_moves_the_prior_but_not_the_seed() -> None:
    kin = _TwoBranchJoint()
    targets = _targets(3)[0]
    targets[1, 1] = 0.0  # B no longer prefers a branch
    seed = np.zeros(7)
    seed[6] = ANGLE_RAD
    anchor = seed.copy()
    anchor[0] = 0.5  # an unobserved root coordinate
    common = {"ground": GROUND, "prior_weight": 1.0, "iterations": 50}
    plain = kin.solve_pose(targets, np.ones(3, dtype=bool), seed, **common)
    anchored = kin.solve_pose(
        targets, np.ones(3, dtype=bool), seed, prior_anchor=anchor, **common
    )
    assert plain.q[0] == pytest.approx(0.0, abs=1e-9)
    assert anchored.q[0] == pytest.approx(0.5, abs=1e-6)
    assert anchored.q[6] > 0  # the seed still chooses the branch


def test_prior_anchor_must_match_the_coordinates() -> None:
    kin = _TwoBranchJoint()
    with pytest.raises(ValueError, match="prior_anchor"):
        kin.solve_pose(
            _targets(3)[0],
            np.ones(3, dtype=bool),
            np.zeros(7),
            ground=GROUND,
            prior_anchor=np.zeros(5),
        )
