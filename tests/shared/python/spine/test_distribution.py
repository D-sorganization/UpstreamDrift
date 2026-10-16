"""Tests for the spine distribution problem (SPN-2b, issue #12742)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.spine.coupling import (
    LinearCouplingLaw,
    StateDependentCouplingLaw,
    level_rotations,
)
from src.shared.python.spine.distribution import (
    ChainOffsets,
    DistributionResult,
    distribute,
    distribute_series,
    distribute_series_detailed,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import SegmentalKinematicsSeries

pytestmark = pytest.mark.unit

CITE = "Invented fixture law, doi:10.0000/test table 1"
LEVELS = (SpinalLevel.L5_S1, SpinalLevel.L4_L5, SpinalLevel.L3_L4)
WEIGHTS = (0.5, 0.3, 0.2)


def _cross_law(n_phi: int = 3) -> LinearCouplingLaw:
    """Three levels, three coordinates, small cross-axis terms."""
    mats = {}
    for lv, w in zip(LEVELS, WEIGHTS, strict=True):
        m = w * np.eye(3)
        m[0, 1] = 0.05 * w
        m[2, 0] = -0.04 * w
        mats[lv] = m
    return LinearCouplingLaw("fixture3", "1", mats, ("fe", "lb", "ar"), CITE)


def _axial_law() -> LinearCouplingLaw:
    mats = {}
    for lv, w in zip(LEVELS, WEIGHTS, strict=True):
        m = np.zeros((3, 1))
        m[2, 0] = w
        mats[lv] = m
    return LinearCouplingLaw("axial1", "2", mats, ("ar",), CITE)


def _rank_deficient_law() -> LinearCouplingLaw:
    """Two coordinates that both drive flexion identically."""
    mats = {}
    for lv, w in zip(LEVELS, WEIGHTS, strict=True):
        m = np.zeros((3, 2))
        m[0, :] = w
        mats[lv] = m
    return LinearCouplingLaw("rankdef", "1", mats, ("a", "b"), CITE)


def _target(law, phi, neutral=None) -> np.ndarray:
    return level_rotations(law, np.asarray(phi, float), neutral).compose()[:3, :3]


def _snapshot(law):
    return copy.deepcopy({lv: m.copy() for lv, m in law.matrices.items()})


# ------------------------------------------------------------------ round trip


def test_round_trip_recovers_phi() -> None:
    law = _cross_law()
    phi_true = np.array([0.35, -0.22, 0.41])
    res = distribute(_target(law, phi_true), law)
    assert isinstance(res, DistributionResult)
    assert res.converged
    np.testing.assert_allclose(res.phi, phi_true, atol=1e-8)
    assert res.residual_rad < 1e-9


def test_round_trip_with_neutral_and_offsets() -> None:
    law = _cross_law()
    neutral = {SpinalLevel.L4_L5: np.eye(4)}
    neutral[SpinalLevel.L4_L5][:3, :3] = Rotation.from_euler("z", 0.12).as_matrix()
    base = Rotation.from_euler("y", 0.2).as_matrix()
    term = Rotation.from_euler("x", -0.1).as_matrix()
    phi_true = np.array([-0.3, 0.25, 0.2])
    target = base @ _target(law, phi_true, neutral) @ term
    res = distribute(target, law, neutral=neutral, fixed=ChainOffsets(base, term))
    np.testing.assert_allclose(res.phi, phi_true, atol=1e-8)
    assert res.residual_rad < 1e-9


def test_result_metadata_and_levels() -> None:
    law = _cross_law()
    res = distribute(_target(law, [0.1, 0.2, 0.3]), law)
    assert res.law_id == "fixture3"
    assert res.law_version == "1"
    assert res.distribution_source == "coupling:fixture3@1"
    assert set(res.level_angles) == set(LEVELS)
    expect = law.expand(res.phi)
    for lv in LEVELS:
        np.testing.assert_allclose(res.level_angles[lv], expect[lv])
    assert res.pose.levels == LEVELS
    assert not res.phi.flags.writeable


def test_state_dependent_law_round_trip() -> None:
    mats = {}
    for lv, w in zip(LEVELS, WEIGHTS, strict=True):
        lo = w * np.eye(3)
        hi = 1.4 * w * np.eye(3)
        mats[lv] = np.stack([lo, hi])
    law = StateDependentCouplingLaw(
        "sd", "1", ("fe", "lb", "ar"), "fe", np.array([-1.0, 1.0]), mats, CITE
    )
    phi_true = np.array([0.3, 0.1, -0.2])
    res = distribute(_target(law, phi_true), law)
    np.testing.assert_allclose(res.phi, phi_true, atol=1e-7)
    assert res.residual_rad < 1e-8


def test_analytic_jacobian_matches_finite_difference() -> None:
    from src.shared.python.spine.distribution import residual_and_jacobian

    law = _cross_law()
    target = (
        _target(law, [0.2, -0.1, 0.3])
        @ Rotation.from_euler("xyz", [0.1, 0.05, -0.07]).as_matrix()
    )
    phi = np.array([0.15, 0.05, -0.2])
    _, jac = residual_and_jacobian(phi, law, target, None, None)
    eps = 1e-6
    fd = np.zeros_like(jac)
    for j in range(3):
        d = np.zeros(3)
        d[j] = eps
        rp, _ = residual_and_jacobian(phi + d, law, target, None, None)
        rm, _ = residual_and_jacobian(phi - d, law, target, None, None)
        fd[:, j] = (rp - rm) / (2 * eps)
    np.testing.assert_allclose(jac, fd, atol=1e-7)


# ----------------------------------------------------------------- unreachable


def test_unreachable_target_reports_known_geodesic_distance() -> None:
    law = _axial_law()
    before = _snapshot(law)
    a, b = 0.6, 0.4
    target = (
        Rotation.from_euler("y", a).as_matrix()
        @ Rotation.from_euler("x", b).as_matrix()
    )
    res = distribute(target, law)
    assert res.converged
    assert res.residual_rad == pytest.approx(b, abs=1e-7)
    np.testing.assert_allclose(res.phi, [a], atol=1e-6)
    after = _snapshot(law)
    for lv in LEVELS:
        assert np.array_equal(before[lv], after[lv])
    assert res.observability.rank == 1


def test_residual_equals_recomputed_geodesic_distance() -> None:
    law = _axial_law()
    target = Rotation.from_euler("zyx", [0.3, 0.5, -0.2]).as_matrix()
    res = distribute(target, law)
    got = res.pose.compose()[:3, :3]
    expect = np.linalg.norm(Rotation.from_matrix(target.T @ got).as_rotvec())
    assert res.residual_rad == pytest.approx(expect, abs=1e-10)


def test_check_tolerance_flags_but_never_rescales() -> None:
    law = _axial_law()
    target = Rotation.from_euler("x", 0.4).as_matrix()
    res = distribute(target, law)
    assert res.check_tolerance(1.0)
    assert not res.check_tolerance(0.1)
    with pytest.raises(ValueError, match="exceeds"):
        res.check_tolerance(0.1, raise_on_exceed=True)
    with pytest.raises(ValueError):
        res.check_tolerance(-1.0)


# --------------------------------------------------------------------- bounds


def test_bounds_respected() -> None:
    law = _cross_law()
    phi_true = np.array([0.5, 0.0, 0.0])
    res = distribute(
        _target(law, phi_true), law, bounds={"fe": (-0.2, 0.2), "lb": (-1, 1)}
    )
    assert res.phi[0] <= 0.2 + 1e-12
    assert res.residual_rad > 1e-3


def test_infeasible_start_rejected() -> None:
    law = _cross_law()
    with pytest.raises(ValueError, match="outside bounds"):
        distribute(
            np.eye(3), law, bounds={"fe": (-0.2, 0.2)}, phi0=np.array([0.5, 0, 0])
        )


@pytest.mark.parametrize(
    "bounds",
    [
        {"fe": (0.2, -0.2)},
        {"fe": (-np.inf, 1.0)},
        {"fe": (np.nan, 1.0)},
        {"nope": (-1.0, 1.0)},
        {"fe": (0.0, 0.0)},
    ],
)
def test_bad_bounds_rejected(bounds) -> None:
    with pytest.raises(ValueError):
        distribute(np.eye(3), _cross_law(), bounds=bounds)


# ---------------------------------------------------------------------- series


def _trajectory(law, n=25):
    t = np.linspace(0.0, 1.0, n)
    phis = np.stack(
        [0.5 * np.sin(2 * np.pi * t), 0.3 * t, 0.9 * np.sin(np.pi * t)], axis=1
    )
    targets = np.stack([_target(law, p) for p in phis])
    return t, phis, targets


def test_series_continuity_and_warm_start_matches_cold() -> None:
    law = _cross_law()
    t, phis, targets = _trajectory(law)
    detail = distribute_series_detailed(targets, t, law)
    got = np.stack([r.phi for r in detail.results])
    np.testing.assert_allclose(got, phis, atol=1e-8)
    assert np.max(np.abs(np.diff(got, axis=0))) < 0.2  # no jumps
    for k in (0, 7, 24):
        cold = distribute(targets[k], law)
        np.testing.assert_allclose(cold.phi, got[k], atol=1e-8)
    assert np.all(detail.residual_rad < 1e-9)


def test_series_returns_kinematics_series() -> None:
    law = _cross_law()
    t, _, targets = _trajectory(law, 6)
    series = distribute_series(targets, t, law)
    assert isinstance(series, SegmentalKinematicsSeries)
    assert series.distribution_source == "coupling:fixture3@1"
    assert series.valid.all()
    assert len(series.poses) == 6
    np.testing.assert_allclose(series.time_s, t)


def test_series_flags_unconverged_frames_invalid_not_repaired() -> None:
    law = _axial_law()
    t = np.array([0.0, 1.0])
    targets = np.stack([np.eye(3), Rotation.from_euler("x", 0.5).as_matrix()])
    detail = distribute_series_detailed(targets, t, law, tolerance_rad=0.1)
    assert list(detail.series.valid) == [True, False]
    assert detail.residual_rad[1] == pytest.approx(0.5, abs=1e-7)


def test_series_preconditions() -> None:
    law = _cross_law()
    with pytest.raises(ValueError, match="shape"):
        distribute_series(np.eye(3), np.array([0.0]), law)
    with pytest.raises(ValueError, match="length"):
        distribute_series(np.stack([np.eye(3)] * 3), np.array([0.0, 1.0]), law)
    with pytest.raises(ValueError, match="increasing"):
        distribute_series(np.stack([np.eye(3)] * 2), np.array([1.0, 0.0]), law)


# --------------------------------------------------------------- observability


def test_full_rank_law_report() -> None:
    law = _cross_law()
    res = distribute(_target(law, [0.1, 0.1, 0.1]), law)
    obs = res.observability
    assert obs.rank == 3
    assert not obs.underdetermined
    assert obs.flagged_coordinates == ()
    assert obs.null_space.shape == (3, 0)


def test_rank_deficient_law_flagged() -> None:
    law = _rank_deficient_law()
    target = Rotation.from_euler("z", 0.5).as_matrix()
    res = distribute(target, law)
    obs = res.observability
    assert obs.rank == 1
    assert obs.underdetermined
    assert set(obs.flagged_coordinates) == {"a", "b"}
    assert obs.null_space.shape == (2, 1)
    np.testing.assert_allclose(
        np.abs(obs.null_space[:, 0]), np.full(2, np.sqrt(0.5)), atol=1e-8
    )
    assert obs.unobservable_target_directions.shape == (3, 2)
    assert res.residual_rad < 1e-8


# --------------------------------------------------------------- preconditions


def test_non_rotation_target_rejected() -> None:
    law = _cross_law()
    with pytest.raises(ValueError, match="proper rotation"):
        distribute(2.0 * np.eye(3), law)
    refl = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="proper rotation"):
        distribute(refl, law)


def test_nan_and_shape_rejected() -> None:
    law = _cross_law()
    bad = np.eye(3)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        distribute(bad, law)
    with pytest.raises(ValueError, match="shape"):
        distribute(np.eye(4), law)
    with pytest.raises(TypeError):
        distribute("abc", law)  # type: ignore[arg-type]


def test_missing_law_rejected() -> None:
    with pytest.raises(TypeError, match="CouplingLaw"):
        distribute(np.eye(3), None)  # type: ignore[arg-type]


def test_phi0_and_solver_args_rejected() -> None:
    law = _cross_law()
    with pytest.raises(ValueError, match="phi0"):
        distribute(np.eye(3), law, phi0=np.zeros(2))
    with pytest.raises(ValueError, match="tol_rad"):
        distribute(np.eye(3), law, tol_rad=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        distribute(np.eye(3), law, max_iter=0)


def test_chain_offsets_must_be_rotations() -> None:
    with pytest.raises(ValueError, match="proper rotation"):
        ChainOffsets(2.0 * np.eye(3), np.eye(3))
