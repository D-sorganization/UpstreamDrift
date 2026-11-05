"""SHOW-C3 (#14559): trust-region impact optimiser on a reduced model.

The reduced plant maps joint torques to impact conditions through the torque
time-integrals (a smooth nonlinear map), so the optimiser logic is exercised
without MuJoCo.  The matched swing misses the straight target (face +3 deg,
path -4 deg, AoA +2.5 deg) and the optimiser must correct it within the trust
region and the torque envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.impact_target import ImpactTarget
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.trajectory_optimiser import (
    ImpactTolerance,
    face_to_path_deg,
    OptimiserConfig,
    RolloutOutcome,
    solve_impact_trajectory,
)
from src.shared.python.impact_optimization.trust_region import TrustRegion
from src.shared.python.optimization.backend_registry import get_backend

pytestmark = pytest.mark.unit

TIMES = np.linspace(0.0, 0.4, 41)
N = 3
NAMES = ("J0", "J1", "J2")
BASE = np.stack(
    [60.0 * np.sin(np.pi * TIMES / 0.4), 30.0 * TIMES / 0.4, 20.0 + 0 * TIMES], axis=1
)
F = np.array([0.020, -0.010, 0.015])
P = np.array([-0.010, 0.030, 0.005])
A = np.array([0.012, 0.004, -0.020])
S = np.array([0.30, 0.20, 0.10])


@dataclass(frozen=True)
class _Impact:
    face_deg: float | None
    path_deg: float | None
    aoa_deg: float
    speed_mph: float
    strike_offset_x_m: float | None = 0.0
    strike_offset_y_m: float | None = 0.0


def _moments(tau: np.ndarray) -> np.ndarray:
    """First time-moment of each torque: independent of the integral."""
    return np.trapezoid(tau * TIMES[:, None], TIMES, axis=0)


def _integrals(tau: np.ndarray) -> np.ndarray:
    return np.trapezoid(tau, TIMES, axis=0)


_I0 = _integrals(BASE)
_M0 = _moments(BASE)
_OFF = (
    3.0 - float(F @ _I0),
    -4.0 - float(P @ _I0),
    2.5 - float(A @ _I0),
)


def plant(tau: np.ndarray) -> RolloutOutcome:
    i = _integrals(tau)
    return RolloutOutcome(
        impact=_Impact(
            face_deg=float(F @ i) + _OFF[0] + 0.05 * np.tanh(float(F @ i)),
            path_deg=float(P @ i) + _OFF[1],
            aoa_deg=float(A @ i) + _OFF[2],
            speed_mph=100.0 + float(S @ (_moments(tau) - _M0)),
        ),
        torque_nm=tau,
        trajectory=np.cumsum(tau, axis=0) * (TIMES[1] - TIMES[0]),
    )


def _envelope(
    limit: float = 5000.0, first: float | None = None
) -> PhysiologicalEnvelope:
    lim = np.full(N, limit)
    if first is not None:
        lim[0] = first
    return PhysiologicalEnvelope(
        names=NAMES,
        tau_max_negative_nm=lim.copy(),
        tau_max_positive_nm=lim.copy(),
        rom_deg=np.tile([-np.inf, np.inf], (N, 1)),
        vel_max_deg_s=np.full(N, 2000.0),
        assumed_torque_slots=(),
        assumed_velocity_slots=(),
    )


def _trust(radius: float) -> TrustRegion:
    return TrustRegion(
        reference=plant(BASE).trajectory, weights=np.ones(N), radius=radius
    )


def _solve(envelope: PhysiologicalEnvelope, radius: float = 2000.0, seed: int = 0):
    return solve_impact_trajectory(
        rollout=plant,
        baseline_torque_nm=BASE,
        times_s=TIMES,
        target=ImpactTarget(speed_mph=100.0),
        envelope=envelope,
        trust_region=_trust(radius),
        config=OptimiserConfig(n_knots=5, seed=seed, max_iterations=80),
    )


def test_matched_swing_misses_the_straight_target() -> None:
    m = plant(BASE).impact
    assert abs(m.face_deg) > 0.5 and abs(m.path_deg) > 1.5


def test_meets_acceptance_tolerances_and_holds_speed() -> None:
    sol = _solve(_envelope())
    imp = sol.outcome.impact
    assert abs(imp.face_deg) <= 0.5
    assert abs(imp.path_deg) <= 1.5
    assert abs(imp.aoa_deg) <= 1.0
    assert abs(imp.speed_mph - 100.0) <= 1.0
    assert sol.tolerances_met is True
    assert sol.deviation < ImpactTarget(speed_mph=100.0).deviation(plant(BASE).impact)


def test_reports_deviation_from_fit_and_stays_in_region() -> None:
    sol = _solve(_envelope(), radius=2000.0)
    assert sol.deviation_from_fit == pytest.approx(
        _trust(2000.0).deviation(sol.outcome.trajectory)
    )
    assert sol.deviation_from_fit <= 2000.0


def test_deterministic_for_a_fixed_seed() -> None:
    a, b = _solve(_envelope(), seed=7), _solve(_envelope(), seed=7)
    np.testing.assert_array_equal(a.knots, b.knots)
    assert a.deviation == b.deviation


def test_binding_joint_is_reported_not_silently_relaxed() -> None:
    tight = _envelope(first=BASE[:, 0].max() * 1.001)
    sol = _solve(tight)
    assert "J0" in sol.binding_joints
    assert sol.max_utilisation <= 1.0 + 1e-6
    assert sol.envelope_limits_nm[0] == pytest.approx(BASE[:, 0].max() * 1.001)


def test_tiny_trust_region_reports_unmet_tolerances() -> None:
    sol = _solve(_envelope(), radius=1e-6)
    assert sol.tolerances_met is False
    assert sol.deviation_from_fit == pytest.approx(
        _trust(1e-6).deviation(sol.outcome.trajectory)
    )


def test_tolerance_defaults_match_the_issue() -> None:
    t = ImpactTolerance()
    assert (t.face_deg, t.path_deg, t.aoa_deg) == (0.5, 1.5, 1.0)


def test_backend_is_registered_in_the_optimiser_registry() -> None:
    spec = get_backend("spline_trust_region")
    assert spec is not None and spec.available()


def test_reports_face_to_path_gap_so_the_trade_off_is_visible() -> None:
    sol = _solve(_envelope())
    imp = sol.outcome.impact
    assert sol.face_to_path_deg == pytest.approx(imp.face_deg - imp.path_deg)


def test_warm_start_from_a_solution_does_not_get_worse() -> None:
    first = _solve(_envelope())
    again = solve_impact_trajectory(
        rollout=plant,
        baseline_torque_nm=BASE,
        times_s=TIMES,
        target=ImpactTarget(speed_mph=100.0),
        envelope=_envelope(),
        trust_region=_trust(2000.0),
        config=OptimiserConfig(n_knots=5, seed=0, max_iterations=1),
        initial_knots=first.knots,
    )
    assert again.deviation <= first.deviation + 1e-9


def test_warm_start_shape_is_checked() -> None:
    with pytest.raises(ContractViolationError):
        solve_impact_trajectory(
            rollout=plant,
            baseline_torque_nm=BASE,
            times_s=TIMES,
            target=ImpactTarget(speed_mph=100.0),
            envelope=_envelope(),
            trust_region=_trust(2000.0),
            config=OptimiserConfig(n_knots=5),
            initial_knots=np.zeros((4, N)),
        )


def test_face_to_path_is_nan_when_face_or_path_unobservable() -> None:
    assert math.isnan(face_to_path_deg(_Impact(None, -4.0, 2.5, 100.0)))
    assert math.isnan(face_to_path_deg(_Impact(3.0, None, 2.5, 100.0)))
    assert face_to_path_deg(_Impact(3.0, -4.0, 2.5, 100.0)) == pytest.approx(7.0)
