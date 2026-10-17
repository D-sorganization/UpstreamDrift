"""Tests for open-loop versus feedback-assisted replay (NSX-31, #12571).

The oracle is the independent planar two-arm golfer.  The reference trajectory
is produced by ``scipy`` DOP853 at ``rtol = 1e-13`` with dense output, never by
the fixed-step RK4 that the replay uses, so the comparison is not an inverse
crime (R13).  The replay torque is the exact torque the reference was driven by,
sampled at the allocation times.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.engines.feedback_comparison import ReplayPolicy
from src.shared.python.loop_nullspace.allocation_spec import (
    AllocationResult,
    ObjectiveSpec,
)
from src.shared.python.loop_nullspace.decomposition import decompose
from src.shared.python.loop_nullspace.constrained_dynamics import (
    constrained_acceleration,
)
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.fixtures import (
    PlanarGolferConfig,
    generic_planar_config,
    load_golden,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    PlanarGolferProvider,
    planar_dynamics,
)
from src.shared.python.loop_nullspace.replay import (
    BaumgarteGains,
    FeedbackLaw,
    ReferenceTrajectory,
    ReplayMode,
    ReplayReport,
    ReplaySettings,
    ReplaySetup,
    feedback_budget,
    replay,
)
from src.shared.python.loop_nullspace.solvers import SolveStatus

pytestmark = pytest.mark.unit

N = 5
S = np.eye(N)
T_END = 0.3
AMPLITUDE = np.array([6.0, 4.0, 2.0, 5.0, 3.0])
OMEGA = 2.0 * np.pi * 3.0
BIAS = np.array([1.0, -0.5, 0.2, 0.7, -0.3])


def _params() -> PlanarGolferParams:
    return PlanarGolferParams.from_golden(load_golden("planar_two_arm_full"))


def _tau(t: float) -> np.ndarray:
    return BIAS + AMPLITUDE * np.sin(OMEGA * t)


def _state0() -> tuple[np.ndarray, np.ndarray]:
    cfg = generic_planar_config()
    return np.array(cfg.q), np.array(cfg.v)


@functools.lru_cache(maxsize=4)
def _reference(tau_of_t, t_end: float = T_END):
    """DOP853 dense solution of the closed-loop dynamics (not the replay RK4)."""
    params = _params()
    q0, v0 = _state0()

    def rhs(t, y):
        dyn = planar_dynamics(y[:N], y[N:], params)
        acc, _ = constrained_acceleration(
            dyn["M"], dyn["h"], tau_of_t(t), dyn["Jc"], dyn["Jdot_v"]
        )
        return np.concatenate([y[N:], acc])

    return solve_ivp(
        rhs,
        (0.0, t_end),
        np.concatenate([q0, v0]),
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
        dense_output=True,
    )


def _allocation(times: np.ndarray, torque: np.ndarray) -> AllocationResult:
    """A hand-built allocation of exact oracle torques (test fixture only)."""
    record = QuantityRecord(
        values=torque,
        units="N m",
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        model_provenance=ModelProvenance("planar_two_arm_golfer_numpy", "1"),
        input_provenance=InputProvenance("test", "exact oracle torque"),
        hypotheses=("H-TORQUE-CAPACITY",),
    )
    return AllocationResult(
        SolveStatus.OPTIMAL, times_s=times, torque=torque, record=record
    )


def _exact_allocation(n_frames: int, tau_of_t=_tau) -> AllocationResult:
    times = np.linspace(0.0, T_END, n_frames)
    return _allocation(times, np.array([tau_of_t(t) for t in times]))


def _setup(reference=None) -> ReplaySetup:
    q0, v0 = _state0()
    return ReplaySetup(actuation=S, q0=q0, v0=v0, reference=reference)


def _error_at_frames(report: ReplayReport, sol) -> float:
    exact = sol.sol(report.times_s)
    return float(np.max(np.abs(report.q - exact[:N].T)))


def _order(errors: list[float]) -> float:
    """Observed order from successive halvings (least-squares slope of log2)."""
    return float(np.mean(np.log2(np.array(errors[:-1]) / np.array(errors[1:]))))


def _open_loop(allocation, provider=None, **settings):
    return replay(
        allocation,
        provider or PlanarGolferProvider(_params()),
        policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        setup=_setup(),
        settings=ReplaySettings(**settings),
    )


# ------------------------------------------------------------ open loop --
def test_open_loop_replay_of_exact_torques_reproduces_the_reference() -> None:
    sol = _reference(_tau)
    report = _open_loop(
        _exact_allocation(31),
        substeps=4,
        interpolation="cubic",
        divergence_tolerance=1.0,
    )
    assert report.mode is ReplayMode.OPEN_LOOP
    # the trajectory moves appreciably, so a small error is not a trivial zero
    assert float(np.max(np.abs(report.q[-1] - report.q[0]))) > 0.05
    # observed 5.2e-7: the cubic truncation of a 3 Hz torque sampled at 100 Hz
    assert _error_at_frames(report, sol) < 2e-6


def test_rk4_is_fourth_order_for_a_held_load() -> None:
    """Constant torque is exactly representable, so the error is the integrator."""
    const = np.array([3.0, -2.0, 1.0, 2.5, -1.5])
    sol = _reference(lambda t: const)
    errors = []
    for substeps in (1, 2, 4):
        report = _open_loop(
            _exact_allocation(7, lambda t: const),
            substeps=substeps,
            interpolation="zoh",
            divergence_tolerance=1.0,
        )
        errors.append(_error_at_frames(report, sol))
    assert errors[0] > errors[1] > errors[2]
    assert 3.6 < _order(errors) < 4.6


@pytest.mark.parametrize(
    ("interpolation", "frames", "low", "high"),
    [
        ("zoh", (7, 13, 25), 0.9, 1.3),
        ("linear", (7, 13, 25), 1.8, 2.2),
        # observed 4.84 then 3.37 as the RK4 floor (~1e-7) is approached
        ("cubic", (13, 25, 49), 3.5, 4.8),
    ],
)
def test_allocation_interpolation_has_its_known_order(
    interpolation: str, frames: tuple[int, ...], low: float, high: float
) -> None:
    """Halving the allocation spacing with the RK4 step held at <= 1 ms."""
    sol = _reference(_tau)
    errors = []
    for n_frames in frames:
        substeps = max(1, round(T_END / (n_frames - 1) / 1e-3))
        report = _open_loop(
            _exact_allocation(n_frames),
            substeps=substeps,
            interpolation=interpolation,
            divergence_tolerance=1.0,
        )
        errors.append(_error_at_frames(report, sol))
    assert low < _order(errors) < high, errors


def test_replay_resets_no_state_and_is_deterministic() -> None:
    a = _open_loop(_exact_allocation(11), substeps=2)
    b = _open_loop(_exact_allocation(11), substeps=2)
    assert a.state_resets == 0
    np.testing.assert_array_equal(a.q, b.q)
    assert a.q[0].tolist() == _state0()[0].tolist()


# ------------------------------------------------------------- closure --
def test_closure_drift_is_reported_in_open_loop_mode() -> None:
    report = _open_loop(_exact_allocation(31), substeps=4, interpolation="cubic")
    drift = report.closure_drift
    assert drift.velocity.shape == report.times_s.shape
    assert drift.position is not None and drift.position.shape == report.times_s.shape
    # observed 1.1e-7 velocity residual: RK4 truncation of the index-reduced loop
    assert drift.velocity_max < 5e-7 and drift.position_max < 5e-7
    assert (
        report.stabilisation is None and report.stabilisation_acceleration_peak == 0.0
    )


def test_baumgarte_is_declared_and_reported_apart_from_the_physics() -> None:
    q0, v0 = _state0()
    provider = PlanarGolferProvider(_params())
    bad_q0 = q0 + np.array([0.0, 0.0, 0.0, 2e-3, -2e-3])  # violates Phi(q0) = 0
    assert float(np.max(np.abs(provider.closure_error(bad_q0)))) > 1e-4
    allocation = _exact_allocation(31)
    runs = {}
    for label, gains in (("plain", None), ("stabilised", BaumgarteGains(30.0, 30.0))):
        runs[label] = replay(
            allocation,
            provider,
            policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
            setup=ReplaySetup(actuation=S, q0=bad_q0, v0=v0),
            settings=ReplaySettings(substeps=4, baumgarte=gains),
        )
    plain, stab = runs["plain"], runs["stabilised"]
    assert plain.stabilisation is None and plain.stabilisation_acceleration_peak == 0.0
    assert stab.stabilisation == BaumgarteGains(30.0, 30.0)
    assert stab.stabilisation_acceleration_peak > 0.0
    # the declared stabilisation pulls the closure back; the plain physics does not
    assert stab.closure_drift.position[-1] < 0.1 * plain.closure_drift.position[-1]
    assert plain.closure_drift.position[-1] > 1e-4


def test_baumgarte_requires_a_position_level_closure_error() -> None:
    class NoPosition:
        def __init__(self, inner):
            self._inner = inner

        def mass_matrix(self, q):
            return self._inner.mass_matrix(q)

        def bias_forces(self, q, v):
            return self._inner.bias_forces(q, v)

        def closure_jacobian(self, q):
            return self._inner.closure_jacobian(q)

        def closure_drift(self, q, v):
            return self._inner.closure_drift(q, v)

    with pytest.raises(ValueError, match="closure_error"):
        replay(
            _exact_allocation(5),
            NoPosition(PlanarGolferProvider(_params())),
            policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
            setup=_setup(),
            settings=ReplaySettings(baumgarte=BaumgarteGains(5.0, 5.0)),
        )


def test_position_drift_is_unavailable_without_a_closure_error_not_zero() -> None:
    class NoPosition:
        def __init__(self, inner):
            self._inner = inner

        def mass_matrix(self, q):
            return self._inner.mass_matrix(q)

        def bias_forces(self, q, v):
            return self._inner.bias_forces(q, v)

        def closure_jacobian(self, q):
            return self._inner.closure_jacobian(q)

        def closure_drift(self, q, v):
            return self._inner.closure_drift(q, v)

    report = replay(
        _exact_allocation(5),
        NoPosition(PlanarGolferProvider(_params())),
        policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        setup=_setup(),
    )
    assert report.closure_drift.position is None
    assert report.closure_drift.velocity.shape == report.times_s.shape


# ------------------------------------------------------------ feedback --
# Gains chosen for explicit-RK4 stability: kd / lambda_min(M) h < 2.78 at h = 2.5 ms.
LAW = FeedbackLaw(kp=np.full(N, 40.0), kd=np.full(N, 1.5))


def _feedback(allocation, provider=None, *, law=LAW, reference=None, **settings):
    sol = _reference(_tau)
    times = np.linspace(0.0, T_END, 61)
    states = sol.sol(times)
    ref = reference or ReferenceTrajectory(times, states[:N].T, states[N:].T)
    return replay(
        allocation,
        provider or PlanarGolferProvider(_params()),
        policy=ReplayPolicy.STATE_FEEDBACK,
        setup=_setup(ref),
        feedback=law,
        settings=ReplaySettings(**settings),
    )


def test_feedback_is_negligible_when_tracking_is_exact() -> None:
    report = _feedback(
        _exact_allocation(31),
        substeps=4,
        interpolation="cubic",
        divergence_tolerance=1.0,
    )
    assert report.mode is ReplayMode.FEEDBACK
    budget = report.feedback_budget
    assert budget is not None
    # observed 1.9e-5 and 5e-11: gains times the cubic-interpolation-level
    # tracking error (~5e-7 rad), against a mismatch case orders larger below
    assert budget.peak_norm_ratio < 1e-4
    assert budget.energy_ratio < 1e-9


def test_zero_gain_feedback_equals_open_loop_and_costs_exactly_nothing() -> None:
    zero = FeedbackLaw(kp=np.zeros(N), kd=np.zeros(N))
    allocation = _exact_allocation(31)
    fb = _feedback(allocation, law=zero, substeps=2)
    ol = _open_loop(allocation, substeps=2)
    np.testing.assert_array_equal(fb.q, ol.q)
    assert fb.feedback_budget.energy_ratio == 0.0
    assert fb.feedback_budget.peak_norm_ratio == 0.0
    assert not np.any(fb.feedback_torque)


def test_feedback_budget_grows_with_model_mismatch() -> None:
    allocation = _exact_allocation(31)
    ratios, errors = [], []
    for scale in (1.0, 1.05, 1.15, 1.4):
        params = dataclasses.replace(_params(), club_mass=_params().club_mass * scale)
        provider = PlanarGolferProvider(params)
        fb = _feedback(allocation, provider, substeps=4, interpolation="cubic")
        ol = _open_loop(allocation, provider, substeps=4, interpolation="cubic")
        ratios.append(fb.feedback_budget.energy_ratio)
        errors.append(float(np.max(np.abs(ol.q - fb.reference_q))))
    assert ratios == sorted(ratios) and ratios[0] < 1e-9 < 1e-3 < ratios[-1]
    assert errors == sorted(errors) and errors[-1] > 100 * errors[0] > 0.0


def test_feedback_torque_is_separate_from_the_allocated_torque() -> None:
    params = dataclasses.replace(_params(), club_mass=_params().club_mass * 1.2)
    allocation = _exact_allocation(31)
    fb = _feedback(allocation, PlanarGolferProvider(params), substeps=4)
    frames = allocation.times_s
    expected = np.column_stack(
        [np.interp(fb.times_s, frames, allocation.torque[:, j]) for j in range(N)]
    )
    np.testing.assert_allclose(fb.allocated_torque, expected, atol=1e-12)
    np.testing.assert_allclose(
        fb.applied_torque, fb.allocated_torque + fb.feedback_torque, atol=1e-12
    )
    assert np.max(np.abs(fb.feedback_torque)) > 1e-3
    ol = _open_loop(allocation, PlanarGolferProvider(params), substeps=4)
    assert ol.feedback_torque is None and ol.feedback_budget is None
    np.testing.assert_array_equal(ol.applied_torque, ol.allocated_torque)


def test_feedback_torque_limit_is_declared_and_counted() -> None:
    params = dataclasses.replace(_params(), club_mass=_params().club_mass * 1.4)
    limited = FeedbackLaw(kp=LAW.kp, kd=LAW.kd, torque_limit=np.full(N, 0.5))
    fb = _feedback(_exact_allocation(31), PlanarGolferProvider(params), law=limited)
    assert np.max(np.abs(fb.feedback_torque)) <= 0.5 + 1e-12
    assert fb.feedback_budget.saturated_fraction > 0.0


# --------------------------------------------------------------- budget --
def test_budget_accounting_on_hand_computable_series() -> None:
    times = np.linspace(0.0, 2.0, 5)
    allocated = np.tile([[2.0, 0.0]], (5, 1))
    feedback = np.tile([[0.0, 1.0]], (5, 1))
    velocity = np.tile([[3.0, -4.0]], (5, 1))
    budget = feedback_budget(times, allocated, feedback, velocity)
    assert budget.peak_norm_ratio == pytest.approx(0.5)
    assert budget.energy_ratio == pytest.approx(0.25)  # (1^2 * 2) / (2^2 * 2)
    assert budget.rms_ratio == pytest.approx(0.5)
    assert budget.abs_work_feedback_j == pytest.approx(8.0)  # |1 * -4| * 2 s
    assert budget.abs_work_allocated_j == pytest.approx(12.0)  # |2 * 3| * 2 s
    assert budget.abs_work_ratio == pytest.approx(8.0 / 12.0)
    np.testing.assert_allclose(budget.per_joint_rms_feedback_nm, [0.0, 1.0])
    np.testing.assert_allclose(budget.per_joint_rms_allocated_nm, [2.0, 0.0])
    assert budget.saturated_fraction == 0.0


def test_budget_ratio_is_infinite_not_nan_for_a_zero_allocation() -> None:
    times = np.linspace(0.0, 1.0, 3)
    zero = np.zeros((3, 2))
    one = np.ones((3, 2))
    budget = feedback_budget(times, zero, one, one)
    assert budget.energy_ratio == float("inf")
    none = feedback_budget(times, zero, zero, one)
    assert none.energy_ratio == 0.0


def test_budget_matches_an_independent_trapezoid_of_the_report_series() -> None:
    params = dataclasses.replace(_params(), club_mass=_params().club_mass * 1.2)
    fb = _feedback(_exact_allocation(31), PlanarGolferProvider(params), substeps=4)
    t = fb.times_s
    num = np.trapezoid(np.sum(fb.feedback_torque**2, axis=1), t)
    den = np.trapezoid(np.sum(fb.allocated_torque**2, axis=1), t)
    assert fb.feedback_budget.energy_ratio == pytest.approx(num / den, rel=1e-12)


# -------------------------------------------------------------- labels --
def test_mode_labels_are_enforced() -> None:
    allocation = _exact_allocation(5)
    provider = PlanarGolferProvider(_params())
    with pytest.raises(ValueError, match="open-loop"):
        replay(
            allocation,
            provider,
            policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
            setup=_setup(),
            feedback=LAW,
        )
    with pytest.raises(ValueError, match="feedback law"):
        replay(allocation, provider, policy=ReplayPolicy.STATE_FEEDBACK, setup=_setup())
    with pytest.raises(ValueError, match="reference"):
        replay(
            allocation,
            provider,
            policy=ReplayPolicy.STATE_FEEDBACK,
            setup=_setup(),
            feedback=LAW,
        )
    with pytest.raises(TypeError, match="ReplayPolicy"):
        replay(allocation, provider, policy="open_loop", setup=_setup())  # type: ignore[arg-type]


def test_reports_carry_their_mode_and_never_present_feedback_as_validation() -> None:
    allocation = _exact_allocation(31)
    ol = _open_loop(allocation, substeps=2)
    fb = _feedback(allocation, substeps=2)
    assert (ol.mode, ol.policy) == (
        ReplayMode.OPEN_LOOP,
        ReplayPolicy.INDEPENDENT_TIME_ONLY,
    )
    assert (fb.mode, fb.policy) == (ReplayMode.FEEDBACK, ReplayPolicy.STATE_FEEDBACK)
    assert ol.evidence_label == "independent_open_loop_replay"
    assert fb.evidence_label == "feedback_assisted_tracking"
    assert not fb.may_present_as_validation
    assert not ol.may_present_as_validation  # replay success is not acceptance
    assert (
        "physical acceptance" in ol.non_claim and "physical acceptance" in fb.non_claim
    )
    for report in (ol, fb):
        for record in report.records.values():
            assert record.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
            assert record.hypotheses == ("H-TORQUE-CAPACITY",)


def test_inconsistent_report_construction_is_refused() -> None:
    ol = _open_loop(_exact_allocation(5))
    fb = _feedback(_exact_allocation(31), substeps=1)
    with pytest.raises(ValueError, match="open-loop"):
        dataclasses.replace(ol, feedback_torque=fb.feedback_torque[:5])
    with pytest.raises(ValueError, match="feedback"):
        dataclasses.replace(fb, feedback_budget=None)
    with pytest.raises(ValueError, match="mode"):
        dataclasses.replace(fb, policy=ReplayPolicy.INDEPENDENT_TIME_ONLY)


# --------------------------------------------------------- divergence --
def test_time_to_divergence_is_reported_against_the_declared_tolerance() -> None:
    params = dataclasses.replace(_params(), club_mass=_params().club_mass * 1.4)
    sol = _reference(_tau)
    times = np.linspace(0.0, T_END, 61)
    states = sol.sol(times)
    ref = ReferenceTrajectory(times, states[:N].T, states[N:].T)
    allocation = _exact_allocation(31)
    ol = replay(
        allocation,
        PlanarGolferProvider(params),
        policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        setup=_setup(ref),
        settings=ReplaySettings(substeps=4, divergence_tolerance=0.02),
    )
    assert ol.divergence_tolerance == 0.02
    assert ol.time_to_divergence_s is not None
    assert 0.0 < ol.time_to_divergence_s < T_END
    exceeding = np.max(np.abs(ol.q - ol.reference_q), axis=1) > 0.02
    assert ol.time_to_divergence_s == pytest.approx(ol.times_s[np.argmax(exceeding)])
    exact = _open_loop(allocation, substeps=4)
    assert exact.time_to_divergence_s is None  # no reference supplied


def test_time_to_divergence_is_none_when_the_replay_stays_inside_tolerance() -> None:
    sol = _reference(_tau)
    times = np.linspace(0.0, T_END, 61)
    states = sol.sol(times)
    ref = ReferenceTrajectory(times, states[:N].T, states[N:].T)
    report = replay(
        _exact_allocation(31),
        PlanarGolferProvider(_params()),
        policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        setup=_setup(ref),
        settings=ReplaySettings(substeps=4, interpolation="cubic"),
    )
    assert report.time_to_divergence_s is None
    assert report.max_state_error < 1e-5


# ------------------------------------------------------ preconditions --
def test_replay_preconditions() -> None:
    provider = PlanarGolferProvider(_params())
    good = _exact_allocation(5)

    def run(allocation=good, setup=None, **kw):
        return replay(
            allocation,
            provider,
            policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
            setup=setup or _setup(),
            **kw,
        )

    with pytest.raises(ValueError, match="OPTIMAL"):
        run(AllocationResult(SolveStatus.INFEASIBLE))
    with pytest.raises(ValueError, match="torque"):
        run(_allocation(good.times_s, good.torque[:, :3]))
    with pytest.raises(ValueError, match="strictly increasing"):
        run(_allocation(good.times_s[::-1].copy(), good.torque))
    with pytest.raises(ValueError, match="substeps"):
        ReplaySettings(substeps=0)
    with pytest.raises(ValueError, match="interpolation"):
        ReplaySettings(interpolation="spline")
    with pytest.raises(ValueError, match="cubic"):
        run(_exact_allocation(3), settings=ReplaySettings(interpolation="cubic"))
    q0, v0 = _state0()
    with pytest.raises(ValueError, match="q0"):
        ReplaySetup(actuation=S, q0=q0[:3], v0=v0)
    with pytest.raises(ValueError, match="selection"):
        ReplaySetup(actuation=np.full((N, N), 0.2), q0=q0, v0=v0)
    with pytest.raises(ValueError, match="QuantityRecord"):
        run(dataclasses.replace(good, record=None))
    with pytest.raises(ValueError, match="gain"):
        FeedbackLaw(kp=-np.ones(N), kd=np.ones(N))
    with pytest.raises(ValueError, match="reference"):
        ReferenceTrajectory(np.array([0.0, 1.0]), np.zeros((3, N)), np.zeros((3, N)))


def test_reference_must_cover_the_replay_window() -> None:
    times = np.linspace(0.0, 0.1, 5)
    ref = ReferenceTrajectory(times, np.zeros((5, N)), np.zeros((5, N)))
    with pytest.raises(ValueError, match="cover"):
        replay(
            _exact_allocation(5),
            PlanarGolferProvider(_params()),
            policy=ReplayPolicy.STATE_FEEDBACK,
            setup=_setup(ref),
            feedback=LAW,
        )


def test_external_load_is_applied_and_excluded_from_the_feedback_budget() -> None:
    """A known external generalised load shifts the motion; it is not feedback."""
    load = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    allocation = _exact_allocation(31)
    q0, v0 = _state0()
    plain = _open_loop(allocation, substeps=2)
    loaded = replay(
        allocation,
        PlanarGolferProvider(_params()),
        policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        setup=ReplaySetup(actuation=S, q0=q0, v0=v0, external_load=lambda t, q: load),
        settings=ReplaySettings(substeps=2),
    )
    assert float(np.max(np.abs(loaded.q - plain.q))) > 1e-4
    assert loaded.feedback_budget is None
    np.testing.assert_array_equal(loaded.applied_torque, loaded.allocated_torque)


# ------------------------------------------------- allocation pipeline --
def _capacity_frame(decomposition, cap: float):
    """Torque-capacity rows only: a feasible set with room in the family."""
    from src.shared.python.loop_nullspace.feasible_rows import FeasibleRow
    from src.shared.python.loop_nullspace.feasible_set import FeasibleSet

    rows = []
    for i in range(decomposition.n_actuated):
        n_i, x_i = decomposition.N[i], decomposition.x_p[i]
        rows.append(
            FeasibleRow(n_i, cap - x_i, f"T:{i}:up", "H-TORQUE-CAPACITY", "torque")
        )
        rows.append(
            FeasibleRow(-n_i, cap + x_i, f"T:{i}:lo", "H-TORQUE-CAPACITY", "torque")
        )
    fs = FeasibleSet.from_rows(decomposition.nullity, rows, ())
    return dataclasses.replace(fs, decomposition=decomposition)


def test_replay_of_a_real_allocation_is_invariant_to_the_family_member() -> None:
    """The allocator picks a different member of the family; the motion is the same.

    NSX-23 returns the minimum-effort torque of the closed-loop family
    ``tau + J_c^T mu``.  It differs from the driving torque by several N m, yet
    the open-loop replay reproduces the reference because ``P_c`` (the KKT
    projector) is blind to that difference.
    """
    pytest.importorskip("highspy")
    pytest.importorskip("clarabel")
    from src.shared.python.loop_nullspace.allocation import allocate_trajectory

    sol = _reference(_tau)
    params = _params()
    times = np.linspace(0.0, T_END, 25)
    frames = []
    for t in times:
        state = sol.sol(t)
        q, v = state[:N], state[N:]
        dyn = planar_dynamics(q, v, params)
        acc, _ = constrained_acceleration(
            dyn["M"], dyn["h"], _tau(t), dyn["Jc"], dyn["Jdot_v"]
        )
        snap = planar_golfer_snapshot(
            PlanarGolferConfig(q, v, acc), actuation="full"
        ).snapshot
        frames.append(_capacity_frame(decompose(snap, weights=np.eye(N)), 500.0))
    result = allocate_trajectory(
        frames,
        ObjectiveSpec(
            times_s=times, torque_scale=np.array([20.0, 35.0, 50.0, 65.0, 80.0])
        ),
    )
    assert result.status is SolveStatus.OPTIMAL
    driving = np.array([_tau(t) for t in times])
    assert float(np.max(np.abs(result.torque - driving))) > 1.0
    report = _open_loop(result, substeps=10, interpolation="cubic")
    assert _error_at_frames(report, sol) < 5e-5  # observed 5.6e-6
    assert report.mode is ReplayMode.OPEN_LOOP
    assert set(report.records["closure_velocity_drift"].hypotheses) == set(
        result.record.hypotheses
    )
