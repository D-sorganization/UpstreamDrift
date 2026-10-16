"""NSX-95 (#12724): strategy evaluators compose the dependencies, never fake them.

Every implemented strategy must call a real evaluator and be reproducible from
it; a strategy whose declared mechanism is absent must show no effect (negative
control); an entry without an evaluator must come back ``NOT_IMPLEMENTED`` or
``UNAVAILABLE`` with its reason and no numbers.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.evidence import (  # noqa: E402
    AllocationDependence,
)
from src.shared.python.loop_nullspace.noise_aware_allocation import (  # noqa: E402
    minimise_face_variance,
)
from src.shared.python.loop_nullspace.strategy_arms import (  # noqa: E402
    ArmSpec,
    ArmStatus,
    ComparisonContext,
)
from src.shared.python.loop_nullspace.strategy_evaluators import (  # noqa: E402
    evaluate_arms,
)
from src.shared.python.loop_nullspace.strategy_spec import StrategyId  # noqa: E402
from tests.unit.loop_nullspace import _strategy_fixtures as sf  # noqa: E402

pytestmark = pytest.mark.unit


def _one(spec: ArmSpec, ctx: ComparisonContext):
    (arm,) = evaluate_arms((spec,), ctx)
    return arm


@pytest.fixture(scope="module")
def ctx() -> ComparisonContext:
    return sf.comparison()


# --------------------------------------------------------------------- S1
def test_s1_reproduces_the_nsx92_optimum_and_lowers_the_variance(ctx) -> None:
    arm = _one(ArmSpec(StrategyId.S1, "s1"), ctx)
    alloc = ctx.allocation
    direct = minimise_face_variance(
        alloc.feasible_sets,
        alloc.times_s,
        alloc.torque_scale,
        alloc.model,
        alloc.declaration,
    )
    assert arm.status is ArmStatus.OK
    assert arm.face_variance == pytest.approx(direct.variance, rel=1e-12)
    assert arm.baseline_variance == pytest.approx(direct.reference_variance, rel=1e-12)
    assert arm.ratio < 1.0 and arm.effect > 0.0
    assert arm.certificate == "global-convex"
    assert arm.chs_mps == sf.NOMINAL_CHS
    assert arm.effort_basis and arm.effort >= arm.baseline_effort


def test_s1_matched_effort_cap_is_respected(ctx) -> None:
    arm = _one(ArmSpec(StrategyId.S1, "s1-cap", effort_cap=sf.EFFORT_CAP), ctx)
    assert arm.effort <= sf.EFFORT_CAP * (1 + 1e-6)
    free = _one(ArmSpec(StrategyId.S1, "s1"), ctx)
    assert free.face_variance <= arm.face_variance + 1e-12 < arm.baseline_variance


def test_negative_control_s1_without_a_declared_mechanism_shows_no_effect() -> None:
    flat = sf.comparison(m1=False)  # additive noise only, no preload
    arm = _one(ArmSpec(StrategyId.S1, "s1-flat"), flat)
    assert arm.status is ArmStatus.NO_MECHANISM
    assert arm.effect == pytest.approx(0.0, abs=1e-15)
    assert arm.ratio == pytest.approx(1.0, abs=1e-15)
    assert arm.record.allocation_dependence is AllocationDependence.INVARIANT
    assert "no declared mechanism" in arm.reason


def test_variance_budget_by_input_class_sums_to_the_total(ctx) -> None:
    arm = _one(ArmSpec(StrategyId.S1, "s1"), ctx)
    assert set(arm.budget) == {"initial_state", "additive_noise", "signal_dependent"}
    assert sum(arm.budget.values()) == pytest.approx(arm.face_variance, rel=1e-12)
    assert arm.budget["signal_dependent"] > arm.budget["additive_noise"]


# --------------------------------------------------------------------- S5
def test_s5_isolates_the_geometric_preload_channel() -> None:
    ctx_m2 = sf.comparison(m1=False, m2=True, sigma0=(0.2, 0.3))
    arm = _one(ArmSpec(StrategyId.S5, "s5"), ctx_m2)
    assert arm.status is ArmStatus.OK
    assert arm.certificate == "local"  # M2 is non-convex: stationary, not global
    assert 0.0 < arm.effect < arm.baseline_variance  # a small, real effect
    assert "H-GEOMETRIC-PRELOAD" in arm.mechanisms


def test_negative_control_s5_without_preload_shows_no_effect(ctx) -> None:
    arm = _one(ArmSpec(StrategyId.S5, "s5"), ctx)  # M1 only, no preload
    assert arm.status is ArmStatus.NO_MECHANISM
    assert arm.ratio == 1.0 and arm.effect == 0.0
    assert "H-GEOMETRIC-PRELOAD" in arm.reason


def test_s1_does_not_borrow_the_preload_channel() -> None:
    both = sf.comparison(m1=True, m2=True)
    s1 = _one(ArmSpec(StrategyId.S1, "s1"), both)
    # M1-only isolation: the frozen-preload problem is convex and global
    assert s1.status is ArmStatus.OK and s1.certificate == "global-convex"
    assert "H-GEOMETRIC-PRELOAD" not in s1.mechanisms
    assert "H-NOISE-SIGNAL-DEPENDENT" in s1.mechanisms


# --------------------------------------------------------------- S2 and S3
def test_s2_impedance_lowers_the_variance_and_costs_effort(ctx) -> None:
    arm = _one(ArmSpec(StrategyId.S2, "s2", impedance=sf.impedance()), ctx)
    assert arm.status is ArmStatus.OK
    assert arm.ratio < 1 / 3  # NSX-93: stiff loop is several times quieter
    assert arm.effort > 0.0 and arm.baseline_effort == 0.0
    assert "H-DECLARED-JOINT-IMPEDANCE" in arm.hypotheses
    assert set(arm.budget) == {"initial_state", "additive_noise"}
    assert sum(arm.budget.values()) == pytest.approx(arm.face_variance, rel=1e-9)


def test_negative_control_zero_impedance_shows_no_effect(ctx) -> None:
    zero = sf.impedance(k=0.0, d=0.0)
    arm = _one(ArmSpec(StrategyId.S2, "s2-zero", impedance=zero), ctx)
    assert arm.status is ArmStatus.NO_MECHANISM
    assert arm.ratio == pytest.approx(1.0, rel=1e-12)
    assert arm.effort == pytest.approx(0.0, abs=1e-18)


def test_s2_noise_inflation_can_make_stiffness_counterproductive(ctx) -> None:
    quiet = _one(ArmSpec(StrategyId.S2, "q", impedance=sf.impedance()), ctx)
    noisy = _one(ArmSpec(StrategyId.S2, "n", impedance=sf.impedance(kappa=0.05)), ctx)
    assert noisy.face_variance > quiet.face_variance  # declared penalty bites


def test_s3_late_window_helps_less_than_full_horizon_stiffness(ctx) -> None:
    t_end = sf.LAYER_STEPS * sf.LAYER_DT
    full = _one(ArmSpec(StrategyId.S2, "full", impedance=sf.impedance()), ctx)
    late = _one(
        ArmSpec(StrategyId.S3, "late", impedance=sf.impedance(0.75 * t_end, t_end)),
        ctx,
    )
    assert late.status is ArmStatus.OK
    # Effort is not assumed: early regulation keeps the state (and so the
    # feedback torque) small, a late window must fight the accumulated error.
    assert late.effort > 0.0 and full.effort > 0.0
    assert late.ratio < 1.0 and late.ratio > full.ratio


def test_s3_requires_a_window_ending_at_impact(ctx) -> None:
    early = sf.impedance(0.0, 0.1)
    with pytest.raises(ValueError, match="impact"):
        _one(ArmSpec(StrategyId.S3, "early", impedance=early), ctx)


def test_s2_cocontraction_traces_the_u_curve_minimum() -> None:
    mctx = sf.comparison(with_muscle=True)
    arm = _one(ArmSpec(StrategyId.S2, "cocon", cocontraction=True), mctx)
    assert arm.status is ArmStatus.OK
    assert arm.ratio < 1.0  # the optimum co-contraction beats the lowest level
    assert arm.diagnostics["level"] > arm.diagnostics["lowest_level"]
    assert arm.effort > arm.baseline_effort
    assert {"muscle_noise", "external_disturbance"} <= set(arm.budget)


# --------------------------------------------------------------------- S7
def test_s7_delayed_feedback_lowers_variance_within_the_margin(ctx) -> None:
    spec = ArmSpec(StrategyId.S7, "s7", feedback_gain=sf.feedback_gain(), delay_steps=2)
    arm = _one(spec, ctx)
    assert arm.status is ArmStatus.OK
    assert arm.ratio < 1.0
    assert arm.diagnostics["delay_margin_s"] > 2 * sf.LAYER_DT
    assert "H-SENSORIMOTOR-DELAY" in arm.hypotheses


def test_s7_beyond_the_delay_margin_is_reported_as_a_failure(ctx) -> None:
    spec = ArmSpec(
        StrategyId.S7, "s7-late", feedback_gain=sf.feedback_gain(), delay_steps=30
    )
    arm = _one(spec, ctx)
    assert arm.status is ArmStatus.UNSTABLE
    assert arm.face_variance is None and "margin" in arm.reason


def test_s7_zero_gain_is_a_negative_control(ctx) -> None:
    spec = ArmSpec(
        StrategyId.S7,
        "s7-zero",
        feedback_gain=np.zeros_like(sf.feedback_gain()),
        delay_steps=3,
    )
    arm = _one(spec, ctx)
    assert arm.status is ArmStatus.NO_MECHANISM
    assert arm.ratio == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------- S8
def test_s8_speed_cases_come_from_the_chs_front(ctx) -> None:
    alloc = ctx.allocation
    cases = (sf.nominal_case(alloc), sf.slow_case(alloc))
    spec = ArmSpec(
        StrategyId.S8,
        "s8",
        speed_cases=cases,
        reference_case="nominal",
        effort_cap=sf.EFFORT_CAP,
    )
    arms = evaluate_arms((spec,), ctx)
    assert [a.label for a in arms] == ["s8:nominal", "s8:slow"]
    nominal, slow = arms
    assert (nominal.chs_mps, slow.chs_mps) == (sf.NOMINAL_CHS, sf.SLOW_CHS)
    assert slow.face_variance < nominal.face_variance  # the tempting shortcut
    assert slow.baseline_variance == nominal.baseline_variance  # common baseline
    assert nominal.effort <= sf.EFFORT_CAP * (1 + 1e-6)


def test_s8_requires_an_effort_cap_and_a_known_reference(ctx) -> None:
    cases = (sf.nominal_case(ctx.allocation), sf.slow_case(ctx.allocation))
    with pytest.raises(ValueError, match="effort_cap"):
        _one(
            ArmSpec(StrategyId.S8, "s8", speed_cases=cases, reference_case="nominal"),
            ctx,
        )
    with pytest.raises(ValueError, match="reference_case"):
        _one(
            ArmSpec(
                StrategyId.S8,
                "s8",
                speed_cases=cases,
                reference_case="nope",
                effort_cap=sf.EFFORT_CAP,
            ),
            ctx,
        )


# ------------------------------------------------- not implemented / missing
@pytest.mark.parametrize(
    ("sid", "status"),
    [
        (StrategyId.S4, ArmStatus.NOT_IMPLEMENTED),
        (StrategyId.S6, ArmStatus.NOT_IMPLEMENTED),
        (StrategyId.S9, ArmStatus.NOT_IMPLEMENTED),
        (StrategyId.S10, ArmStatus.UNAVAILABLE),
        (StrategyId.S11, ArmStatus.NOT_IMPLEMENTED),
        (StrategyId.S12, ArmStatus.NOT_IMPLEMENTED),
    ],
)
def test_unimplemented_strategies_return_a_reason_and_no_numbers(
    ctx, sid, status
) -> None:
    arm = _one(ArmSpec(sid, "x"), ctx)
    assert arm.status is status
    assert arm.reason and arm.face_variance is None and arm.record is None
    assert math.isnan(arm.ratio) and math.isnan(arm.effect)


def test_a_missing_representation_context_is_unavailable_not_guessed() -> None:
    no_layer = dataclasses.replace(sf.comparison(), layer=None)
    arm = _one(ArmSpec(StrategyId.S2, "s2", impedance=sf.impedance()), no_layer)
    assert arm.status is ArmStatus.UNAVAILABLE and "layer" in arm.reason


def test_spec_preconditions() -> None:
    with pytest.raises(ValueError, match="label"):
        ArmSpec(StrategyId.S1, "")
    with pytest.raises(ValueError, match="delay_steps"):
        ArmSpec(StrategyId.S7, "x", feedback_gain=sf.feedback_gain(), delay_steps=-1)
    with pytest.raises(ValueError, match="effort_cap"):
        ArmSpec(StrategyId.S1, "x", effort_cap=-1.0)
    with pytest.raises(ValueError, match="S2.*impedance"):
        _one(ArmSpec(StrategyId.S2, "x"), sf.comparison())
