"""NSX-95 (#12724): controlled strategy comparisons, rankings and reversals.

The controls are the point: a slower swing must not win a matched-speed
comparison however quiet it is, a strategy compared at an effort it exceeds is
excluded rather than relaxed, and a ranking that does not survive another prior
or the held-out seeds is reported as a reversal.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.noise_aware_allocation import (  # noqa: E402
    minimise_face_variance,
)
from src.shared.python.loop_nullspace.paired_mc_spec import (  # noqa: E402
    CampaignSpec,
    IntegratorSignature,
)
from src.shared.python.loop_nullspace.paired_monte_carlo import Strategy  # noqa: E402
from src.shared.python.loop_nullspace.replay_spec import ReplayMode  # noqa: E402
from src.shared.python.loop_nullspace.strategy_arms import (  # noqa: E402
    ArmSpec,
    ArmStatus,
)
from src.shared.python.loop_nullspace.strategy_campaign import (  # noqa: E402
    ComparisonMode,
    Exclusion,
    MatchSettings,
    run_comparison,
    variance_budget_table,
)
from src.shared.python.loop_nullspace.strategy_stability import (  # noqa: E402
    StabilityStatus,
    heldout_stability,
    rank_under_priors,
)
from src.shared.python.loop_nullspace.strategy_evaluators import (  # noqa: E402
    evaluate_arms,
)
from src.shared.python.loop_nullspace.strategy_spec import (  # noqa: E402
    NON_CLAIM,
    StrategyId,
)
from src.shared.python.perturbation.disturbance import (  # noqa: E402
    DisturbanceSpec,
    InjectionStage,
)
from tests.unit.loop_nullspace import _noise_aware_fixtures as fx  # noqa: E402
from tests.unit.loop_nullspace import _strategy_fixtures as sf  # noqa: E402

pytestmark = pytest.mark.unit

T_END = sf.LAYER_STEPS * sf.LAYER_DT


def _specs(ctx):
    alloc = ctx.allocation
    return (
        ArmSpec(StrategyId.S1, "S1 capped", effort_cap=sf.EFFORT_CAP),
        ArmSpec(StrategyId.S1, "S1 free"),
        ArmSpec(StrategyId.S2, "S2 impedance", impedance=sf.impedance()),
        ArmSpec(StrategyId.S3, "S3 late", impedance=sf.impedance(0.75 * T_END, T_END)),
        ArmSpec(
            StrategyId.S7,
            "S7 delayed",
            feedback_gain=sf.feedback_gain(),
            delay_steps=2,
        ),
        ArmSpec(
            StrategyId.S7,
            "S7 too late",
            feedback_gain=sf.feedback_gain(),
            delay_steps=30,
        ),
        ArmSpec(
            StrategyId.S8,
            "S8",
            speed_cases=(sf.nominal_case(alloc), sf.slow_case(alloc)),
            reference_case="nominal",
            effort_cap=sf.EFFORT_CAP,
        ),
        ArmSpec(StrategyId.S4, "S4"),
    )


@pytest.fixture(scope="module")
def ctx():
    return sf.comparison()


@pytest.fixture(scope="module")
def arms(ctx):
    return evaluate_arms(_specs(ctx), ctx)


def _row(report, label):
    return next(r for r in report.rows if r.arm.label == label)


MATCH = MatchSettings(reference_chs_mps=sf.NOMINAL_CHS, chs_rel_tol=0.02)


# --------------------------------------------------------- matched speed
def test_a_slower_swing_cannot_win_the_matched_speed_comparison(arms) -> None:
    # comparators the quiet slow swing beats when speed is left free
    pool = [
        a for a in arms if a.label in ("S1 capped", "S3 late") or a.label[:2] == "S8"
    ]
    free = run_comparison(pool, ComparisonMode.UNCONSTRAINED, MATCH)
    slow = _row(free, "S8:slow")
    assert slow.rank == 1  # unconstrained, the quiet slow swing "wins"

    matched = run_comparison(pool, ComparisonMode.MATCHED_CHS, MATCH)
    row = _row(matched, "S8:slow")
    assert row.rank is None and not row.admitted
    assert row.exclusion is Exclusion.SPEED_MISMATCH
    assert matched.winner is not None and matched.winner != "S8:slow"
    assert all(r.arm.chs_mps == sf.NOMINAL_CHS for r in matched.ranked)
    # the exclusion is what mattered: the slow arm was better than the winner
    assert slow.arm.ratio < _row(matched, matched.winner).arm.ratio
    # and it is excluded from the full pool too
    every = run_comparison(arms, ComparisonMode.MATCHED_CHS, MATCH)
    assert _row(every, "S8:slow").exclusion is Exclusion.SPEED_MISMATCH


def test_a_faster_swing_outside_the_tolerance_is_not_matched_either(arms) -> None:
    fast = dataclasses.replace(
        next(a for a in arms if a.label == "S1 capped"), label="fast", chs_mps=44.0
    )
    report = run_comparison((*arms, fast), ComparisonMode.MATCHED_CHS, MATCH)
    assert _row(report, "fast").exclusion is Exclusion.SPEED_MISMATCH
    near = dataclasses.replace(fast, label="near", chs_mps=40.4)  # inside 2 %
    report = run_comparison((*arms, near), ComparisonMode.MATCHED_CHS, MATCH)
    assert _row(report, "near").admitted


def test_matched_effort_excludes_other_bases_and_arms_over_the_cap(arms) -> None:
    capped = arms[0]
    settings = dataclasses.replace(
        MATCH, effort_cap=sf.EFFORT_CAP, effort_basis=capped.effort_basis
    )
    report = run_comparison(arms, ComparisonMode.MATCHED_EFFORT, settings)
    assert _row(report, "S1 capped").admitted
    assert _row(report, "S1 free").exclusion is Exclusion.EFFORT_EXCEEDED
    assert _row(report, "S2 impedance").exclusion is Exclusion.EFFORT_BASIS_MISMATCH
    assert _row(report, "S8:slow").exclusion is Exclusion.SPEED_MISMATCH
    assert all(r.arm.effort <= sf.EFFORT_CAP * (1 + 1e-6) for r in report.ranked)


def test_matched_effort_requires_a_cap_and_a_basis() -> None:
    with pytest.raises(ValueError, match="effort_cap"):
        run_comparison((), ComparisonMode.MATCHED_EFFORT, MATCH)
    with pytest.raises(ValueError, match="effort_cap"):
        MatchSettings(40.0, effort_cap=-1.0)
    with pytest.raises(ValueError, match="chs_rel_tol"):
        MatchSettings(40.0, chs_rel_tol=0.0)


def test_unconstrained_fronts_show_the_speed_accuracy_tradeoff(arms) -> None:
    pool = [
        a for a in arms if a.label in ("S1 capped", "S3 late") or a.label[:2] == "S8"
    ]
    free = run_comparison(pool, ComparisonMode.UNCONSTRAINED, MATCH)
    chs_front = free.fronts["variance_vs_chs"]
    assert {"S8:slow", "S8:nominal"} <= set(chs_front)  # quiet-but-slow is on it
    assert "S3 late" not in chs_front  # dominated: noisier at the same speed
    assert all(r.admitted for r in free.rows)
    assert any(k.startswith("variance_vs_effort:") for k in free.fronts)
    # a faster-and-quieter arm removes the slow swing from the front
    everything = run_comparison(arms, ComparisonMode.UNCONSTRAINED, MATCH)
    assert "S8:slow" not in everything.fronts["variance_vs_chs"]


# --------------------------------------------------- report honesty
def test_report_states_assumptions_failures_and_the_non_claim(arms) -> None:
    report = run_comparison(arms, ComparisonMode.MATCHED_CHS, MATCH)
    assert report.non_claim == NON_CLAIM and "not coaching advice" in NON_CLAIM
    assert report.primary_outcome == "face_angle_variance"
    assert report.ranking_basis == "variance_ratio_to_strategy_off_baseline"
    failed = [r for r in report.rows if r.arm.status is ArmStatus.UNSTABLE]
    assert [r.arm.label for r in failed] == ["S7 too late"]
    assert failed[0].exclusion is Exclusion.FAILED
    assert report.failure_rate == pytest.approx(1 / 8)  # 8 attempted arms, 1 failed
    assert report.not_attempted == ("S4",)
    assert any("linear" in a for a in report.assumptions)
    assert all(r.ratio_record is not None for r in report.ranked)
    assert all(r.ratio_record.units == "1" for r in report.ranked)


def test_a_strategy_that_does_not_beat_the_baseline_does_not_win(ctx) -> None:
    flat_ctx = sf.comparison(m1=False)
    arms = evaluate_arms(
        (
            ArmSpec(StrategyId.S1, "S1"),
            ArmSpec(StrategyId.S2, "zero", impedance=sf.impedance(k=0.0, d=0.0)),
        ),
        flat_ctx,
    )
    report = run_comparison(arms, ComparisonMode.MATCHED_CHS, MATCH)
    assert report.winner is None
    assert all(not r.beats_baseline for r in report.ranked)


def test_variance_budget_table_has_input_class_fractions(arms) -> None:
    table = variance_budget_table(arms)
    assert "S1 capped" in table.rows and "S4" not in table.rows
    for label, fractions in table.rows.items():
        assert sum(fractions.values()) == pytest.approx(1.0, abs=1e-9), label
    assert {"initial_state", "additive_noise", "signal_dependent"} <= set(table.classes)
    assert table.rows["S1 capped"]["signal_dependent"] > 0.5


# --------------------------------------------------------- priors
def _weak_vs_strong():
    strong = sf.comparison()
    weak = sf.comparison(k=(0.02, 0.04))
    specs = (
        ArmSpec(StrategyId.S1, "S1"),
        ArmSpec(StrategyId.S2, "S2 soft", impedance=sf.impedance(k=1.0, d=0.5)),
    )
    return specs, {"strong signal noise": strong, "weak signal noise": weak}


def test_rankings_are_reported_per_prior_with_reversals() -> None:
    specs, priors = _weak_vs_strong()
    result = rank_under_priors(specs, priors, ComparisonMode.MATCHED_CHS)
    assert set(result.reports) == set(priors)
    strong, weak = (result.reports[k] for k in priors)
    assert strong.winner == "S1" and weak.winner == "S2 soft"
    assert len(result.reversals) == 1
    rev = result.reversals[0]
    assert {rev.better_in_first, rev.better_in_second} == {"S1", "S2 soft"}
    assert (rev.first_prior, rev.second_prior) == tuple(priors)
    assert not result.stable
    assert "not coaching advice" in result.non_claim


def test_a_ranking_that_does_not_depend_on_the_prior_has_no_reversal() -> None:
    specs = (
        ArmSpec(StrategyId.S1, "S1"),
        ArmSpec(StrategyId.S2, "S2 stiff", impedance=sf.impedance()),
    )
    priors = {"a": sf.comparison(), "b": sf.comparison(k=(0.4, 0.9))}
    result = rank_under_priors(specs, priors, ComparisonMode.MATCHED_CHS)
    assert result.reversals == () and result.stable


def test_rank_under_priors_needs_two_priors() -> None:
    specs, priors = _weak_vs_strong()
    with pytest.raises(ValueError, match="two priors"):
        rank_under_priors(specs, {"only": priors["strong signal noise"]}, MATCH_MODE)


MATCH_MODE = ComparisonMode.MATCHED_CHS


# ------------------------------------------------------ held-out seeds
INTEGRATOR = IntegratorSignature("explicit Euler, reduced-coordinate pendulum", fx.DT)


def _mc_spec(sp: DisturbanceSpec, n: int) -> CampaignSpec:
    return fx.mc_campaign_spec(sp, n, n)


@pytest.fixture(scope="module")
def optimum_pair():
    sp = fx.strong_spec()
    alloc = sf.allocation()
    res = minimise_face_variance(
        alloc.feasible_sets,
        alloc.times_s,
        alloc.torque_scale,
        alloc.model,
        alloc.declaration,
    )
    return sp, res


def test_ranking_is_stable_on_held_out_seeds_or_the_reversal_is_reported(
    ctx, optimum_pair
) -> None:
    sp, res = optimum_pair
    arms = evaluate_arms((ArmSpec(StrategyId.S1, "S1"),), ctx)
    # a second arm that is the minimum-effort reference (strategy off)
    ref_arm = dataclasses.replace(
        arms[0],
        label="reference",
        face_variance=arms[0].baseline_variance,
        effort=arms[0].baseline_effort,
    )
    report = run_comparison((*arms, ref_arm), ComparisonMode.MATCHED_CHS, MATCH)
    assert [r.arm.label for r in report.ranked] == ["S1", "reference"]
    factory = fx.mc_runner_factory(sp)
    strategies = {
        "S1": Strategy(
            "S1", factory(res.torque, res.z), ReplayMode.OPEN_LOOP, INTEGRATOR
        ),
        "reference": Strategy(
            "reference",
            factory(res.reference_torque, res.reference_z),
            ReplayMode.OPEN_LOOP,
            INTEGRATOR,
        ),
    }
    check = heldout_stability(
        report, strategies, _mc_spec(sp, 1200), outcome="y", n_boot=200
    )
    (pair,) = check.pairs
    assert (pair.better, pair.worse) == ("S1", "reference")
    assert pair.status is StabilityStatus.CONFIRMED
    assert pair.paired_sd_diff < 0.0 and pair.holdout_sd_diff < 0.0
    assert check.stable and check.reversals == ()
    assert check.failure_rates["S1"] == 0.0


def _scaled_runner(scale: float):
    """Outcome ``scale * x0[0]`` of the persistent initial-state draw."""

    def runner(draw):
        return [scale * float(draw.realise("x0")[0])]

    return runner


def test_a_held_out_reversal_is_reported_not_hidden(ctx, optimum_pair) -> None:
    """The linear model ranks A over B; the independent MC says the opposite."""
    sp, _ = optimum_pair
    arms = evaluate_arms(
        (
            ArmSpec(StrategyId.S1, "A"),
            ArmSpec(StrategyId.S2, "B", impedance=sf.impedance()),
        ),
        ctx,
    )
    report = run_comparison(arms, ComparisonMode.UNCONSTRAINED, MATCH)
    better, worse = (r.arm.label for r in report.ranked)
    strategies = {
        better: Strategy(better, _scaled_runner(2.0), ReplayMode.OPEN_LOOP, INTEGRATOR),
        worse: Strategy(worse, _scaled_runner(1.0), ReplayMode.OPEN_LOOP, INTEGRATOR),
    }
    check = heldout_stability(
        report, strategies, _mc_spec(sp, 400), outcome="y", n_boot=100
    )
    (pair,) = check.pairs
    assert pair.status is StabilityStatus.REVERSED
    assert not check.stable and check.reversals == (pair,)
    assert pair.holdout_sd_diff > 0.0  # the "better" arm has the larger MC SD


def test_identical_runners_are_inconclusive_not_confirmed(ctx, optimum_pair) -> None:
    sp, _ = optimum_pair
    arms = evaluate_arms(
        (
            ArmSpec(StrategyId.S1, "A"),
            ArmSpec(StrategyId.S2, "B", impedance=sf.impedance()),
        ),
        ctx,
    )
    report = run_comparison(arms, ComparisonMode.UNCONSTRAINED, MATCH)
    names = [r.arm.label for r in report.ranked]
    strategies = {
        n: Strategy(n, _scaled_runner(1.0), ReplayMode.OPEN_LOOP, INTEGRATOR)
        for n in names
    }
    check = heldout_stability(
        report, strategies, _mc_spec(sp, 400), outcome="y", n_boot=100
    )
    assert check.pairs[0].status is StabilityStatus.INCONCLUSIVE
    assert not check.stable and check.reversals == ()


def test_heldout_preconditions(ctx, optimum_pair) -> None:
    sp, _ = optimum_pair
    arms = evaluate_arms((ArmSpec(StrategyId.S1, "A"),), ctx)
    report = run_comparison(arms, ComparisonMode.UNCONSTRAINED, MATCH)
    with pytest.raises(ValueError, match="two ranked"):
        heldout_stability(report, {}, _mc_spec(sp, 50), outcome="y")
