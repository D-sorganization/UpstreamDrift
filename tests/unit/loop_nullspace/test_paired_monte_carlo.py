"""NSX-91 (#12720): paired nonlinear Monte Carlo runner (CRN, retained failures).

Statistical tolerances are derived, not chosen.  For ``N`` Gaussian draws the
sample standard deviation has relative standard error ``1 / sqrt(2 (N - 1))``,
so a two-sided ``(1 - alpha)`` interval has half-width ``z s / sqrt(2 (N - 1))``
(``Z99``); the tests compare the bootstrap interval with that closed form.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.loop_nullspace import paired_monte_carlo as pmc
from src.shared.python.loop_nullspace.paired_mc_spec import (
    CampaignSpec,
    GeneratorSpec,
    IntegratorSignature,
    NoiseRelation,
    OutcomeChannel,
    OutcomeKind,
    compare_noise,
)
from src.shared.python.loop_nullspace.replay_spec import ReplayMode
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)

pytestmark = pytest.mark.unit

Z99 = 2.5758293035489004  # two-sided 99 % normal quantile
DT = 0.01
N_STEPS = 5
EULER = IntegratorSignature("explicit Euler", DT)
RK4 = IntegratorSignature("RK4", DT / 4.0)


def _noise(scale: float = 1.0, name: str = "torque") -> DisturbanceSpec:
    return DisturbanceSpec(
        name=name,
        stage=InjectionStage.TORQUE,
        channels=("u1", "u2"),
        units="N m",
        covariance=np.diag([0.04, 0.09]) * scale,
        spectrum=Spectrum.WHITE,
    )


def _offset(scale: float = 1.0) -> DisturbanceSpec:
    return DisturbanceSpec(
        name="x0",
        stage=InjectionStage.INITIAL_STATE,
        channels=("a",),
        units="rad",
        covariance=np.array([[0.01 * scale]]),
        persistent=True,
    )


def _spec(
    scale: float = 1.0, n_trials: int = 40, generator: GeneratorSpec | None = None
) -> CampaignSpec:
    return CampaignSpec(
        disturbances=(_noise(scale), _offset(scale)),
        n_steps=N_STEPS,
        dt=DT,
        n_trials=n_trials,
        n_holdout=n_trials,
        paired_seed=11,
        holdout_seed=9001,
        outcomes=(
            OutcomeChannel("face_angle", "deg", OutcomeKind.ANGLE, tolerance=1.5),
            OutcomeChannel("chs", "m/s", OutcomeKind.SCALAR),
        ),
        generator=generator,
    )


def _linear_runner(gain: float = 1.0, log: list | None = None):
    """Outcomes linear in the draw: face (deg) and speed (m/s)."""

    def run(draw: pmc.TrialDraw) -> np.ndarray:
        w = draw.realise("torque")
        a = draw.realise("x0")
        if log is not None:
            log.append((draw.seed_set, draw.index, w.copy(), a.copy()))
        return np.array(
            [gain * (3.0 * w.sum() * DT + 40.0 * a[0]), 45.0 + gain * w[:, 0].sum()]
        )

    return run


def _strategy(name="A", runner=None, mode=ReplayMode.OPEN_LOOP, sig=EULER):
    return pmc.Strategy(name, runner or _linear_runner(), mode, sig)


# --------------------------------------------------------------- spec checks
def test_campaign_spec_preconditions() -> None:
    base = {
        "disturbances": (_noise(),),
        "n_steps": 3,
        "dt": DT,
        "n_trials": 10,
        "n_holdout": 10,
        "paired_seed": 1,
        "holdout_seed": 2,
        "outcomes": (OutcomeChannel("chs", "m/s", OutcomeKind.SCALAR),),
    }
    CampaignSpec(**base)
    for patch, match in (
        ({"holdout_seed": 1}, "hold-out"),
        ({"n_trials": 1}, "n_trials"),
        ({"n_holdout": 0}, "n_holdout"),
        ({"dt": 0.0}, "dt"),
        ({"disturbances": ()}, "disturbance"),
        ({"disturbances": (_noise(), _noise())}, "unique"),
        ({"outcomes": ()}, "outcome"),
    ):
        with pytest.raises(ValueError, match=match):
            CampaignSpec(**{**base, **patch})
    with pytest.raises(ValueError, match="tolerance"):
        OutcomeChannel("face", "deg", OutcomeKind.ANGLE)
    with pytest.raises(ValueError, match="unit"):
        OutcomeChannel("face", "furlong", OutcomeKind.ANGLE, tolerance=1.0)


# ------------------------------------------------------- common random numbers
def test_draws_are_common_across_strategies_and_independent_across_seed_sets() -> None:
    spec = _spec()
    log_a: list = []
    log_b: list = []
    strategies = [
        _strategy("A", _linear_runner(1.0, log_a)),
        _strategy("B", _linear_runner(0.5, log_b)),
    ]
    result = pmc.run_campaign(spec, strategies)
    assert len(log_a) == len(log_b) == spec.n_trials + spec.n_holdout
    for (set_a, i_a, w_a, x_a), (set_b, i_b, w_b, x_b) in zip(
        log_a, log_b, strict=True
    ):
        assert (set_a, i_a) == (set_b, i_b)
        np.testing.assert_array_equal(w_a, w_b)  # common random numbers
        np.testing.assert_array_equal(x_a, x_b)
    paired = [e for e in log_a if e[0] == "paired"]
    holdout = [e for e in log_a if e[0] == "holdout"]
    assert not any(np.array_equal(p[2], h[2]) for p in paired for h in holdout)
    assert result.paired.seed_set == "paired" and result.holdout.seed_set == "holdout"


def test_a_draw_depends_only_on_seed_set_index_and_disturbance() -> None:
    spec = _spec()
    first = pmc.TrialDraw(spec, "paired", 7).realise("torque")
    pmc.TrialDraw(spec, "paired", 3).realise("torque")
    again = pmc.TrialDraw(spec, "paired", 7).realise("torque")
    np.testing.assert_array_equal(first, again)
    assert first.shape == (N_STEPS, 2)
    assert pmc.TrialDraw(spec, "paired", 7).realise("x0").shape == (1,)
    other_set = pmc.TrialDraw(spec, "holdout", 7).realise("torque")
    assert not np.array_equal(first, other_set)
    with pytest.raises(ValueError, match="unknown disturbance"):
        pmc.TrialDraw(spec, "paired", 0).realise("nope")
    with pytest.raises(ValueError, match="seed_set"):
        pmc.TrialDraw(spec, "validation", 0)
    with pytest.raises(ValueError, match="index"):
        pmc.TrialDraw(spec, "paired", spec.n_trials)


def test_white_draw_variance_is_timestep_consistent() -> None:
    """``Var(w) = S / dt`` per step (V6); checked with the closed-form SE."""
    spec = _spec(n_trials=4000)
    draws = np.array(
        [pmc.TrialDraw(spec, "paired", i).realise("torque")[:, 0] for i in range(4000)]
    )
    n = draws.size
    rel_se = 1.0 / math.sqrt(2.0 * (n - 1))  # ignoring within-trial independence
    assert abs(draws.var() * DT / 0.04 - 1.0) < Z99 * rel_se


def test_serial_and_batched_runs_are_identical_given_a_seed() -> None:
    spec = _spec()
    strategies = [_strategy("A"), _strategy("B", _linear_runner(0.3))]
    serial = pmc.run_campaign(spec, strategies, batch_size=1)
    for batch in (7, 40, 1000):
        batched = pmc.run_campaign(spec, strategies, batch_size=batch)
        for name in ("A", "B"):
            for got, want in (
                (batched.paired, serial.paired),
                (batched.holdout, serial.holdout),
            ):
                np.testing.assert_array_equal(
                    got.outcome_matrix(name), want.outcome_matrix(name)
                )
    again = pmc.run_campaign(spec, strategies)
    np.testing.assert_array_equal(
        again.paired.outcome_matrix("A"), serial.paired.outcome_matrix("A")
    )


# ------------------------------------------------------------- retained failures
def test_failures_misses_and_non_admissible_rollouts_are_retained_and_counted() -> None:
    spec = _spec(n_trials=30)

    def runner(draw: pmc.TrialDraw) -> np.ndarray:
        k = draw.index % 6
        if k == 1:
            raise ValueError("solver diverged")
        if k == 2:
            raise pmc.TrialMiss("no face-ball crossing")
        if k == 3:
            raise pmc.TrialNonAdmissible("contact cone violated")
        if k == 4:
            return np.array([np.nan, 40.0])
        return np.array([0.1 * k, 40.0 + k])

    result = pmc.run_campaign(spec, [_strategy("A", runner)])
    records = result.paired.records["A"]
    assert len(records) == spec.n_trials  # nothing dropped
    tally = pmc.tally(records)
    assert tally.counts[pmc.TrialStatus.FAILED] == 10  # k = 1 and k = 4
    assert tally.counts[pmc.TrialStatus.MISSED] == 5
    assert tally.counts[pmc.TrialStatus.NON_ADMISSIBLE] == 5
    assert tally.counts[pmc.TrialStatus.OK] == 10
    assert sum(tally.counts.values()) == spec.n_trials
    assert tally.failure_rate == pytest.approx(20 / 30)
    lo, hi = tally.failure_rate_ci
    assert 0.0 <= lo < tally.failure_rate < hi <= 1.0
    assert result.paired.outcome_matrix("A").shape == (10, 2)
    failed = [r for r in records if r.status is pmc.TrialStatus.FAILED]
    assert all(np.isnan(r.outcomes).all() for r in failed)
    assert any("solver diverged" in (r.reason or "") for r in failed)
    assert any("non-finite" in (r.reason or "") for r in failed)


def test_programming_errors_are_not_swallowed_as_trial_failures() -> None:
    def runner(draw: pmc.TrialDraw) -> np.ndarray:
        raise TypeError("bug")

    with pytest.raises(TypeError, match="bug"):
        pmc.run_campaign(_spec(), [_strategy("A", runner)])


def test_wrong_outcome_length_is_a_failure_not_a_silent_truncation() -> None:
    result = pmc.run_campaign(
        _spec(n_trials=5), [_strategy("A", lambda d: np.array([1.0]))]
    )
    assert pmc.tally(result.paired.records["A"]).counts[pmc.TrialStatus.FAILED] == 5


# ------------------------------------------------------------ R13 honesty gates
def test_a_run_using_the_synthesizers_integrator_or_step_is_rejected() -> None:
    gen = GeneratorSpec(integrator=EULER, noise=(_noise(2.0),))
    spec = _spec(generator=gen)
    with pytest.raises(ValueError, match="inverse crime"):
        pmc.run_campaign(spec, [_strategy("A", sig=EULER)])
    same_name = IntegratorSignature("explicit euler", DT / 3.0)
    with pytest.raises(ValueError, match="inverse crime"):
        pmc.run_campaign(spec, [_strategy("A", sig=same_name)])
    same_step = IntegratorSignature("RK4", DT)
    with pytest.raises(ValueError, match="inverse crime"):
        pmc.run_campaign(spec, [_strategy("A", sig=same_step)])
    pmc.run_campaign(spec, [_strategy("A", sig=RK4)])  # different both: accepted
    adaptive = IntegratorSignature("DOP853", None)
    pmc.run_campaign(spec, [_strategy("A", sig=adaptive)])


def test_identical_generator_noise_is_circular_and_rejected() -> None:
    gen = GeneratorSpec(integrator=EULER, noise=(_noise(1.0), _offset(1.0)))
    with pytest.raises(ValueError, match="circular"):
        _spec(generator=gen)


def test_mismatched_noise_is_flagged_with_its_differences() -> None:
    gen = GeneratorSpec(integrator=EULER, noise=(_noise(2.0), _offset(1.0)))
    spec = _spec(generator=gen)
    assert (
        compare_noise(spec.disturbances, gen.noise).relation is NoiseRelation.MISMATCHED
    )
    result = pmc.run_campaign(spec, [_strategy("A", sig=RK4)])
    assert "noise_mismatched_to_generator" in result.flags
    assert any("torque" in d for d in result.noise_mismatch.differences)
    plain = pmc.run_campaign(_spec(), [_strategy("A", sig=RK4)])
    assert "noise_mismatched_to_generator" not in plain.flags
    assert plain.noise_mismatch is None


def test_noise_comparison_sees_each_declared_field() -> None:
    base = _noise()
    changed = DisturbanceSpec(
        name="torque",
        stage=InjectionStage.EXTERNAL_HAND_WRENCH,
        channels=("u1", "u2"),
        units="N m",
        covariance=base.covariance,
        spectrum=Spectrum.WHITE,
    )
    diff = compare_noise((base,), (changed,))
    assert diff.relation is NoiseRelation.MISMATCHED and "stage" in diff.differences[0]
    missing = compare_noise((base,), ())
    assert missing.relation is NoiseRelation.MISMATCHED


# ---------------------------------------------------------- campaign validation
def test_strategy_and_campaign_preconditions() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="at least one"):
        pmc.run_campaign(spec, [])
    with pytest.raises(ValueError, match="unique"):
        pmc.run_campaign(spec, [_strategy("A"), _strategy("A")])
    with pytest.raises(TypeError, match="ReplayMode"):
        pmc.Strategy("A", _linear_runner(), "open_loop", EULER)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch_size"):
        pmc.run_campaign(spec, [_strategy("A")], batch_size=0)
    with pytest.raises(ValueError, match="step_s"):
        IntegratorSignature("x", -1.0)
