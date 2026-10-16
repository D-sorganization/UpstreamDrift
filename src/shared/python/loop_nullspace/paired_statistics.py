"""Dispersion, paired comparison and the NSX-90 cross-check of a campaign (NSX-91).

Every number here comes from the retained trials of
:mod:`paired_monte_carlo`.  Declared statistical choices (the specification fixes
none):

* **Intervals** are seeded percentile bootstrap intervals (``ci_level``,
  ``n_boot``).  For ``N`` Gaussian outcomes the sample SD has relative standard
  error ``1 / sqrt(2 (N - 1))``, so a two-sided ``z`` interval has half-width
  about ``z s / sqrt(2 (N - 1))``; the tests check the bootstrap against this
  closed form instead of using a chosen tolerance.
* **Face angle (and any ``ANGLE`` outcome)** goes through
  :func:`outcome_dispersion.compute_outcome_dispersion` (wrapped statistics, V3).
  Scalar outcomes get mean and SD with bootstrap intervals
  (:class:`ScalarDispersion`).  The coefficient of variation is never reported.
* **Paired differences** resample trial indices *jointly* for both strategies,
  so the common random numbers cancel in the difference.  Only trials where both
  strategies succeeded enter a difference; trials where one failed are counted in
  ``failed_only_a`` / ``failed_only_b`` and in each strategy's own tally.
* **Failure rate** is every non-``OK`` trial, with a Wilson interval.
* **Linear cross-check** (V7) compares the MC SD with the NSX-90 prediction and
  *always* reports the gap; agreement is claimed only when the nonlinearity gate
  ran and passed for that outcome.

Non-claim: the MC SD is conditional on the successful trials; compare it with
the failure rate, never alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.nonlinearity_gate import NonlinearityReport
from src.shared.python.loop_nullspace.paired_mc_spec import (
    CampaignSpec,
    OutcomeChannel,
    OutcomeKind,
)
from src.shared.python.loop_nullspace.paired_monte_carlo import (
    FailureTally,
    SeedSetResult,
    TrialStatus,
    tally,
)
from src.shared.python.loop_nullspace.replay_spec import ReplayMode
from src.shared.python.perturbation.outcome_dispersion import (
    OutcomeDispersion,
    compute_outcome_dispersion,
    wrap_angle,
)

__all__ = [
    "CrossCheckStatus",
    "LinearCrossCheck",
    "OutcomeDifference",
    "PairedComparison",
    "RecordProvenance",
    "ScalarDispersion",
    "StrategySummary",
    "compare_strategies",
    "cross_check_linear",
    "holdout_consistent",
    "sd_records",
    "summarise",
]

Interval = tuple[float, float]
_PERIOD = {"rad": 2.0 * math.pi, "deg": 360.0}
_CHUNK_ELEMENTS = 2_000_000
_LABEL = {
    ReplayMode.OPEN_LOOP: "independent_open_loop_replay",
    ReplayMode.FEEDBACK: "feedback_assisted_tracking",
}


@dataclass(frozen=True)
class ScalarDispersion:
    """Mean and population SD of a non-angle outcome with bootstrap intervals."""

    unit: str
    n_samples: int
    mean: float
    sd: float
    mean_ci: Interval
    sd_ci: Interval
    ci_level: float
    n_boot: int
    seed: int


@dataclass(frozen=True)
class StrategySummary:
    """A strategy's tally and per-outcome dispersion on one seed set.

    ``dispersions[name]`` is ``None`` when no trial succeeded (never zero).
    """

    strategy: str
    mode: ReplayMode
    seed_set: str
    tally: FailureTally
    dispersions: dict[str, OutcomeDispersion | ScalarDispersion | None]
    hypotheses: tuple[str, ...] = ()

    @property
    def evidence_label(self) -> str:
        """The only label under which this summary may be shown (R10)."""
        return _LABEL[self.mode]


# ------------------------------------------------------------------ bootstrap
def _validate_boot(ci_level: float, n_boot: int) -> None:
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if n_boot < 20:
        raise ValueError(f"n_boot must be >= 20, got {n_boot}")


def _resample_rows(
    n: int, n_boot: int, rng: np.random.Generator, stat: Callable[[np.ndarray], Array]
) -> Array:
    """``stat`` of index-resample matrices, chunked to bound memory."""
    chunk = max(1, _CHUNK_ELEMENTS // max(n, 1))
    parts = []
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        parts.append(stat(rng.integers(0, n, size=(stop - start, n))))
    return np.concatenate(parts)


def _percentile_interval(boot: Array, estimate: float, ci_level: float) -> Interval:
    lo, hi = np.percentile(boot, [50.0 * (1.0 - ci_level), 50.0 * (1.0 + ci_level)])
    return (min(float(lo), estimate), max(float(hi), estimate))


def _scalar_dispersion(
    values: Array, channel: OutcomeChannel, ci_level: float, n_boot: int, seed: int
) -> ScalarDispersion:
    mean, sd = float(values.mean()), float(values.std())
    rng = np.random.default_rng(seed)
    both = _resample_rows(
        values.size,
        n_boot,
        rng,
        lambda idx: np.stack([values[idx].mean(axis=1), values[idx].std(axis=1)], 1),
    )
    return ScalarDispersion(
        unit=channel.unit,
        n_samples=int(values.size),
        mean=mean,
        sd=sd,
        mean_ci=_percentile_interval(both[:, 0], mean, ci_level),
        sd_ci=_percentile_interval(both[:, 1], sd, ci_level),
        ci_level=ci_level,
        n_boot=n_boot,
        seed=seed,
    )


def _angle_dispersion(
    values: Array, channel: OutcomeChannel, ci_level: float, n_boot: int, seed: int
) -> OutcomeDispersion:
    assert channel.tolerance is not None  # validated by OutcomeChannel
    return compute_outcome_dispersion(
        values,
        channel.target,
        channel.tolerance,
        miss_half_width=channel.miss_half_width,
        angle_unit=channel.unit,
        ci_level=ci_level,
        n_boot=n_boot,
        seed=seed,
    )


def summarise(
    spec: CampaignSpec,
    result: SeedSetResult,
    strategy: str,
    *,
    ci_level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> StrategySummary:
    """Tally and per-outcome dispersion of ``strategy`` on one seed set.

    Pre: ``strategy`` is in ``result``; ``0 < ci_level < 1``; ``n_boot >= 20``.
    Post: ``tally.n_trials`` equals the trials run (failures included);
    dispersions use the ``OK`` trials only and are ``None`` if there are none;
    identical for an identical ``seed``.
    """
    _validate_boot(ci_level, n_boot)
    if strategy not in result.records:
        raise ValueError(f"unknown strategy {strategy!r}")
    matrix = result.outcome_matrix(strategy)
    dispersions: dict[str, OutcomeDispersion | ScalarDispersion | None] = {}
    for col, channel in enumerate(spec.outcomes):
        if matrix.shape[0] == 0:
            dispersions[channel.name] = None
        elif channel.kind is OutcomeKind.ANGLE:
            dispersions[channel.name] = _angle_dispersion(
                matrix[:, col], channel, ci_level, n_boot, seed
            )
        else:
            dispersions[channel.name] = _scalar_dispersion(
                matrix[:, col], channel, ci_level, n_boot, seed
            )
    return StrategySummary(
        strategy=strategy,
        mode=result.modes[strategy],
        seed_set=result.seed_set,
        tally=tally(result.records[strategy], ci_level),
        dispersions=dispersions,
        hypotheses=result.hypotheses[strategy],
    )


# ----------------------------------------------------------------- comparison
@dataclass(frozen=True)
class OutcomeDifference:
    """Paired ``A - B`` statistics of one outcome over the common successful trials."""

    name: str
    unit: str
    n_pairs: int
    mean_diff: float
    mean_diff_ci: Interval
    sd_a: float
    sd_b: float
    sd_diff: float
    sd_diff_ci: Interval


@dataclass(frozen=True)
class PairedComparison:
    """Strategy ``A`` against ``B`` on the same draws of one seed set."""

    strategy_a: str
    strategy_b: str
    seed_set: str
    n_trials: int
    n_pairs: int
    failed_only_a: int
    failed_only_b: int
    failed_both: int
    tally_a: FailureTally
    tally_b: FailureTally
    outcomes: dict[str, OutcomeDifference]
    cross_mode: bool = False
    ci_level: float = 0.95
    flags: tuple[str, ...] = field(default=())


def _circular_mean(x: np.ndarray, period: float) -> np.ndarray:
    theta = x * (2.0 * math.pi / period)
    return np.arctan2(np.sin(theta).mean(-1), np.cos(theta).mean(-1)) * (
        period / (2.0 * math.pi)
    )


def _sd_rows(x: np.ndarray, period: float | None) -> np.ndarray:
    """Population SD over the last axis (wrapped about the circular mean)."""
    if period is None:
        return np.asarray(x.std(axis=-1))
    dev = wrap_angle(x - _circular_mean(x, period)[..., None], period)
    return np.asarray(np.sqrt(np.mean(dev * dev, axis=-1)))


def _mean_rows(x: np.ndarray, period: float | None) -> np.ndarray:
    return np.asarray(x.mean(axis=-1) if period is None else _circular_mean(x, period))


def _difference(
    channel: OutcomeChannel,
    a: Array,
    b: Array,
    ci_level: float,
    n_boot: int,
    seed: int,
) -> OutcomeDifference:
    period = _PERIOD[channel.unit] if channel.kind is OutcomeKind.ANGLE else None
    if period is not None:
        a, b = (
            wrap_angle(a - channel.target, period),
            wrap_angle(b - channel.target, period),
        )
    diff = a - b if period is None else wrap_angle(a - b, period)

    def stat(idx: np.ndarray) -> Array:
        return np.stack(
            [
                _mean_rows(diff[idx], period),
                _sd_rows(a[idx], period) - _sd_rows(b[idx], period),
            ],
            1,
        )

    boot = _resample_rows(a.size, n_boot, np.random.default_rng(seed), stat)
    mean_d = float(_mean_rows(diff, period))
    sd_a, sd_b = float(_sd_rows(a, period)), float(_sd_rows(b, period))
    return OutcomeDifference(
        name=channel.name,
        unit=channel.unit,
        n_pairs=int(a.size),
        mean_diff=mean_d,
        mean_diff_ci=_percentile_interval(boot[:, 0], mean_d, ci_level),
        sd_a=sd_a,
        sd_b=sd_b,
        sd_diff=sd_a - sd_b,
        sd_diff_ci=_percentile_interval(boot[:, 1], sd_a - sd_b, ci_level),
    )


def _pairs(result: SeedSetResult, a: str, b: str) -> tuple[Array, Array, int, int, int]:
    """Outcome matrices on the trials both succeeded and the failure splits."""
    rec_a, rec_b = result.records[a], result.records[b]
    ok_a = np.array([r.status is TrialStatus.OK for r in rec_a])
    ok_b = np.array([r.status is TrialStatus.OK for r in rec_b])
    both = ok_a & ok_b
    out_a = np.array([r.outcomes for r in rec_a])[both]
    out_b = np.array([r.outcomes for r in rec_b])[both]
    return (
        out_a,
        out_b,
        int(np.sum(~ok_a & ok_b)),
        int(np.sum(ok_a & ~ok_b)),
        int(np.sum(~ok_a & ~ok_b)),
    )


def compare_strategies(
    spec: CampaignSpec,
    result: SeedSetResult,
    strategy_a: str,
    strategy_b: str,
    *,
    allow_cross_mode: bool = False,
    ci_level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> PairedComparison:
    """Paired comparison of two strategies on the shared draws of one seed set.

    Each strategy's :class:`ReplayMode` is read from ``result``.  Comparing an open-loop strategy with a feedback-assisted one
    raises ``ValueError`` unless ``allow_cross_mode`` is set, which is recorded
    in ``cross_mode`` (R10).  Post: every trial of both strategies is counted in
    the tallies; differences use the trials both completed; with fewer than two
    pairs the outcome differences are empty and ``flags`` says so.
    """
    _validate_boot(ci_level, n_boot)
    for name in (strategy_a, strategy_b):
        if name not in result.records:
            raise ValueError(f"unknown strategy {name!r}")
    cross = result.modes[strategy_a] is not result.modes[strategy_b]
    if cross and not allow_cross_mode:
        raise ValueError(
            "open-loop and feedback-assisted strategies are reported apart (R10); "
            "pass allow_cross_mode=True to compare them under their labels"
        )
    out_a, out_b, only_a, only_b, both_failed = _pairs(result, strategy_a, strategy_b)
    outcomes: dict[str, OutcomeDifference] = {}
    flags: tuple[str, ...] = ()
    if out_a.shape[0] >= 2:
        for col, channel in enumerate(spec.outcomes):
            outcomes[channel.name] = _difference(
                channel, out_a[:, col], out_b[:, col], ci_level, n_boot, seed
            )
    else:
        flags = ("fewer_than_two_common_successes",)
    return PairedComparison(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        seed_set=result.seed_set,
        n_trials=len(result.records[strategy_a]),
        n_pairs=int(out_a.shape[0]),
        failed_only_a=only_a,
        failed_only_b=only_b,
        failed_both=both_failed,
        tally_a=tally(result.records[strategy_a], ci_level),
        tally_b=tally(result.records[strategy_b], ci_level),
        outcomes=outcomes,
        cross_mode=cross,
        ci_level=ci_level,
        flags=flags,
    )


def holdout_consistent(
    paired: PairedComparison, holdout: PairedComparison
) -> dict[str, bool]:
    """Per outcome: does the hold-out reproduce the paired SD ranking?

    True when both comparisons name the same ordered strategies, the SD
    differences have the same sign and their intervals overlap.  A swapped or
    sign-flipped ranking (a reversal) is ``False``.
    """
    if (paired.strategy_a, paired.strategy_b) != (
        holdout.strategy_a,
        holdout.strategy_b,
    ):
        return dict.fromkeys(paired.outcomes, False)
    out: dict[str, bool] = {}
    for name, first in paired.outcomes.items():
        second = holdout.outcomes.get(name)
        if second is None:
            out[name] = False
            continue
        overlap = (
            first.sd_diff_ci[0] <= second.sd_diff_ci[1]
            and second.sd_diff_ci[0] <= first.sd_diff_ci[1]
        )
        out[name] = bool(overlap and first.sd_diff * second.sd_diff > 0.0)
    return out


# ------------------------------------------------------------ linear cross-check
class CrossCheckStatus(str, Enum):
    """Whether the NSX-90 prediction and the MC agree, and on what basis."""

    AGREES = "agrees"
    DISAGREES = "disagrees"
    NOT_LINEAR_VALID = "not_linear_valid"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class LinearCrossCheck:
    """One outcome's MC SD against the linear prediction; the gap is always kept."""

    outcome: str
    predicted_sd: float
    mc_sd: float
    mc_sd_ci: Interval
    gap: float
    relative_gap: float
    within_ci: bool
    failure_rate: float
    status: CrossCheckStatus


def _status(
    gate: NonlinearityReport | None, col: int, within: bool
) -> CrossCheckStatus:
    if gate is None:
        return CrossCheckStatus.NOT_ASSESSED
    if gate.exceeded[col]:
        return CrossCheckStatus.NOT_LINEAR_VALID
    return CrossCheckStatus.AGREES if within else CrossCheckStatus.DISAGREES


def _sd_and_ci(disp: OutcomeDispersion | ScalarDispersion) -> tuple[float, Interval]:
    return disp.sd, disp.sd_ci


def cross_check_linear(
    summary: StrategySummary,
    spec: CampaignSpec,
    linear_covariance: ArrayLike,
    *,
    gate: NonlinearityReport | None,
) -> list[LinearCrossCheck]:
    """Compare the MC SDs with ``sqrt(diag(Cov(y)))`` of the NSX-90 prediction.

    Pre: ``linear_covariance`` is ``(ny, ny)`` for the campaign's ``ny``
    outcomes in the outcomes' units; ``gate`` (if given) has ``ny`` outputs.
    Post: ``gap = mc_sd - predicted_sd`` is reported for every outcome;
    ``AGREES`` needs the gate to have passed and the prediction to lie inside the
    MC interval; without a gate the status is ``NOT_ASSESSED`` (V7).  An outcome
    with no successful trial is ``NOT_ASSESSED`` with NaN gap.
    """
    cov = np.asarray(linear_covariance, dtype=np.float64)
    n_out = len(spec.outcomes)
    if cov.shape != (n_out, n_out):
        raise ValueError(
            f"linear_covariance must be ({n_out}, {n_out}), got {cov.shape}"
        )
    if gate is not None and len(gate.exceeded) != n_out:
        raise ValueError(f"gate must have {n_out} outputs, got {len(gate.exceeded)}")
    variances = np.diag(cov)
    if not np.all(variances >= 0.0):
        raise ValueError("linear_covariance must have non-negative variances")
    checks = []
    for col, channel in enumerate(spec.outcomes):
        pred = math.sqrt(float(variances[col]))
        disp = summary.dispersions[channel.name]
        if disp is None:
            checks.append(
                LinearCrossCheck(
                    channel.name,
                    pred,
                    math.nan,
                    (math.nan, math.nan),
                    math.nan,
                    math.nan,
                    False,
                    summary.tally.failure_rate,
                    CrossCheckStatus.NOT_ASSESSED,
                )
            )
            continue
        mc_sd, interval = _sd_and_ci(disp)
        within = interval[0] <= pred <= interval[1]
        rel = (
            (mc_sd - pred) / pred if pred > 0.0 else (0.0 if mc_sd == 0.0 else math.inf)
        )
        checks.append(
            LinearCrossCheck(
                channel.name,
                pred,
                mc_sd,
                interval,
                mc_sd - pred,
                rel,
                within,
                summary.tally.failure_rate,
                _status(gate, col, within),
            )
        )
    return checks


# ------------------------------------------------------------------- records
@dataclass(frozen=True)
class RecordProvenance:
    """Provenance and conditioning shared by the records of one summary."""

    model: ModelProvenance
    inputs: InputProvenance
    conditioning: tuple[str, ...] = ()


def sd_records(
    summary: StrategySummary,
    spec: CampaignSpec,
    provenance: RecordProvenance,
    *,
    hypotheses: tuple[str, ...] | None = None,
) -> dict[str, QuantityRecord]:
    """``QuantityRecord`` of each outcome's SD and of the failure rate.

    ``INVARIANT`` without allocation hypotheses, ``FAMILY_DEPENDENT`` with them
    (default: the summary's); the replay mode label is a convention, and the
    noise hypotheses ride on ``conditioning``.  Always ``UNVALIDATED``.
    Raises ``ValueError`` if no outcome has a successful trial.
    """
    hyps = summary.hypotheses if hypotheses is None else hypotheses
    dependence = (
        AllocationDependence.FAMILY_DEPENDENT
        if hyps
        else AllocationDependence.INVARIANT
    )
    conventions = (summary.evidence_label, f"seed_set={summary.seed_set}", "paired_mc")

    def make(value: float, units: str) -> QuantityRecord:
        return QuantityRecord(
            values=np.array([value]),
            units=units,
            allocation_dependence=dependence,
            model_provenance=provenance.model,
            input_provenance=provenance.inputs,
            conventions=conventions,
            hypotheses=hyps,
            conditioning=provenance.conditioning,
        )

    records = {
        ch.name: make(disp.sd, ch.unit)
        for ch in spec.outcomes
        if (disp := summary.dispersions[ch.name]) is not None
    }
    if not records:
        raise ValueError("no successful trial: the campaign has no SD to record")
    records["failure_rate"] = make(summary.tally.failure_rate, "1")
    return records
