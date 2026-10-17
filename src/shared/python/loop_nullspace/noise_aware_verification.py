"""Held-out paired Monte Carlo check of a noise-aware optimum (NSX-92, #12721).

The linear variance model is a *prediction*.  This module confirms (or
refutes) it with the NSX-91 campaign machinery: the minimum-variance allocation
and the minimum-effort reference run on the **same** random draws (common random
numbers), and the SD ranking is then re-checked on an independent hold-out seed
set.  Nothing here re-implements a campaign:

* :func:`~.paired_monte_carlo.run_campaign` runs both strategies on both seed
  sets (R13 gates included),
* :func:`~.paired_statistics.compare_strategies` /
  :func:`~.paired_statistics.holdout_consistent` give the paired SD difference
  and whether the hold-out reproduces its sign (a reversal is reported),
* :func:`~.paired_statistics.cross_check_linear` keeps the linear-versus-MC gap
  for every strategy and only says ``AGREES`` when the NSX-90 nonlinearity gate
  passed (V7).

Rule V8: a confirmed ranking is a finding conditional on the declared noise and
mechanisms, not a causal or physiological statement.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.noise_aware_spec import (
    NON_CLAIM,
    NoiseAwareResult,
    NoiseAwareStatus,
)
from src.shared.python.loop_nullspace.nonlinearity_gate import NonlinearityReport
from src.shared.python.loop_nullspace.paired_mc_spec import (
    CampaignSpec,
    IntegratorSignature,
)
from src.shared.python.loop_nullspace.paired_monte_carlo import (
    CampaignResult,
    Strategy,
    TrialRunner,
    run_campaign,
)
from src.shared.python.loop_nullspace.paired_statistics import (
    LinearCrossCheck,
    PairedComparison,
    compare_strategies,
    cross_check_linear,
    holdout_consistent,
    summarise,
)
from src.shared.python.loop_nullspace.replay_spec import ReplayMode

__all__ = [
    "OPTIMUM",
    "REFERENCE",
    "RunnerFactory",
    "VerificationReport",
    "verify_against_monte_carlo",
]

OPTIMUM = "min-variance"
REFERENCE = "min-effort"
RunnerFactory = Callable[[Array, tuple[Array, ...]], TrialRunner]


@dataclass(frozen=True)
class VerificationReport:
    """Paired and hold-out evidence for one optimum.

    ``confirmed`` is True only when the paired and hold-out SD differences
    (optimum minus reference, outcome ``outcome``) are both negative with
    intervals that exclude zero on the hold-out set and the hold-out ranking
    agrees with the paired one.  ``predicted_*`` are the linear-model SDs;
    ``cross_checks`` keeps the linear-versus-MC gap per strategy (V7).
    """

    outcome: str
    paired: PairedComparison
    holdout: PairedComparison
    holdout_agrees: bool
    confirmed: bool
    predicted_sd: dict[str, float]
    cross_checks: dict[str, list[LinearCrossCheck]]
    flags: tuple[str, ...]
    non_claim: str = NON_CLAIM


def _strategies(
    result: NoiseAwareResult,
    make_runner: RunnerFactory,
    integrator: IntegratorSignature,
    hyps: tuple[str, ...],
) -> list[Strategy]:
    ref_tau, tau = result.reference_torque, result.torque
    assert ref_tau is not None and tau is not None
    return [
        Strategy(
            OPTIMUM,
            make_runner(tau, result.z),
            ReplayMode.OPEN_LOOP,
            integrator,
            hyps,
        ),
        Strategy(
            REFERENCE,
            make_runner(ref_tau, result.reference_z),
            ReplayMode.OPEN_LOOP,
            integrator,
            hyps,
        ),
    ]


def _excludes_zero_below(comp: PairedComparison, outcome: str) -> bool:
    diff = comp.outcomes.get(outcome)
    return diff is not None and diff.sd_diff < 0.0 and diff.sd_diff_ci[1] < 0.0


def verify_against_monte_carlo(
    result: NoiseAwareResult,
    spec: CampaignSpec,
    make_runner: RunnerFactory,
    integrator: IntegratorSignature,
    *,
    outcome: str,
    gate: NonlinearityReport | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> VerificationReport:
    """Run optimum and reference on common draws, then on hold-out seeds.

    ``make_runner(torque, z)`` builds the campaign runner of one allocation (a
    callable ``TrialDraw -> outcomes``) from its ``(frames, n_a)`` torque and
    per-frame ``z``; the runner must use a forward model independent of the
    kernels that produced the prediction (an inverse crime is refused through
    ``integrator`` when the spec declares a generator).  ``outcome`` names the
    face-angle channel of ``spec``.

    Preconditions (``ValueError``/``TypeError``): ``result`` carries an
    allocation (OPTIMAL, LOCAL_STATIONARY or flat), ``outcome`` is a channel of
    ``spec``, the spec's outcomes number one (the linear prediction is a scalar
    variance).  Postconditions: ``cross_checks`` has an entry per strategy with
    the gap reported whatever the status; ``confirmed`` never holds without the
    hold-out reproducing the paired ranking; a flat objective can never be
    ``confirmed`` (there is no ranking to confirm).
    """
    if not isinstance(result, NoiseAwareResult):
        raise TypeError("result must be a NoiseAwareResult")
    if result.status not in (
        NoiseAwareStatus.OPTIMAL,
        NoiseAwareStatus.LOCAL_STATIONARY,
        NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT,
    ):
        raise ValueError(f"nothing to verify: status {result.status.value}")
    if result.variance is None or result.reference_variance is None:
        raise ValueError("nothing to verify: the result carries no variances")
    if not callable(make_runner):
        raise TypeError("make_runner must be callable")
    names = [o.name for o in spec.outcomes]
    if outcome not in names:
        raise ValueError(f"unknown outcome {outcome!r}; spec has {names}")
    if len(names) != 1:
        raise ValueError("the linear prediction is a scalar variance: one outcome")
    camp: CampaignResult = run_campaign(
        spec, _strategies(result, make_runner, integrator, result.hypotheses)
    )
    paired = compare_strategies(
        spec, camp.paired, OPTIMUM, REFERENCE, n_boot=n_boot, seed=seed
    )
    hold = compare_strategies(
        spec, camp.holdout, OPTIMUM, REFERENCE, n_boot=n_boot, seed=seed
    )
    agrees = holdout_consistent(paired, hold).get(outcome, False)
    flat = result.status is NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT
    confirmed = (
        not flat
        and agrees
        and _excludes_zero_below(paired, outcome)
        and _excludes_zero_below(hold, outcome)
    )
    predicted = {
        OPTIMUM: math.sqrt(float(result.variance)),
        REFERENCE: math.sqrt(float(result.reference_variance)),
    }
    checks = {
        name: cross_check_linear(
            summarise(spec, camp.holdout, name, n_boot=n_boot, seed=seed),
            spec,
            np.array([[predicted[name] ** 2]]),
            gate=gate,
        )
        for name in (OPTIMUM, REFERENCE)
    }
    return VerificationReport(
        outcome=outcome,
        paired=paired,
        holdout=hold,
        holdout_agrees=bool(agrees),
        confirmed=bool(confirmed),
        predicted_sd=predicted,
        cross_checks=checks,
        flags=camp.flags,
    )
