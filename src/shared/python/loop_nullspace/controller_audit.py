"""Controller synergy audit on Option A-2 swing-and-recovery traces (#13884).

Policy-agnostic instrument: given terminal states ``x_T`` of a set of trials
and a declared task map ``y = C x_T``, report the uncontrolled-manifold split
(:func:`~.variability_ucm.ucm_decompose`), the tolerance-noise-covariation
costs (:func:`~.variability_tnc.tnc_decompose`) and, when the controller's
gains and plant are given, the minimum-intervention signature
``max_t ||K_t N_t||`` (:func:`~.minimum_intervention.task_irrelevant_gain_norm`).
Nothing here redefines those quantities.

Recovered/unrecoverable labels from the #13837 traces are carried only as
strata (counts in the report); they never enter the task variable. TNC fails
closed below ``min_tnc_trials`` (default 30): the report then holds no TNC
result and says why.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics
from src.shared.python.humanoid_robotics.option_a.trace_contract import (
    validate_trace_bundle,
)
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.minimum_intervention import (
    task_irrelevant_gain_norm,
)
from src.shared.python.loop_nullspace.variability_tnc import (
    DEFAULT_SEED,
    TNCResult,
    linear_outcome_fn,
    tnc_decompose,
)
from src.shared.python.loop_nullspace.variability_ucm import (
    RANK_RTOL,
    UCMResult,
    finite_matrix,
    ucm_decompose,
)

__all__ = [
    "AuditInputs",
    "ControllerAuditReport",
    "ControllerAuditSpec",
    "audit_controller",
    "audit_inputs_from_traces",
    "predicted_ucm_variances",
]

MIN_TNC_TRIALS = 30


@dataclass(frozen=True)
class ControllerAuditSpec:
    """Pre-registered task map ``C`` (m, n), ``state_scale`` (n,) and coordinates.

    Preconditions (``ValueError``): finite ``C`` with ``m < n``, a positive
    finite ``state_scale`` of length ``n``, ``n`` coordinate names and
    ``min_tnc_trials >= 2``.
    """

    task_jacobian: Array
    state_scale: Array
    coordinates: tuple[str, ...]
    seed: int = DEFAULT_SEED
    min_tnc_trials: int = MIN_TNC_TRIALS
    tnc_permutations: int = 200

    def __post_init__(self) -> None:
        jac = finite_matrix(self.task_jacobian, "task_jacobian")
        m, n = jac.shape
        if m >= n:
            raise ValueError(f"task needs dim(y) < dim(x); got {m} >= {n}")
        if self.state_scale is None:
            raise ValueError("state_scale must be pre-registered")
        scale = np.array(self.state_scale, dtype=np.float64)
        if scale.shape != (n,) or not np.all(np.isfinite(scale) & (scale > 0.0)):
            raise ValueError("state_scale must be a positive finite vector of length n")
        if len(self.coordinates) != n:
            raise ValueError(f"coordinates must name all {n} state dimensions")
        if self.min_tnc_trials < 2:
            raise ValueError("min_tnc_trials must be >= 2")
        object.__setattr__(self, "task_jacobian", jac)
        object.__setattr__(self, "state_scale", scale)
        object.__setattr__(self, "coordinates", tuple(self.coordinates))


@dataclass(frozen=True)
class AuditInputs:
    """Terminal states ``(n_trials, n)`` and one stratum label per trial."""

    states: Array
    strata: tuple[str, ...]

    def __post_init__(self) -> None:
        states = finite_matrix(self.states, "states")
        if len(self.strata) != states.shape[0]:
            raise ValueError("strata must hold one label per trial")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "strata", tuple(str(s) for s in self.strata))


@dataclass(frozen=True)
class ControllerAuditReport:
    """UCM split, TNC costs (or why withheld) and the null-space gain norm."""

    spec: ControllerAuditSpec
    ucm: UCMResult
    tnc: TNCResult | None
    tnc_withheld: str | None
    null_space_gain_norm: float | None
    strata: dict[str, int]

    def as_record(self) -> dict[str, Any]:
        """JSON-ready record holding ``C``, the coordinates and the seed."""
        return {
            "task_jacobian": self.spec.task_jacobian.tolist(),
            "state_scale": self.spec.state_scale.tolist(),
            "coordinates": list(self.spec.coordinates),
            "seed": self.spec.seed,
            "ucm": vars(self.ucm),
            "tnc": None if self.tnc is None else vars(self.tnc),
            "tnc_withheld": self.tnc_withheld,
            "null_space_gain_norm": self.null_space_gain_norm,
            "strata": dict(self.strata),
        }


def audit_inputs_from_traces(bundle: Mapping[str, Any]) -> AuditInputs:
    """Last recorded state of each trial in a #13837 trace bundle.

    The bundle is validated first (:func:`validate_trace_bundle`, fail closed).
    ``outcome_label`` becomes the strata, in trial order.
    """
    validate_trace_bundle(bundle)
    trials = np.asarray(bundle["trial_index"])
    x_k = np.asarray(bundle["x_k"], dtype=np.float64)
    ids = np.unique(trials)
    last = [int(np.flatnonzero(trials == i)[-1]) for i in ids]
    return AuditInputs(x_k[last], tuple(np.asarray(bundle["outcome_label"])))


def predicted_ucm_variances(
    spec: ControllerAuditSpec, covariance: ArrayLike
) -> tuple[float, float]:
    """``(V_UCM, V_ORT)`` of a state covariance projected on ``N(C)``, ``R(C^T)``.

    Uses the same scaling as :func:`ucm_decompose` (``z = x / state_scale``),
    so a Lyapunov covariance gives the population value of its Monte Carlo
    estimate.
    """
    inv = 1.0 / spec.state_scale
    cov_z = finite_matrix(covariance, "covariance") * np.outer(inv, inv)
    _, sv, vt = np.linalg.svd(spec.task_jacobian * spec.state_scale)
    rank = int(np.sum(sv > RANK_RTOL * max(1.0, float(sv[0]))))
    ort, ucm = vt[:rank], vt[rank:]
    return float(np.trace(ucm @ cov_z @ ucm.T)), float(np.trace(ort @ cov_z @ ort.T))


def audit_controller(
    spec: ControllerAuditSpec,
    inputs: AuditInputs,
    *,
    gains: ArrayLike | None = None,
    plant: LinearizedDynamics | None = None,
) -> ControllerAuditReport:
    """Run the audit; ``gains`` ``(T, nu, n)`` and ``plant`` must come together.

    Postcondition: ``tnc`` is ``None`` exactly when ``tnc_withheld`` is set.
    """
    if (gains is None) != (plant is None):
        raise ValueError("gains and plant must be given together")
    n_trials = inputs.states.shape[0]
    ucm = ucm_decompose(inputs.states, spec.task_jacobian, spec.state_scale)
    tnc, withheld = None, None
    if n_trials < spec.min_tnc_trials:
        withheld = f"TNC needs {spec.min_tnc_trials} trials; got {n_trials} < {spec.min_tnc_trials}"
    else:
        tnc = tnc_decompose(
            inputs.states / spec.state_scale,
            linear_outcome_fn(spec.task_jacobian * spec.state_scale),
            np.zeros(spec.task_jacobian.shape[0]),
            n_permutations=spec.tnc_permutations,
            seed=spec.seed,
        )
    gain_norm = None
    if gains is not None and plant is not None:
        rows = spec.task_jacobian
        gain_norm = task_irrelevant_gain_norm(gains, plant, rows, np.eye(rows.shape[0]))
    return ControllerAuditReport(
        spec, ucm, tnc, withheld, gain_norm, dict(Counter(inputs.strata))
    )
