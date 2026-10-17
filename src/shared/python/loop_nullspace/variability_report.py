"""MOSAIC multi-swing variability consumer and falsification result (NSX-97).

Orchestrates, on held-out test sessions only: UCM/ORT about ``ker C_imp``,
the Muller-Sternad TNC decomposition, and the comparison of predicted versus
measured outcome-covariance directions.  Predicted covariance uses noise
amplitudes fitted on **training sessions only** (leakage raises).  Phase and
physical-time residuals are reported together when trials are supplied.

The result characterises variability structure.  It is UNVALIDATED software
output, not a scientific qualification.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.estimation.mosaic.swing_registration import MultiSwingTrial

from .evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from .variability_directions import (
    DirectionComparison,
    compare_outcome_directions,
    predicted_outcome_covariance,
)
from .variability_registration import (
    RegistrationResiduals,
    registration_residual_report,
)
from .variability_split import (
    NoiseFit,
    SessionSplit,
    SwingSample,
    assert_fit_not_leaked,
    fit_noise_amplitudes,
)
from .variability_tnc import DEFAULT_SEED, TNCResult, linear_outcome_fn, tnc_decompose
from .variability_ucm import UCMResult, ucm_decompose

Array = NDArray[np.float64]
MODEL = ModelProvenance("variability_falsification", "1")
NON_CLAIMS: tuple[str, ...] = (
    "UCM and TNC characterise variability structure; they do not identify "
    "neural intent or a control strategy.",
    "Neither measure proves a co-contraction benefit, minimum-variance "
    "control or any ranking of strategies as coaching advice.",
    "Linear (first-order) kernels of the impact outcome map; near-grazing "
    "contact requires Monte Carlo.",
    "Real-capture falsification is a deferred validation; no capture result "
    "is implied by this software.",
)


@dataclass(frozen=True)
class FalsificationOptions:
    """Declared analysis options (all defaults are recorded in the result)."""

    k: int = 2
    n_permutations: int = 200
    n_rotations: int = 200
    seed: int = DEFAULT_SEED
    state_scale: ArrayLike | None = None
    outcome_weights: ArrayLike | None = None
    outcome_scale: ArrayLike | None = None
    registration_key: str = "inputs"
    registration_bins: int = 50


@dataclass(frozen=True)
class VariabilityFalsificationResult:
    """UCM, TNC, direction comparison, residuals, evidence and non-claims."""

    ucm: UCMResult
    tnc: TNCResult
    directions: DirectionComparison
    registration: RegistrationResiduals | None
    noise_fit: NoiseFit
    split: SessionSplit
    n_test_swings: int
    outcome_source: str
    seed: int
    records: dict[str, QuantityRecord]
    non_claims: tuple[str, ...] = field(default=NON_CLAIMS)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary: scalars and shapes only, no raw swing data."""
        out = {
            "seed": self.seed,
            "outcome_source": self.outcome_source,
            "n_test_swings": self.n_test_swings,
            "train_sessions": len(self.split.train_sessions),
            "test_sessions": len(self.split.test_sessions),
            "ucm": _plain(self.ucm),
            "tnc": _plain(self.tnc),
            "directions": _plain(self.directions),
            "noise_amplitudes": self.noise_fit.amplitudes.tolist(),
            "non_claims": list(self.non_claims),
            "validation_status": "UNVALIDATED",
        }
        reg = self.registration
        if reg is not None:
            out["registration"] = {
                "phase_rms": reg.phase_rms,
                "physical_rms": reg.physical_rms,
                "event_duration_sd_s": reg.event_duration_sd.tolist(),
                "variance_raw": reg.variance_raw,
                "variance_registered": reg.variance_registered,
            }
        return out


def _plain(obj: Any) -> Any:
    return {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in dataclasses.asdict(obj).items()
    }


def _records(
    ucm: UCMResult, tnc: TNCResult, dirs: DirectionComparison, seed: int
) -> dict[str, QuantityRecord]:
    inputs = InputProvenance(
        "measured_swings", f"held-out sessions; seed {seed}", "unvalidated"
    )
    conv = ("first-order kernel of the impact outcome map",)
    items = {
        "delta_v": (ucm.delta_v, "1"),
        "delta_v_z": (ucm.delta_v_z, "1"),
        "v_ucm_per_dim": (ucm.v_ucm_per_dim, "state_unit^2"),
        "v_ort_per_dim": (ucm.v_ort_per_dim, "state_unit^2"),
        "tolerance_cost": (tnc.tolerance_cost, "outcome_unit^2"),
        "noise_cost": (tnc.noise_cost, "outcome_unit^2"),
        "covariation_cost": (tnc.covariation_cost, "outcome_unit^2"),
        "mean_principal_angle_deg": (dirs.mean_angle_deg, "deg"),
    }
    return {
        name: QuantityRecord(
            values=np.array([val]),
            units=units,
            allocation_dependence=AllocationDependence.INVARIANT,
            model_provenance=MODEL,
            input_provenance=inputs,
            conventions=conv,
        )
        for name, (val, units) in items.items()
    }


def falsify_variability(
    samples: Sequence[SwingSample],
    split: SessionSplit,
    jacobian: ArrayLike,
    target: ArrayLike,
    *,
    test_outcomes: ArrayLike | None = None,
    options: FalsificationOptions | None = None,
    registration_trials: Sequence[MultiSwingTrial] | None = None,
) -> VariabilityFalsificationResult:
    """Run the NSX-97 falsification on held-out sessions.

    ``test_outcomes`` (rows aligned with the test swings in ``samples``
    order) are the measured outcomes; when omitted they are linearised from
    the states and labelled ``linearised_from_state``.

    Raises:
        LeakageError: If a test swing reaches the noise fit.
        ValueError: If no test swings exist or shapes disagree.
    """
    opt = options or FalsificationOptions()
    train = [s for s in samples if s.session_id in split.train_sessions]
    test = [s for s in samples if s.session_id in split.test_sessions]
    if len(test) < 3:
        raise ValueError("need at least three test swings from the test sessions")
    fit = fit_noise_amplitudes(train, split)
    assert_fit_not_leaked(fit, split, (s.swing_id for s in test))
    jac = np.asarray(jacobian, dtype=np.float64)
    states = np.stack([s.state for s in test])
    ucm = ucm_decompose(states, jac, opt.state_scale)
    tnc = tnc_decompose(
        states,
        linear_outcome_fn(jac),
        target,
        weights=opt.outcome_weights,
        n_permutations=opt.n_permutations,
        n_rotations=opt.n_rotations,
        seed=opt.seed,
    )
    if test_outcomes is None:
        outcomes, source = states @ jac.T, "linearised_from_state"
    else:
        outcomes, source = np.asarray(test_outcomes, dtype=np.float64), "measured"
    if outcomes.shape != (len(test), jac.shape[0]):
        raise ValueError("test_outcomes must have one row per test swing")
    dirs = compare_outcome_directions(
        predicted_outcome_covariance(jac, fit.covariance),
        outcomes,
        k=min(opt.k, jac.shape[0] - 1),
        outcome_scale=opt.outcome_scale,
        n_permutations=opt.n_permutations,
        seed=opt.seed,
    )
    reg = (
        registration_residual_report(
            registration_trials, opt.registration_key, opt.registration_bins
        )
        if registration_trials
        else None
    )
    return VariabilityFalsificationResult(
        ucm=ucm,
        tnc=tnc,
        directions=dirs,
        registration=reg,
        noise_fit=fit,
        split=split,
        n_test_swings=len(test),
        outcome_source=source,
        seed=opt.seed,
        records=_records(ucm, tnc, dirs, opt.seed),
    )
