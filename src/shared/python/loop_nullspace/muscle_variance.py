"""Variance-versus-co-contraction curve of a muscle-actuated loop (NSX-94, #12723).

For each swept co-contraction level ``c`` (:mod:`muscle_cocontraction`) the
linearised tangent dynamics of the closed loop are rebuilt and an outcome
variance is propagated with the NSX-90 machinery (``variance_propagation`` /
``variance_records``):

* mechanism M3 (``H-MUSCLE-SHORT-RANGE-STIFFNESS``): the level's joint stiffness
  ``K(c)`` enters ``F`` through :func:`apply_muscle_stiffness`;
* mechanism M1 (``H-NOISE-SIGNAL-DEPENDENT``): the level's muscle-force noise
  ``S(a*(c))`` enters through the input map ``G_mus = (dxi'' / dtau) R^T``;
* an *external* (activation-independent) torque disturbance enters through the
  coordinate torques.

Input channels are ``[muscle noise (m) | external (n)]`` in two
:class:`InputClass` groups, so the budget separates the two.  The result is the
classic U-curve: low co-contraction leaves the loop soft (large response to the
external disturbance, ``Var ~ 1/K``), high co-contraction stiffens it but the
signal-dependent noise grows with ``a`` (``Var ~ c^2 / c``).  Whether a real
golfer sits anywhere on it is a non-claim.

Fail closed: an unqualified frame returns ``UNAVAILABLE`` records; a *declared*
stiffness the model cannot supply makes the variance ``UNAVAILABLE`` with that
reason rather than silently falling back to the stiffness-free loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.derivative_capability import ClosedLoopDerivatives
from src.shared.python.loop_nullspace.configuration_manifold import (
    ConfigurationManifold,
)
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.muscle_cocontraction import (
    CocontractionSweep,
    SweepConfig,
    SweepStatus,
    cocontraction_sweep,
)
from src.shared.python.loop_nullspace.muscle_fiber import CocontractionIndex
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame
from src.shared.python.loop_nullspace.muscle_stiffness import (
    StiffnessNotQualified,
    apply_muscle_stiffness,
    muscle_stiffness,
)
from src.shared.python.loop_nullspace.tangent_linearization import (
    tangent_state_space,
)
from src.shared.python.loop_nullspace.variance_propagation import (
    InputClass,
    VarianceProblem,
    euler_dynamics,
)
from src.shared.python.loop_nullspace.variance_records import (
    VarianceAssessment,
    VarianceDeclaration,
    assess_variance,
    declared_mechanisms,
    noise_hypotheses,
)
from src.shared.python.perturbation.disturbance import DisturbanceSpec

__all__ = [
    "CocontractionVarianceCurve",
    "VarianceModel",
    "cocontraction_variance_curve",
]


@dataclass(frozen=True)
class VarianceModel:
    """The loop the variance is propagated through (all declared inputs).

    ``derivs`` are the closed-loop derivatives at the nominal ``(q, v, tau)``,
    whose ``q`` and ``v`` are also declared here;
    ``coordinate_indices`` places the muscle coordinates among the ``nv``
    velocity coordinates; ``external`` is a white torque disturbance with one
    channel per muscle coordinate; the state is the scalar tangent frame
    ``(xi, eta)`` (tangent dimension one, NSX-84), ``outcome_rows`` is
    ``(ny, 2)`` and ``initial_covariance`` ``(2, 2)``.
    """

    derivs: ClosedLoopDerivatives
    q: ArrayLike
    v: ArrayLike
    manifold: ConfigurationManifold
    basis_field: Callable[[Array], Array]
    coordinate_indices: tuple[int, ...]
    dt: float
    n_steps: int
    outcome_rows: ArrayLike
    external: DisturbanceSpec
    initial_covariance: ArrayLike
    outcome_units: str

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if not self.outcome_units:
            raise ValueError("outcome_units must be non-empty")


@dataclass(frozen=True)
class CocontractionVarianceCurve:
    """Outcome variance per co-contraction level, with the sweep it rests on.

    ``variance`` has shape ``(L, ny)``; it is ``UNAVAILABLE`` (one NaN, with the
    reason) when the sweep is not qualified or a declared stiffness failed.
    """

    sweep: CocontractionSweep
    variance: QuantityRecord
    assessments: tuple[VarianceAssessment, ...]

    @property
    def argmin_level(self) -> float | None:
        """Level minimising the first outcome's variance, or ``None``."""
        if self.variance.allocation_dependence is AllocationDependence.UNAVAILABLE:
            return None
        return float(self.sweep.levels[int(np.argmin(self.variance.values[:, 0]))])


def _embedding(indices: Sequence[int], nv: int) -> Array:
    emb = np.zeros((nv, len(indices)))
    emb[list(indices), np.arange(len(indices))] = 1.0
    return emb


def _unavailable(frame: MuscleFrame, reason: str, units: str) -> QuantityRecord:
    return QuantityRecord(
        values=np.array([np.nan]),
        units=units,
        allocation_dependence=AllocationDependence.UNAVAILABLE,
        model_provenance=frame.model_provenance,
        input_provenance=frame.input_provenance,
        reason=reason,
    )


def _level_assessment(
    frame: MuscleFrame,
    activation: Array,
    config: SweepConfig,
    model: VarianceModel,
) -> VarianceAssessment:
    idx = model.coordinate_indices
    nv = model.derivs.nv
    derivs = model.derivs
    stiffness_model = config.stiffness_model
    declared_impedance = stiffness_model is not None
    if stiffness_model is not None:
        stiff = muscle_stiffness(
            frame, activation, stiffness_model, force_gain=config.force_gain
        )
        derivs = apply_muscle_stiffness(derivs, stiff.joint, coordinate_indices=idx)
    r_full = np.zeros((frame.n_muscles, nv))
    r_full[:, list(idx)] = frame.moment_arms
    common: dict[str, Any] = {
        "q": model.q,
        "v": model.v,
        "manifold": model.manifold,
        "basis_field": model.basis_field,
    }
    ss_mus = tangent_state_space(derivs, selection=r_full, **common)
    ss_ext = tangent_state_space(derivs, selection=_embedding(idx, nv).T, **common)
    g_all = np.hstack([ss_mus.G, ss_ext.G])
    dyn = euler_dynamics([(ss_mus.F, g_all)] * model.n_steps, model.dt)
    cov_mus = config.noise.discrete_covariance(
        model.dt, activation if config.noise.has_signal_law else None
    )
    cov_ext = model.external.discrete_covariance(model.dt)
    sigma = np.zeros((cov_mus.shape[0] + cov_ext.shape[0],) * 2)
    sigma[: cov_mus.shape[0], : cov_mus.shape[0]] = cov_mus
    sigma[cov_mus.shape[0] :, cov_mus.shape[0] :] = cov_ext
    n_mus = frame.n_muscles
    problem = VarianceProblem(
        dynamics=dyn,
        initial_covariance=np.asarray(model.initial_covariance, dtype=np.float64),
        step_covariance=np.repeat(sigma[None], model.n_steps, axis=0),
        outcome_rows=np.asarray(model.outcome_rows, dtype=np.float64),
        noise_classes=(
            InputClass("muscle_noise", tuple(range(n_mus))),
            InputClass("external", tuple(range(n_mus, g_all.shape[1]))),
        ),
        initial_classes=(InputClass("state", (0, 1)),),
    )
    specs = [config.noise, model.external]
    decl = VarianceDeclaration(
        mechanisms=declared_mechanisms(
            specs, geometric_preload=False, muscle_impedance=declared_impedance
        ),
        noise_hypotheses=noise_hypotheses(specs),
        model_provenance=frame.model_provenance,
        input_provenance=frame.input_provenance,
    )
    return assess_variance(problem, decl, outcome_units=model.outcome_units)


def cocontraction_variance_curve(
    frame: MuscleFrame,
    torque: ArrayLike,
    index: CocontractionIndex,
    config: SweepConfig,
    model: VarianceModel,
) -> CocontractionVarianceCurve:
    """Outcome variance at each declared co-contraction level (module doc).

    Preconditions: those of :func:`cocontraction_sweep` plus a
    :class:`VarianceModel` whose ``external`` has one channel per muscle
    coordinate and whose ``dt`` equals ``config.dt`` (``ValueError``).
    Postcondition: the variance record's dependence is ``FAMILY_DEPENDENT`` iff
    a mechanism (M1 signal-dependent noise, M3 stiffness) is declared, else
    ``INVARIANT`` conditioned on the noise hypotheses.
    """
    if model.dt != config.dt:
        raise ValueError(f"model.dt {model.dt} differs from config.dt {config.dt}")
    if model.external.n_channels != frame.n_coordinates:
        raise ValueError("external must have one channel per muscle coordinate")
    units = f"({model.outcome_units})^2"
    sweep = cocontraction_sweep(frame, torque, index, config)
    if sweep.status is not SweepStatus.COMPUTED:
        why = "NOT_QUALIFIED: " + "; ".join(sweep.reasons)
        return CocontractionVarianceCurve(sweep, _unavailable(frame, why, units), ())
    try:
        runs = tuple(
            _level_assessment(frame, act, config, model) for act in sweep.activations
        )
    except StiffnessNotQualified as exc:
        return CocontractionVarianceCurve(
            sweep, _unavailable(frame, f"stiffness: {exc}", units), ()
        )
    first = runs[0].variance
    record = QuantityRecord(
        values=np.array([r.variance.values for r in runs]),
        units=first.units,
        allocation_dependence=first.allocation_dependence,
        model_provenance=first.model_provenance,
        input_provenance=first.input_provenance,
        validation_status=first.validation_status,
        hypotheses=first.hypotheses,
        conditioning=first.conditioning,
    )
    return CocontractionVarianceCurve(sweep, record, runs)
