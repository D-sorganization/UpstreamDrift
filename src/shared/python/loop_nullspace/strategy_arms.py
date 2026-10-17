"""Arm specifications, contexts and results of the strategy comparison (NSX-95).

An *arm* is one strategy at one declared setting evaluated in one
:class:`ComparisonContext`.  The context bundles the representations of the same
swing that the dependencies provide (an allocation problem for NSX-92, a tangent
layer for NSX-93, a muscle frame for NSX-94); an arm that needs a representation
the context lacks is ``UNAVAILABLE``, never guessed.

:class:`ArmResult` carries the face-angle variance (the primary outcome), the
variance of the *same representation with the strategy off* (``baseline``), the
clubhead speed (an allocation-invariant input), the effort in its own basis, the
typed NSX-81 record, the variance budget by input class and the honest status.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.impedance_spec import JointImpedance
from src.shared.python.loop_nullspace.layer_linear import LayerLinearization
from src.shared.python.loop_nullspace.noise_aware_front import ChsCase
from src.shared.python.loop_nullspace.noise_aware_model import NoiseAwareModel
from src.shared.python.loop_nullspace.strategy_spec import NON_CLAIM, StrategyId
from src.shared.python.loop_nullspace.variance_records import VarianceDeclaration

__all__ = [
    "AllocationContext",
    "ArmResult",
    "ArmSpec",
    "ArmStatus",
    "ComparisonContext",
    "LayerContext",
    "MuscleContext",
    "skipped_arm",
    "typed_variance_record",
]


class ArmStatus(str, Enum):
    """Termination class of one arm.

    ``NO_MECHANISM`` is a *result* (the strategy's declared mechanism is absent,
    so it has no effect; the numbers are the baseline's).  ``INFEASIBLE``,
    ``NOT_CONVERGED`` and ``UNSTABLE`` are failures and are counted in the failure
    rate.  ``UNAVAILABLE`` and ``NOT_IMPLEMENTED`` are not attempts.
    """

    OK = "ok"
    NO_MECHANISM = "no_mechanism"
    INFEASIBLE = "infeasible"
    NOT_CONVERGED = "not_converged"
    UNSTABLE = "unstable"
    UNAVAILABLE = "unavailable"
    NOT_IMPLEMENTED = "not_implemented"

    @property
    def evaluated(self) -> bool:
        return self in (ArmStatus.OK, ArmStatus.NO_MECHANISM)

    @property
    def failed(self) -> bool:
        return self in (
            ArmStatus.INFEASIBLE,
            ArmStatus.NOT_CONVERGED,
            ArmStatus.UNSTABLE,
        )

    @property
    def attempted(self) -> bool:
        return self.evaluated or self.failed


@dataclass(frozen=True)
class AllocationContext:
    """NSX-92 inputs: frames, times, torque scale, variance model, declaration."""

    feasible_sets: tuple[Any, ...]
    times_s: Array
    torque_scale: Array
    model: NoiseAwareModel
    declaration: VarianceDeclaration

    def __post_init__(self) -> None:
        if len(self.feasible_sets) == 0:
            raise ValueError("an allocation context needs at least one frame")
        if not isinstance(self.model, NoiseAwareModel):
            raise TypeError("model must be a NoiseAwareModel")
        if not isinstance(self.declaration, VarianceDeclaration):
            raise TypeError("declaration must be a VarianceDeclaration")
        object.__setattr__(self, "feasible_sets", tuple(self.feasible_sets))


@dataclass(frozen=True)
class LayerContext:
    """NSX-93 inputs: the controller-free tangent chain and its disturbances.

    ``selection``, ``basis`` and ``offset`` map the tangent state to the actuated
    errors (:func:`~.layer_linear.impedance_state_gain`); ``sigma`` is the
    per-step noise covariance before any declared impedance noise penalty.
    """

    linearization: LayerLinearization
    times_s: Array
    selection: Array
    basis: Array
    offset: Array
    p0: Array
    sigma: Array
    outcome_rows: Array
    declaration: VarianceDeclaration
    outcome_units: str

    def __post_init__(self) -> None:
        if self.linearization.impedance_gain is not None or (
            self.linearization.feedback_gain is not None
        ):
            raise ValueError("the layer context must be the controller-free chain")
        if np.asarray(self.times_s).shape != (self.linearization.n_steps,):
            raise ValueError("times_s needs one entry per linearization step")
        if not self.outcome_units:
            raise ValueError("outcome_units must be non-empty")


@dataclass(frozen=True)
class MuscleContext:
    """NSX-94 inputs for the co-contraction curve (see ``cocontraction_variance_curve``)."""

    frame: Any
    torque: Array
    index: Any
    config: Any
    model: Any


@dataclass(frozen=True)
class ComparisonContext:
    """One swing in up to three representations; ``chs_mps`` is its clubhead speed.

    Clubhead speed is an allocation-invariant input of the motion (NSX-92), never
    something a strategy computed here.
    """

    label: str
    chs_mps: float
    allocation: AllocationContext | None = None
    layer: LayerContext | None = None
    muscle: MuscleContext | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a context needs a label")
        if not (math.isfinite(self.chs_mps) and self.chs_mps > 0.0):
            raise ValueError(f"chs_mps must be positive and finite, got {self.chs_mps}")


@dataclass(frozen=True)
class ArmSpec:
    """One strategy at one declared setting.

    Which fields apply depends on the strategy: ``effort_cap`` (S1, S5, S8),
    ``impedance`` (S2, S3), ``cocontraction`` / ``cocontraction_level`` (S2 on a
    muscle model; ``None`` level means the variance-minimising level of the
    computed curve), ``feedback_gain`` / ``delay_steps`` (S7) and
    ``speed_cases`` / ``reference_case`` (S8, which expands to one arm per case).
    """

    strategy: StrategyId
    label: str
    effort_cap: float | None = None
    impedance: JointImpedance | None = None
    cocontraction: bool = False
    cocontraction_level: float | None = None
    feedback_gain: Array | None = None
    delay_steps: int = 0
    speed_cases: tuple[ChsCase, ...] = ()
    reference_case: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, StrategyId):
            raise TypeError("strategy must be a StrategyId")
        if not self.label.strip():
            raise ValueError("an arm needs a non-empty label")
        if self.effort_cap is not None and not (
            math.isfinite(self.effort_cap) and self.effort_cap > 0.0
        ):
            raise ValueError(f"effort_cap must be positive, got {self.effort_cap}")
        if self.delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {self.delay_steps}")
        object.__setattr__(self, "speed_cases", tuple(self.speed_cases))


@dataclass(frozen=True)
class ArmResult:
    """Outcome of one arm (see the module docstring).

    ``face_variance`` and ``baseline_variance`` are in ``outcome_units``^2 of the
    linear model; ``effort`` and ``baseline_effort`` are in ``effort_basis`` and
    comparable only within that basis.  ``budget`` maps the canonical input
    classes (``initial_state``, ``additive_noise``, ``signal_dependent``,
    ``muscle_noise``, ``external_disturbance``) to variance parts that sum to
    ``face_variance``.  ``diagnostics`` holds strategy-specific scalars such as
    the delay margin.
    """

    strategy: StrategyId
    label: str
    status: ArmStatus
    reason: str = ""
    face_variance: float | None = None
    baseline_variance: float | None = None
    chs_mps: float | None = None
    effort: float | None = None
    baseline_effort: float | None = None
    effort_basis: str = ""
    certificate: str = ""
    budget: Mapping[str, float] = field(default_factory=dict)
    mechanisms: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    unverified_hypothesis_ids: tuple[str, ...] = ()
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    record: QuantityRecord | None = None
    non_claim: str = NON_CLAIM

    @property
    def ratio(self) -> float:
        """Variance over the strategy-off baseline (lower is better), else NaN."""
        if self.face_variance is None or not self.baseline_variance:
            return math.nan
        return float(self.face_variance / self.baseline_variance)

    @property
    def effect(self) -> float:
        """Baseline minus variance (positive: the strategy helped), else NaN."""
        if self.face_variance is None or self.baseline_variance is None:
            return math.nan
        return float(self.baseline_variance - self.face_variance)


def skipped_arm(spec: ArmSpec, status: ArmStatus, reason: str) -> ArmResult:
    """A non-attempt (``UNAVAILABLE``/``NOT_IMPLEMENTED``): reason, no numbers."""
    if status.attempted or not reason.strip():
        raise ValueError("a skipped arm needs a non-attempt status and a reason")
    return ArmResult(spec.strategy, spec.label, status, reason)


def typed_variance_record(
    value: float,
    units: str,
    decl: VarianceDeclaration,
    extra_conditioning: tuple[str, ...] = (),
) -> QuantityRecord:
    """NSX-81 record of a linear variance under ``decl``.

    ``FAMILY_DEPENDENT`` (with the mechanism ids) iff a V1 mechanism is declared,
    else ``INVARIANT`` and conditioned on the noise hypotheses.
    """
    ids = decl.ids()
    return QuantityRecord(
        values=np.array([value]),
        units=units,
        allocation_dependence=(
            AllocationDependence.FAMILY_DEPENDENT
            if ids
            else AllocationDependence.INVARIANT
        ),
        model_provenance=decl.model_provenance,
        input_provenance=decl.input_provenance,
        hypotheses=ids,
        conditioning=tuple(
            dict.fromkeys((*decl.noise_hypotheses, *extra_conditioning))
        ),
    )
