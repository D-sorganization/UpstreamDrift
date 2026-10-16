"""Consistency strategy catalogue S1-S12 as data (NSX-95, #12724).

The owner's question is which strategies make a golfer more *consistent* (lower
face-angle variance), under which disturbances, at what cost.  This module is
the catalogue only: a frozen registry of what each strategy *is*, which model
class can represent it, what it costs and which evaluator computes it.  The
numbers come from the evaluators of NSX-92/93/94/91/25 that the catalogue
*composes* (:mod:`.strategy_evaluators`); nothing here estimates a variance.

Honesty rules enforced at construction time:

* ``IMPLEMENTED`` requires dotted paths of real evaluators (resolved lazily by
  :func:`resolve_evaluators`, so a typo or a deleted function fails a test);
* ``NOT_IMPLEMENTED`` (the mechanism is defined but nothing computes it yet) and
  ``UNAVAILABLE`` (this model class cannot represent it, e.g. shaft-flex phase
  matching on a rigid club) require a reason and carry no evaluator;
* the entry-level availability must agree with its per-model-class
  representations, and every model class appears exactly once.

Non-claim: a ranking built on this catalogue is conditional on the declared
priors and is not coaching advice until externally validated.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

__all__ = [
    "CATALOGUE",
    "EFFORT_ACTIVATION",
    "EFFORT_ALLOCATION",
    "EFFORT_LAYER",
    "NON_CLAIM",
    "Availability",
    "CostDeclaration",
    "ModelClass",
    "Representation",
    "StrategyEntry",
    "StrategyId",
    "entry_for",
    "resolve_evaluators",
]

NON_CLAIM = (
    "Rankings are conditional on the declared disturbance priors, mechanisms and "
    "the linear tangent model at the nominal motion; they compare mechanisms "
    "inside a model, not golfers, and are not coaching advice until externally "
    "validated."
)

#: Effort bases.  They are different physical lines (rule R10) and are never
#: summed or compared across bases.
EFFORT_ALLOCATION = "capacity_weighted_allocation_effort"
EFFORT_LAYER = "expected_layer_l2_effort"
EFFORT_ACTIVATION = "muscle_activation_l2"

_LN = "src.shared.python.loop_nullspace"
_MINIMISE = f"{_LN}.noise_aware_allocation.minimise_face_variance"
_LAYER = f"{_LN}.layer_linear.layer_variance"
_COCON = f"{_LN}.muscle_variance.cocontraction_variance_curve"
_CHS_FRONT = f"{_LN}.noise_aware_front.chs_variance_front"
_EVAL = f"{_LN}.strategy_evaluators"


class StrategyId(str, Enum):
    """The twelve catalogued strategies, in the order of the issue."""

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    S7 = "S7"
    S8 = "S8"
    S9 = "S9"
    S10 = "S10"
    S11 = "S11"
    S12 = "S12"


class Availability(str, Enum):
    """Whether (and why not) a strategy can be evaluated."""

    IMPLEMENTED = "implemented"
    NOT_IMPLEMENTED = "not_implemented"
    UNAVAILABLE = "unavailable"

    @property
    def implemented(self) -> bool:
        return self is Availability.IMPLEMENTED


class ModelClass(str, Enum):
    """Model classes a strategy can be represented on.

    A flexible-shaft class is deliberately absent: the stack has no flexible
    shaft model, which is exactly why S10 is ``UNAVAILABLE`` rather than
    estimated.
    """

    RIGID_JOINT_TORQUE = "rigid_joint_torque"
    MUSCLE_DRIVEN = "muscle_driven"


@dataclass(frozen=True)
class Representation:
    """How one model class represents a strategy (or why it cannot)."""

    model_class: ModelClass
    availability: Availability
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.model_class, ModelClass):
            raise TypeError("model_class must be a ModelClass")
        if not self.availability.implemented and not self.reason.strip():
            raise ValueError(
                f"a {self.availability.value} representation needs a reason"
            )


@dataclass(frozen=True)
class CostDeclaration:
    """What the strategy costs and in which (non-commensurable) effort basis."""

    basis: str
    description: str

    def __post_init__(self) -> None:
        if not self.basis.strip() or not self.description.strip():
            raise ValueError("a cost needs a basis and a description")


@dataclass(frozen=True)
class StrategyEntry:
    """One catalogue entry (see the module docstring for the contract).

    ``mechanism_ids`` are the declared mechanism / hypothesis ids any of which
    must be in force for the strategy to have an effect at all (the negative
    control: no declared mechanism, no effect).  ``evaluators`` are dotted paths
    of the dependency functions the adapter composes; ``adapter`` is the
    catalogue's own thin wrapper.
    """

    id: StrategyId
    name: str
    mechanism: str
    mechanism_ids: tuple[str, ...]
    representations: tuple[Representation, ...]
    cost: CostDeclaration
    availability: Availability
    evaluators: tuple[str, ...] = ()
    adapter: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.mechanism.strip():
            raise ValueError(f"{self.id.value}: name and mechanism must be declared")
        classes = [r.model_class for r in self.representations]
        if sorted(c.value for c in classes) != sorted(m.value for m in ModelClass):
            raise ValueError(
                f"{self.id.value}: every model class needs exactly one representation"
            )
        if self.availability.implemented:
            if not (self.evaluators and self.adapter):
                raise ValueError(
                    f"{self.id.value}: an IMPLEMENTED entry needs an evaluator and "
                    "an adapter"
                )
        elif self.evaluators or self.adapter:
            raise ValueError(
                f"{self.id.value}: a {self.availability.value} entry has no evaluator"
            )
        if self.availability is not _derived_availability(self.representations):
            raise ValueError(
                f"{self.id.value}: availability disagrees with its representations"
            )


def _derived_availability(reps: tuple[Representation, ...]) -> Availability:
    if any(r.availability.implemented for r in reps):
        return Availability.IMPLEMENTED
    if all(r.availability is Availability.UNAVAILABLE for r in reps):
        return Availability.UNAVAILABLE
    return Availability.NOT_IMPLEMENTED


def _impl() -> Representation:
    return Representation(ModelClass.RIGID_JOINT_TORQUE, Availability.IMPLEMENTED)


def _reps(
    rigid: Representation, muscle: Representation
) -> tuple[Representation, Representation]:
    return (rigid, muscle)


def _no(model: ModelClass, reason: str) -> Representation:
    return Representation(model, Availability.NOT_IMPLEMENTED, reason)


def _none(model: ModelClass, reason: str) -> Representation:
    return Representation(model, Availability.UNAVAILABLE, reason)


_RIGID, _MUSCLE = ModelClass.RIGID_JOINT_TORQUE, ModelClass.MUSCLE_DRIVEN
_SIGNAL = "H-NOISE-SIGNAL-DEPENDENT"
_PRELOAD = "H-GEOMETRIC-PRELOAD"
_IMPEDANCE = "H-DECLARED-JOINT-IMPEDANCE"
_SRS = "H-MUSCLE-SHORT-RANGE-STIFFNESS"
_DELAY = "H-SENSORIMOTOR-DELAY"
_NO_MUSCLE = "no muscle-level evaluator exists for this mechanism"


def _entry(
    sid: StrategyId,
    name: str,
    mechanism: str,
    ids: tuple[str, ...],
    reps: tuple[Representation, ...],
    cost: CostDeclaration,
    evaluators: tuple[str, ...] = (),
    adapter: str | None = None,
) -> StrategyEntry:
    return StrategyEntry(
        sid,
        name,
        mechanism,
        ids,
        reps,
        cost,
        _derived_availability(reps),
        evaluators,
        adapter,
    )


_ALLOC_COST = CostDeclaration(
    EFFORT_ALLOCATION,
    "capacity-weighted quadratic effort of the allocated torque history",
)
_LAYER_COST = CostDeclaration(
    EFFORT_LAYER,
    "expected quadratic effort of the declared impedance / feedback torque "
    "(a separate line from the allocated torque, rule R10)",
)
_UNDEFINED = CostDeclaration(
    "undefined", "no cost model exists until the strategy is implemented"
)


def _catalogue() -> dict[StrategyId, StrategyEntry]:
    s = StrategyId
    entries = [
        _entry(
            s.S1,
            "Null-space redistribution",
            "Move the closure-family allocation z to the minimum-face-variance "
            "point of the NSX-92 problem, with the signal-dependent noise (M1) as "
            "the only allocation-to-variance channel.",
            (_SIGNAL,),
            _reps(
                _impl(),
                _no(_MUSCLE, "muscle redundancy is not wired into the NSX-92 solve"),
            ),
            _ALLOC_COST,
            (_MINIMISE,),
            f"{_EVAL}.evaluate_allocation_arm",
        ),
        _entry(
            s.S2,
            "Co-contraction and impedance",
            "Raise joint stiffness/damping (declared impedance windows, NSX-93) or "
            "antagonist co-contraction (short-range stiffness against "
            "signal-dependent noise, NSX-94).",
            (_IMPEDANCE, _SRS),
            _reps(
                Representation(_RIGID, Availability.IMPLEMENTED),
                Representation(_MUSCLE, Availability.IMPLEMENTED),
            ),
            _LAYER_COST,
            (_LAYER, _COCON),
            f"{_EVAL}.evaluate_layer_arm",
        ),
        _entry(
            s.S3,
            "Distal stabilization near impact",
            "A declared impedance window that ends at impact (NSX-93), "
            "stabilizing late in the downswing only.",
            (_IMPEDANCE,),
            _reps(
                _impl(),
                _no(_MUSCLE, _NO_MUSCLE),
            ),
            _LAYER_COST,
            (_LAYER,),
            f"{_EVAL}.evaluate_layer_arm",
        ),
        _entry(
            s.S4,
            "Proximal-distal effort shift",
            "Shift effort between proximal and distal joints at fixed clubhead speed.",
            (_SIGNAL,),
            _reps(
                _no(_RIGID, "no per-joint effort-share parameterisation exists"),
                _no(_MUSCLE, _NO_MUSCLE),
            ),
            _UNDEFINED,
        ),
        _entry(
            s.S5,
            "Grip pressure and hand-split preload",
            "Redistribute the allocation through the geometric preload of the "
            "closure (M2) with additive noise, isolated from M1.",
            (_PRELOAD,),
            _reps(
                _impl(),
                _no(_MUSCLE, "the hand-club closure has no muscle-level model"),
            ),
            _ALLOC_COST,
            (_MINIMISE,),
            f"{_EVAL}.evaluate_allocation_arm",
        ),
        _entry(
            s.S6,
            "Kinematic redundancy (UCM)",
            "Confine joint-space variability to the uncontrolled manifold of the "
            "clubhead outcome.",
            (),
            _reps(
                _no(_RIGID, "no UCM computation exists in the repository"),
                _no(_MUSCLE, _NO_MUSCLE),
            ),
            _UNDEFINED,
        ),
        _entry(
            s.S7,
            "Delayed feedback",
            "Delayed state feedback with a declared gain schedule and "
            "sensorimotor delay (NSX-93), limited by its delay margin.",
            (_DELAY,),
            _reps(
                _impl(),
                _no(_MUSCLE, _NO_MUSCLE),
            ),
            _LAYER_COST,
            (_LAYER,),
            f"{_EVAL}.evaluate_layer_arm",
        ),
        _entry(
            s.S8,
            "Speed-accuracy trade-off",
            "Trade clubhead speed for consistency: the variance of cases with "
            "different nominal speed at a common effort cap (NSX-92 CHS front).",
            (_SIGNAL,),
            _reps(
                _impl(),
                _no(_MUSCLE, _NO_MUSCLE),
            ),
            _ALLOC_COST,
            (_CHS_FRONT, _MINIMISE),
            f"{_EVAL}.evaluate_speed_accuracy",
        ),
        _entry(
            s.S9,
            "Biarticular coupling",
            "Use muscles that span two joints to couple joint torques.",
            (),
            _reps(
                _none(
                    _RIGID,
                    "a joint-torque model has no biarticular muscles to couple",
                ),
                _no(_MUSCLE, "no biarticular comparison is built on the NSX-94 frame"),
            ),
            _UNDEFINED,
        ),
        _entry(
            s.S10,
            "Shaft-flex phase matching",
            "Time the release to the shaft's flexural phase at impact.",
            (),
            _reps(
                _none(_RIGID, "the rigid club has no shaft flexibility to phase-match"),
                _none(
                    _MUSCLE,
                    "the rigid club in the muscle-driven model has no shaft "
                    "flexibility to phase-match",
                ),
            ),
            _UNDEFINED,
        ),
        _entry(
            s.S11,
            "Timing and sequencing robustness",
            "Tolerate timing jitter of the sequencing of segment peaks.",
            (),
            _reps(
                _no(_RIGID, "no timing-disturbance variance evaluator exists"),
                _no(_MUSCLE, _NO_MUSCLE),
            ),
            _UNDEFINED,
        ),
        _entry(
            s.S12,
            "Address-state consistency",
            "Reduce the variability of the address posture (initial state).",
            (),
            _reps(
                _no(_RIGID, "no address-consistency cost model exists"),
                _no(_MUSCLE, _NO_MUSCLE),
            ),
            _UNDEFINED,
        ),
    ]
    return {e.id: e for e in entries}


CATALOGUE: Mapping[StrategyId, StrategyEntry] = MappingProxyType(_catalogue())


def entry_for(strategy: StrategyId) -> StrategyEntry:
    """Catalogue entry of ``strategy`` (``ValueError`` for a non-member)."""
    if not isinstance(strategy, StrategyId):
        raise ValueError(f"unknown strategy {strategy!r}")
    return CATALOGUE[strategy]


def _import_dotted(path: str) -> Callable[..., object]:
    module_name, _, attr = path.rpartition(".")
    target = getattr(importlib.import_module(module_name), attr)
    if not callable(target):
        raise TypeError(f"{path} is not callable")
    return target  # type: ignore[no-any-return]


def resolve_evaluators(entry: StrategyEntry) -> tuple[Callable[..., object], ...]:
    """The dependency evaluators of ``entry`` followed by its adapter.

    Raises ``ValueError`` (naming the missing evaluator) for an entry that is not
    implemented, and ``AttributeError``/``ImportError`` for a dangling path.
    """
    if not entry.availability.implemented or entry.adapter is None:
        raise ValueError(
            f"{entry.id.value} has no evaluator: it is {entry.availability.value}"
        )
    return tuple(_import_dotted(p) for p in (*entry.evaluators, entry.adapter))
