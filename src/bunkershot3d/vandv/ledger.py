"""The validation ledger: what holds each factor down, and what would lift it.

The problem this solves
-----------------------

"Validation is 0 of 4 and nothing is measured" is a true statement that a
reader can do nothing with.  It does not say *which* measurement would
change it, made how, to what tolerance, or which of the eight
NASA-STD-7009B factors it would move.  Written as prose it also drifts:
the assessment lives in one file and the plan in another, and the two
disagree quietly until somebody notices the published table says 0 while
the roadmap talks as though it were 2.

This module makes the roadmap the *source* of the assessment.  Each
:class:`LedgerEntry` records the level a factor sits at, what holds it
there, and -- when a measurement is what is missing -- an ordered chain of
:class:`LevelStep`, each naming the :class:`MeasurementSpec` that must be
satisfied to climb one level.  :meth:`ValidationLedger.assessment` then
*derives* the score from the ledger and the measurements on hand.  With no
measurements the derived score is the level the ledger holds, which is the
level the credibility statement already published.  There is one number,
so there is nothing to drift.

Three kinds of blocker, and only one of them is a measurement
-------------------------------------------------------------

Most of the gap is not measurement-limited, and pretending otherwise
would be its own dishonesty.  Verification needs a method of manufactured
solutions and coverage of the F1-F3 tiers; robustness needs a sensitivity
study over ``lambda``'s published 1.0-2.8 spread and an independent
review; use history accrues only by being used.  :class:`Blocker` records
which kind of work each factor is waiting on, and
:class:`LedgerEntry` refuses to attach a measurement to a factor a
measurement cannot move.

What "leverage" means here
--------------------------

:meth:`ValidationLedger.leverage_ranking` ranks the specs by credit per
unit of effort.  Both halves are declared conventions, not estimates:
:attr:`EffortClass.cost_units` is an ordinal cost scale, and the credit a
step carries is *the gap that still stands below its threshold* -- a level
bought on a factor three short of its threshold is worth three times a
level bought on a factor one short.  A step's credit is split evenly
across the measurements it still lacks, so a spec that is one of two
missing prerequisites earns half.  Argue with the weights; they are in one
place.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .exceptions import VerificationError
from .measurement import MeasurementRecord, MeasurementRegister

__all__ = [
    "MAX_CREDIBILITY_LEVEL",
    "AcceptanceCriterion",
    "Blocker",
    "CredibilityFactor",
    "EffortClass",
    "FactorAssessment",
    "LedgerEntry",
    "LeverageEntry",
    "LevelStep",
    "MeasurementSpec",
    "ValidationLedger",
]

MAX_CREDIBILITY_LEVEL = 4
"""Top of the NASA-STD-7009B 0-4 scale."""

_MIN_STATEMENT_CHARS = 40
"""Shortest a justification may be before it stops being one."""


class CredibilityFactor(StrEnum):
    """The eight NASA-STD-7009B credibility factors."""

    VERIFICATION = "verification"
    VALIDATION = "validation"
    INPUT_PEDIGREE = "input_pedigree"
    RESULTS_UNCERTAINTY = "results_uncertainty"
    RESULTS_ROBUSTNESS = "results_robustness"
    USE_HISTORY = "use_history"
    MS_MANAGEMENT = "ms_management"
    PEOPLE_QUALIFICATIONS = "people_qualifications"

    @property
    def label(self) -> str:
        """Human-readable factor name, as NASA-STD-7009B writes it."""
        if self is CredibilityFactor.MS_MANAGEMENT:
            return "M&S Management"
        return self.value.replace("_", " ").title()


class Blocker(StrEnum):
    """What kind of work a factor is waiting on.

    The distinction matters because only one of these can be bought with an
    experiment.  A roadmap that lists a measurement against every factor is
    selling instrument time that would change nothing.
    """

    MEASUREMENT = "measurement"
    """A measurement would move it, and the ledger names which one."""

    ANALYSIS = "analysis"
    """Code, proof or study work would move it. No experiment required."""

    USE = "use"
    """Accrues only by the model being used to make a real decision."""

    NOT_SELF_ASSESSABLE = "not_self_assessable"
    """Cannot honestly be self-scored, so it is not scored."""


class EffortClass(StrEnum):
    """An ordinal cost scale for a measurement campaign.

    These are **declared conventions for ranking**, not cost estimates.
    They exist so that "cheap" and "expensive" mean something a test can
    check, and so the ordering can be argued with in one place rather than
    re-litigated in prose every time the roadmap is read.
    """

    BENCH_HOUR = "bench_hour"
    """A balance, a mould and a sample. No golfer, no bunker, no rig."""

    BENCH_DAY = "bench_day"
    """A laboratory apparatus, repeats, and a reduction step."""

    FIELD_SESSION = "field_session"
    """A course, players, and a session that cannot be repeated identically."""

    INSTRUMENTED_RIG = "instrumented_rig"
    """Synchronised high-rate instrumentation and its calibration."""

    @property
    def cost_units(self) -> int:
        """Relative cost, on the declared ordinal scale."""
        return _EFFORT_UNITS[self]


_EFFORT_UNITS: Mapping[EffortClass, int] = MappingProxyType(
    {
        EffortClass.BENCH_HOUR: 1,
        EffortClass.BENCH_DAY: 3,
        EffortClass.FIELD_SESSION: 8,
        EffortClass.INSTRUMENTED_RIG: 30,
    }
)


def _require_levels_in_range(
    owner: object,
    names: tuple[str, ...],
    *,
    label: str,
    optional: bool = False,
) -> None:
    """Raise unless each named attribute is a credibility level on the 0-4 scale.

    Args:
        owner: Object carrying the level attributes.
        names: Attribute names to validate.
        label: Message prefix, e.g. ``"level step "`` or ``f"{factor.value}."``.
        optional: When true, a ``None`` level is accepted and skipped.

    Raises:
        VerificationError: naming the offending level.
    """
    for name in names:
        level = getattr(owner, name)
        if level is None and optional:
            continue
        if not isinstance(level, int) or not 0 <= level <= MAX_CREDIBILITY_LEVEL:
            raise VerificationError(
                f"{label}{name} must be an integer in 0-"
                f"{MAX_CREDIBILITY_LEVEL}, got {level!r}"
            )


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    """What a measurement has to achieve before it counts.

    The two numeric gates are deliberately procedural rather than about the
    value itself.  Whether a measured number happens to agree with the
    model is the *result* of validation; whether the measurement is good
    enough for the comparison to mean anything is the *precondition*, and
    it is the precondition people skip.

    Attributes:
        statement: What the measurement must demonstrate, in full.
        min_samples: Fewest independent samples that will be accepted.
        max_relative_expanded_uncertainty: Largest expanded uncertainty
            (``k = 2``), as a fraction of the value, that still leaves the
            comparison able to distinguish model from constant.
    """

    statement: str
    min_samples: int
    max_relative_expanded_uncertainty: float

    def __post_init__(self) -> None:
        """Validate the criterion.

        Raises:
            VerificationError: If the statement is too short to be one, or
                either gate is out of range.
        """
        if len(self.statement.strip()) < _MIN_STATEMENT_CHARS:
            raise VerificationError(
                "an acceptance criterion must state what the measurement has "
                f"to demonstrate, got {self.statement!r}"
            )
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise VerificationError(
                f"min_samples must be a positive integer, got {self.min_samples!r}"
            )
        bound = self.max_relative_expanded_uncertainty
        if not math.isfinite(bound) or not 0.0 < bound < 1.0:
            raise VerificationError(
                "max_relative_expanded_uncertainty must be a fraction in "
                f"(0, 1), got {bound!r}"
            )

    def is_met_by(self, record: MeasurementRecord) -> bool:
        """True when ``record`` clears both gates."""
        return not self.shortfall(record)

    def shortfall(self, record: MeasurementRecord) -> str:
        """Say what stops ``record`` counting, or return an empty string."""
        problems: list[str] = []
        if record.sample_count < self.min_samples:
            problems.append(
                f"{record.sample_count} sample(s) against the {self.min_samples} "
                "this criterion requires"
            )
        if record.relative_expanded_uncertainty > (
            self.max_relative_expanded_uncertainty
        ):
            problems.append(
                f"an expanded uncertainty of "
                f"{record.relative_expanded_uncertainty:.3g} against the "
                f"{self.max_relative_expanded_uncertainty:.3g} this criterion "
                "allows"
            )
        return "; ".join(problems)

    def summary(self) -> str:
        """The two gates, rendered for a table cell."""
        return (
            f"n >= {self.min_samples}, U_rel <= "
            f"{self.max_relative_expanded_uncertainty:.3g} (k = 2)"
        )


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    """The minimum measurement that would raise one factor by one level.

    Attributes:
        key: Stable identifier, used by :class:`LevelStep` and by an
            incoming :class:`~.measurement.MeasurementRecord`.
        quantity: What is measured, SI-suffixed.
        unit: SI unit symbol. A record in any other unit does not count.
        conditions: The conditions the measurement must be made under.
        instrument_class: The class of instrument, never a brand.
        acceptance: What the measurement has to achieve.
        effort: Ordinal cost class of the campaign.
        provenance_keys: Sand properties whose
            :class:`~bunkershot3d.sand.provenance.ProvenanceBasis` this
            measurement would move to ``MEASURED``. Empty when the
            measurement constrains the model without directly measuring a
            model constant.
        note: Why this measurement, rather than an easier one.
    """

    key: str
    quantity: str
    unit: str
    conditions: str
    instrument_class: str
    acceptance: AcceptanceCriterion
    effort: EffortClass
    provenance_keys: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the spec.

        Raises:
            VerificationError: If a field a reader would need is empty.
        """
        for name in ("key", "quantity", "unit", "conditions", "instrument_class"):
            if not str(getattr(self, name)).strip():
                raise VerificationError(
                    f"measurement spec {self.key!r} has an empty {name}; a "
                    "measurement nobody can go and make is not a roadmap"
                )

    def is_satisfied_by(self, record: MeasurementRecord) -> bool:
        """True when ``record`` is offered against this spec and clears it."""
        if record.spec_key != self.key or record.unit != self.unit:
            return False
        return self.acceptance.is_met_by(record)

    def best_record(self, register: MeasurementRegister) -> MeasurementRecord | None:
        """Return the satisfying record with the smallest uncertainty.

        Ties break on the source string, so the choice is deterministic
        across runs and platforms.
        """
        candidates = [r for r in register.records if self.is_satisfied_by(r)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda r: (r.relative_expanded_uncertainty, r.source),
        )


@dataclass(frozen=True, slots=True)
class LevelStep:
    """One level of one factor, and the measurements that buy it.

    Attributes:
        from_level: The level this step starts at.
        to_level: ``from_level + 1``. Levels are climbed one at a time
            because NASA-STD-7009B's levels are not commensurable: two
            level-1 measurements do not make a level 2.
        requires: Spec keys, **all** of which must be satisfied.
        rationale: Why these measurements, and why this is the level they
            buy rather than a higher one.
    """

    from_level: int
    to_level: int
    requires: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        """Validate the step.

        Raises:
            VerificationError: If the levels are not adjacent and in range,
                if no measurement is named, or if the rationale is missing.
        """
        _require_levels_in_range(
            self, ("from_level", "to_level"), label="level step "
        )
        if self.to_level != self.from_level + 1:
            raise VerificationError(
                "a level step climbs exactly one level, got "
                f"{self.from_level} to {self.to_level}"
            )
        if not self.requires:
            raise VerificationError(
                f"the step from level {self.from_level} names no measurement; "
                "a step nothing can satisfy is a wish, not a roadmap entry"
            )
        if len(self.rationale.strip()) < _MIN_STATEMENT_CHARS:
            raise VerificationError(
                f"the step from level {self.from_level} must say why these "
                "measurements buy this level and not another"
            )

    def outstanding(self, satisfied: frozenset[str]) -> tuple[str, ...]:
        """Return the required spec keys ``satisfied`` does not cover."""
        return tuple(key for key in self.requires if key not in satisfied)

    def label(self, factor: CredibilityFactor) -> str:
        """``"Validation 0 -> 1"``, for a table cell."""
        return f"{factor.label} {self.from_level} to {self.to_level}"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One factor: where it sits, what holds it there, and the way up.

    Attributes:
        factor: Which credibility factor.
        held_level: The level the factor sits at with **no** measurements
            supplied, or ``None`` when it is not self-assessable.
        threshold_level: The level the intended use demands.
        blocker: What kind of work the factor is waiting on.
        held_because: What specifically holds it at ``held_level``.
        evidence: What ``held_level`` rests on.
        gap_statement: What is missing, stated as work.
        steps: The measurement chain, ascending from ``held_level``. Empty
            unless ``blocker`` is :attr:`Blocker.MEASUREMENT`.
    """

    factor: CredibilityFactor
    held_level: int | None
    threshold_level: int
    blocker: Blocker
    held_because: str
    evidence: str
    gap_statement: str
    steps: tuple[LevelStep, ...] = ()

    def __post_init__(self) -> None:
        """Validate the entry.

        Raises:
            VerificationError: If a level is out of range, a justification
                is missing, the blocker and the steps disagree, or the
                steps do not start where the factor actually sits.
        """
        self._require_levels()
        self._require_statements()
        self._require_blocker_agrees_with_steps()
        self._require_steps_are_a_chain()

    def _require_levels(self) -> None:
        """Raise unless both levels sit on the 0-4 scale.

        Raises:
            VerificationError: naming the offending level.
        """
        _require_levels_in_range(
            self,
            ("held_level", "threshold_level"),
            label=f"{self.factor.value}.",
            optional=True,
        )

    def _require_statements(self) -> None:
        """Raise unless every justification says something.

        Raises:
            VerificationError: naming the empty statement.
        """
        for name in ("held_because", "evidence", "gap_statement"):
            if len(str(getattr(self, name)).strip()) < _MIN_STATEMENT_CHARS:
                raise VerificationError(
                    f"{self.factor.value} has an empty {name}; an unexplained "
                    "credibility level is a number without evidence"
                )

    def _require_blocker_agrees_with_steps(self) -> None:
        """Raise unless the blocker and the presence of steps agree.

        Raises:
            VerificationError: If a non-measurement factor names a
                measurement, or a measurement-limited one names none.
        """
        if self.blocker is Blocker.NOT_SELF_ASSESSABLE and self.held_level is not None:
            raise VerificationError(
                f"{self.factor.value} is marked not self-assessable but "
                f"carries level {self.held_level}; pick one"
            )
        if self.held_level is None and self.blocker is not Blocker.NOT_SELF_ASSESSABLE:
            raise VerificationError(
                f"{self.factor.value} has no level but is not marked "
                "not-self-assessable; a blank level needs a reason"
            )
        if self.steps and self.blocker is not Blocker.MEASUREMENT:
            raise VerificationError(
                f"{self.factor.value} is blocked on {self.blocker.value} but "
                "names a measurement: only a measurement-limited factor may "
                "promise that an experiment moves it"
            )
        if self.blocker is Blocker.MEASUREMENT and not self.steps:
            raise VerificationError(
                f"{self.factor.value} is marked measurement-limited and so "
                "must name at least one level step; otherwise it is a claim "
                "that data would help, with no statement of which data"
            )

    def _require_steps_are_a_chain(self) -> None:
        """Raise unless the steps ascend contiguously from ``held_level``.

        Raises:
            VerificationError: If the chain starts above the current level
                or skips one.
        """
        if not self.steps:
            return
        if self.steps[0].from_level != self.held_level:
            raise VerificationError(
                f"{self.factor.value} sits at level {self.held_level} but its "
                f"first step starts at {self.steps[0].from_level}; a roadmap "
                "that starts above the current level is a wish list"
            )
        for lower, upper in zip(self.steps, self.steps[1:], strict=False):
            if upper.from_level != lower.to_level:
                raise VerificationError(
                    f"{self.factor.value} steps are not contiguous: "
                    f"{lower.to_level} then {upper.from_level}"
                )

    @property
    def is_measurement_limited(self) -> bool:
        """True when an experiment is what stands between here and higher."""
        return self.blocker is Blocker.MEASUREMENT

    def achieved_level(self, satisfied: frozenset[str]) -> int | None:
        """The level reached given the spec keys in ``satisfied``.

        Steps are climbed in order and stop at the first one whose
        measurements are not all present.  A later step being satisfied
        does not skip an earlier one: the levels are not commensurable.
        """
        if self.held_level is None:
            return None
        level = self.held_level
        for step in self.steps:
            if step.outstanding(satisfied):
                break
            level = step.to_level
        return level

    def next_step(self, level: int | None) -> LevelStep | None:
        """The step that starts at ``level``, if there is one."""
        for step in self.steps:
            if step.from_level == level:
                return step
        return None


@dataclass(frozen=True)
class FactorAssessment:
    """One factor's achieved level, its threshold, and the gap between them.

    Attributes:
        factor: Which factor.
        achieved_level: 0-4, or ``None`` when the factor cannot honestly
            be self-assessed.
        threshold_level: The level the intended use -- choosing between
            two wedge sole geometries and believing the answer -- demands.
        evidence: What the achieved level rests on.
        gap_statement: What is missing, stated as work rather than as a
            euphemism.
    """

    factor: CredibilityFactor
    achieved_level: int | None
    threshold_level: int
    evidence: str
    gap_statement: str

    def __post_init__(self) -> None:
        """Validate the levels.

        Raises:
            VerificationError: If a level falls outside 0-4, or the
                evidence or gap statement is empty.
        """
        _require_levels_in_range(
            self,
            ("achieved_level", "threshold_level"),
            label=f"{self.factor.value}.",
            optional=True,
        )
        for name in ("evidence", "gap_statement"):
            if not getattr(self, name).strip():
                raise VerificationError(
                    f"{self.factor.value} has an empty {name}; an unexplained "
                    "credibility level is a number without evidence"
                )

    @property
    def is_assessed(self) -> bool:
        """False when the factor was deliberately not self-scored."""
        return self.achieved_level is not None

    @property
    def gap(self) -> int | None:
        """``threshold - achieved``, never negative; ``None`` if unassessed."""
        if self.achieved_level is None:
            return None
        return max(self.threshold_level - self.achieved_level, 0)

    @property
    def meets_threshold(self) -> bool:
        """True only when the factor is assessed and clears its threshold."""
        return self.achieved_level is not None and self.achieved_level >= (
            self.threshold_level
        )

    def level_text(self) -> str:
        """The achieved level rendered for a table cell."""
        if self.achieved_level is None:
            return "not assessed"
        return f"{self.achieved_level} / {MAX_CREDIBILITY_LEVEL}"

    def gap_text(self) -> str:
        """The gap rendered for a table cell."""
        if self.gap is None:
            return "n/a"
        return "met" if self.gap == 0 else f"{self.gap} level(s) short"


@dataclass(frozen=True, slots=True)
class LeverageEntry:
    """One measurement's credit per unit of effort, at a point in time.

    Attributes:
        spec: The measurement.
        credit: Weighted share of the levels it would immediately unlock.
            Zero when every step it appears in is out of reach, or already
            satisfied.
        unlocks: Labels of the steps that credit came from.
        leverage: ``credit / spec.effort.cost_units``.
    """

    spec: MeasurementSpec
    credit: float
    unlocks: tuple[str, ...]
    leverage: float

    @property
    def is_reachable(self) -> bool:
        """True when this measurement would move something today."""
        return self.credit > 0.0


def _leverage_sort_key(item: LeverageEntry) -> tuple[float, float, str]:
    """Sort key for leverage ranking: best leverage, then cheapest, then key.

    Bound as a named function rather than an inline lambda so the effort fields
    are reached through a local `spec`, keeping each access within the repo's
    two-hop Law-of-Demeter budget (`scripts/ci/check_lod.py`).
    """
    spec = item.spec
    return (-item.leverage, spec.effort.cost_units, spec.key)


@dataclass(frozen=True)
class ValidationLedger:
    """Every factor's entry and every measurement spec, in one object.

    Attributes:
        entries: One entry per credibility factor.
        specs: Spec key to spec.
    """

    entries: tuple[LedgerEntry, ...]
    specs: Mapping[str, MeasurementSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze the ledger.

        Raises:
            VerificationError: If a factor is missing or duplicated, a step
                names a spec that does not exist, a spec is named by no
                step, or two specs claim the same sand property.
        """
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))
        self._require_every_factor_once()
        self._require_steps_and_specs_agree()
        self._require_provenance_targets_are_disjoint()

    def _require_every_factor_once(self) -> None:
        """Raise unless each factor appears exactly once.

        Raises:
            VerificationError: naming the missing or duplicated factors.
        """
        seen = [entry.factor for entry in self.entries]
        missing = sorted(
            factor.value for factor in CredibilityFactor if factor not in seen
        )
        if missing:
            raise VerificationError(
                "the validation ledger is missing an entry for: " + ", ".join(missing)
            )
        if len(seen) != len(set(seen)):
            raise VerificationError(
                "the validation ledger scores a factor more than once, which "
                "is how two levels for one factor start disagreeing"
            )

    def _require_steps_and_specs_agree(self) -> None:
        """Raise unless steps and specs name each other exactly.

        Raises:
            VerificationError: naming the orphan on whichever side it is.
        """
        required: set[str] = set()
        for entry in self.entries:
            for step in entry.steps:
                for key in step.requires:
                    if key not in self.specs:
                        raise VerificationError(
                            f"{entry.factor.value} requires the unknown "
                            f"measurement {key!r}"
                        )
                    required.add(key)
        orphans = sorted(set(self.specs) - required)
        if orphans:
            raise VerificationError(
                "these measurement specs are named by no level step, so "
                "nothing would move if they were made: " + ", ".join(orphans)
            )

    def _require_provenance_targets_are_disjoint(self) -> None:
        """Raise unless at most one spec claims each sand property.

        Raises:
            VerificationError: naming the contested property.
        """
        owner: dict[str, str] = {}
        for key in sorted(self.specs):
            for prop in self.specs[key].provenance_keys:
                if prop in owner:
                    raise VerificationError(
                        f"{key!r} and {owner[prop]!r} both claim to measure "
                        f"{prop!r}; two specs claiming one property is how a "
                        "provenance flip starts racing itself"
                    )
                owner[prop] = key

    def entry(self, factor: CredibilityFactor) -> LedgerEntry:
        """Return one factor's entry.

        Raises:
            VerificationError: If the factor is not in the ledger.
        """
        for item in self.entries:
            if item.factor is factor:
                return item
        raise VerificationError(f"{factor.value} is not in the validation ledger")

    def spec(self, key: str) -> MeasurementSpec:
        """Return one measurement spec.

        Raises:
            VerificationError: If the key is not in the ledger.
        """
        try:
            return self.specs[key]
        except KeyError:
            raise VerificationError(
                f"unknown measurement {key!r}; the ledger knows "
                + ", ".join(sorted(self.specs))
            ) from None

    def step_specs(self, step: LevelStep) -> tuple[MeasurementSpec, ...]:
        """Return the specs one step requires.

        Raises:
            VerificationError: If the step names a spec the ledger lacks.
        """
        return tuple(self.spec(key) for key in step.requires)

    def satisfied_spec_keys(self, register: MeasurementRegister) -> frozenset[str]:
        """The spec keys ``register`` actually satisfies."""
        return frozenset(
            key
            for key, spec in self.specs.items()
            if any(spec.is_satisfied_by(record) for record in register.records)
        )

    def achieved_level(
        self, factor: CredibilityFactor, register: MeasurementRegister
    ) -> int | None:
        """The level ``factor`` reaches given the measurements on hand."""
        return self.entry(factor).achieved_level(self.satisfied_spec_keys(register))

    def assessment(self, register: MeasurementRegister) -> tuple[FactorAssessment, ...]:
        """Derive the NASA-STD-7009B assessment from the ledger.

        With an empty register this returns the levels the ledger holds --
        which is the whole point.  Writing the roadmap does not raise the
        score; supplying a measurement does.
        """
        satisfied = self.satisfied_spec_keys(register)
        return tuple(
            FactorAssessment(
                factor=entry.factor,
                achieved_level=entry.achieved_level(satisfied),
                threshold_level=entry.threshold_level,
                evidence=entry.evidence,
                gap_statement=entry.gap_statement,
            )
            for entry in self.entries
        )

    def step_weight(
        self, factor: CredibilityFactor, from_level: int | None = None
    ) -> int:
        """How much one level on ``factor`` is worth, on the declared scale.

        The gap that still stands below the threshold: a level bought on a
        factor three short is worth three times a level bought on a factor
        one short.  Never less than one, so a level above threshold still
        counts for something.
        """
        entry = self.entry(factor)
        level = entry.held_level if from_level is None else from_level
        if level is None:
            return 1
        return max(entry.threshold_level - level, 1)

    def reachable_steps(
        self, register: MeasurementRegister
    ) -> tuple[tuple[CredibilityFactor, LevelStep], ...]:
        """The one step per factor that the current level sits at.

        A step two levels up is not reachable, and a measurement that only
        feeds one has no leverage today however important it is later.
        """
        satisfied = self.satisfied_spec_keys(register)
        found: list[tuple[CredibilityFactor, LevelStep]] = []
        for entry in self.entries:
            step = entry.next_step(entry.achieved_level(satisfied))
            if step is not None:
                found.append((entry.factor, step))
        return tuple(found)

    def leverage_ranking(
        self, register: MeasurementRegister
    ) -> tuple[LeverageEntry, ...]:
        """Rank every spec by credit per unit of effort, highest first.

        Ties break on effort (cheaper first) and then on the key, so the
        ranking is stable across runs and can be published.
        """
        satisfied = self.satisfied_spec_keys(register)
        credit: dict[str, float] = dict.fromkeys(self.specs, 0.0)
        unlocks: dict[str, list[str]] = {key: [] for key in self.specs}
        for factor, step in self.reachable_steps(register):
            outstanding = step.outstanding(satisfied)
            if not outstanding:
                continue
            share = self.step_weight(factor, step.from_level) / len(outstanding)
            for key in outstanding:
                credit[key] += share
                unlocks[key].append(step.label(factor))
        entries = [
            LeverageEntry(
                spec=spec,
                credit=credit[key],
                unlocks=tuple(unlocks[key]),
                leverage=credit[key] / spec.effort.cost_units,
            )
            for key, spec in self.specs.items()
        ]
        entries.sort(key=_leverage_sort_key)
        return tuple(entries)

    def measurement_limited_factors(self) -> tuple[CredibilityFactor, ...]:
        """The factors an experiment could actually move."""
        return tuple(
            entry.factor for entry in self.entries if entry.is_measurement_limited
        )

    def ordered_specs(self) -> Sequence[MeasurementSpec]:
        """Every spec, in key order, for stable rendering."""
        return [self.specs[key] for key in sorted(self.specs)]
