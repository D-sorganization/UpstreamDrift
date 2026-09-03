"""The record a real measurement arrives as, and the register that holds them.

Why a record type at all
------------------------

The credibility statement (:mod:`.credibility`) says validation is at
**0 of 4** and that nine launch-side quantities have no published
measurement (issue #8616).  Saying so is cheap.  What is expensive -- and
what this module supplies -- is a *defined way to stop saying it*: a
structure a measurement can be supplied in, such that supplying one moves
the assessment and nothing else does.

A record is deliberately not a number
-------------------------------------

A :class:`MeasurementRecord` carries the value, but the ledger gates on
the *procedural* half: how many samples, at what expanded uncertainty, in
what units, under what conditions, on what instrument.  That is the half
that decides whether a comparison can distinguish the model's prediction
from the borrowed constant it currently uses, and it is the half people
leave out.

Two bases, and only two
-----------------------

:attr:`MeasurementBasis.INSTRUMENT` is a real measurement and **must**
carry a value and a date.  :attr:`MeasurementBasis.SYNTHETIC_FIXTURE`
exists so the intake path can be tested end to end, and it **must not**
carry a value at all.  A fixture that cannot hold a number cannot be
quoted, copied into a report, or mistaken for data six months later --
which is the failure mode issue #7999 already caught once in this package,
when a hand-written shear-cell line shipped as a "calibration".

Note the asymmetry with :class:`~bunkershot3d.sand.provenance.ProvenanceBasis`:
that enum records where a *model constant* came from, and includes
``CONVENTION`` and ``BORROWED_ANALOGUE`` because most constants here are
not measurements.  This enum records where a *datum* came from, and has no
such shading, because a datum either was measured or was not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from .exceptions import VandVError

__all__ = [
    "SYNTHETIC_SOURCE_MARKER",
    "MeasurementBasis",
    "MeasurementIntakeError",
    "MeasurementRecord",
    "MeasurementRegister",
]

SYNTHETIC_SOURCE_MARKER = "SYNTHETIC FIXTURE NOT A MEASUREMENT"
"""Every synthetic record must say this in its own source string.

Shouted, unpunctuated and impossible to mistake for a laboratory name.
A grep for this string finds every place in the repository where the
apparatus is being exercised rather than used.
"""


class MeasurementIntakeError(VandVError):
    """A measurement record was malformed, or was not a measurement at all.

    Raised for a record that omits a required field, for a synthetic
    fixture that tries to carry a value, for an instrument record that
    does not, and for a measurement document whose schema version this
    package does not understand.
    """


class MeasurementBasis(StrEnum):
    """Whether a record came off an instrument or exists to test the path."""

    INSTRUMENT = "instrument"
    """A real measurement, made with the named instrument on the named sand."""

    SYNTHETIC_FIXTURE = "synthetic_fixture"
    """Exercises the intake path. Carries no value, and never can."""


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    """One measurement offered against one :class:`~.ledger.MeasurementSpec`.

    Attributes:
        spec_key: The ledger spec this record is offered against.
        basis: Instrument measurement, or synthetic fixture.
        source: Laboratory report, dataset identifier or, for a fixture,
            :data:`SYNTHETIC_SOURCE_MARKER` and where the fixture lives.
        instrument: The instrument the value was read from.
        conditions: The conditions the measurement was made under.
        sample_count: How many independent samples the value summarises.
        relative_expanded_uncertainty: Expanded uncertainty at ``k = 2``,
            as a fraction of the value. Strictly positive: a measurement
            reported without an uncertainty is not a measurement.
        unit: SI unit symbol, which must match the spec's exactly.
        value: The measured value, or ``None`` for a synthetic fixture.
        measured_on: ISO date. Required for an instrument record.
        note: Anything a reader of the record needs and the fields lack.
    """

    spec_key: str
    basis: MeasurementBasis
    source: str
    instrument: str
    conditions: str
    sample_count: int
    relative_expanded_uncertainty: float
    unit: str
    value: float | None = None
    measured_on: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the record.

        Raises:
            MeasurementIntakeError: If a required field is empty, a count or
                uncertainty is out of range, or the basis and the value
                disagree.
        """
        self._require_text()
        self._require_counts()
        self._require_basis_consistency()

    def _require_text(self) -> None:
        """Raise unless every text field carries something.

        Raises:
            MeasurementIntakeError: naming the empty field.
        """
        for name in ("spec_key", "source", "instrument", "conditions", "unit"):
            if not str(getattr(self, name)).strip():
                raise MeasurementIntakeError(
                    f"measurement record has an empty {name}; a record that "
                    "cannot say what it measured, on what, is not intake"
                )

    def _require_counts(self) -> None:
        """Raise unless the sample count and uncertainty are usable.

        Raises:
            MeasurementIntakeError: If either is out of range.
        """
        if not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise MeasurementIntakeError(
                f"{self.spec_key}: sample_count must be a positive integer, "
                f"got {self.sample_count!r}"
            )
        uncertainty = self.relative_expanded_uncertainty
        if not math.isfinite(uncertainty) or uncertainty <= 0.0:
            raise MeasurementIntakeError(
                f"{self.spec_key}: relative_expanded_uncertainty must be a "
                f"positive finite fraction, got {uncertainty!r}. A value "
                "reported without an uncertainty cannot be compared against "
                "anything, so it cannot validate anything."
            )

    def _require_basis_consistency(self) -> None:
        """Raise unless the basis and the value agree.

        Raises:
            MeasurementIntakeError: If a fixture carries a value, if an
                instrument record does not, if a fixture fails to declare
                itself, or if an instrument record is undated.
        """
        if self.basis is MeasurementBasis.SYNTHETIC_FIXTURE:
            if self.value is not None:
                raise MeasurementIntakeError(
                    f"{self.spec_key}: a synthetic fixture may not carry a "
                    "value. Inventing a plausible-looking number, even as an "
                    "example, is how a fixture becomes a citation."
                )
            if SYNTHETIC_SOURCE_MARKER not in self.source:
                raise MeasurementIntakeError(
                    f"{self.spec_key}: a synthetic record must contain "
                    f"'{SYNTHETIC_SOURCE_MARKER}' in its source, so that a "
                    "reader who sees only the source knows it is not data"
                )
            return
        if self.value is None or not math.isfinite(self.value):
            raise MeasurementIntakeError(
                f"{self.spec_key}: an instrument record must carry a value, "
                f"got {self.value!r}"
            )
        if SYNTHETIC_SOURCE_MARKER in self.source:
            raise MeasurementIntakeError(
                f"{self.spec_key}: an instrument record must not claim to be "
                "a synthetic fixture"
            )
        self._require_date()

    def _require_date(self) -> None:
        """Raise unless an instrument record carries an ISO date.

        Raises:
            MeasurementIntakeError: If the date is missing or malformed.
        """
        if not self.measured_on.strip():
            raise MeasurementIntakeError(
                f"{self.spec_key}: an instrument record must carry "
                "measured_on as an ISO date; an undated measurement cannot be "
                "traced back to a sand state or a calibration"
            )
        try:
            date.fromisoformat(self.measured_on)
        except ValueError as exc:
            raise MeasurementIntakeError(
                f"{self.spec_key}: measured_on must be an ISO date, got "
                f"{self.measured_on!r}"
            ) from exc

    @property
    def is_synthetic(self) -> bool:
        """True when this record exists to exercise the intake path."""
        return self.basis is MeasurementBasis.SYNTHETIC_FIXTURE

    def describe(self) -> str:
        """A one-line statement fit for a manifest or a report."""
        value = "no value (synthetic fixture)" if self.is_synthetic else f"{self.value}"
        return (
            f"{self.spec_key}: {value} {self.unit} "
            f"(n = {self.sample_count}, U_rel = "
            f"{self.relative_expanded_uncertainty:.3g} at k = 2) "
            f"from {self.source}"
        )


@dataclass(frozen=True, slots=True)
class MeasurementRegister:
    """Every measurement currently offered to the ledger.

    Attributes:
        records: The records, in the order they were supplied.
    """

    records: tuple[MeasurementRecord, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the records.

        Raises:
            MeasurementIntakeError: If anything in ``records`` is not a
                :class:`MeasurementRecord`.
        """
        frozen = tuple(self.records)
        for record in frozen:
            if not isinstance(record, MeasurementRecord):
                raise MeasurementIntakeError(
                    "a measurement register holds MeasurementRecord values, "
                    f"got {type(record).__name__}"
                )
        object.__setattr__(self, "records", frozen)

    @property
    def is_empty(self) -> bool:
        """True when no measurement has been supplied."""
        return not self.records

    @property
    def has_synthetic_records(self) -> bool:
        """True when any record exists only to exercise the intake path."""
        return any(record.is_synthetic for record in self.records)

    def records_for(self, spec_key: str) -> tuple[MeasurementRecord, ...]:
        """Return every record offered against ``spec_key``."""
        return tuple(r for r in self.records if r.spec_key == spec_key)

    def merged_with(self, other: MeasurementRegister) -> MeasurementRegister:
        """Return a register holding this register's records and ``other``'s."""
        return MeasurementRegister(records=self.records + other.records)
