"""The intake path: how a real measurement gets into the assessment.

There is exactly one way in.  A measurement document is a small YAML file
naming the ledger spec it answers, the conditions and instrument it was
made with, the sample count and the expanded uncertainty; it is loaded
here into :class:`~.measurement.MeasurementRecord` values; those form a
:class:`~.measurement.MeasurementRegister`; and
:func:`~.credibility.credibility_assessment` derives the NASA-STD-7009B
score from the ledger and that register.  Nothing else moves the score.
A prose claim in a docstring does not, a comment does not, and neither
does this module existing.

Where documents live
--------------------

:data:`MEASUREMENTS_DIR` ships **empty** and must keep shipping empty
until something is measured.  :func:`shipped_register` reads it and
**refuses** any document containing a synthetic fixture, so the fixtures
that prove the apparatus works can never reach the published score.  If
the directory is missing entirely -- an installed wheel that did not carry
the package data -- the register is empty, which errs toward understating
rather than overstating.  That is the only direction an error here is
allowed to point.

Provenance, and the rank that PR #9238 established
--------------------------------------------------

:func:`provenance_updates` turns satisfied specs into
:class:`~bunkershot3d.sand.provenance.PropertyProvenance` records with
:attr:`~bunkershot3d.sand.provenance.ProvenanceBasis.MEASURED`.  It is the
only thing in the package that produces that basis, which is what makes
"nothing is measured" a checkable statement rather than a habit.

:data:`EVIDENTIAL_RANK` encodes the distinction PR #9238 drew and that
this package has to keep: ``BORROWED_ANALOGUE``, ``ESTIMATED`` and
``CONVENTION`` all sit at the same rank.  Fitting a constant to a
declared, *simulated* target makes it more checkable, not better
evidenced, and :func:`is_provenance_upgrade` says so.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ..sand.provenance import PropertyProvenance, ProvenanceBasis
from .ledger import ValidationLedger
from .measurement import (
    MeasurementBasis,
    MeasurementIntakeError,
    MeasurementRecord,
    MeasurementRegister,
)
from .roadmap import VALIDATION_LEDGER

__all__ = [
    "EVIDENTIAL_RANK",
    "MEASUREMENTS_DIR",
    "SUPPORTED_SCHEMA_VERSION",
    "describe_register",
    "is_provenance_upgrade",
    "load_measurement_document",
    "load_measurements",
    "measured_property_names",
    "provenance_updates",
    "shipped_register",
]

MEASUREMENTS_DIR = Path(__file__).resolve().parent / "measurements"
"""Where a real measurement document goes. Ships empty, and must stay empty."""

SUPPORTED_SCHEMA_VERSION = 1
"""The only measurement-document schema version this package understands."""

_DOCUMENT_SUFFIXES = (".yaml", ".yml", ".json")
"""Recognised document extensions. JSON is read by the YAML parser."""

_REQUIRED_FIELDS = (
    "spec_key",
    "basis",
    "source",
    "instrument",
    "conditions",
    "sample_count",
    "relative_expanded_uncertainty",
    "unit",
)
"""Fields a record must carry. ``value`` is required only of instrument records."""

EVIDENTIAL_RANK: Mapping[ProvenanceBasis, int] = MappingProxyType(
    {
        ProvenanceBasis.BORROWED_ANALOGUE: 0,
        ProvenanceBasis.ESTIMATED: 0,
        ProvenanceBasis.CONVENTION: 0,
        ProvenanceBasis.SPECIFICATION: 1,
        ProvenanceBasis.MEASURED: 2,
    }
)
"""How much evidence each provenance basis carries. Ties are deliberate.

The three at rank 0 are three different ways of not having measured
something.  A constant fitted to a simulated target (``CONVENTION``) is
more checkable than one lifted from a hardware-store analogue
(``BORROWED_ANALOGUE``) and that is worth having, but it is not better
evidence, and a ranking that scored it higher would let a package walk
itself up a credibility scale without ever touching a sample.
"""


def is_provenance_upgrade(old: ProvenanceBasis, new: ProvenanceBasis) -> bool:
    """True only when ``new`` carries strictly more evidence than ``old``.

    Lateral moves -- ``BORROWED_ANALOGUE`` to ``CONVENTION`` and back --
    return ``False`` in both directions.
    """
    return EVIDENTIAL_RANK[new] > EVIDENTIAL_RANK[old]


def load_measurement_document(
    path: Path, ledger: ValidationLedger = VALIDATION_LEDGER
) -> tuple[MeasurementRecord, ...]:
    """Load one measurement document.

    Args:
        path: The document.
        ledger: The ledger whose spec keys the records must name.

    Returns:
        The records, in document order.

    Raises:
        MeasurementIntakeError: If the file cannot be parsed, declares an
            unsupported schema version, is not shaped as a document, omits
            a required field, or names a spec the ledger does not have.
    """
    payload = _parse(path)
    version = payload.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise MeasurementIntakeError(
            f"{path}: schema_version must be {SUPPORTED_SCHEMA_VERSION}, got "
            f"{version!r}. A document this package does not understand is "
            "refused rather than partly read."
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise MeasurementIntakeError(
            f"{path}: 'records' must be a non-empty list; a measurement "
            "document with no measurements in it is a filing error"
        )
    return tuple(
        _record(path, index, item, ledger) for index, item in enumerate(raw_records)
    )


def load_measurements(
    directory: Path, ledger: ValidationLedger = VALIDATION_LEDGER
) -> tuple[MeasurementRecord, ...]:
    """Load every measurement document in ``directory``.

    Args:
        directory: Where the documents live. A missing directory yields no
            records rather than raising: an installed wheel that did not
            carry the package data must understate, never overstate.
        ledger: The ledger whose spec keys the records must name.

    Returns:
        Every record, ordered by file name and then by document order.

    Raises:
        MeasurementIntakeError: propagated from any document that is
            malformed.
    """
    if not directory.is_dir():
        return ()
    records: list[MeasurementRecord] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in _DOCUMENT_SUFFIXES:
            records.extend(load_measurement_document(path, ledger))
    return tuple(records)


def shipped_register(
    directory: Path | None = None, ledger: ValidationLedger = VALIDATION_LEDGER
) -> MeasurementRegister:
    """The register the published credibility assessment is derived from.

    Args:
        directory: Override for :data:`MEASUREMENTS_DIR`, for tests.
        ledger: The ledger whose spec keys the records must name.

    Returns:
        A register holding every real measurement on file. Empty today.

    Raises:
        MeasurementIntakeError: If any document holds a synthetic fixture.
            A fixture exists to prove the intake path works; letting one
            reach the published score would make the apparatus the exact
            defect it was built to prevent.
    """
    root = MEASUREMENTS_DIR if directory is None else directory
    records = load_measurements(root, ledger)
    synthetic = [r.spec_key for r in records if r.is_synthetic]
    if synthetic:
        raise MeasurementIntakeError(
            f"{root} holds synthetic fixture record(s) for "
            + ", ".join(sorted(synthetic))
            + ". A synthetic record may exercise the intake path in a test, "
            "and may never reach the published credibility assessment."
        )
    return MeasurementRegister(records=records)


def provenance_updates(
    register: MeasurementRegister, ledger: ValidationLedger = VALIDATION_LEDGER
) -> Mapping[str, PropertyProvenance]:
    """The sand properties this register moves to ``MEASURED``.

    A spec contributes only when a record actually satisfies its acceptance
    criterion; a record that misses the sample count or the uncertainty
    bound flips nothing, because a measurement too coarse to compare
    against is not a measurement of the model's input either.

    Args:
        register: The measurements on hand.
        ledger: The ledger the specs come from.

    Returns:
        Property name to its new provenance record. Empty when nothing has
        been measured, which is the shipped state.
    """
    updates: dict[str, PropertyProvenance] = {}
    for key in sorted(ledger.satisfied_spec_keys(register)):
        spec = ledger.spec(key)
        record = spec.best_record(register)
        if record is None:  # pragma: no cover - satisfied implies a record
            continue
        for prop in spec.provenance_keys:
            updates[prop] = PropertyProvenance(
                basis=ProvenanceBasis.MEASURED,
                source=record.source,
                note=(
                    f"measured on golf bunker sand as {spec.quantity} "
                    f"({record.sample_count} sample(s), U_rel = "
                    f"{record.relative_expanded_uncertainty:.3g} at k = 2) "
                    f"under the conditions of ledger spec '{spec.key}'"
                ),
            )
    return MappingProxyType(updates)


def _parse(path: Path) -> dict[str, Any]:
    """Read one document into a mapping.

    Raises:
        MeasurementIntakeError: If the file is missing, unparseable, or not
            a mapping at the top level.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MeasurementIntakeError(f"{path}: cannot be read: {exc}") from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MeasurementIntakeError(f"{path}: is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise MeasurementIntakeError(
            f"{path}: a measurement document is a mapping with "
            "'schema_version' and 'records'"
        )
    return payload


def _record(
    path: Path, index: int, item: Any, ledger: ValidationLedger
) -> MeasurementRecord:
    """Build one record from one document entry.

    Raises:
        MeasurementIntakeError: If the entry is not a mapping, omits a
            required field, carries an unknown basis, or names a spec the
            ledger does not have.
    """
    where = f"{path} record {index}"
    if not isinstance(item, dict):
        raise MeasurementIntakeError(f"{where}: must be a mapping")
    missing = [name for name in _REQUIRED_FIELDS if item.get(name) is None]
    if missing:
        raise MeasurementIntakeError(f"{where}: is missing " + ", ".join(missing))
    spec_key = str(item["spec_key"])
    if spec_key not in ledger.specs:
        raise MeasurementIntakeError(
            f"{where}: names the measurement {spec_key!r}, which is not in "
            "the validation ledger. Add the spec to the ledger first, so "
            "that what the measurement would buy is stated before it is "
            "made rather than after."
        )
    try:
        basis = MeasurementBasis(str(item["basis"]))
    except ValueError as exc:
        raise MeasurementIntakeError(
            f"{where}: basis must be one of "
            + ", ".join(member.value for member in MeasurementBasis)
            + f", got {item['basis']!r}"
        ) from exc
    value = item.get("value")
    return MeasurementRecord(
        spec_key=spec_key,
        basis=basis,
        source=str(item["source"]),
        instrument=str(item["instrument"]),
        conditions=str(item["conditions"]),
        sample_count=_as_int(where, "sample_count", item["sample_count"]),
        relative_expanded_uncertainty=_as_float(
            where,
            "relative_expanded_uncertainty",
            item["relative_expanded_uncertainty"],
        ),
        unit=str(item["unit"]),
        value=None if value is None else _as_float(where, "value", value),
        measured_on=str(item.get("measured_on", "")),
        note=str(item.get("note", "")),
    )


def _as_int(where: str, name: str, raw: Any) -> int:
    """Coerce a document field to an int.

    Raises:
        MeasurementIntakeError: If it is not an integer.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise MeasurementIntakeError(f"{where}: {name} must be an integer, got {raw!r}")
    return raw


def _as_float(where: str, name: str, raw: Any) -> float:
    """Coerce a document field to a float.

    Raises:
        MeasurementIntakeError: If it is not a number.
    """
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise MeasurementIntakeError(f"{where}: {name} must be a number, got {raw!r}")
    return float(raw)


def measured_property_names(
    register: MeasurementRegister, ledger: ValidationLedger = VALIDATION_LEDGER
) -> tuple[str, ...]:
    """The sand properties this register has measured, in name order."""
    return tuple(sorted(provenance_updates(register, ledger)))


def describe_register(records: Iterable[MeasurementRecord]) -> str:
    """A multi-line statement of what has been measured, for a manifest."""
    lines = [record.describe() for record in records]
    if not lines:
        return "no measurements on file: NASA-STD-7009B validation is 0 of 4"
    return "\n".join(lines)
