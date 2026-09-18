"""Measurement-to-prediction program for the sand-to-ball transfer (issue #9543).

What this module is for
-----------------------

:mod:`bunkershot3d.ball.splash` partitions the delivered sand impulse into a
ball launch through :class:`~bunkershot3d.ball.splash.MomentumTransfer`, whose
every parameter is a stated placeholder, and floors every launch verdict at
``BEYOND_VALIDATION`` because per issue #8616 no published measurement of
ball speed, launch angle or spin out of sand exists. Issue #9239 showed what
that costs: correcting one assumption (the accelerated mass, issue #8659)
moved the nominal carry from 11.8 m to 1.6 m and emptied the workbench's
playability window, and nothing in the package could say which of the two
numbers was closer to a real shot.

This module is the defined way to stop saying "uncalibrated". It holds the
*contract* of the program -- what a measured stroke must carry, how the
intended-use matrix and the measurement protocol are registered, which
thresholds are fixed before any held-out analysis, and what the versioned
evidence looks like once it exists. :mod:`.qualification_fit` holds the
arithmetic: the bounded fit on the calibration subset, the held-out
comparison under ASME V&V 20, and the report.

Nothing here is a measurement. The shipped register of measured strokes is
**empty** and stays empty until real strokes are measured; the objects below
are the shapes those strokes will arrive in, and the tests exercise them with
fixtures that are refused the moment they claim to be data.

The rules the contract enforces
-------------------------------

* **No synthetic stroke can qualify anything.** A
  :class:`MeasuredStroke` carries its launch quantities as
  :class:`~bunkershot3d.vandv.measurement.MeasurementRecord` values and
  refuses a synthetic fixture at construction, so the intake path can be
  tested but a fixture can never reach a fit.
* **No train/test leakage.** The calibration subset is designated by
  *session* before any fit, sessions are disjoint by construction, and two
  strokes with the same raw-data digest are refused as one stroke filed
  twice.
* **Thresholds precede analysis.** :data:`PRACTICAL_TOLERANCES` is a frozen
  constant with its rationale in the source; a qualification records the
  tolerances it was judged against, and a regime that fails them is
  recorded as rejected rather than dropped.
* **Sand-only measurements cannot qualify ball prediction.** The sand
  characterisation register admits a batch through the ledger's own
  acceptance criteria (issue #9286), but a stroke without a ball-speed
  record is not a stroke.
* **The floor is lifted per regime, by version.** A
  :class:`TransferQualification` names the regimes it qualified and the
  evidence digest it rests on; :meth:`TransferQualification.statement_for`
  answers ``WITHIN`` only for a strike inside a qualified regime and keeps
  the ``BEYOND_VALIDATION`` floor everywhere else.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from types import MappingProxyType

from ..exceptions import BunkerShot3DValueError
from ..metrics.playability import DEFAULT_CARRY_TOLERANCE_FRACTION
from ..provenance.hashing import canonical_json
from ..solvers.envelope import EnvelopeStatus
from ..vandv.ledger import ValidationLedger
from ..vandv.measurement import MeasurementRecord, MeasurementRegister
from ..vandv.roadmap import VALIDATION_LEDGER
from ..vandv.validation import NumericalUncertainty
from .lie import BallLie, BallLieType, BallProperties
from .rig_capability import (
    THREE_CAMERA_RIG_CAPABILITY,
    RigCapability,
    rig_capability_markdown,
)
from .splash import MomentumTransfer, SandDelivery

__all__ = [
    "BALL_LAUNCH_REFERENCE_SPECS",
    "INTENDED_USE_MATRIX",
    "MEASUREMENT_PROTOCOL",
    "PRACTICAL_TOLERANCES",
    "QUALIFICATION_EVIDENCE_SCHEMA",
    "THREE_CAMERA_RIG_CAPABILITY",
    "FitOutcome",
    "FitStatus",
    "IntendedUseMatrix",
    "MeasuredStroke",
    "MeasurementProtocol",
    "ObjectiveDisposition",
    "PracticalTolerances",
    "QualificationDataset",
    "RegimeVerdict",
    "RigCapability",
    "TransferQualification",
    "TransferQualificationError",
    "UseRegime",
    "objective_disposition",
    "rig_capability_markdown",
]

QUALIFICATION_EVIDENCE_SCHEMA = "transfer-qualification/1"
"""Schema of the versioned evidence a :class:`TransferQualification` is."""

BALL_LAUNCH_REFERENCE_SPECS: Mapping[str, str] = MappingProxyType(
    {
        "ball_launch_speed_m_s": "m/s",
        "ball_launch_angle_rad": "rad",
        "ball_spin_rate_rad_s": "rad/s",
    }
)
"""The explicit ball-launch reference records, key to required unit.

These extend the ledger's seven sand-side specs (:mod:`bunkershot3d.vandv.roadmap`)
with the launch-side quantities the ledger's video spec names but does not
key individually. A :class:`MeasuredStroke` offers one
:class:`~bunkershot3d.vandv.measurement.MeasurementRecord` against each,
in exactly this unit."""

_DIGEST_HEX_LENGTH = 64
"""Length of a SHA-256 hex digest, the only raw-data digest accepted."""


class TransferQualificationError(BunkerShot3DValueError):
    """A stroke, dataset or qualification was malformed or would leak."""


class FitStatus(StrEnum):
    """How a calibration fit ended. Every value is preserved in the evidence."""

    CONVERGED = "converged"
    """The bounded fit converged and every declared parameter is identifiable."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Too few calibration strokes, or a declared parameter's quantity is unmeasured."""

    UNIDENTIFIABLE = "unidentifiable"
    """The calibration subset cannot separate the declared parameters."""

    NOT_CONVERGED = "not_converged"
    """The optimiser stopped without converging."""


class ObjectiveDisposition(StrEnum):
    """What may be done with the carry objective of issue #9239."""

    UNAVAILABLE_UNCALIBRATED = "unavailable-uncalibrated"
    """No qualification exists: ranking on carry is unavailable."""

    UNAVAILABLE_OUTSIDE_QUALIFIED_REGIME = "unavailable-outside-qualified-regime"
    """A qualification exists but does not cover this strike."""

    DEGENERATE_TARGET = "degenerate-target"
    """Qualified, but the nominal carry lies outside the window at this target."""

    SUPPORTED = "supported"
    """Qualified, and the window is non-empty at this target."""


def _require_text(owner: str, name: str, value: str) -> str:
    """Return ``value`` stripped, or raise if it is empty.

    Raises:
        TransferQualificationError: naming the empty field.
    """
    text = str(value).strip()
    if not text:
        raise TransferQualificationError(f"{owner} has an empty {name}")
    return text


def _require_ordered_unit_range(
    owner: str, name: str, bounds: tuple[float, float], upper_limit: float
) -> tuple[float, float]:
    """Return ``bounds`` as floats, or raise if not ordered inside the limit.

    Raises:
        TransferQualificationError: If the pair is malformed.
    """
    try:
        low, high = (float(v) for v in bounds)
    except (TypeError, ValueError) as error:
        raise TransferQualificationError(
            f"{owner}: {name} must be a (low, high) pair, got {bounds!r}"
        ) from error
    if not (math.isfinite(low) and math.isfinite(high)) or not (
        0.0 <= low < high <= upper_limit
    ):
        raise TransferQualificationError(
            f"{owner}: {name} must satisfy 0 <= low < high <= {upper_limit:g}, "
            f"got ({low!r}, {high!r})"
        )
    return (low, high)


@dataclass(frozen=True, slots=True)
class UseRegime:
    """One cell of the intended-use matrix.

    Attributes:
        key: Stable identifier the evidence names the regime by.
        lie_type: The lie class the regime covers, from
            :attr:`~bunkershot3d.ball.lie.BallLie.lie_type`.
        relative_density_range: ``(low, high)`` of bed relative density.
        entry_speed_range_m_s: ``(low, high)`` of head entry speed [m/s].
    """

    key: str
    lie_type: BallLieType
    relative_density_range: tuple[float, float]
    entry_speed_range_m_s: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate the regime.

        Raises:
            TransferQualificationError: If the key is empty, the lie type is
                not a :class:`~bunkershot3d.ball.lie.BallLieType`, or a range
                is malformed.
        """
        object.__setattr__(self, "key", _require_text("use regime", "key", self.key))
        if not isinstance(self.lie_type, BallLieType):
            raise TransferQualificationError(
                f"use regime {self.key!r}: lie_type must be a BallLieType, got "
                f"{self.lie_type!r}"
            )
        object.__setattr__(
            self,
            "relative_density_range",
            _require_ordered_unit_range(
                f"use regime {self.key!r}",
                "relative_density_range",
                self.relative_density_range,
                1.0,
            ),
        )
        object.__setattr__(
            self,
            "entry_speed_range_m_s",
            _require_ordered_unit_range(
                f"use regime {self.key!r}",
                "entry_speed_range_m_s",
                self.entry_speed_range_m_s,
                math.inf,
            ),
        )

    def covers(self, delivery: SandDelivery, lie: BallLie) -> bool:
        """True when the strike falls inside this regime."""
        low_d, high_d = self.relative_density_range
        low_v, high_v = self.entry_speed_range_m_s
        return (
            lie.lie_type is self.lie_type
            and low_d <= delivery.bed_relative_density <= high_d
            and low_v <= delivery.entry_speed_m_s <= high_v
        )

    def describe(self) -> str:
        """One line naming the regime's bounds."""
        return (
            f"{self.key}: {self.lie_type.value} lie, D_r in "
            f"[{self.relative_density_range[0]:g}, {self.relative_density_range[1]:g}], "
            f"entry speed in [{self.entry_speed_range_m_s[0]:g}, "
            f"{self.entry_speed_range_m_s[1]:g}] m/s"
        )


@dataclass(frozen=True, slots=True)
class IntendedUseMatrix:
    """The regimes the program intends to qualify, registered before intake.

    Attributes:
        regimes: The cells, with unique keys. A strike is assigned to the
            first regime that covers it, so overlapping cells are ordered.
    """

    regimes: tuple[UseRegime, ...]

    def __post_init__(self) -> None:
        """Validate the matrix.

        Raises:
            TransferQualificationError: If it is empty or a key repeats.
        """
        regimes = tuple(self.regimes)
        if not regimes:
            raise TransferQualificationError(
                "an intended-use matrix must name at least one regime"
            )
        keys = [r.key for r in regimes]
        if len(keys) != len(set(keys)):
            raise TransferQualificationError(
                "intended-use matrix regime keys must be unique, got " + ", ".join(keys)
            )
        object.__setattr__(self, "regimes", regimes)

    def regime_for(self, delivery: SandDelivery, lie: BallLie) -> UseRegime | None:
        """The first regime covering the strike, or ``None``."""
        for regime in self.regimes:
            if regime.covers(delivery, lie):
                return regime
        return None

    def regime(self, key: str) -> UseRegime:
        """Return the regime with ``key``.

        Raises:
            TransferQualificationError: If no regime has that key.
        """
        for regime in self.regimes:
            if regime.key == key:
                return regime
        raise TransferQualificationError(f"no use regime {key!r}")


@dataclass(frozen=True, slots=True)
class MeasurementProtocol:
    """The registered measurement protocol every stroke is made under.

    Attributes:
        name: Protocol name.
        version: Protocol version; a change of apparatus is a new version.
        apparatus: What measured the strokes, as a class and a configuration
            rather than a brand.
        calibration_reference: How the apparatus was calibrated and when.
        sand_characterisation_spec_keys: Ledger spec keys each sand batch
            must satisfy with an instrument record before its strokes count.
            Every key must exist in the ledger.
        note: What the protocol cannot measure, and anything else a reader
            of a stroke needs.
    """

    name: str
    version: str
    apparatus: str
    calibration_reference: str
    sand_characterisation_spec_keys: tuple[str, ...]
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the protocol against the ledger.

        Raises:
            TransferQualificationError: If a field is empty, or a spec key is
                not in the validation ledger.
        """
        for name in ("name", "version", "apparatus", "calibration_reference"):
            object.__setattr__(
                self,
                name,
                _require_text("measurement protocol", name, getattr(self, name)),
            )
        keys = tuple(str(k) for k in self.sand_characterisation_spec_keys)
        if not keys:
            raise TransferQualificationError(
                "a measurement protocol must require at least one sand "
                "characterisation spec; a stroke on uncharacterised sand cannot "
                "attribute its error to the model"
            )
        unknown = sorted(set(keys) - set(VALIDATION_LEDGER.specs))
        if unknown:
            raise TransferQualificationError(
                "measurement protocol names sand specs the ledger lacks: "
                + ", ".join(unknown)
            )
        object.__setattr__(self, "sand_characterisation_spec_keys", keys)


INTENDED_USE_MATRIX = IntendedUseMatrix(
    regimes=(
        UseRegime("standard-lie-firm", BallLieType.STANDARD, (0.6, 1.0), (20.0, 27.0)),
        UseRegime("standard-lie-loose", BallLieType.STANDARD, (0.0, 0.6), (20.0, 27.0)),
        UseRegime("buried-lie", BallLieType.BURIED, (0.0, 1.0), (20.0, 27.0)),
        UseRegime("plugged-lie", BallLieType.PLUGGED, (0.0, 1.0), (20.0, 27.0)),
    )
)
"""The registered intended-use matrix (issue #9543, step 1).

Four lie regimes over the 20-27 m/s delivery band the tool is designed for
(:mod:`bunkershot3d.vandv.roadmap`). The standard lie is split at ``D_r = 0.6``
because that is where the lie-dependent efficiency of issue #8704 moves the
answer most; the buried and plugged lies are single cells because the
intercepted fraction, not the packing, dominates there. A regime is
qualified or rejected on its own held-out strokes; none is qualified today."""

MEASUREMENT_PROTOCOL = MeasurementProtocol(
    name="splash-shot launch reference",
    version="1.0.0",
    apparatus=(
        "three global-shutter cameras at 1920x1200 and 60 fps over a calibrated "
        "target volume for ball launch speed and direction; a synchronised "
        "delivery measurement (optical or IMU) for head entry and exit states; "
        "a divot cast per stroke; bench characterisation of every sand batch"
    ),
    calibration_reference=(
        "camera intrinsics and extrinsics from the calibrated target volume on "
        "each session; the delivery instrument under its own certificate; both "
        "recorded in the raw-data manifest the stroke digest covers"
    ),
    sand_characterisation_spec_keys=(
        "bunker_sand_bulk_density_kg_m3",
        "bunker_sand_drained_friction_angle_deg",
    ),
    note=(
        "spin, ejecta motion and head force are unavailable on this rig (see "
        "THREE_CAMERA_RIG_CAPABILITY) and stay missing in the stroke record; "
        "particle size distribution and moisture travel with the bulk-density "
        "record's conditions until the ledger keys them separately"
    ),
)
"""The registered measurement protocol (issue #9543, step 1). Every stroke on
file must be made under it; a change of apparatus is a new version."""


def _require_reference_record(
    owner: str, spec_key: str, record: MeasurementRecord
) -> None:
    """Refuse a launch reference record that cannot stand as data.

    Raises:
        TransferQualificationError: If the record is not a
            :class:`~bunkershot3d.vandv.measurement.MeasurementRecord`, is
            synthetic, is offered against another key, or is in the wrong
            unit.
    """
    if not isinstance(record, MeasurementRecord):
        raise TransferQualificationError(
            f"{owner}: {spec_key} must be a MeasurementRecord, got "
            f"{type(record).__name__}"
        )
    if record.is_synthetic:
        raise TransferQualificationError(
            f"{owner}: {spec_key} is a synthetic fixture. A fixture may exercise "
            "the intake path and may never enter a calibration or a held-out "
            "comparison (issue #9543)"
        )
    if record.spec_key != spec_key:
        raise TransferQualificationError(
            f"{owner}: record offered against {record.spec_key!r} where "
            f"{spec_key!r} is required"
        )
    unit = BALL_LAUNCH_REFERENCE_SPECS[spec_key]
    if record.unit != unit:
        raise TransferQualificationError(
            f"{owner}: {spec_key} must be in {unit!r}, got {record.unit!r}"
        )


@dataclass(frozen=True, slots=True)
class MeasuredStroke:
    """One measured stroke: the model's inputs and the launch it produced.

    The model-side inputs are a :class:`~bunkershot3d.ball.splash.SandDelivery`
    built from the solver run on the *measured* delivery, so the prediction
    for this stroke is exactly what the shipped pipeline would produce. The
    measured launch is carried as instrument records so that every value
    arrives with its instrument, date and uncertainty or does not arrive.

    Attributes:
        stroke_id: Unique identifier within the dataset.
        session_id: The measurement session; the calibration/holdout split
            is by session.
        sand_batch_id: The sand batch, which must be characterised in the
            dataset's sand register.
        measured_on: ISO date.
        raw_data_digest: SHA-256 hex digest of the raw capture the launch
            was reduced from. Two strokes with one digest are one stroke.
        delivery: The solver's and metrics layer's account of the strike.
        lie: The ball's lie for the strike.
        club_loft_rad: Effective loft at delivery [rad].
        ball_speed: Instrument record against ``ball_launch_speed_m_s``.
        numerical: ``u_num`` of the predicted ball speed for this stroke
            [m/s], from the solver's grid study.
        launch_angle: Instrument record against ``ball_launch_angle_rad``,
            or ``None`` when the apparatus could not resolve it.
        spin_rate: Instrument record against ``ball_spin_rate_rad_s``, or
            ``None``. Missing quantities stay missing.
        u_input_m_s: Standard uncertainty of the predicted ball speed from
            the sand-state inputs [m/s].
        ball: Ball properties.
        note: Anything a reader of the stroke needs.
    """

    stroke_id: str
    session_id: str
    sand_batch_id: str
    measured_on: str
    raw_data_digest: str
    delivery: SandDelivery
    lie: BallLie
    club_loft_rad: float
    ball_speed: MeasurementRecord
    numerical: NumericalUncertainty
    launch_angle: MeasurementRecord | None = None
    spin_rate: MeasurementRecord | None = None
    u_input_m_s: float = 0.0
    ball: BallProperties = field(default_factory=BallProperties)
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the stroke.

        Raises:
            TransferQualificationError: If an identifier is empty, the date
                or digest is malformed, the loft is not in ``(0, pi/2)``, the
                input uncertainty is negative, or a launch record is not an
                instrument record in the required unit.
        """
        for name in ("stroke_id", "session_id", "sand_batch_id"):
            object.__setattr__(
                self, name, _require_text("measured stroke", name, getattr(self, name))
            )
        owner = f"stroke {self.stroke_id!r}"
        try:
            date.fromisoformat(self.measured_on)
        except (TypeError, ValueError) as error:
            raise TransferQualificationError(
                f"{owner}: measured_on must be an ISO date, got {self.measured_on!r}"
            ) from error
        digest = str(self.raw_data_digest).strip().lower()
        if len(digest) != _DIGEST_HEX_LENGTH or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise TransferQualificationError(
                f"{owner}: raw_data_digest must be a SHA-256 hex digest of the "
                f"raw capture, got {self.raw_data_digest!r}"
            )
        object.__setattr__(self, "raw_data_digest", digest)
        if not isinstance(self.delivery, SandDelivery):
            raise TransferQualificationError(
                f"{owner}: delivery must be a SandDelivery"
            )
        if not isinstance(self.lie, BallLie):
            raise TransferQualificationError(f"{owner}: lie must be a BallLie")
        loft = float(self.club_loft_rad)
        if not math.isfinite(loft) or not 0.0 < loft < math.pi / 2:
            raise TransferQualificationError(
                f"{owner}: club_loft_rad must be in (0, pi/2), got {loft!r}"
            )
        if not isinstance(self.numerical, NumericalUncertainty):
            raise TransferQualificationError(
                f"{owner}: numerical must be a NumericalUncertainty"
            )
        u_input = float(self.u_input_m_s)
        if not math.isfinite(u_input) or u_input < 0.0:
            raise TransferQualificationError(
                f"{owner}: u_input_m_s must be finite and non-negative, got {u_input!r}"
            )
        _require_reference_record(owner, "ball_launch_speed_m_s", self.ball_speed)
        if float(self.ball_speed.value) <= 0.0:  # type: ignore[arg-type]
            raise TransferQualificationError(
                f"{owner}: a measured launch speed must be positive, got "
                f"{self.ball_speed.value!r}; a ball that did not launch is not a "
                "splash-shot reference"
            )
        if self.launch_angle is not None:
            _require_reference_record(owner, "ball_launch_angle_rad", self.launch_angle)
        if self.spin_rate is not None:
            _require_reference_record(owner, "ball_spin_rate_rad_s", self.spin_rate)

    @property
    def lie_type(self) -> BallLieType:
        """The lie class of the stroke."""
        return self.lie.lie_type

    @property
    def measured_ball_speed_m_s(self) -> float:
        """The measured launch speed [m/s]."""
        return float(self.ball_speed.value)  # type: ignore[arg-type]

    @property
    def ball_speed_standard_uncertainty_m_s(self) -> float:
        """``u_exp`` of the launch speed [m/s]: expanded ``U_rel`` at ``k = 2``, halved."""
        return (
            0.5
            * self.ball_speed.relative_expanded_uncertainty
            * abs(self.measured_ball_speed_m_s)
        )


@dataclass(frozen=True, slots=True)
class QualificationDataset:
    """Every measured stroke on file, with its split designated up front.

    Attributes:
        protocol: The protocol every stroke was made under.
        intended_use: The regimes being qualified.
        sand_batches: Sand batch id to its characterisation register.
        strokes: Every measured stroke.
        calibration_session_ids: The sessions designated for the fit. Every
            other session is held out. Designated before the fit, never
            after it.
    """

    protocol: MeasurementProtocol
    intended_use: IntendedUseMatrix
    sand_batches: Mapping[str, MeasurementRegister]
    strokes: tuple[MeasuredStroke, ...]
    calibration_session_ids: frozenset[str]

    def __post_init__(self) -> None:
        """Validate the dataset and refuse anything that would leak.

        Raises:
            TransferQualificationError: If there are no strokes, an id or
                digest repeats, a batch is uncharacterised or characterised
                by a synthetic record, a stroke lies outside the matrix, a
                designated session has no strokes, or either subset is
                empty.
        """
        strokes = tuple(self.strokes)
        if not strokes:
            raise TransferQualificationError("a qualification dataset has no strokes")
        object.__setattr__(self, "strokes", strokes)
        object.__setattr__(
            self, "sand_batches", MappingProxyType(dict(self.sand_batches))
        )
        object.__setattr__(
            self, "calibration_session_ids", frozenset(self.calibration_session_ids)
        )
        self._require_unique_strokes()
        self._require_characterised_batches()
        self._require_strokes_inside_matrix()
        self._require_split()

    def _require_unique_strokes(self) -> None:
        """Raise if a stroke id or a raw-data digest repeats."""
        ids = [s.stroke_id for s in self.strokes]
        if len(ids) != len(set(ids)):
            raise TransferQualificationError("stroke ids must be unique")
        digests: dict[str, str] = {}
        for stroke in self.strokes:
            other = digests.get(stroke.raw_data_digest)
            if other is not None:
                raise TransferQualificationError(
                    f"strokes {other!r} and {stroke.stroke_id!r} carry the same "
                    "raw-data digest; one capture filed twice would leak between "
                    "calibration and holdout"
                )
            digests[stroke.raw_data_digest] = stroke.stroke_id

    def _require_characterised_batches(self) -> None:
        """Raise unless every batch satisfies every required sand spec."""
        ledger: ValidationLedger = VALIDATION_LEDGER
        for batch in sorted({s.sand_batch_id for s in self.strokes}):
            register = self.sand_batches.get(batch)
            if not isinstance(register, MeasurementRegister):
                raise TransferQualificationError(
                    f"sand batch {batch!r} has no characterisation register; a "
                    "stroke on uncharacterised sand cannot attribute its error "
                    "to the model"
                )
            if register.has_synthetic_records:
                raise TransferQualificationError(
                    f"sand batch {batch!r} is characterised by a synthetic fixture"
                )
            missing = [
                key
                for key in self.protocol.sand_characterisation_spec_keys
                if ledger.spec(key).best_record(register) is None
            ]
            if missing:
                raise TransferQualificationError(
                    f"sand batch {batch!r} has no record satisfying "
                    + ", ".join(missing)
                    + " under the ledger's acceptance criteria (issue #9286)"
                )

    def _require_strokes_inside_matrix(self) -> None:
        """Raise if a stroke falls outside every regime."""
        for stroke in self.strokes:
            if self.intended_use.regime_for(stroke.delivery, stroke.lie) is None:
                raise TransferQualificationError(
                    f"stroke {stroke.stroke_id!r} lies outside every regime of the "
                    "intended-use matrix; extend the matrix before intake, not after"
                )

    def _require_split(self) -> None:
        """Raise unless both subsets are non-empty and the designation is real."""
        sessions = {s.session_id for s in self.strokes}
        unknown = sorted(self.calibration_session_ids - sessions)
        if unknown:
            raise TransferQualificationError(
                "calibration sessions with no strokes: " + ", ".join(unknown)
            )
        if not self.calibration_session_ids:
            raise TransferQualificationError(
                "no calibration session designated; the split is declared before "
                "the fit, not chosen by it"
            )
        if not sessions - self.calibration_session_ids:
            raise TransferQualificationError(
                "every session is designated for calibration, leaving nothing "
                "held out; a fit checked on its own data qualifies nothing"
            )

    @property
    def calibration_strokes(self) -> tuple[MeasuredStroke, ...]:
        """Strokes from the designated calibration sessions."""
        return tuple(
            s for s in self.strokes if s.session_id in self.calibration_session_ids
        )

    @property
    def holdout_strokes(self) -> tuple[MeasuredStroke, ...]:
        """Strokes from every other session."""
        return tuple(
            s for s in self.strokes if s.session_id not in self.calibration_session_ids
        )

    @property
    def unseen_holdout_batches(self) -> tuple[str, ...]:
        """Held-out sand batches that no calibration stroke was made on."""
        seen = {s.sand_batch_id for s in self.calibration_strokes}
        return tuple(
            sorted(
                {
                    s.sand_batch_id
                    for s in self.holdout_strokes
                    if s.sand_batch_id not in seen
                }
            )
        )

    def evidence_digest(self) -> str:
        """SHA-256 over what the qualification rests on, for its version."""
        payload = {
            "schema": QUALIFICATION_EVIDENCE_SCHEMA,
            "protocol": [self.protocol.name, self.protocol.version],
            "calibration_sessions": sorted(self.calibration_session_ids),
            "strokes": [
                [s.stroke_id, s.session_id, s.sand_batch_id, s.raw_data_digest]
                for s in self.strokes
            ],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PracticalTolerances:
    """Thresholds fixed before any held-out analysis, with their rationale.

    Attributes:
        max_abs_relative_bias: Largest ``|mean(E / D)|`` of held-out ball
            speed a regime may show.
        max_relative_rms_error: Largest ``sqrt(mean((E / D)^2))``.
        min_interval_coverage: Smallest share of held-out strokes whose
            ``|E|`` falls inside the expanded ``U = k u_val``.
        max_launch_angle_bias_rad: Largest ``|mean(E)|`` of launch angle,
            over the strokes that carry one.
        min_holdout_strokes_per_regime: Fewest held-out strokes a regime
            needs before its statistics mean anything.
        rationale: Why these numbers.
    """

    max_abs_relative_bias: float
    max_relative_rms_error: float
    min_interval_coverage: float
    max_launch_angle_bias_rad: float
    min_holdout_strokes_per_regime: int
    rationale: str

    def __post_init__(self) -> None:
        """Validate the tolerances.

        Raises:
            TransferQualificationError: If a fraction is outside ``(0, 1)``,
                the angle is not positive, the count is below one, or the
                rationale is missing.
        """
        for name in (
            "max_abs_relative_bias",
            "max_relative_rms_error",
            "min_interval_coverage",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise TransferQualificationError(
                    f"{name} must be a fraction in (0, 1), got {value!r}"
                )
        if not math.isfinite(self.max_launch_angle_bias_rad) or (
            self.max_launch_angle_bias_rad <= 0.0
        ):
            raise TransferQualificationError(
                "max_launch_angle_bias_rad must be positive, got "
                f"{self.max_launch_angle_bias_rad!r}"
            )
        if (
            not isinstance(self.min_holdout_strokes_per_regime, int)
            or self.min_holdout_strokes_per_regime < 1
        ):
            raise TransferQualificationError(
                "min_holdout_strokes_per_regime must be a positive integer"
            )
        _require_text("practical tolerances", "rationale", self.rationale)


PRACTICAL_TOLERANCES = PracticalTolerances(
    max_abs_relative_bias=0.05,
    max_relative_rms_error=0.10,
    min_interval_coverage=0.80,
    max_launch_angle_bias_rad=math.radians(3.0),
    min_holdout_strokes_per_regime=5,
    rationale=(
        "The playability window (issue #8614) accepts carry within +/-10 % of "
        "the target, and carry scales close to the square of ball speed over a "
        "greenside shot, so a systematic ball-speed error of 5 % consumes the "
        "whole half-width; the bias bound is that. The RMS bound is the window "
        "half-width itself: a model whose held-out scatter exceeds the band it "
        "is judged in cannot rank designs on that band (issue #9243). Coverage "
        "of 0.80 for a nominal 95 % interval is the floor below which the stated "
        "uncertainty is understated by more than the ranking can absorb. Three "
        "degrees of launch-angle bias moves carry by under 2 % near the loft "
        "the tool is designed for. Five held-out strokes per regime is the "
        "fewest at which a coverage fraction resolves to 0.2, and is half the "
        "ten-stroke sample the ledger's video spec already demands."
    ),
)
"""The predeclared tolerances (issue #9543, step 4). Fixed here, not per run."""


@dataclass(frozen=True, slots=True)
class FitOutcome:
    """How the calibration fit ended, converged or not. Always preserved.

    Attributes:
        status: The outcome.
        parameters: The parameter names that were declared for fitting.
        transfer: The frozen parameters, or ``None`` unless converged.
        standard_errors: Parameter name to its standard error.
        sensitivities: Parameter name to the mean normalised local
            sensitivity ``(p / v) dv/dp`` of predicted ball speed.
        covariance: Row-major ``p x p`` parameter covariance, or ``()``.
        residual_rms: RMS of the weighted residuals ``(S - D) / u_exp``.
        calibration_stroke_ids: The strokes the fit used.
        reasons: Why the fit ended as it did.
    """

    status: FitStatus
    parameters: tuple[str, ...]
    transfer: MomentumTransfer | None
    standard_errors: Mapping[str, float]
    sensitivities: Mapping[str, float]
    covariance: tuple[tuple[float, ...], ...]
    residual_rms: float | None
    calibration_stroke_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze the mappings and keep status and transfer consistent.

        Raises:
            TransferQualificationError: If a converged fit carries no
                transfer or a failed one carries one.
        """
        object.__setattr__(
            self, "standard_errors", MappingProxyType(dict(self.standard_errors))
        )
        object.__setattr__(
            self, "sensitivities", MappingProxyType(dict(self.sensitivities))
        )
        converged = self.status is FitStatus.CONVERGED
        if converged != (self.transfer is not None):
            raise TransferQualificationError(
                f"a fit with status {self.status.value!r} "
                + ("must" if converged else "must not")
                + " carry frozen parameters"
            )

    @property
    def succeeded(self) -> bool:
        """True only for a converged, identifiable fit."""
        return self.status is FitStatus.CONVERGED


@dataclass(frozen=True, slots=True)
class RegimeVerdict:
    """The held-out verdict for one regime of the intended-use matrix.

    Attributes:
        regime: The regime.
        qualified: Whether every predeclared tolerance was met.
        holdout_stroke_ids: The held-out strokes in the regime.
        holdout_session_ids: Their sessions, all independent of calibration.
        independent_sand_batches: Whether every held-out batch was unseen
            in calibration.
        relative_bias: ``mean(E / D)`` of ball speed, or ``None``.
        relative_rms_error: ``sqrt(mean((E / D)^2))``, or ``None``.
        interval_coverage: Share of strokes with ``|E| <= U``, or ``None``.
        launch_angle_bias_rad: ``mean(E)`` of launch angle over the strokes
            that carry one, or ``None``.
        noise_limited_count: Strokes whose ``|E| <= u_val``.
        model_error_detected_count: Strokes whose ``|E| > u_val``.
        dominant_uncertainty: ``"numerical"``, ``"measurement"``,
            ``"input"`` or ``"none"``: the largest mean standard uncertainty.
        reasons: Why the regime was rejected, or what qualified it.
    """

    regime: UseRegime
    qualified: bool
    holdout_stroke_ids: tuple[str, ...]
    holdout_session_ids: tuple[str, ...]
    independent_sand_batches: bool
    relative_bias: float | None
    relative_rms_error: float | None
    interval_coverage: float | None
    launch_angle_bias_rad: float | None
    noise_limited_count: int
    model_error_detected_count: int
    dominant_uncertainty: str
    reasons: tuple[str, ...]

    def describe(self) -> str:
        """One line fit for a verdict reason or a report row."""
        if self.relative_bias is None:
            stats = "no held-out statistics"
        else:
            stats = (
                f"relative bias {self.relative_bias:+.3g}, relative RMS "
                f"{self.relative_rms_error:.3g}, interval coverage "
                f"{self.interval_coverage:.2f}"
            )
        return (
            f"{self.regime.key}: {'qualified' if self.qualified else 'rejected'} "
            f"on {len(self.holdout_stroke_ids)} held-out stroke(s) in "
            f"{len(self.holdout_session_ids)} independent session(s); {stats}"
        )


@dataclass(frozen=True, slots=True)
class TransferQualification:
    """The versioned evidence a launch verdict may lift its floor on.

    Attributes:
        version: ``transfer-qualification/1+<12 hex>`` of the evidence digest.
        evidence_digest: SHA-256 over the dataset the fit and the held-out
            comparison were made on.
        protocol: The measurement protocol.
        intended_use: The regimes that were assessed.
        tolerances: The predeclared tolerances the regimes were judged by.
        fit: The calibration fit, converged or not.
        verdicts: One verdict per regime, qualified or rejected.
        calibration_stroke_ids: The strokes the fit used.
        holdout_stroke_ids: The strokes the verdicts used. Disjoint.
    """

    version: str
    evidence_digest: str
    protocol: MeasurementProtocol
    intended_use: IntendedUseMatrix
    tolerances: PracticalTolerances
    fit: FitOutcome
    verdicts: tuple[RegimeVerdict, ...]
    calibration_stroke_ids: tuple[str, ...]
    holdout_stroke_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the evidence.

        Raises:
            TransferQualificationError: If the version does not carry the
                digest, the subsets overlap, a regime is qualified on a
                failed fit, or a verdict names a regime outside the matrix.
        """
        if (
            self.version
            != f"{QUALIFICATION_EVIDENCE_SCHEMA}+{self.evidence_digest[:12]}"
        ):
            raise TransferQualificationError(
                f"version {self.version!r} does not name the evidence digest"
            )
        if set(self.calibration_stroke_ids) & set(self.holdout_stroke_ids):
            raise TransferQualificationError(
                "calibration and held-out stroke sets overlap; this evidence leaks"
            )
        keys = {r.key for r in self.intended_use.regimes}
        fit_status = self.fit.status
        for verdict in self.verdicts:
            if verdict.regime.key not in keys:
                raise TransferQualificationError(
                    f"verdict for {verdict.regime.key!r} names a regime outside "
                    "the intended-use matrix"
                )
            if verdict.qualified and fit_status is not FitStatus.CONVERGED:
                raise TransferQualificationError(
                    f"regime {verdict.regime.key!r} cannot be qualified on a fit "
                    f"that ended {fit_status.value!r}"
                )

    @property
    def transfer(self) -> MomentumTransfer | None:
        """The frozen parameters, or ``None`` when the fit failed."""
        return self.fit.transfer

    @property
    def qualified_regimes(self) -> tuple[UseRegime, ...]:
        """Regimes whose held-out comparison met every tolerance."""
        return tuple(v.regime for v in self.verdicts if v.qualified)

    @property
    def rejected_verdicts(self) -> tuple[RegimeVerdict, ...]:
        """Regimes that failed, with the reasons preserved."""
        return tuple(v for v in self.verdicts if not v.qualified)

    def verdict_for(self, delivery: SandDelivery, lie: BallLie) -> RegimeVerdict | None:
        """The verdict of the regime covering the strike, or ``None``."""
        regime = self.intended_use.regime_for(delivery, lie)
        if regime is None:
            return None
        for verdict in self.verdicts:
            if verdict.regime.key == regime.key:
                return verdict
        return None

    def covers(self, delivery: SandDelivery, lie: BallLie) -> bool:
        """True when the strike is inside a qualified regime."""
        verdict = self.verdict_for(delivery, lie)
        return verdict is not None and verdict.qualified

    def statement_for(
        self, delivery: SandDelivery, lie: BallLie
    ) -> tuple[EnvelopeStatus, tuple[str, ...]]:
        """The launch model's own status for one strike, and why.

        ``WITHIN`` only inside a qualified regime; ``BEYOND_VALIDATION``
        everywhere else, with the reason naming what this evidence does and
        does not cover. The solver's verdict is combined with it by the
        caller, so a carry never reads better than the shot behind it.
        """
        verdict = self.verdict_for(delivery, lie)
        if verdict is None:
            return (
                EnvelopeStatus.BEYOND_VALIDATION,
                (
                    f"{self.version} does not cover this strike: it lies outside "
                    "every regime of the intended-use matrix, so the "
                    "uncalibrated floor is preserved",
                ),
            )
        if verdict.qualified:
            return (
                EnvelopeStatus.WITHIN,
                (
                    f"ball launch partition qualified under {self.version} for "
                    f"regime {verdict.describe()}; judged against the "
                    "predeclared tolerances, on sessions independent of the "
                    "calibration subset",
                ),
            )
        return (
            EnvelopeStatus.BEYOND_VALIDATION,
            (
                f"{self.version} rejected regime {verdict.describe()}: "
                + "; ".join(verdict.reasons)
                + ". The uncalibrated floor is preserved",
            ),
        )


def objective_disposition(
    qualification: TransferQualification | None,
    delivery: SandDelivery,
    lie: BallLie,
    *,
    nominal_carry_m: float,
    target_carry_m: float,
    tolerance_fraction: float = DEFAULT_CARRY_TOLERANCE_FRACTION,
) -> ObjectiveDisposition:
    """The disposition of the carry objective for issue #9239.

    Ranking on carry is **unavailable** until a qualification covers the
    strike; once it does, a target the qualified model cannot reach at the
    nominal delivery leaves the window empty and the objective degenerate,
    and only a target inside the window is supported.

    Args:
        qualification: The evidence, or ``None`` when none exists.
        delivery: The nominal strike.
        lie: Its lie.
        nominal_carry_m: Carry the qualified model predicts at the nominal
            delivery [m].
        target_carry_m: The target the window is judged against [m].
        tolerance_fraction: Half-width of the window as a fraction of the
            target.

    Returns:
        The disposition.

    Raises:
        TransferQualificationError: If the target, carry or tolerance is
            not usable.
    """
    if not math.isfinite(target_carry_m) or target_carry_m <= 0.0:
        raise TransferQualificationError(
            f"target_carry_m must be positive, got {target_carry_m!r}"
        )
    if not math.isfinite(nominal_carry_m) or nominal_carry_m < 0.0:
        raise TransferQualificationError(
            f"nominal_carry_m must be finite and non-negative, got {nominal_carry_m!r}"
        )
    if not 0.0 < tolerance_fraction <= 1.0:
        raise TransferQualificationError(
            f"tolerance_fraction must be in (0, 1], got {tolerance_fraction!r}"
        )
    if qualification is None:
        return ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED
    if not qualification.covers(delivery, lie):
        return ObjectiveDisposition.UNAVAILABLE_OUTSIDE_QUALIFIED_REGIME
    if abs(nominal_carry_m - target_carry_m) > tolerance_fraction * target_carry_m:
        return ObjectiveDisposition.DEGENERATE_TARGET
    return ObjectiveDisposition.SUPPORTED
