"""AddBiomechanics calibration import for bounded inertia priors.

AddBiomechanics force-plate calibration sessions can identify subject-specific
segment masses and inertias that markerless-only fitting cannot recover. This
module keeps that path in the canonical anthropometrics layer: imported values
are validated as :class:`SegmentProperties`, then exposed as deterministic
``theta_prior`` parameter specs for the estimator's bounded inertia block.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._subject_anthropometrics import SubjectAnthropometrics
from .segment_properties import SegmentProperties

PRIOR_SCHEMA_VERSION = 1
_INERTIA_COMPONENTS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("ixx", (0, 0)),
    ("iyy", (1, 1)),
    ("izz", (2, 2)),
    ("ixy", (0, 1)),
    ("ixz", (0, 2)),
    ("iyz", (1, 2)),
)
_SourceFormat = Literal["addbiomechanics", "subject", "prior_set"]


@dataclass(frozen=True)
class InertiaPriorParameter:
    """One bounded scalar prior derived from a calibration segment."""

    name: str
    segment_name: str
    component: str
    prior: float
    lower: float
    upper: float
    prior_scale: float
    source_session_id: str
    kind: str = "inertia"

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_empty(self.segment_name, "segment_name")
        _require_non_empty(self.component, "component")
        _require_non_empty(self.source_session_id, "source_session_id")
        for field_name in ("prior", "lower", "upper", "prior_scale"):
            _require_finite(float(getattr(self, field_name)), field_name)
        if self.lower >= self.upper:
            raise ValueError(f"{self.name}: lower must be < upper")
        if not (self.lower <= self.prior <= self.upper):
            raise ValueError(f"{self.name}: prior must lie inside bounds")
        if self.prior_scale <= 0.0:
            raise ValueError(f"{self.name}: prior_scale must be positive")

    def to_estimator_spec_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload matching ``SharedParameterSpec`` fields."""
        return {
            "name": self.name,
            "initial": self.prior,
            "kind": self.kind,
            "lower": self.lower,
            "upper": self.upper,
            "prior": self.prior,
            "prior_scale": self.prior_scale,
            "locked": False,
        }


@dataclass(frozen=True)
class CalibrationInertiaPriorSet:
    """Validated inertia priors from one AddBiomechanics calibration session."""

    subject_id: str
    source_session_id: str
    parameters: tuple[InertiaPriorParameter, ...]
    source_method: str = "addbiomechanics_force_plate"

    def __post_init__(self) -> None:
        _require_non_empty(self.subject_id, "subject_id")
        _require_non_empty(self.source_session_id, "source_session_id")
        _require_non_empty(self.source_method, "source_method")
        if not self.parameters:
            raise ValueError("parameters must be non-empty")
        names = [param.name for param in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe prior-set payload."""
        return {
            "schema_version": PRIOR_SCHEMA_VERSION,
            "subject_id": self.subject_id,
            "source_session_id": self.source_session_id,
            "source_method": self.source_method,
            "parameters": [asdict(param) for param in self.parameters],
        }

    def to_estimator_parameter_block_payload(self) -> dict[str, Any]:
        """Return the estimator parameter-block payload for ``theta_prior``."""
        return {
            "parameters": [
                param.to_estimator_spec_payload() for param in self.parameters
            ]
        }


def build_inertia_priors_from_subject(
    subject: SubjectAnthropometrics,
    *,
    source_session_id: str,
    correction_fraction: float = 0.10,
    prior_scale_fraction: float = 0.02,
) -> CalibrationInertiaPriorSet:
    """Build bounded inertia priors from calibrated segment properties.

    Args:
        subject: Validated AddBiomechanics-scaled anthropometrics.
        source_session_id: Stable calibration-session identifier.
        correction_fraction: Symmetric relative bound around every prior.
        prior_scale_fraction: Gaussian prior scale as a relative fraction.

    Returns:
        A deterministic prior set ordered by subject segment order and inertia
        component order.
    """
    if not isinstance(subject, SubjectAnthropometrics):
        raise TypeError(
            f"subject must be a SubjectAnthropometrics, got {type(subject).__name__}"
        )
    _require_non_empty(source_session_id, "source_session_id")
    _require_fraction(correction_fraction, "correction_fraction")
    _require_fraction(prior_scale_fraction, "prior_scale_fraction")

    parameters: list[InertiaPriorParameter] = []
    for segment_name, segment in subject.segments:
        tensor = np.asarray(segment.inertia_tensor, dtype=float)
        for component, index in _INERTIA_COMPONENTS:
            prior = float(tensor[index])
            parameters.append(
                _make_inertia_parameter(
                    segment_name=segment_name,
                    component=component,
                    prior=prior,
                    source_session_id=source_session_id,
                    correction_fraction=correction_fraction,
                    prior_scale_fraction=prior_scale_fraction,
                )
            )

    return CalibrationInertiaPriorSet(
        subject_id=subject.subject_id,
        source_session_id=source_session_id,
        source_method=subject.source_method,
        parameters=tuple(parameters),
    )


def load_addbiomechanics_inertia_priors(
    path: Path | str,
    *,
    correction_fraction: float = 0.10,
    prior_scale_fraction: float = 0.02,
) -> CalibrationInertiaPriorSet:
    """Load AddBiomechanics calibration output or a saved prior-set JSON file.

    The importer accepts three bounded formats:

    * this module's saved prior-set schema;
    * canonical ``SubjectAnthropometrics`` JSON from
      :mod:`anthropometrics.persistence`;
    * an AddBiomechanics export with subject metadata plus scaled segment
      masses, CoMs, and 3x3 inertia tensors.
    """
    payload = _read_json_object(path)
    source_format = _classify_payload(payload)
    if source_format == "prior_set":
        return _prior_set_from_dict(payload)
    if source_format == "subject":
        from .persistence import load_subject

        subject = load_subject(Path(path))
        session_id = _session_id_from_payload(payload, fallback=subject.subject_id)
        return build_inertia_priors_from_subject(
            subject,
            source_session_id=session_id,
            correction_fraction=correction_fraction,
            prior_scale_fraction=prior_scale_fraction,
        )
    subject = _subject_from_addbiomechanics_payload(payload)
    session_id = _session_id_from_payload(payload, fallback=subject.subject_id)
    return build_inertia_priors_from_subject(
        subject,
        source_session_id=session_id,
        correction_fraction=correction_fraction,
        prior_scale_fraction=prior_scale_fraction,
    )


def save_inertia_priors(priors: CalibrationInertiaPriorSet, path: Path | str) -> None:
    """Serialize a validated prior set to deterministic JSON."""
    if not isinstance(priors, CalibrationInertiaPriorSet):
        raise TypeError(
            f"priors must be a CalibrationInertiaPriorSet, got {type(priors).__name__}"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(priors.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _make_inertia_parameter(
    *,
    segment_name: str,
    component: str,
    prior: float,
    source_session_id: str,
    correction_fraction: float,
    prior_scale_fraction: float,
) -> InertiaPriorParameter:
    base = max(abs(prior), 1.0e-6)
    half_width = correction_fraction * base
    scale = max(prior_scale_fraction * base, 1.0e-9)
    name = f"theta_prior.{segment_name}.inertia.{component}"
    return InertiaPriorParameter(
        name=name,
        segment_name=segment_name,
        component=component,
        prior=prior,
        lower=prior - half_width,
        upper=prior + half_width,
        prior_scale=scale,
        source_session_id=source_session_id,
    )


def _subject_from_addbiomechanics_payload(
    payload: Mapping[str, Any],
) -> SubjectAnthropometrics:
    subject_payload = payload.get("subject")
    subject_data = subject_payload if isinstance(subject_payload, Mapping) else payload
    subject_id = str(subject_data.get("subject_id", payload.get("subject_id", "")))
    height_m = float(subject_data["height_m"])
    mass_kg = float(subject_data["mass_kg"])
    segments = _segments_from_payload(payload, height_m=height_m, mass_kg=mass_kg)
    return SubjectAnthropometrics(
        subject_id=subject_id,
        height_m=height_m,
        mass_kg=mass_kg,
        segments=segments,
        source_method="addbiomechanics_force_plate",
        age_years=_optional_float(subject_data.get("age_years")),
        sex=str(subject_data.get("sex", "unspecified")),
    )


def _segments_from_payload(
    payload: Mapping[str, Any],
    *,
    height_m: float,
    mass_kg: float,
) -> tuple[tuple[str, SegmentProperties], ...]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, str):
        raise ValueError("AddBiomechanics payload must contain a segments array")
    segments: list[tuple[str, SegmentProperties]] = []
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            raise ValueError("each AddBiomechanics segment must be an object")
        name = str(raw["name"])
        tensor = np.asarray(raw["inertia_tensor"], dtype=float)
        segment = SegmentProperties(
            name=name,
            body_part_id=str(raw.get("body_part_id", name)),
            length_m=float(raw.get("length_m", 1.0)),
            proximal_marker=_optional_str(raw.get("proximal_marker")),
            distal_marker=_optional_str(raw.get("distal_marker")),
            mass_kg=float(raw["mass_kg"]),
            com_xyz_m=np.asarray(raw.get("com_xyz_m", [0.0, 0.0, 0.0]), dtype=float),
            inertia_tensor=tensor,
            source_method="addbiomechanics_force_plate",
            source_subject_height_m=height_m,
            source_subject_mass_kg=mass_kg,
        )
        segments.append((name, segment))
    return tuple(segments)


def _prior_set_from_dict(payload: Mapping[str, Any]) -> CalibrationInertiaPriorSet:
    version = payload.get("schema_version")
    if version != PRIOR_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported inertia-prior schema_version {version!r}; "
            f"expected {PRIOR_SCHEMA_VERSION}"
        )
    raw_parameters = payload.get("parameters")
    if not isinstance(raw_parameters, Sequence):
        raise ValueError("prior set payload must contain a parameters array")
    return CalibrationInertiaPriorSet(
        subject_id=str(payload["subject_id"]),
        source_session_id=str(payload["source_session_id"]),
        source_method=str(payload.get("source_method", "addbiomechanics_force_plate")),
        parameters=tuple(
            InertiaPriorParameter(
                name=str(item["name"]),
                segment_name=str(item["segment_name"]),
                component=str(item["component"]),
                prior=float(item["prior"]),
                lower=float(item["lower"]),
                upper=float(item["upper"]),
                prior_scale=float(item["prior_scale"]),
                source_session_id=str(item["source_session_id"]),
                kind=str(item.get("kind", "inertia")),
            )
            for item in raw_parameters
        ),
    )


def _read_json_object(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} is not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: top-level JSON value must be an object")
    return payload


def _classify_payload(payload: Mapping[str, Any]) -> _SourceFormat:
    if (
        payload.get("schema_version") == PRIOR_SCHEMA_VERSION
        and "parameters" in payload
    ):
        return "prior_set"
    if "source_method" in payload and "segments" in payload and "height_m" in payload:
        return "subject"
    return "addbiomechanics"


def _session_id_from_payload(payload: Mapping[str, Any], *, fallback: str) -> str:
    for key in ("source_session_id", "session_id", "trial_id"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw
    return fallback


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _require_non_empty(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _require_finite(value: float, label: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _require_fraction(value: float, label: str) -> None:
    _require_finite(float(value), label)
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{label} must be in the range (0, 1]")


__all__ = [
    "CalibrationInertiaPriorSet",
    "InertiaPriorParameter",
    "PRIOR_SCHEMA_VERSION",
    "build_inertia_priors_from_subject",
    "load_addbiomechanics_inertia_priors",
    "save_inertia_priors",
]
