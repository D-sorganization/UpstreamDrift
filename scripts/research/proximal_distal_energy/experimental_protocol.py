"""Fail-closed contracts for proximal-distal experimental falsification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


_IDENTITY_FIELDS = frozenset(
    {"name", "email", "birth_date", "address", "phone", "medical_record_id"}
)
_DATASET_STATUSES = frozenset({"synthetic_dry_run", "governed_human_data"})


def _text(name: str, value: object) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclass(frozen=True, slots=True)
class ModalityRequirement:
    """One preregistered measurement stream."""

    name: str
    units: tuple[str, ...]
    minimum_rate_hz: float
    maximum_sync_uncertainty_ms: float
    required_for: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text("modality name", self.name))
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "required_for", tuple(self.required_for))
        if not self.units or not self.required_for:
            raise ValueError("modality units and required_for must be non-empty")
        if self.minimum_rate_hz <= 0 or self.maximum_sync_uncertainty_ms < 0:
            raise ValueError(
                "modality rate must be positive and uncertainty non-negative"
            )


@dataclass(frozen=True, slots=True)
class ExperimentalPrediction:
    """One fixed human-data estimand and decision rule."""

    prediction_id: str
    hypothesis_id: str
    estimand: str
    expected_sign_or_interval: str
    falsifier: str
    analysis_level: str
    required_modalities: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "prediction_id",
            "hypothesis_id",
            "estimand",
            "expected_sign_or_interval",
            "falsifier",
            "analysis_level",
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "required_modalities", tuple(self.required_modalities))
        if not self.required_modalities:
            raise ValueError("prediction required_modalities must be non-empty")


@dataclass(frozen=True, slots=True)
class ProtocolManifest:
    """Frozen acquisition, processing, split, and inference contract."""

    protocol_id: str
    schema_version: str
    registration_status: str
    frozen_at_utc: str
    inclusion_criteria: tuple[str, ...]
    exclusion_criteria: tuple[str, ...]
    modalities: tuple[ModalityRequirement, ...]
    predictions: tuple[ExperimentalPrediction, ...]
    held_out_fraction: float
    split_unit: str
    identity_policy: str
    filtering_contract: dict[str, Any]
    calibration_contract: dict[str, Any]
    residual_thresholds: dict[str, float]
    missing_data_contract: dict[str, Any]
    inference_boundary: str

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "schema_version",
            "registration_status",
            "frozen_at_utc",
            "split_unit",
            "identity_policy",
            "inference_boundary",
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        if self.registration_status != "frozen_before_human_outcomes":
            raise ValueError("protocol must be frozen before human outcomes")
        if self.split_unit != "participant":
            raise ValueError("held-out split unit must be participant")
        if not 0.2 <= self.held_out_fraction < 0.5:
            raise ValueError("held_out_fraction must be in [0.2, 0.5)")
        names = {item.name for item in self.modalities}
        if len(names) != len(self.modalities):
            raise ValueError("modality names must be unique")
        for prediction in self.predictions:
            missing = set(prediction.required_modalities) - names
            if missing:
                raise ValueError(
                    f"prediction references unknown modalities: {sorted(missing)}"
                )

    @classmethod
    def from_json(cls, path: str | Path) -> ProtocolManifest:
        """Load and validate a protocol manifest."""
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        record["inclusion_criteria"] = tuple(record["inclusion_criteria"])
        record["exclusion_criteria"] = tuple(record["exclusion_criteria"])
        modalities = []
        for item in record["modalities"]:
            values = dict(item)
            values["units"] = tuple(values["units"])
            values["required_for"] = tuple(values["required_for"])
            modalities.append(ModalityRequirement(**values))
        record["modalities"] = tuple(modalities)
        predictions = []
        for item in record["predictions"]:
            values = dict(item)
            values["required_modalities"] = tuple(values["required_modalities"])
            predictions.append(ExperimentalPrediction(**values))
        record["predictions"] = tuple(predictions)
        return cls(**record)


@dataclass(frozen=True, slots=True)
class DatasetReadiness:
    """Fail-closed assessment of whether claims may be evaluated."""

    status: str
    pipeline_ready: bool
    claims_evaluable: bool
    participant_count: int
    held_out_count: int
    limitations: tuple[str, ...]


def evaluate_dataset_readiness(
    protocol: ProtocolManifest, record: dict[str, Any]
) -> DatasetReadiness:
    """Validate provenance and determine whether human claims may be tested."""
    status = _text("dataset_status", record.get("dataset_status"))
    if status not in _DATASET_STATUSES:
        raise ValueError(f"dataset_status must be one of {sorted(_DATASET_STATUSES)}")

    participants = record.get("participants")
    if not isinstance(participants, list) or len(participants) < 2:
        raise ValueError("participants must contain at least two records")
    pseudonyms: list[str] = []
    splits: list[str] = []
    for participant in participants:
        forbidden = _IDENTITY_FIELDS.intersection(participant)
        if forbidden:
            raise ValueError(
                f"identity-bearing field is prohibited: {sorted(forbidden)}"
            )
        if participant.get("pseudonym_source") != "governed_random_assignment":
            raise ValueError("pseudonyms must use governed_random_assignment")
        pseudonyms.append(_text("pseudonym", participant.get("pseudonym")))
        split = _text("split", participant.get("split"))
        if split not in {"training", "held_out"}:
            raise ValueError("split must be training or held_out")
        splits.append(split)
    if len(pseudonyms) != len(set(pseudonyms)):
        raise ValueError("participant pseudonyms must be unique")
    held_out_count = splits.count("held_out")
    if not held_out_count or not splits.count("training"):
        raise ValueError("both training and held_out participants are required")

    streams = record.get("modalities")
    if not isinstance(streams, dict):
        raise ValueError("modalities must be an object")
    for requirement in protocol.modalities:
        stream = streams.get(requirement.name)
        if not isinstance(stream, dict):
            raise ValueError(f"required modality is missing: {requirement.name}")
        if tuple(stream.get("units", ())) != requirement.units:
            raise ValueError(f"{requirement.name} units do not match protocol")
        digest = _text("sha256", stream.get("sha256"))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{requirement.name} sha256 must be lowercase hex")
        if float(stream.get("sample_rate_hz", 0)) < requirement.minimum_rate_hz:
            raise ValueError(f"{requirement.name} sample rate is below protocol")
        if (
            float(stream.get("sync_uncertainty_ms", float("inf")))
            > requirement.maximum_sync_uncertainty_ms
        ):
            raise ValueError(
                f"{requirement.name} synchronization uncertainty exceeds protocol"
            )

    if status == "governed_human_data":
        for key in (
            "ethics_approval_reference",
            "consent_basis",
            "data_authority_path",
        ):
            _text(key, record.get(key))
        authority = str(record["data_authority_path"]).replace("\\", "/").lower()
        if "/upstreamdrift/" in authority or authority.endswith("/upstreamdrift"):
            raise ValueError(
                "governed human data authority must remain outside the public repository"
            )
        return DatasetReadiness(
            status="governed_human_data_ready",
            pipeline_ready=True,
            claims_evaluable=True,
            participant_count=len(participants),
            held_out_count=held_out_count,
            limitations=(protocol.inference_boundary,),
        )

    return DatasetReadiness(
        status="synthetic_dry_run_only",
        pipeline_ready=True,
        claims_evaluable=False,
        participant_count=len(participants),
        held_out_count=held_out_count,
        limitations=(
            "Synthetic records qualify schema and processing readiness only.",
            "No human, physiological, skill, or causal claim may be evaluated.",
        ),
    )
