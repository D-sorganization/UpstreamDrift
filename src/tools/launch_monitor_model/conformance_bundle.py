"""Versioned, data-free conformance bundle for launch-monitor consumers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.tools.launch_monitor_model.contract_v2 import (
    BackingRecordV2,
    LaunchMonitorAnalysisResultV2,
    MetricUnitsV2,
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
    SourceFileReferenceV2,
)
from src.tools.launch_monitor_model.longitudinal_types import (
    LongitudinalSessionResultV1,
)
from src.tools.launch_monitor_model.player_covariation_types import (
    PlayerCovariationResultV1,
)
from src.tools.launch_monitor_model.strokes_gained_types import (
    OutcomeProxyResultV1,
    StrokesGainedAnalysisResultV1,
)

LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION: Literal[
    "launch-monitor-analytics-conformance/1.0.0"
] = "launch-monitor-analytics-conformance/1.0.0"

AnalysisKind = Literal[
    "analysis_v2",
    "player_covariation",
    "attested_longitudinal",
    "source_backed_strokes_gained",
    "distance_target_proxy",
]
ExpectedStatus = Literal["available", "unavailable"]
ConformancePayload = Annotated[
    LaunchMonitorAnalysisResultV2
    | PlayerCovariationResultV1
    | LongitudinalSessionResultV1
    | StrokesGainedAnalysisResultV1
    | OutcomeProxyResultV1,
    Field(discriminator="contract_version"),
]

_PAYLOAD_TYPES: dict[str, type[BaseModel]] = {
    "analysis_v2": LaunchMonitorAnalysisResultV2,
    "player_covariation": PlayerCovariationResultV1,
    "attested_longitudinal": LongitudinalSessionResultV1,
    "source_backed_strokes_gained": StrokesGainedAnalysisResultV1,
    "distance_target_proxy": OutcomeProxyResultV1,
}
_REQUIRED_CASES = frozenset(
    (kind, status) for kind in _PAYLOAD_TYPES for status in ("available", "unavailable")
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LaunchMonitorConformanceScenarioV1(_StrictModel):
    """One synthetic output case with uniform consumer evidence."""

    scenario_id: str = Field(min_length=1)
    analysis_kind: AnalysisKind
    expected_status: ExpectedStatus
    description: str = Field(min_length=1)
    units: dict[str, MetricUnitsV2] = Field(min_length=1)
    claims: dict[str, bool | str] = Field(min_length=1)
    player_identity: PlayerIdentityV2
    session_identity: SessionIdentityV2
    order_evidence: OrderEvidenceV2
    sources: tuple[SourceFileReferenceV2, ...] = Field(min_length=1)
    backing_records: tuple[BackingRecordV2, ...] = Field(min_length=1)
    exclusions: dict[str, int]
    payload: ConformancePayload
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("exclusions")
    @classmethod
    def require_nonnegative_exclusions(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key or count < 0 for key, count in value.items()):
            raise ValueError("exclusion keys must be non-empty and counts non-negative")
        return value

    @model_validator(mode="after")
    def validate_case(self) -> LaunchMonitorConformanceScenarioV1:
        expected_type = _PAYLOAD_TYPES[self.analysis_kind]
        if not isinstance(self.payload, expected_type):
            raise ValueError("analysis_kind does not match the result payload contract")
        if self.payload.status != self.expected_status:
            raise ValueError("expected_status does not match the result payload status")
        source_ids = {source.source_id for source in self.sources}
        if any(record.source_id not in source_ids for record in self.backing_records):
            raise ValueError("every backing record must join to a declared source_id")
        if self.claims.get("causal_inference") is not False:
            raise ValueError("conformance scenarios must forbid causal inference")
        if self.scenario_sha256 != launch_monitor_conformance_scenario_sha256(self):
            raise ValueError(
                "scenario_sha256 does not match canonical scenario content"
            )
        return self


class LaunchMonitorConformanceBundleV1(_StrictModel):
    """Complete consumer bundle with canonical content-address verification."""

    bundle_version: Literal["launch-monitor-analytics-conformance/1.0.0"] = (
        LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION
    )
    description: str = Field(min_length=1)
    data_classification: Literal["synthetic_contract_fixture_no_private_rows"] = (
        "synthetic_contract_fixture_no_private_rows"
    )
    input_records_embedded: Literal[False] = False
    scenarios: tuple[LaunchMonitorConformanceScenarioV1, ...] = Field(min_length=10)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_bundle(self) -> LaunchMonitorConformanceBundleV1:
        identities = [scenario.scenario_id for scenario in self.scenarios]
        if len(identities) != len(set(identities)):
            raise ValueError("scenario_id values must be unique")
        cases = {
            (scenario.analysis_kind, scenario.expected_status)
            for scenario in self.scenarios
        }
        if cases != _REQUIRED_CASES or len(self.scenarios) != len(_REQUIRED_CASES):
            raise ValueError(
                "bundle must contain exactly one required conformance case"
            )
        if self.bundle_sha256 != launch_monitor_conformance_bundle_sha256(self):
            raise ValueError("bundle_sha256 does not match canonical bundle content")
        return self


def launch_monitor_conformance_scenario_sha256(
    scenario: LaunchMonitorConformanceScenarioV1 | dict[str, Any],
) -> str:
    """Hash a scenario while excluding its self-referential hash field."""

    payload = (
        scenario.model_dump(mode="json")
        if isinstance(scenario, LaunchMonitorConformanceScenarioV1)
        else dict(scenario)
    )
    payload.pop("scenario_sha256", None)
    return _canonical_sha256(payload)


def launch_monitor_conformance_bundle_sha256(
    bundle: LaunchMonitorConformanceBundleV1 | dict[str, Any],
) -> str:
    """Hash a bundle while excluding its self-referential hash field."""

    payload = (
        bundle.model_dump(mode="json")
        if isinstance(bundle, LaunchMonitorConformanceBundleV1)
        else dict(bundle)
    )
    payload.pop("bundle_sha256", None)
    return _canonical_sha256(payload)


def launch_monitor_conformance_bundle_json_schema() -> dict[str, Any]:
    """Return the strict OpenAPI-compatible conformance bundle schema."""

    return LaunchMonitorConformanceBundleV1.model_json_schema()


__all__ = [
    "LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION",
    "LaunchMonitorConformanceBundleV1",
    "LaunchMonitorConformanceScenarioV1",
    "launch_monitor_conformance_bundle_json_schema",
    "launch_monitor_conformance_bundle_sha256",
    "launch_monitor_conformance_scenario_sha256",
]
