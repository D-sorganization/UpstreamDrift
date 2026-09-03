"""Data-free golden conformance bundle for launch-monitor analytics consumers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.generate_launch_monitor_contract import launch_monitor_conformance_bundle
from scripts.launch_monitor_conformance_fixture import _portable_snapshot_value
from src.tools.launch_monitor_model import (
    LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION,
    LaunchMonitorConformanceBundleV1,
    launch_monitor_conformance_bundle_json_schema,
    launch_monitor_conformance_bundle_sha256,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_ROOT = REPO_ROOT / "docs" / "api" / "contracts"
SCHEMA_PATH = CONTRACT_ROOT / "launch-monitor-conformance-bundle-v1.schema.json"
GOLDEN_PATH = (
    CONTRACT_ROOT / "fixtures" / "launch-monitor-conformance-bundle-v1.golden.json"
)


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(map(_walk_keys, value.values())))
    if isinstance(value, list):
        return set().union(*(map(_walk_keys, value)))
    return set()


def test_bundle_spans_available_and_unavailable_cases_without_input_rows() -> None:
    bundle = launch_monitor_conformance_bundle()
    scenarios = {
        (item.analysis_kind, item.expected_status) for item in bundle.scenarios
    }

    assert bundle.bundle_version == LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION
    assert bundle.data_classification == "synthetic_contract_fixture_no_private_rows"
    assert bundle.input_records_embedded is False
    assert scenarios == {
        ("analysis_v2", "available"),
        ("analysis_v2", "unavailable"),
        ("player_covariation", "available"),
        ("player_covariation", "unavailable"),
        ("attested_longitudinal", "available"),
        ("attested_longitudinal", "unavailable"),
        ("source_backed_strokes_gained", "available"),
        ("source_backed_strokes_gained", "unavailable"),
        ("distance_target_proxy", "available"),
        ("distance_target_proxy", "unavailable"),
    }
    payload = bundle.model_dump(mode="json")
    assert "records" not in _walk_keys(payload)
    assert "restricted_internal" not in json.dumps(payload)


def test_scenarios_retain_units_claims_evidence_lineage_and_exclusions() -> None:
    bundle = launch_monitor_conformance_bundle()

    for scenario in bundle.scenarios:
        assert scenario.units
        assert scenario.claims["causal_inference"] is False
        assert scenario.sources
        assert scenario.backing_records
        source_ids = {source.source_id for source in scenario.sources}
        assert all(
            record.source_id in source_ids for record in scenario.backing_records
        )
        assert all(
            len(record.record_sha256) == 64 for record in scenario.backing_records
        )
        assert sum(scenario.exclusions.values()) >= 0

    longitudinal = next(
        item
        for item in bundle.scenarios
        if item.scenario_id == "attested-longitudinal-available"
    )
    assert longitudinal.player_identity.trust_level == "explicit_user_attested"
    assert longitudinal.session_identity.trust_level == "source_reported"
    assert longitudinal.session_identity.evidence
    assert longitudinal.order_evidence.trust_level == "explicit_user_attested"
    assert longitudinal.payload.claims.causal_improvement is False

    strokes_gained = next(
        item
        for item in bundle.scenarios
        if item.scenario_id == "source-backed-strokes-gained-available"
    )
    proxy = next(
        item
        for item in bundle.scenarios
        if item.scenario_id == "distance-target-proxy-available"
    )
    assert strokes_gained.claims["is_strokes_gained"] is True
    assert strokes_gained.claims["source_backed"] is True
    assert proxy.claims["is_strokes_gained"] is False
    assert proxy.claims["source_backed"] is False


def test_payload_and_bundle_hashes_fail_closed_after_mutation() -> None:
    payload = launch_monitor_conformance_bundle().model_dump(mode="json")
    payload["scenarios"][0]["description"] = "tampered scenario"

    with pytest.raises(ValidationError, match="scenario_sha256"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)

    payload = launch_monitor_conformance_bundle().model_dump(mode="json")
    payload["description"] = "tampered bundle"
    with pytest.raises(ValidationError, match="bundle_sha256"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)


def test_canonical_bundle_hash_and_published_artifacts_match_authority() -> None:
    bundle = launch_monitor_conformance_bundle()
    published = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    published_schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert published == bundle.model_dump(mode="json")
    assert published_schema == launch_monitor_conformance_bundle_json_schema()
    assert bundle.bundle_sha256 == launch_monitor_conformance_bundle_sha256(bundle)
    assert published_schema["properties"]["bundle_version"]["const"] == (
        "launch-monitor-analytics-conformance/1.0.0"
    )
    assert published_schema["additionalProperties"] is False


def test_numeric_snapshot_is_quantized_for_cross_platform_invariance() -> None:
    bundle = launch_monitor_conformance_bundle()
    longitudinal = next(
        scenario
        for scenario in bundle.scenarios
        if scenario.scenario_id == "attested-longitudinal-available"
    )
    pooled = longitudinal.payload.pooled_association

    assert pooled is not None
    assert pooled.standard_error == 0.16183472
    assert pooled.confidence_interval_low == 0.9849697
    assert pooled.confidence_interval_high == 2.0150303
    assert pooled.p_value == 0.0026577006

    windows_tail = {
        "standard_error": 0.16183471874253738,
        "confidence_interval_low": 0.984969697271183,
        "confidence_interval_high": 2.0150303027288152,
        "p_value": 0.0026577005664792496,
    }
    linux_tail = {
        "standard_error": 0.1618347187425374,
        "confidence_interval_low": 0.9849696972710931,
        "confidence_interval_high": 2.0150303027289054,
        "p_value": 0.0026577005664792513,
    }
    assert _portable_snapshot_value(windows_tail) == _portable_snapshot_value(
        linux_tail
    )
