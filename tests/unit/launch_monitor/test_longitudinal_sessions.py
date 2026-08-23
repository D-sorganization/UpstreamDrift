"""Contract tests for attested, session-unit longitudinal analysis."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.shared.python.launch_monitor import (
    AnalysisContextV2,
    LongitudinalSessionRequestV1,
    analyze_longitudinal_sessions,
    longitudinal_session_contract_json_schema,
)


pytestmark = pytest.mark.unit


FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "launch_monitor"
    / "longitudinal_attested_v1.json"
)
SOURCE_FIXTURE = FIXTURE.with_suffix(".csv")


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _analyze(payload: dict[str, Any]):
    return analyze_longitudinal_sessions(
        pd.DataFrame.from_records(payload["records"]),
        LongitudinalSessionRequestV1.model_validate(payload["request"]),
        context=AnalysisContextV2.model_validate(payload["context"]),
    )


def _projection(result: Any) -> dict[str, Any]:
    return {
        "contract_version": result.contract_version,
        "status": result.status,
        "primary_unit": result.claims.primary_unit,
        "causal_improvement": result.claims.causal_improvement,
        "association_scope": result.claims.association_scope,
        "session_cells": len(result.session_aggregates),
        "player_count": len(result.player_associations),
        "player_directions": [item.direction for item in result.player_associations],
        "pooled_method": result.pooled_association.method,
        "pooled_clusters": result.pooled_association.cluster_count,
        "pooled_estimate": round(result.pooled_association.estimate_per_order_unit, 8),
        "order_unit": result.order_evidence.unit,
        "backing_records": len(result.lineage.backing_records),
    }


def test_attested_fixture_matches_golden_session_level_contract() -> None:
    payload = _fixture()

    result = _analyze(payload)

    assert _projection(result) == payload["expected"]
    assert result.pooled_association.uncertainty_state == "available"
    assert result.pooled_association.confidence_interval_low is not None
    assert result.pooled_association.confidence_interval_high is not None
    assert len(result.lineage.dataset_fingerprint_sha256) == 64
    assert all(item.shot_count == 3 for item in result.session_aggregates)


def test_golden_source_reference_is_content_addressed_and_exact() -> None:
    payload = _fixture()
    source = payload["context"]["sources"][0]

    source_rows = list(
        csv.DictReader(SOURCE_FIXTURE.read_text(encoding="utf-8").splitlines())
    )

    assert sha256(SOURCE_FIXTURE.read_bytes()).hexdigest() == source["file_sha256"]
    assert source["source_uri"] == (
        "tests/fixtures/launch_monitor/longitudinal_attested_v1.csv"
    )
    assert source_rows == [
        {key: str(value) for key, value in row.items()} for row in payload["records"]
    ]


def test_duplicate_shots_do_not_reweight_session_level_association() -> None:
    payload = _fixture()
    baseline = _analyze(payload)
    repeated = deepcopy(payload)
    duplicated = [
        row
        for row in repeated["records"]
        if row["player_id"] == "player-1" and row["session_id"] == "p1-s1"
    ]
    repeated["records"].extend(duplicated * 4)

    result = _analyze(repeated)

    assert len(result.session_aggregates) == len(baseline.session_aggregates)
    assert result.pooled_association.estimate_per_order_unit == pytest.approx(
        baseline.pooled_association.estimate_per_order_unit
    )
    assert len(result.lineage.backing_records) == len(repeated["records"])


@pytest.mark.parametrize(
    ("context_path", "reason_code"),
    [
        (("player_identity", "trust_level"), "untrusted_player_identity"),
        (("session_identity", "trust_level"), "untrusted_session_identity"),
        (("order_evidence", "trust_level"), "untrusted_order_evidence"),
    ],
)
def test_analysis_fails_closed_without_trusted_identity_and_order_evidence(
    context_path: tuple[str, str], reason_code: str
) -> None:
    payload = _fixture()
    payload["context"][context_path[0]][context_path[1]] = "untrusted_inferred"

    result = _analyze(payload)

    assert result.status == "unavailable"
    assert result.session_aggregates == ()
    assert result.pooled_association is None
    assert result.availability[0].reason_code == reason_code
    assert len(result.lineage.backing_records) == len(payload["records"])


def test_pooled_uncertainty_is_unavailable_with_too_few_player_clusters() -> None:
    payload = _fixture()
    payload["records"] = [
        row
        for row in payload["records"]
        if row["player_id"] in {"player-1", "player-2"}
    ]
    payload["context"]["sources"][0]["session_ids"] = sorted(
        {row["session_id"] for row in payload["records"]}
    )

    result = _analyze(payload)

    assert result.status == "partial"
    assert len(result.player_associations) == 2
    assert all(item.direction == "increasing" for item in result.player_associations)
    assert result.pooled_association is None
    pooled = next(
        item for item in result.availability if item.result_path == "pooled_association"
    )
    assert pooled.reason_code == "insufficient_player_clusters"
    assert pooled.observed_count == 2
    assert pooled.required_count == 4


def test_nonconstant_order_within_session_is_structured_unavailable() -> None:
    payload = _fixture()
    payload["records"][0]["session_number"] = 99

    result = _analyze(payload)

    assert result.status == "unavailable"
    assert result.session_aggregates == ()
    assert result.availability[0].reason_code == "nonconstant_session_order"


def test_analysis_is_unavailable_when_no_complete_finite_shots_remain() -> None:
    payload = _fixture()
    for row in payload["records"]:
        row["carry_distance"] = float("inf")

    result = _analyze(payload)

    assert result.status == "unavailable"
    assert result.session_aggregates == ()
    assert result.availability[0].reason_code == "no_complete_finite_shots"
    assert result.missingness.excluded_by_reason == {
        "incomplete_or_nonfinite_selected_fields": len(payload["records"])
    }
    assert len(result.lineage.backing_records) == len(payload["records"])


def test_blank_attested_identity_values_are_excluded_not_grouped() -> None:
    payload = _fixture()
    payload["records"][0]["player_id"] = "   "

    result = _analyze(payload)

    assert result.status == "available"
    assert result.missingness.included_shot_count == len(payload["records"]) - 1
    assert result.missingness.excluded_by_reason == {
        "incomplete_or_nonfinite_selected_fields": 1
    }


def test_declared_strata_and_confounders_are_explicit_design_terms() -> None:
    payload = _fixture()
    payload["request"]["strata"] = ["club"]
    payload["request"]["confounders"] = ["temperature_c"]

    result = _analyze(payload)

    assert result.design.strata == ("club",)
    assert result.design.confounders == ("temperature_c",)
    assert result.design.session_aggregate == "mean"
    assert result.claims.confounder_control_is_causal is False
    assert all(item.stratum == {"club": "7i"} for item in result.session_aggregates)


def test_contract_schema_is_versioned_and_forbids_extra_fields() -> None:
    schema = longitudinal_session_contract_json_schema()

    assert schema["properties"]["contract_version"]["const"] == (
        "launch-monitor-longitudinal-session/1.0.0"
    )
    assert schema["additionalProperties"] is False
    assert (
        schema["$defs"]["LongitudinalSessionRequestV1"]["additionalProperties"] is False
    )


def test_published_longitudinal_schema_matches_python_authority() -> None:
    schema_path = (
        Path(__file__).parents[3]
        / "docs"
        / "api"
        / "contracts"
        / "launch-monitor-longitudinal-session-v1.schema.json"
    )

    published = json.loads(schema_path.read_text(encoding="utf-8"))

    assert published == longitudinal_session_contract_json_schema()


def test_request_rejects_overlapping_or_duplicate_design_terms() -> None:
    with pytest.raises(ValueError, match="unique and disjoint"):
        LongitudinalSessionRequestV1(
            metric="carry_distance",
            strata=("club", "club"),
            confounders=("club",),
        )
