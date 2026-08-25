"""Canonical source-backed strokes-gained and outcome-proxy contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.shared.python.launch_monitor import (
    CourseStateColumnsV1,
    ExpectedStrokesBaselineV2,
    ExpectedStrokesStateV2,
    GroupingDimensionV1,
    LongitudinalDimensionV1,
    OutcomeProxyRequestV1,
    StrokesGainedRequestV1,
    analyze_outcome_proxy,
    analyze_source_backed_strokes_gained,
    baseline_table_sha256,
    strokes_gained_contract_json_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _states() -> tuple[ExpectedStrokesStateV2, ...]:
    return (
        ExpectedStrokesStateV2(
            lie="fairway",
            context="standard",
            target="hole-1",
            distance_yards=100.0,
            expected_strokes=2.8,
            standard_error=0.10,
        ),
        ExpectedStrokesStateV2(
            lie="fairway",
            context="standard",
            target="hole-1",
            distance_yards=200.0,
            expected_strokes=3.8,
            standard_error=0.14,
        ),
        ExpectedStrokesStateV2(
            lie="green",
            context="standard",
            target="hole-1",
            distance_yards=0.0,
            expected_strokes=0.0,
            standard_error=0.0,
        ),
        ExpectedStrokesStateV2(
            lie="green",
            context="standard",
            target="hole-1",
            distance_yards=20.0,
            expected_strokes=1.5,
            standard_error=0.08,
        ),
    )


def _baseline() -> ExpectedStrokesBaselineV2:
    states = _states()
    return ExpectedStrokesBaselineV2(
        baseline_id="licensed-test-baseline",
        version="2026.1",
        source_url="https://example.org/expected-strokes-method",
        license="test-only",
        table_sha256=baseline_table_sha256(states),
        states=states,
    )


def _request(*, min_samples: int = 3) -> StrokesGainedRequestV1:
    return StrokesGainedRequestV1(
        start=CourseStateColumnsV1(
            lie_column="start_lie",
            context_column="start_context",
            target_column="target",
            distance_column="start_distance",
            distance_unit="yd",
        ),
        finish=CourseStateColumnsV1(
            lie_column="finish_lie",
            context_column="finish_context",
            target_column="target",
            distance_column="finish_distance_m",
            distance_unit="m",
        ),
        shot_id_column="shot_id",
        confidence_level=0.95,
        min_samples=min_samples,
        summaries=(
            GroupingDimensionV1(
                dimension="player",
                column="player_id",
                trust_level="pseudonymous_stable",
                evidence="Stable study pseudonym supplied by the owner.",
            ),
            GroupingDimensionV1(
                dimension="session",
                column="session_id",
                trust_level="explicit_user_attested",
                evidence="The user attested the exported session identifier.",
            ),
            GroupingDimensionV1(
                dimension="club",
                column="club",
                trust_level="verified_external",
                evidence="Club identity was verified against capture metadata.",
            ),
        ),
        longitudinal=LongitudinalDimensionV1(
            order_column="session_order",
            order_unit="session",
            group_column="player_id",
            group_dimension="player",
            trust_level="pseudonymous_stable",
            evidence="Stable player pseudonym and chronological session order.",
            min_samples=3,
        ),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(6)],
            "player_id": ["player-a"] * 3 + ["player-b"] * 3,
            "session_id": ["s1", "s2", "s3", "s1", "s2", "s3"],
            "session_order": [1, 2, 3, 1, 2, 3],
            "club": ["7i"] * 6,
            "start_lie": ["fairway"] * 6,
            "start_context": ["standard"] * 6,
            "target": ["hole-1"] * 6,
            "start_distance": [150.0, 160.0, 170.0, 150.0, 160.0, 170.0],
            "finish_lie": ["green"] * 6,
            "finish_context": ["standard"] * 6,
            "finish_distance_m": [18.288, 13.716, 9.144, 18.288, 13.716, 9.144],
        }
    )


def test_baseline_hash_is_numeric_and_row_order_canonical() -> None:
    states = _states()
    dictionaries = [state.model_dump(mode="json") for state in reversed(states)]

    assert baseline_table_sha256(states) == baseline_table_sha256(dictionaries)


def test_baseline_rejects_tamper_and_duplicate_course_state() -> None:
    states = _states()
    with pytest.raises(ValidationError, match="table_sha256"):
        ExpectedStrokesBaselineV2(
            baseline_id="tampered",
            version="1",
            source_url="https://example.org/method",
            license="test-only",
            table_sha256="0" * 64,
            states=states,
        )
    with pytest.raises(ValidationError, match="duplicate"):
        ExpectedStrokesBaselineV2(
            baseline_id="duplicate",
            version="1",
            source_url="https://example.org/method",
            license="test-only",
            table_sha256=baseline_table_sha256((*states, states[0])),
            states=(*states, states[0]),
        )


def test_source_backed_sg_reports_traceable_rows_uncertainty_and_summaries() -> None:
    result = analyze_source_backed_strokes_gained(
        _frame(),
        _baseline(),
        _request(),
    )

    assert result.status == "available"
    assert result.metric_name == "source_backed_strokes_gained"
    assert result.unit == "strokes"
    assert result.value_summary.count == 6
    assert result.value_summary.mean == pytest.approx(1.275, abs=1e-4)
    assert result.value_summary.confidence_interval is not None
    assert result.uncertainty.benchmark_method == "interpolated-state-standard-errors"
    assert result.baseline.table_sha256 == _baseline().table_sha256
    assert result.row_results[0].expected_start == pytest.approx(3.3)
    assert result.row_results[0].expected_finish == pytest.approx(1.5)
    assert result.row_results[0].strokes_gained == pytest.approx(0.8)
    assert result.row_results[0].start.target == "hole-1"
    assert result.row_results[0].input_record_sha256 is not None
    assert result.exclusions.total_excluded == 0
    assert {summary.dimension for summary in result.group_summaries} == {
        "player",
        "session",
        "club",
    }
    assert len(result.longitudinal_summaries) == 2
    assert result.longitudinal_summaries[0].slope_unit == "strokes/session"
    assert result.claims.is_strokes_gained is True
    assert result.claims.source_backed is True
    assert result.claims.causal_inference is False
    assert "target/hole" in " ".join(result.limitations).lower()


def test_source_backed_sg_excludes_bad_rows_and_fails_minimum_closed() -> None:
    frame = _frame().iloc[:3].copy()
    frame.loc[0, "start_context"] = ""
    frame.loc[1, "start_distance"] = 250.0

    partial = analyze_source_backed_strokes_gained(
        frame,
        _baseline(),
        _request(min_samples=1),
    )
    assert partial.status == "partial"
    assert partial.value_summary.count == 1
    assert partial.exclusions.by_reason == {
        "missing_course_state": 1,
        "outside_baseline": 1,
    }
    assert len(partial.excluded_rows) == 2

    unavailable = analyze_source_backed_strokes_gained(
        frame,
        _baseline(),
        _request(min_samples=2),
    )
    assert unavailable.status == "unavailable"
    assert unavailable.availability.reason_code == "insufficient_complete_rows"
    assert unavailable.availability.observed_count == 1
    assert unavailable.availability.required_count == 2
    assert unavailable.value_summary.count == 1
    assert unavailable.value_summary.mean is None
    assert unavailable.group_summaries == ()
    assert unavailable.longitudinal_summaries == ()


def test_grouped_and_longitudinal_summaries_require_explicit_evidence() -> None:
    with pytest.raises(ValidationError, match="trust_level"):
        GroupingDimensionV1(
            dimension="player",
            column="player_id",
            trust_level="untrusted_inferred",  # type: ignore[arg-type]
            evidence="Guessed from row order.",
        )
    with pytest.raises(ValidationError, match="evidence"):
        LongitudinalDimensionV1(
            order_column="session_order",
            order_unit="session",
            trust_level="explicit_user_attested",
            evidence="",
        )


def test_launch_monitor_proxy_is_never_labeled_strokes_gained() -> None:
    result = analyze_outcome_proxy(
        pd.DataFrame(
            {
                "carry_m": [137.16, 140.0],
                "lateral_m": [-9.144, 4.572],
            }
        ),
        OutcomeProxyRequestV1(
            carry_column="carry_m",
            lateral_column="lateral_m",
            carry_unit="m",
            lateral_unit="m",
            target_distance_yards=150.0,
        ),
    )

    assert result.metric_name == "expected_proximity_dispersion_proxy"
    assert result.unit == "yd"
    assert result.claims.is_strokes_gained is False
    assert result.claims.source_backed is False
    assert result.row_results[0].lateral_yards == pytest.approx(-10.0)
    assert "not strokes gained" in " ".join(result.limitations).lower()


def test_strokes_gained_schema_is_versioned() -> None:
    schema = strokes_gained_contract_json_schema()
    assert schema["title"] == "StrokesGainedAnalysisResultV1"
    assert schema["properties"]["contract_version"]["const"] == (
        "launch-monitor-strokes-gained-analysis/1.0.0"
    )


def test_published_strokes_gained_schema_matches_python_authority() -> None:
    path = (
        Path(__file__).parents[3]
        / "docs"
        / "api"
        / "contracts"
        / "launch-monitor-strokes-gained-v1.schema.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == (
        strokes_gained_contract_json_schema()
    )
