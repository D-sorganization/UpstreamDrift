"""Build the synthetic, aggregate-only launch-monitor conformance bundle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from src.shared.python.launch_monitor import (
    AnalysisContextV2,
    BackingRecordV2,
    CourseStateColumnsV1,
    ExpectedStrokesBaselineV2,
    ExpectedStrokesStateV2,
    FlexibleAnalysisRequest,
    LaunchMonitorConformanceBundleV1,
    LaunchMonitorConformanceScenarioV1,
    LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION,
    LongitudinalSessionRequestV1,
    MetricUnitsV2,
    OutcomeProxyRequestV1,
    PlayerCovariationRequestV1,
    PlayerIdentityV2,
    SourceFileReferenceV2,
    StrokesGainedRequestV1,
    analyze_longitudinal_sessions,
    analyze_outcome_proxy,
    analyze_player_covariation_v1,
    analyze_source_backed_strokes_gained,
    analyze_variables_v2,
    baseline_table_sha256,
    build_analysis_lineage_v2,
    launch_monitor_conformance_bundle_sha256,
    launch_monitor_conformance_scenario_sha256,
)

_SOURCE_ID = "synthetic-conformance-source"


def _source(frame: pd.DataFrame) -> SourceFileReferenceV2:
    records = frame.to_dict(orient="records")
    content = json.dumps(
        records,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return SourceFileReferenceV2(
        source_id=_SOURCE_ID,
        file_sha256=sha256(content).hexdigest(),
        rights_status="public_redistributable",
    )


def _context(
    frame: pd.DataFrame, *, player_identity: PlayerIdentityV2 | None = None
) -> AnalysisContextV2:
    return AnalysisContextV2(
        sources=(_source(frame),),
        player_identity=player_identity or PlayerIdentityV2(),
        source_units={
            "ball_speed": "mph",
            "club_speed": "mph",
            "face_angle": "deg",
            "club_path": "deg",
            "carry_distance": "yd",
        },
    )


def _analysis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"analysis-{index}" for index in range(6)],
            "source_id": [_SOURCE_ID] * 6,
            "source_row": list(range(6)),
            "monitor_vendor": ["TrackMan"] * 6,
            "monitor_model": ["synthetic-comparable"] * 6,
            "software_version": ["fixture-1"] * 6,
            "club_speed": [90, 91, 92, 93, 94, 95],
            "ball_speed": [130, 132, 133, 135, 136, 138],
        }
    )


def _covariation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"covariation-{index}" for index in range(8)],
            "source_id": [_SOURCE_ID] * 8,
            "source_row": list(range(8)),
            "player_id": ["player-a"] * 4 + ["player-b"] * 4,
            "face_angle": [0, 1, 2, 3, 10, 11, 12, 13],
            "club_path": [3, 2, 1, 0, 13, 12, 11, 10],
            "monitor_vendor": ["Foresight"] * 8,
            "monitor_model": ["synthetic-comparable"] * 8,
            "software_version": ["fixture-1"] * 8,
        }
    )


def _covariation_context(frame: pd.DataFrame) -> AnalysisContextV2:
    return _context(
        frame,
        player_identity=PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="player_id",
            evidence="Synthetic fixture identities are attested by construction.",
        ),
    )


def _longitudinal_inputs(root: Path) -> tuple[pd.DataFrame, Any, AnalysisContextV2]:
    path = (
        root / "tests" / "fixtures" / "launch_monitor" / "longitudinal_attested_v1.json"
    )
    fixture = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame.from_records(fixture["records"])
    frame["source_id"] = _SOURCE_ID
    context_payload = fixture["context"]
    context_payload["sources"] = [_source(frame).model_dump(mode="json")]
    return (
        frame,
        LongitudinalSessionRequestV1.model_validate(fixture["request"]),
        AnalysisContextV2.model_validate(context_payload),
    )


def _baseline() -> ExpectedStrokesBaselineV2:
    states = (
        ExpectedStrokesStateV2(
            lie="fairway",
            context="standard",
            target="hole-1",
            distance_yards=100,
            expected_strokes=2.8,
            standard_error=0.1,
        ),
        ExpectedStrokesStateV2(
            lie="fairway",
            context="standard",
            target="hole-1",
            distance_yards=200,
            expected_strokes=3.8,
            standard_error=0.14,
        ),
        ExpectedStrokesStateV2(
            lie="green",
            context="standard",
            target="hole-1",
            distance_yards=0,
            expected_strokes=0,
            standard_error=0,
        ),
        ExpectedStrokesStateV2(
            lie="green",
            context="standard",
            target="hole-1",
            distance_yards=20,
            expected_strokes=1.5,
            standard_error=0.08,
        ),
    )
    return ExpectedStrokesBaselineV2(
        baseline_id="synthetic-published-method",
        version="fixture-1",
        source_url="https://example.org/synthetic-expected-strokes-method",
        license="CC0-1.0 synthetic fixture",
        table_sha256=baseline_table_sha256(states),
        states=states,
    )


def _strokes_gained_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"sg-{index}" for index in range(3)],
            "source_id": [_SOURCE_ID] * 3,
            "start_lie": ["fairway"] * 3,
            "start_context": ["standard"] * 3,
            "finish_lie": ["green"] * 3,
            "finish_context": ["standard"] * 3,
            "target": ["hole-1"] * 3,
            "start_distance": [140, 150, 160],
            "finish_distance": [20, 15, 10],
        }
    )


def _strokes_gained_request(min_samples: int) -> StrokesGainedRequestV1:
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
            distance_column="finish_distance",
            distance_unit="yd",
        ),
        shot_id_column="shot_id",
        min_samples=min_samples,
    )


def _unit_map(values: Mapping[str, object]) -> dict[str, MetricUnitsV2]:
    units: dict[str, MetricUnitsV2] = {}
    for name, value in values.items():
        if isinstance(value, MetricUnitsV2):
            units[name] = value
        else:
            unit = str(value)
            units[name] = MetricUnitsV2(
                canonical_unit=unit,
                display_unit=unit,
                authority="source_declared",
            )
    return units


def _scenario(**values: object) -> LaunchMonitorConformanceScenarioV1:
    values["scenario_sha256"] = launch_monitor_conformance_scenario_sha256(values)
    return LaunchMonitorConformanceScenarioV1.model_validate(values)


def _lineage_scenario_values(
    *, result: Any, context: AnalysisContextV2, units: Mapping[str, object]
) -> dict[str, object]:
    lineage = result.lineage
    return {
        "units": _unit_map(units),
        "player_identity": context.player_identity,
        "session_identity": context.session_identity,
        "order_evidence": context.order_evidence,
        "sources": lineage.sources,
        "backing_records": lineage.backing_records,
    }


def _analysis_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    frame = _analysis_frame()
    context = _context(frame)
    request = FlexibleAnalysisRequest(
        outcome="ball_speed",
        predictors=("club_speed",),
        analysis_mode="correlation",
        min_samples=4,
    )
    results = (
        analyze_variables_v2(frame, request, context=context),
        analyze_variables_v2(frame.iloc[:2], request, context=context),
    )
    return tuple(
        _scenario(
            scenario_id=f"analysis-v2-{result.status}",
            analysis_kind="analysis_v2",
            expected_status=result.status,
            description=f"Synthetic analysis-v2 {result.status} result.",
            claims=result.claims.model_dump(mode="json"),
            exclusions=result.missingness.excluded_by_reason,
            payload=result,
            **_lineage_scenario_values(
                result=result, context=context, units=result.units
            ),
        )
        for result in results
    )


def _covariation_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    frame = _covariation_frame()
    context = _covariation_context(frame)
    request = PlayerCovariationRequestV1(
        x_column="face_angle",
        y_column="club_path",
        player_column="player_id",
    )
    results = (
        analyze_player_covariation_v1(frame, request, context=context),
        analyze_player_covariation_v1(frame.iloc[:2], request, context=context),
    )
    return tuple(
        _scenario(
            scenario_id=f"player-covariation-{result.status}",
            analysis_kind="player_covariation",
            expected_status=result.status,
            description=f"Synthetic player-covariation {result.status} result.",
            claims=result.claims.model_dump(mode="json"),
            exclusions=result.missingness.excluded_by_reason,
            payload=result,
            **_lineage_scenario_values(
                result=result, context=context, units=result.units
            ),
        )
        for result in results
    )


def _longitudinal_scenarios(
    root: Path,
) -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    frame, request, context = _longitudinal_inputs(root)
    unavailable_context = context.model_copy(
        update={
            "order_evidence": context.order_evidence.model_copy(
                update={"trust_level": "untrusted_inferred"}
            )
        }
    )
    pairs = (
        (analyze_longitudinal_sessions(frame, request, context=context), context),
        (
            analyze_longitudinal_sessions(frame, request, context=unavailable_context),
            unavailable_context,
        ),
    )
    scenarios = []
    for result, scenario_context in pairs:
        claims = result.claims.model_dump(mode="json")
        claims["causal_inference"] = False
        scenarios.append(
            _scenario(
                scenario_id=f"attested-longitudinal-{result.status}",
                analysis_kind="attested_longitudinal",
                expected_status=result.status,
                description=f"Synthetic attested-longitudinal {result.status} result.",
                claims=claims,
                exclusions=result.missingness.excluded_by_reason,
                payload=result,
                **_lineage_scenario_values(
                    result=result,
                    context=scenario_context,
                    units={
                        request.metric: scenario_context.source_units[request.metric]
                    },
                ),
            )
        )
    return tuple(scenarios)


def _derived_lineage(
    frame: pd.DataFrame, context: AnalysisContextV2
) -> dict[str, object]:
    lineage = build_analysis_lineage_v2(frame, context)
    return {
        "player_identity": context.player_identity,
        "session_identity": context.session_identity,
        "order_evidence": context.order_evidence,
        "sources": lineage.sources,
        "backing_records": lineage.backing_records,
    }


def _strokes_gained_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    frame = _strokes_gained_frame()
    context = _context(frame)
    results = (
        analyze_source_backed_strokes_gained(
            frame,
            _baseline(),
            _strokes_gained_request(3),
            context=context,
        ),
        analyze_source_backed_strokes_gained(
            frame,
            _baseline(),
            _strokes_gained_request(4),
            context=context,
        ),
    )
    return tuple(
        _scenario(
            scenario_id=f"source-backed-strokes-gained-{result.status}",
            analysis_kind="source_backed_strokes_gained",
            expected_status=result.status,
            description=f"Synthetic source-backed strokes-gained {result.status} result.",
            units=_unit_map(result.units),
            claims=result.claims.model_dump(mode="json"),
            exclusions=result.exclusions.by_reason,
            payload=result,
            **_derived_lineage(frame, context),
        )
        for result in results
    )


def _proxy_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    available_frame = pd.DataFrame(
        {
            "shot_id": ["proxy-0", "proxy-1"],
            "source_id": [_SOURCE_ID, _SOURCE_ID],
            "carry": [150.0, 155.0],
            "lateral": [-5.0, 2.0],
        }
    )
    unavailable_frame = pd.DataFrame(
        {
            "shot_id": ["proxy-0", "proxy-1"],
            "source_id": [_SOURCE_ID, _SOURCE_ID],
            "carry": [150.0, "missing"],
            "lateral": [-5.0, 2.0],
        }
    )
    requests = (
        OutcomeProxyRequestV1(
            carry_column="carry",
            lateral_column="lateral",
            carry_unit="yd",
            lateral_unit="yd",
            target_distance_yards=150,
            shot_id_column="shot_id",
            min_samples=1,
        ),
        OutcomeProxyRequestV1(
            carry_column="carry",
            lateral_column="lateral",
            carry_unit="yd",
            lateral_unit="yd",
            target_distance_yards=150,
            shot_id_column="shot_id",
            min_samples=2,
        ),
    )
    pairs = tuple(
        (analyze_outcome_proxy(frame, request), frame, _context(frame))
        for frame, request in zip(
            (available_frame, unavailable_frame), requests, strict=True
        )
    )
    return tuple(
        _scenario(
            scenario_id=f"distance-target-proxy-{result.status}",
            analysis_kind="distance_target_proxy",
            expected_status=result.status,
            description=f"Synthetic distance/target proxy {result.status} result.",
            units=_unit_map(result.units),
            claims=result.claims.model_dump(mode="json"),
            exclusions=result.exclusions.by_reason,
            payload=result,
            **_derived_lineage(frame, context),
        )
        for result, frame, context in pairs
    )


def launch_monitor_conformance_bundle(root: Path) -> LaunchMonitorConformanceBundleV1:
    """Return all deterministic consumer cases without embedding input rows."""

    scenarios = (
        *_analysis_scenarios(),
        *_covariation_scenarios(),
        *_longitudinal_scenarios(root),
        *_strokes_gained_scenarios(),
        *_proxy_scenarios(),
    )
    values: dict[str, object] = {
        "bundle_version": LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION,
        "description": (
            "Synthetic, data-free consumer outputs for canonical launch-monitor "
            "analytics; no observed or private shot rows are embedded."
        ),
        "data_classification": "synthetic_contract_fixture_no_private_rows",
        "input_records_embedded": False,
        "scenarios": scenarios,
    }
    values["bundle_sha256"] = launch_monitor_conformance_bundle_sha256(values)
    return LaunchMonitorConformanceBundleV1.model_validate(values)


__all__ = ["launch_monitor_conformance_bundle"]
