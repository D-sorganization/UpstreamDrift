"""Contract tests for flexible, provenance-preserving statistical analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tools.launch_monitor_model import (
    FlexibleAnalysisRequest,
    analyze_variables,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _shots(count: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    club_speed = np.linspace(35.0, 52.0, count)
    attack_angle = rng.normal(-0.03, 0.02, count)
    ball_speed = 1.48 * club_speed + 3.2 * attack_angle
    ball_speed += rng.normal(0.0, 0.35, count)
    carry_distance = 3.35 * ball_speed + rng.normal(0.0, 1.5, count)
    return pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(count)],
            "session_id": np.where(np.arange(count) < count / 2, "a", "b"),
            "monitor_vendor": np.where(np.arange(count) % 2, "Garmin", "TrackMan"),
            "source_row": np.arange(2, count + 2),
            "club_speed": club_speed,
            "attack_angle": attack_angle,
            "ball_speed": ball_speed,
            "carry_distance": carry_distance,
            "source::custom_numeric": club_speed * 2.0,
        }
    )


def test_comprehensive_analysis_reports_uncertainty_diagnostics_and_lineage() -> None:
    result = analyze_variables(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            correlation_method="pearson",
            analysis_mode="comprehensive",
            min_samples=20,
        ),
    )

    assert result.dataset.row_count == 120
    assert result.dataset.complete_row_count == 120
    assert result.dataset.monitor_vendors == ("Garmin", "TrackMan")
    assert len(result.dataset.fingerprint_sha256) == 64
    assert result.regression is not None
    assert result.regression.r_squared > 0.99
    assert result.regression.adjusted_r_squared > 0.99
    assert result.regression.residual_diagnostics.rmse < 0.5
    assert result.regression.coefficients["club_speed"].estimate == pytest.approx(
        1.48, rel=0.02
    )
    assert all(
        item.ci_lower <= item.coefficient <= item.ci_upper
        for item in result.correlations
    )
    adjusted = {item.predictor: item.adjusted_p_value for item in result.correlations}
    assert adjusted["club_speed"] <= 0.05
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())
    payload = result.to_dict()
    assert payload["request"]["outcome"] == "ball_speed"
    assert payload["dataset"]["fingerprint_sha256"] == result.dataset.fingerprint_sha256


def test_pairwise_missing_policy_preserves_per_relationship_counts() -> None:
    frame = _shots(40)
    frame.loc[:4, "attack_angle"] = np.nan
    frame.loc[5:7, "club_speed"] = np.nan

    result = analyze_variables(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="correlation",
            missing_policy="pairwise",
            min_samples=10,
        ),
    )

    counts = {item.predictor: item.sample_count for item in result.correlations}
    assert counts == {"club_speed": 37, "attack_angle": 35}
    assert result.dataset.complete_row_count == 32


def test_dataset_fingerprint_matches_tools_contract_and_ignores_frame_index() -> None:
    frame = pd.DataFrame(
        {
            "shot_id": ["a", "b", "c"],
            "session_id": ["s", "s", "s"],
            "monitor_vendor": ["TrackMan", "TrackMan", "TrackMan"],
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.1],
        },
        index=[10, 20, 30],
    )
    request = FlexibleAnalysisRequest(
        outcome="y",
        predictors=("x",),
        analysis_mode="correlation",
        min_samples=3,
    )

    indexed = analyze_variables(frame, request)
    reset = analyze_variables(frame.reset_index(drop=True), request)

    expected_tools_fingerprint = (
        "6bdb2a22ab06a0fac0b7b0a085f099783759a109d1c38d49eb23f3473c9efff9"
    )
    assert indexed.dataset.fingerprint_sha256 == expected_tools_fingerprint
    assert reset.dataset.fingerprint_sha256 == expected_tools_fingerprint
    assert indexed.to_dict()["contract_version"] == "1.0.0"


def test_grouped_analysis_keeps_monitor_results_separate() -> None:
    result = analyze_variables(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="carry_distance",
            predictors=("ball_speed",),
            group_by="monitor_vendor",
            analysis_mode="comprehensive",
            min_samples=20,
        ),
    )

    assert [group.group_value for group in result.groups] == ["Garmin", "TrackMan"]
    assert all(group.regression is not None for group in result.groups)
    assert all(group.row_count == 60 for group in result.groups)


def test_custom_source_field_cannot_be_pooled_across_monitor_vendors() -> None:
    with pytest.raises(ValueError, match="source fields.*multiple monitors"):
        analyze_variables(
            _shots(),
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("source::custom_numeric",),
                analysis_mode="correlation",
            ),
        )


def test_aggregate_records_never_enter_regression() -> None:
    frame = _shots(30)
    frame["observation_kind"] = "aggregate"
    with pytest.raises(
        ValueError, match="Aggregate observations cannot enter regression"
    ):
        analyze_variables(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed",),
                analysis_mode="regression",
                allow_aggregate=True,
            ),
        )


def test_explicit_aggregate_correlation_is_labeled_descriptive() -> None:
    frame = _shots(30)
    frame["observation_kind"] = "aggregate"
    result = analyze_variables(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            allow_aggregate=True,
        ),
    )

    assert any("ecological" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize(
    ("analysis_request", "message"),
    [
        (
            FlexibleAnalysisRequest(outcome="ball_speed", predictors=("ball_speed",)),
            "outcome cannot also be a predictor",
        ),
        (
            FlexibleAnalysisRequest(outcome="ball_speed", predictors=("constant",)),
            "Constant variables",
        ),
    ],
)
def test_invalid_analysis_contracts_fail_closed(
    analysis_request: FlexibleAnalysisRequest, message: str
) -> None:
    frame = _shots(30)
    frame["constant"] = 1.0
    with pytest.raises(ValueError, match=message):
        analyze_variables(frame, analysis_request)
