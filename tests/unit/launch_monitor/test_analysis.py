from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tools.launch_monitor_model import (
    FilterRule,
    TreatmentConfig,
    analyze_dispersion,
    analyze_trend,
    apply_treatment,
    compare_monitors,
    compute_correlations,
    compute_pca,
    compute_vif,
    fit_predictive_model,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _shots(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    club = np.linspace(35.0, 50.0, n)
    attack = rng.normal(-0.04, 0.025, n)
    ball = 1.47 * club + 3.0 * attack + rng.normal(0.0, 0.7, n)
    return pd.DataFrame(
        {
            "shot_id": [f"s{i}" for i in range(n)],
            "session_id": np.where(np.arange(n) < n / 2, "a", "b"),
            "monitor_vendor": np.where(np.arange(n) % 2, "Garmin", "TrackMan"),
            "captured_at": pd.date_range("2026-01-01", periods=n, freq="D"),
            "club_speed": club,
            "attack_angle": attack,
            "ball_speed": ball,
            "smash_factor": ball / club,
            "carry_distance": 3.4 * ball + rng.normal(0.0, 2.0, n),
            "lateral_carry": rng.normal(2.0, 8.0, n),
        }
    )


def test_treatment_flags_duplicates_missing_and_robust_outliers() -> None:
    frame = _shots(20)
    frame.loc[2, "ball_speed"] = np.nan
    frame.loc[3, "club_speed"] = 999.0
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = apply_treatment(
        frame,
        TreatmentConfig(
            required_metrics=("club_speed", "ball_speed"),
            duplicate_columns=("shot_id",),
            outlier_metrics=("club_speed",),
            robust_z_threshold=3.5,
            exclude_flagged=True,
        ),
    )
    assert {"missing_required", "duplicate", "robust_outlier"} <= set(
        result.flags["flag_type"]
    )
    assert len(result.data) == len(frame) - 3
    assert len(result.audit_log) >= 3


def test_treatment_filters_and_labels_derived_metrics() -> None:
    frame = _shots(20).drop(columns=["smash_factor"])
    result = apply_treatment(
        frame,
        TreatmentConfig(
            filters=(FilterRule("monitor_vendor", "eq", "TrackMan"),),
        ),
    )
    assert set(result.data["monitor_vendor"]) == {"TrackMan"}
    assert result.data["smash_factor"].notna().all()
    assert set(result.data["status::smash_factor"]) == {"derived"}
    assert any(item["action"] == "filter" for item in result.audit_log)
    assert any(item["action"] == "derive_metric" for item in result.audit_log)


def test_correlations_include_counts_significance_and_derived_warning() -> None:
    result = compute_correlations(
        _shots(),
        metrics=("club_speed", "ball_speed", "smash_factor", "attack_angle"),
        method="spearman",
        controls=("attack_angle",),
    )
    assert result.coefficients.loc["club_speed", "ball_speed"] > 0.95
    assert result.p_values.loc["club_speed", "ball_speed"] < 0.001
    assert result.pair_counts.loc["club_speed", "ball_speed"] == 80
    assert result.adjusted_p_values is not None
    assert result.partial_coefficients is not None
    assert "smash_factor" in result.derived_metrics
    assert result.edges


def test_pca_and_vif_expose_multicollinearity() -> None:
    frame = _shots(100)
    metrics = ("club_speed", "ball_speed", "carry_distance", "attack_angle")
    pca = compute_pca(frame, metrics=metrics)
    vif = compute_vif(frame, metrics=metrics)
    assert pca.sample_count == 100
    assert pca.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert pca.loadings.shape == (4, 4)
    assert vif.sample_count == 100
    assert vif.values["ball_speed"] > 5
    assert "ball_speed" in vif.warning_metrics


@pytest.mark.parametrize("model", ["linear", "ridge", "lasso", "elastic_net"])
def test_regression_models_are_reproducible(model: str) -> None:
    frame = _shots(120)
    first = fit_predictive_model(
        frame,
        target="ball_speed",
        features=("club_speed", "attack_angle"),
        model=model,
        random_seed=7,
        group_column="session_id",
    )
    second = fit_predictive_model(
        frame,
        target="ball_speed",
        features=("club_speed", "attack_angle"),
        model=model,
        random_seed=7,
        group_column="session_id",
    )
    assert first.metrics == second.metrics
    assert first.metrics["r2"] > 0.9
    assert first.predictions.equals(second.predictions)


def test_shallow_mlp_is_reproducible_when_analysis_extra_is_installed() -> None:
    pytest.importorskip("sklearn")
    result = fit_predictive_model(
        _shots(160),
        target="ball_speed",
        features=("club_speed", "attack_angle"),
        model="mlp",
        random_seed=11,
    )
    assert result.metrics["r2"] > 0.85
    assert result.coefficients is None


def test_model_rejects_identity_derived_leakage() -> None:
    with pytest.raises(ValueError, match="leakage"):
        fit_predictive_model(
            _shots(),
            target="ball_speed",
            features=("club_speed", "smash_factor"),
            model="ridge",
        )


def test_matched_monitor_comparison_recovers_bias_and_slope() -> None:
    x = np.linspace(30.0, 50.0, 30)
    frame = pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(30), 2),
            "monitor_vendor": np.tile(["A", "B"], 30),
            "ball_speed": np.column_stack([x, 1.03 * x + 0.8]).ravel(),
        }
    )
    result = compare_monitors(
        frame,
        metric="ball_speed",
        monitor_column="monitor_vendor",
        match_column="match_id",
        reference_monitor="A",
    )
    comparison = result.pairwise[0]
    assert comparison.matched is True
    assert comparison.slope == pytest.approx(1.03, rel=0.01)
    assert comparison.intercept == pytest.approx(0.8, rel=0.1)
    assert comparison.mean_bias > 1.0


def test_dispersion_and_longitudinal_trend_capture_change() -> None:
    frame = _shots(80)
    dispersion = analyze_dispersion(
        frame, forward="carry_distance", lateral="lateral_carry"
    )
    assert dispersion.sample_count == 80
    assert dispersion.ellipse_major >= dispersion.ellipse_minor > 0
    assert dispersion.area_95 > 0

    trend_frame = frame.copy()
    trend_frame["club_speed"] += np.where(np.arange(80) >= 40, 4.0, 0.0)
    trend = analyze_trend(
        trend_frame,
        metric="club_speed",
        time_column="captured_at",
        rolling_window=10,
    )
    assert trend.slope_per_day > 0
    assert trend.change_candidates
    assert trend.latest_mean > trend.earliest_mean
