"""Tests for src.shared.python.validation_pkg.kaggle_validation (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.shared.python.validation_pkg.kaggle_validation import (
    ShotRecord,
    get_clean_shots,
    get_dataset_statistics,
)

# ---------------------------------------------------------------------------
# ShotRecord dataclass
# ---------------------------------------------------------------------------


class TestShotRecord:
    def _make(self, **kwargs) -> ShotRecord:
        defaults = {
            "ball_speed_mph": 150.0,
            "launch_angle_deg": 12.0,
            "launch_direction_deg": 0.0,
            "backspin_rpm": 2500.0,
            "sidespin_rpm": 0.0,
            "spin_rate_rpm": 2500.0,
            "spin_axis_deg": 0.0,
            "carry_distance_yards": 250.0,
            "total_distance_yards": 270.0,
            "apex_height_ft": 100.0,
            "air_density_g_l": 1.225,
            "temperature_f": 70.0,
            "air_pressure_kpa": 101.3,
        }
        defaults.update(kwargs)
        return ShotRecord(**defaults)

    def test_ball_speed_mps_positive(self) -> None:
        sr = self._make(ball_speed_mph=100.0)
        assert sr.ball_speed_mps > 0.0

    def test_ball_speed_mps_conversion(self) -> None:
        sr = self._make(ball_speed_mph=1.0)
        # 1 mph ≈ 0.44704 m/s
        assert abs(sr.ball_speed_mps - 0.44704) < 0.001

    def test_carry_distance_m_positive(self) -> None:
        sr = self._make(carry_distance_yards=200.0)
        assert sr.carry_distance_m > 0.0

    def test_carry_distance_m_greater_than_yards(self) -> None:
        # 1 yard = 0.9144 m; 200 yards > 100m
        sr = self._make(carry_distance_yards=200.0)
        assert sr.carry_distance_m > 100.0

    def test_apex_height_m_conversion(self) -> None:
        sr = self._make(apex_height_ft=1.0)
        # 1 ft ≈ 0.3048 m
        assert abs(sr.apex_height_m - 0.3048) < 0.001

    def test_air_density_kg_m3(self) -> None:
        sr = self._make(air_density_g_l=1.225)
        assert sr.air_density_kg_m3 == 1.225


# ---------------------------------------------------------------------------
# get_clean_shots
# ---------------------------------------------------------------------------


class TestGetCleanShots:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ball_speed_mph": [150.0, 80.0, 200.0, -10.0, np.nan],
                "launch_angle_deg": [12.0, 30.0, 10.0, 5.0, 8.0],
                "carry_distance_yards": [250.0, 150.0, 300.0, 100.0, 200.0],
                "spin_rate_rpm": [2500.0, 3000.0, 2000.0, 1000.0, 2800.0],
                "apex_height_ft": [100.0, 80.0, 120.0, 50.0, 90.0],
            }
        )

    def test_returns_dataframe(self) -> None:
        df = self._make_df()
        result = get_clean_shots(df)
        assert isinstance(result, pd.DataFrame)

    def test_filters_negative_speed(self) -> None:
        df = self._make_df()
        result = get_clean_shots(df)
        if "ball_speed_mph" in result.columns:
            assert (result["ball_speed_mph"] >= 0).all()


# ---------------------------------------------------------------------------
# get_dataset_statistics
# ---------------------------------------------------------------------------


class TestGetDatasetStatistics:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ball_speed_mph": [150.0, 160.0, 140.0],
                "launch_angle_deg": [10.0, 12.0, 11.0],
                "carry_distance_yards": [240.0, 260.0, 250.0],
                "spin_rate_rpm": [2400.0, 2600.0, 2500.0],
            }
        )

    def test_kaggle_validation_returns_dict(self) -> None:
        df = self._make_df()
        result = get_dataset_statistics(df)
        assert isinstance(result, dict)

    def test_stats_include_mean(self) -> None:
        df = self._make_df()
        result = get_dataset_statistics(df)
        # At least one column should have a "mean" key
        has_mean = any("mean" in stats for stats in result.values())
        assert has_mean

    def test_non_empty_result(self) -> None:
        df = self._make_df()
        result = get_dataset_statistics(df)
        assert len(result) > 0
