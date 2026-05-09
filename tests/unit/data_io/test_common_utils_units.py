"""Tests for src.shared.python.data_io.common_utils (Issues #1949, #1744)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.shared.python.data_io.common_utils import (
    CONVERSION_FACTORS,
    convert_units,
    load_golf_data,
    normalize_z_score,
    save_golf_data,
    standardize_joint_angles,
)

# ---------------------------------------------------------------------------
# CONVERSION_FACTORS dict
# ---------------------------------------------------------------------------


class TestConversionFactors:
    def test_common_utils_units_is_dict(self) -> None:
        assert isinstance(CONVERSION_FACTORS, dict)

    def test_common_utils_units_non_empty(self) -> None:
        assert len(CONVERSION_FACTORS) > 0

    def test_keys_are_tuples(self) -> None:
        for key in CONVERSION_FACTORS:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_values_are_floats(self) -> None:
        for v in CONVERSION_FACTORS.values():
            assert isinstance(v, float)

    def test_deg_rad_pair_exists(self) -> None:
        assert ("deg", "rad") in CONVERSION_FACTORS
        assert ("rad", "deg") in CONVERSION_FACTORS

    def test_deg_rad_round_trip(self) -> None:
        factor = CONVERSION_FACTORS[("deg", "rad")] * CONVERSION_FACTORS[("rad", "deg")]
        assert abs(factor - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# convert_units
# ---------------------------------------------------------------------------


class TestConvertUnits:
    def test_same_unit_returns_value(self) -> None:
        assert convert_units(42.0, "m", "m") == 42.0

    def test_zero_value(self) -> None:
        assert convert_units(0.0, "deg", "rad") == 0.0

    def test_degrees_to_radians(self) -> None:
        result = convert_units(180.0, "deg", "rad")
        assert abs(result - math.pi) < 1e-10

    def test_radians_to_degrees(self) -> None:
        result = convert_units(math.pi, "rad", "deg")
        assert abs(result - 180.0) < 1e-10

    def test_m_to_mm(self) -> None:
        assert convert_units(1.0, "m", "mm") == pytest.approx(1000.0)

    def test_mm_to_m(self) -> None:
        assert convert_units(1000.0, "mm", "m") == pytest.approx(1.0)

    def test_kg_to_lb(self) -> None:
        result = convert_units(1.0, "kg", "lb")
        assert result > 2.0  # 1 kg ≈ 2.205 lb

    def test_unsupported_conversion_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            convert_units(1.0, "kg", "km")

    def test_round_trip_velocity(self) -> None:
        mph_val = convert_units(1.0, "m/s", "mph")
        back = convert_units(mph_val, "mph", "m/s")
        assert abs(back - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# normalize_z_score
# ---------------------------------------------------------------------------


class TestNormalizeZScore:
    def test_common_utils_units_returns_ndarray(self) -> None:
        result = normalize_z_score(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_mean_near_zero(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_z_score(data)
        assert abs(np.mean(result)) < 1e-7

    def test_std_near_one(self) -> None:
        data = np.random.rand(100)
        result = normalize_z_score(data)
        # With epsilon, std may be slightly off but should be close to 1
        assert abs(np.std(result) - 1.0) < 0.01

    def test_constant_array_no_divide_by_zero(self) -> None:
        data = np.ones(10)
        result = normalize_z_score(data)  # should not raise
        assert np.all(np.isfinite(result))

    def test_same_length_output(self) -> None:
        data = np.arange(10, dtype=float)
        result = normalize_z_score(data)
        assert len(result) == len(data)


# ---------------------------------------------------------------------------
# standardize_joint_angles
# ---------------------------------------------------------------------------


class TestStandardizeJointAngles:
    def test_returns_dataframe(self) -> None:
        angles = np.zeros((10, 3))
        result = standardize_joint_angles(angles)
        assert isinstance(result, pd.DataFrame)

    def test_time_column_present(self) -> None:
        angles = np.zeros((5, 2))
        df = standardize_joint_angles(angles)
        assert "time" in df.columns

    def test_time_starts_at_zero(self) -> None:
        angles = np.zeros((5, 2))
        df = standardize_joint_angles(angles)
        assert df["time"].iloc[0] == pytest.approx(0.0)

    def test_auto_named_columns(self) -> None:
        angles = np.zeros((5, 3))
        df = standardize_joint_angles(angles)
        assert "joint_0" in df.columns
        assert "joint_2" in df.columns

    def test_named_columns_used(self) -> None:
        angles = np.zeros((5, 2))
        df = standardize_joint_angles(angles, angle_names=["hip", "knee"])
        assert "hip" in df.columns
        assert "knee" in df.columns

    def test_row_count_matches(self) -> None:
        angles = np.zeros((20, 4))
        df = standardize_joint_angles(angles)
        assert len(df) == 20


# ---------------------------------------------------------------------------
# load_golf_data / save_golf_data
# ---------------------------------------------------------------------------


class TestLoadSaveGolfData:
    def test_load_csv(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("a,b\n1,2\n3,4\n")
            path = f.name
        try:
            df = load_golf_data(path)
            assert list(df.columns) == ["a", "b"]
            assert len(df) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_json(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write('[{"x": 1}, {"x": 2}]')
            path = f.name
        try:
            df = load_golf_data(path)
            assert "x" in df.columns
        finally:
            Path(path).unlink(missing_ok=True)

    def test_common_utils_units_unsupported_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            load_golf_data("/tmp/data.xyz")

    def test_save_csv_creates_file(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_golf_data(df, path, format="csv")
            assert Path(path).exists()
            loaded = pd.read_csv(path)
            assert list(loaded.columns) == ["a", "b"]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_json_creates_file(self) -> None:
        df = pd.DataFrame({"x": [10, 20]})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_golf_data(df, path, format="json")
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_unsupported_format_raises(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unsupported"):
            save_golf_data(df, "/tmp/out.xyz", format="xyz")
