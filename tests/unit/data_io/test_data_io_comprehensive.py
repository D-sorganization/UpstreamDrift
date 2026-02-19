"""Comprehensive tests for src.shared.python.data_io package.

Covers common_utils (convert_units, normalize_z_score, standardize_joint_angles,
CONVERSION_FACTORS), path_utils (get_repo_root, ensure_directory, find_file_in_parents,
get_relative_path), and reproducibility (set_seeds, get_rng, log_execution_time).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shared.python.data_io.common_utils import (
    CONVERSION_FACTORS,
    convert_units,
    normalize_z_score,
    standardize_joint_angles,
)
from src.shared.python.data_io.path_utils import (
    ensure_directory,
    find_file_in_parents,
    get_data_dir,
    get_docs_dir,
    get_engines_dir,
    get_relative_path,
    get_repo_root,
    get_shared_dir,
    get_shared_python_root,
    get_src_root,
    get_tests_root,
)
from src.shared.python.data_io.reproducibility import (
    DEFAULT_SEED,
    MAX_SEED,
    get_rng,
    log_execution_time,
    set_seeds,
)

# ============================================================================
# Tests for common_utils: convert_units
# ============================================================================


class TestConvertUnits:
    """Tests for unit conversion utility."""

    @pytest.mark.parametrize(
        "value, from_u, to_u, expected",
        [
            (42.0, "deg", "deg", 42.0),
            (1.5, "m", "m", 1.5),
            (180.0, "deg", "rad", np.pi),
            (np.pi, "rad", "deg", 180.0),
            (1.0, "m", "ft", 3.28084),
            (3.28084, "ft", "m", 1.0),
        ],
        ids=["deg-identity", "m-identity", "deg-to-rad", "rad-to-deg",
             "m-to-ft", "ft-to-m"],
    )
    def test_unit_conversion(
        self, value: float, from_u: str, to_u: str, expected: float
    ) -> None:
        """Unit conversions should produce the expected result."""
        assert convert_units(value, from_u, to_u) == pytest.approx(expected, rel=0.01)

    def test_roundtrip(self) -> None:
        """Converting A→B→A should recover original value."""
        original = 100.0
        intermediate = convert_units(original, "m/s", "mph")
        recovered = convert_units(intermediate, "mph", "m/s")
        assert recovered == pytest.approx(original, rel=1e-10)

    @pytest.mark.parametrize(
        "from_u,to_u",
        [
            ("deg", "rad"),
            ("rad", "deg"),
            ("m", "ft"),
            ("ft", "m"),
            ("m/s", "mph"),
            ("mph", "m/s"),
            ("kg", "lb"),
            ("lb", "kg"),
        ],
    )
    def test_all_supported_conversions(self, from_u: str, to_u: str) -> None:
        """Every supported conversion should work without error."""
        result = convert_units(1.0, from_u, to_u)
        assert result > 0

    def test_unsupported_conversion_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            convert_units(1.0, "parsec", "lightyear")

    def test_conversion_factors_populated(self) -> None:
        assert len(CONVERSION_FACTORS) >= 10  # We have many conversions


# ============================================================================
# Tests for common_utils: normalize_z_score
# ============================================================================


class TestNormalizeZScore:
    """Tests for Z-score normalization."""

    def test_basic_normalization(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_z_score(data)
        assert normed.mean() == pytest.approx(0.0, abs=1e-10)
        assert normed.std() == pytest.approx(1.0, abs=0.01)

    def test_constant_data(self) -> None:
        """Constant data: std=0, epsilon prevents division by zero."""
        data = np.ones(10) * 5.0
        normed = normalize_z_score(data)
        # All values should be ~0 (numerator = 0)
        np.testing.assert_allclose(normed, 0.0, atol=1e-6)

    def test_negative_values(self) -> None:
        data = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        normed = normalize_z_score(data)
        assert normed.mean() == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Tests for common_utils: standardize_joint_angles
# ============================================================================


class TestStandardizeJointAngles:
    """Tests for joint angle standardization."""

    def test_basic(self) -> None:
        angles = np.random.default_rng(42).random((10, 3))
        df = standardize_joint_angles(angles)
        assert isinstance(df, pd.DataFrame)
        assert "time" in df.columns
        assert len(df) == 10

    def test_custom_names(self) -> None:
        angles = np.zeros((5, 2))
        df = standardize_joint_angles(angles, angle_names=["shoulder", "elbow"])
        assert "shoulder" in df.columns
        assert "elbow" in df.columns

    def test_default_names(self) -> None:
        angles = np.zeros((5, 3))
        df = standardize_joint_angles(angles)
        assert "joint_0" in df.columns
        assert "joint_2" in df.columns

    def test_time_column_spacing(self) -> None:
        angles = np.zeros((100, 2))
        df = standardize_joint_angles(angles, time_step=0.01)
        # Time should start at 0 and increment by 0.01
        assert df["time"].iloc[0] == pytest.approx(0.0)
        assert df["time"].iloc[1] == pytest.approx(0.01)


# ============================================================================
# Tests for path_utils
# ============================================================================


class TestPathUtils:
    """Tests for path resolution utilities."""

    def test_get_repo_root_returns_path(self) -> None:
        root = get_repo_root()
        assert isinstance(root, Path)
        assert root.exists()

    def test_get_repo_root_has_pyproject(self) -> None:
        """Repo root should contain pyproject.toml."""
        root = get_repo_root()
        assert (root / "pyproject.toml").exists()

    @pytest.mark.parametrize(
        "getter, expected_name",
        [
            (get_src_root, "src"),
            (get_tests_root, "tests"),
            (get_data_dir, "data"),
            (get_docs_dir, "docs"),
            (get_engines_dir, "engines"),
            (get_shared_dir, "shared"),
        ],
        ids=["src", "tests", "data", "docs", "engines", "shared"],
    )
    def test_directory_getter_returns_expected_name(
        self, getter: object, expected_name: str
    ) -> None:
        """All directory getters should return a Path with the correct name."""
        result = getter()
        assert isinstance(result, Path)
        assert result.name == expected_name

    def test_get_shared_python_root(self) -> None:
        sp = get_shared_python_root()
        assert sp.name == "python"
        assert sp.parent.name == "shared"

    def test_ensure_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "test_subdir" / "nested"
        result = ensure_directory(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_ensure_directory_idempotent(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "idempotent"
        ensure_directory(new_dir)
        ensure_directory(new_dir)  # Should not raise
        assert new_dir.exists()

    def test_get_relative_path(self) -> None:
        root = get_repo_root()
        full_path = root / "src" / "shared" / "python" / "core"
        rel = get_relative_path(full_path)
        assert str(rel) == str(Path("src") / "shared" / "python" / "core")

    def test_get_relative_path_custom_base(self, tmp_path: Path) -> None:
        child = tmp_path / "a" / "b" / "c.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        rel = get_relative_path(child, base=tmp_path)
        assert str(rel) == str(Path("a") / "b" / "c.txt")

    def test_find_file_in_parents(self) -> None:
        """Should find pyproject.toml starting from deep in repo."""
        result = find_file_in_parents("pyproject.toml")
        assert result is not None
        assert result.name == "pyproject.toml"

    def test_find_file_in_parents_not_found(self) -> None:
        result = find_file_in_parents("definitely_not_a_real_file_xyz.txt")
        assert result is None


# ============================================================================
# Tests for reproducibility
# ============================================================================


class TestReproducibility:
    """Tests for reproducibility utilities."""

    def test_default_seed(self) -> None:
        assert DEFAULT_SEED == 42

    def test_max_seed(self) -> None:
        assert np.iinfo(np.uint32).max == MAX_SEED

    def test_set_seeds_deterministic(self) -> None:
        """After set_seeds, random output should be reproducible."""
        set_seeds(42)
        a1 = random.random()
        np1 = np.random.random()  # noqa: NPY002

        set_seeds(42)
        a2 = random.random()
        np2 = np.random.random()  # noqa: NPY002

        assert a1 == a2
        assert np1 == np2

    def test_set_seeds_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Seed"):
            set_seeds(-1)

    def test_set_seeds_no_validate_accepts_valid(self) -> None:
        """With validate=False, valid seeds still work."""
        set_seeds(0, validate=False)  # Should not raise

    def test_set_seeds_no_validate_negative_numpy_error(self) -> None:
        """Negative seeds bypass Python check but numpy still rejects."""
        with pytest.raises(ValueError):
            set_seeds(-1, validate=False)

    def test_get_rng_with_seed(self) -> None:
        rng = get_rng(42)
        assert isinstance(rng, np.random.Generator)
        val1 = rng.random()

        rng2 = get_rng(42)
        val2 = rng2.random()
        assert val1 == val2

    def test_get_rng_without_seed(self) -> None:
        rng = get_rng()
        assert isinstance(rng, np.random.Generator)
        # Just verify it works
        _ = rng.random()

    def test_log_execution_time(self) -> None:
        """Context manager should time an operation without error."""
        with log_execution_time("test_operation"):
            time.sleep(0.01)
        # No assertion needed — just verify it doesn't crash

    def test_log_execution_time_custom_logger(self) -> None:
        import logging

        custom_logger = logging.getLogger("test_logger")
        with log_execution_time("custom_op", logger_obj=custom_logger):
            pass
