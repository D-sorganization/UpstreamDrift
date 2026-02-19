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

    def test_identity_conversion(self) -> None:
        """Same from/to unit should return unchanged value."""
        assert convert_units(42.0, "deg", "deg") == 42.0
        assert convert_units(1.5, "m", "m") == 1.5

    def test_deg_to_rad(self) -> None:
        result = convert_units(180.0, "deg", "rad")
        assert result == pytest.approx(np.pi)

    def test_rad_to_deg(self) -> None:
        result = convert_units(np.pi, "rad", "deg")
        assert result == pytest.approx(180.0)

    def test_m_to_ft(self) -> None:
        result = convert_units(1.0, "m", "ft")
        assert result == pytest.approx(3.28084, rel=0.01)

    def test_ft_to_m(self) -> None:
        result = convert_units(3.28084, "ft", "m")
        assert result == pytest.approx(1.0, rel=0.01)

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

    def test_get_src_root(self) -> None:
        src = get_src_root()
        assert isinstance(src, Path)
        assert src.name == "src"

    def test_get_tests_root(self) -> None:
        tests = get_tests_root()
        assert tests.name == "tests"
        assert tests.parent == get_repo_root()

    def test_get_data_dir(self) -> None:
        data = get_data_dir()
        assert data.name == "data"

    def test_get_docs_dir(self) -> None:
        docs = get_docs_dir()
        assert docs.name == "docs"

    def test_get_engines_dir(self) -> None:
        engines = get_engines_dir()
        assert engines.name == "engines"

    def test_get_shared_dir(self) -> None:
        shared = get_shared_dir()
        assert shared.name == "shared"

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
