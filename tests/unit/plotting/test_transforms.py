"""Tests for src.shared.python.plotting.transforms.DataManager (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.plotting.transforms import DataManager


class _StubRecorder:
    """Minimal stub implementing get_time_series for testing DataManager."""

    def __init__(self) -> None:
        t = np.linspace(0.0, 1.0, 50)
        self._data: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "joint_positions": (t, np.sin(t)),
            "joint_velocities": (t, np.cos(t)),
            "kinetic_energy": (t, 0.5 * np.sin(t) ** 2),
        }

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        if field_name not in self._data:
            raise KeyError(f"Field {field_name!r} not found")
        return self._data[field_name]

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, 1, 50)
        return t, np.zeros(50)


class TestDataManagerConstruction:
    def test_transforms_valid_construction(self) -> None:
        dm = DataManager(_StubRecorder())
        assert dm is not None

    def test_stores_recorder(self) -> None:
        rec = _StubRecorder()
        dm = DataManager(rec)
        assert dm.recorder is rec

    def test_default_joint_names_empty(self) -> None:
        dm = DataManager(_StubRecorder())
        assert dm.joint_names == []

    def test_custom_joint_names(self) -> None:
        names = ["Hip", "Knee", "Ankle"]
        dm = DataManager(_StubRecorder(), joint_names=names)
        assert dm.joint_names == names

    def test_enable_cache_default_true(self) -> None:
        dm = DataManager(_StubRecorder())
        assert dm.enable_cache is True

    def test_cache_disabled(self) -> None:
        dm = DataManager(_StubRecorder(), enable_cache=False)
        assert dm.enable_cache is False


class TestDataManagerGetSeries:
    def setup_method(self) -> None:
        self.dm = DataManager(_StubRecorder(), enable_cache=False)

    def test_transforms_returns_tuple(self) -> None:
        result = self.dm.get_series("joint_positions")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_times_are_array(self) -> None:
        times, _ = self.dm.get_series("joint_positions")
        assert isinstance(times, np.ndarray)

    def test_values_are_array(self) -> None:
        _, values = self.dm.get_series("joint_velocities")
        assert isinstance(values, np.ndarray)

    def test_transforms_missing_field_raises(self) -> None:
        with pytest.raises((KeyError, Exception)):
            self.dm.get_series("nonexistent_field")


class TestDataManagerCaching:
    def test_cached_result_consistent(self) -> None:
        dm = DataManager(_StubRecorder(), enable_cache=True)
        t1, v1 = dm.get_series("joint_positions")
        t2, v2 = dm.get_series("joint_positions")
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)

    def test_clear_cache_refetches(self) -> None:
        dm = DataManager(_StubRecorder(), enable_cache=True)
        dm.get_series("joint_positions")
        dm.clear_cache()
        # After clearing, fetching again should still work
        result = dm.get_series("joint_positions")
        assert result is not None


class TestDataManagerJointNames:
    def test_get_joint_name_in_range(self) -> None:
        dm = DataManager(_StubRecorder(), joint_names=["Hip", "Knee"])
        assert dm.get_joint_name(0) == "Hip"
        assert dm.get_joint_name(1) == "Knee"

    def test_get_joint_name_out_of_range(self) -> None:
        dm = DataManager(_StubRecorder(), joint_names=["Hip"])
        result = dm.get_joint_name(5)
        assert "5" in result  # Should include the index

    def test_get_joint_name_no_names(self) -> None:
        dm = DataManager(_StubRecorder())
        result = dm.get_joint_name(0)
        assert "0" in result

    def test_get_aligned_label_no_names(self) -> None:
        dm = DataManager(_StubRecorder())
        result = dm.get_aligned_label(2, 5)
        assert "2" in result

    def test_get_aligned_label_exact_match(self) -> None:
        dm = DataManager(_StubRecorder(), joint_names=["A", "B", "C"])
        result = dm.get_aligned_label(1, 3)
        assert result == "B"

    def test_get_aligned_label_mismatch(self) -> None:
        # Data has more dims than names → aligns from end
        dm = DataManager(_StubRecorder(), joint_names=["A", "B"])
        result = dm.get_aligned_label(0, 5)
        assert isinstance(result, str)
