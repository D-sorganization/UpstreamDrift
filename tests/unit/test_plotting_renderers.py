"""Tests for plotting renderers (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.plotting.core import GolfSwingPlotter
from src.shared.python.plotting.renderers.club import ClubRenderer
from src.shared.python.plotting.renderers.comparison import ComparisonRenderer
from src.shared.python.plotting.renderers.coordination import CoordinationRenderer
from src.shared.python.plotting.renderers.dashboard import DashboardRenderer
from src.shared.python.plotting.renderers.energy import EnergyRenderer
from src.shared.python.plotting.renderers.kinematics import KinematicsRenderer
from src.shared.python.plotting.renderers.kinetics import KineticsRenderer
from src.shared.python.plotting.renderers.signal import SignalRenderer
from src.shared.python.plotting.renderers.stability import StabilityRenderer
from src.shared.python.plotting.renderers.vectors import VectorOverlayRenderer
from src.shared.python.plotting.transforms import DataManager


class _MockRecorder:
    """Minimal recorder mock satisfying RecorderInterface."""

    engine = None

    def get_time_series(self, field_name) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(0, 1, 10), np.zeros((10, 3))

    def get_induced_acceleration_series(self, source) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(0, 1, 10), np.zeros((10, 3))

    def set_analysis_config(self, config) -> None:
        pass


def _make_dm() -> DataManager:
    return DataManager(_MockRecorder(), enable_cache=False)


class TestDataManager:
    def test_plotting_renderers_construction(self) -> None:
        dm = _make_dm()
        assert dm is not None

    def test_get_series_returns_arrays(self) -> None:
        dm = _make_dm()
        t, v = dm.get_series("joint_positions")
        assert isinstance(t, np.ndarray)
        assert isinstance(v, np.ndarray)


class TestClubRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = ClubRenderer(_make_dm())
        assert renderer is not None


class TestEnergyRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = EnergyRenderer(_make_dm())
        assert renderer is not None


class TestKinematicsRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = KinematicsRenderer(_make_dm())
        assert renderer is not None


class TestKineticsRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = KineticsRenderer(_make_dm())
        assert renderer is not None


class TestStabilityRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = StabilityRenderer(_make_dm())
        assert renderer is not None


class TestVectorsRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = VectorOverlayRenderer(_make_dm())
        assert renderer is not None


class TestSignalRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = SignalRenderer(_make_dm())
        assert renderer is not None


class TestComparisonRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = ComparisonRenderer(_make_dm())
        assert renderer is not None


class TestCoordinationRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = CoordinationRenderer(_make_dm())
        assert renderer is not None


class TestDashboardRenderer:
    def test_plotting_renderers_construction(self) -> None:
        renderer = DashboardRenderer(_make_dm())
        assert renderer is not None


class TestGolfSwingPlotter:
    def test_plotting_renderers_construction(self) -> None:
        plotter = GolfSwingPlotter(_MockRecorder(), joint_names=["hip", "knee"])
        assert plotter is not None
