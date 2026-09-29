"""Tests for plotting.animation, plotting.base, plotting.energy, plotting.kinematics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.plotting.animation import AnimationConfig, SwingAnimator
from src.shared.python.plotting.base import RecorderInterface
from src.shared.python.plotting.energy import plot_energy_overview
from src.shared.python.plotting.kinematics import plot_joint_positions


class _MockRecorder:
    engine = None

    def get_time_series(self, field_name) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(0, 1, 10), np.zeros((10, 3))

    def get_induced_acceleration_series(self, source) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(0, 1, 10), np.zeros((10, 3))

    def set_analysis_config(self, config) -> None:
        pass


class TestAnimationConfig:
    def test_plotting_extras_default_construction(self) -> None:
        config = AnimationConfig()
        assert config is not None

    def test_has_fps(self) -> None:
        config = AnimationConfig()
        assert hasattr(config, "fps")

    def test_custom_fps(self) -> None:
        config = AnimationConfig(fps=30)
        assert config.fps == 30


class TestSwingAnimator:
    def test_plotting_extras_construction(self) -> None:
        animator = SwingAnimator(_MockRecorder())
        assert animator is not None


class TestRecorderInterfaceBase:
    def test_protocol_importable(self) -> None:
        assert RecorderInterface is not None


class TestEnergyModuleImport:
    def test_plot_energy_overview_callable(self) -> None:
        assert callable(plot_energy_overview)


class TestKinematicsModuleImport:
    def test_plot_joint_positions_callable(self) -> None:
        assert callable(plot_joint_positions)
