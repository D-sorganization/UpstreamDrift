"""Tests for plotting.animation, plotting.base, plotting.energy, plotting.kinematics (Issues #1949, #1744)."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.shared.python.plotting.animation import AnimationConfig, SwingAnimator
from src.shared.python.plotting.base import RecorderInterface
from src.shared.python.plotting.energy import plot_energy_overview
from src.shared.python.plotting.kinematics import plot_joint_positions


class _MockRecorder:
    engine = None

    def get_time_series(self, field_name):
        return np.linspace(0, 1, 10), np.zeros((10, 3))

    def get_induced_acceleration_series(self, source):
        return np.linspace(0, 1, 10), np.zeros((10, 3))

    def set_analysis_config(self, config):
        pass


class _TrajectoryRecorder:
    def __init__(self, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self._series = series

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self._series[field_name]


class TestAnimationConfig:
    def test_default_construction(self) -> None:
        config = AnimationConfig()
        assert config is not None

    def test_has_fps(self) -> None:
        config = AnimationConfig()
        assert hasattr(config, "fps")

    def test_custom_fps(self) -> None:
        config = AnimationConfig(fps=30)
        assert config.fps == 30


class TestSwingAnimator:
    def test_construction(self) -> None:
        animator = SwingAnimator(_MockRecorder())
        assert animator is not None

    def test_gather_trajectory_data(self) -> None:
        recorder = _TrajectoryRecorder(
            {
                "club_head_position": (
                    np.array([0.0, 0.5, 1.0]),
                    np.array([[0.0, 0.0, 0.0], [0.4, 0.1, 0.2], [0.8, 0.3, 0.4]]),
                ),
                "r_hand_position": (
                    np.array([0.0, 0.5]),
                    np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.1]]),
                ),
            }
        )
        animator = SwingAnimator(recorder)

        body_data, times = animator._gather_trajectory_data(  # noqa: SLF001
            ["club_head", "r_hand"]
        )

        assert set(body_data) == {"club_head", "r_hand"}
        np.testing.assert_array_equal(times, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_array_equal(
            body_data["club_head"],
            np.array([[0.0, 0.0, 0.0], [0.4, 0.1, 0.2], [0.8, 0.3, 0.4]]),
        )

    def test_create_trajectory_animation_handles_empty_data(self) -> None:
        recorder = _TrajectoryRecorder(
            {
                "club_head_position": (
                    np.array([]),
                    np.empty((0, 3)),
                )
            }
        )
        animator = SwingAnimator(recorder)

        anim = animator.create_trajectory_animation()

        assert isinstance(anim, FuncAnimation)
        anim._draw_was_started = True  # type: ignore[attr-defined]
        plt.close("all")


class TestRecorderInterfaceBase:
    def test_protocol_importable(self) -> None:
        assert RecorderInterface is not None


class TestEnergyModuleImport:
    def test_plot_energy_overview_callable(self) -> None:
        assert callable(plot_energy_overview)


class TestKinematicsModuleImport:
    def test_plot_joint_positions_callable(self) -> None:
        assert callable(plot_joint_positions)
