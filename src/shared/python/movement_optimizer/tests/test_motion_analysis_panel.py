# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the motion analysis panel and swing/chain plot renderers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from movement_optimizer.gui.plot_renderer import (
    plot_chain_curvature,
    plot_chain_energy,
    plot_chain_tension,
    plot_chain_tip_speed,
    plot_swing_angle,
    plot_swing_com_height,
    plot_swing_com_path,
    plot_swing_energy,
    plot_swing_joint_power,
    plot_swing_joint_torques,
)
from movement_optimizer.models.chain_forces import ChainForceHistory
from movement_optimizer.models.swingset import SWING_POLICY_JOINT_NAMES
from movement_optimizer.models.swingset_forces import SwingForceHistory

_T = 12
_N = 5
_NJOINTS = len(SWING_POLICY_JOINT_NAMES)


@pytest.fixture
def mock_ax():
    return MagicMock()


@pytest.fixture
def swing_history() -> SwingForceHistory:
    time = np.linspace(0.0, 1.0, _T)
    return SwingForceHistory(
        time_s=time,
        joint_torque_nm=np.ones((_T, _NJOINTS)),
        joint_power_w=np.ones((_T, _NJOINTS)),
        swing_angle_rad=np.linspace(-0.5, 0.5, _T),
        com_height_m=np.linspace(0.0, 0.3, _T),
        com_path_m=np.column_stack([np.sin(time), np.cos(time)]),
        energy_j=np.linspace(0.0, 10.0, _T),
    )


@pytest.fixture
def chain_history() -> ChainForceHistory:
    return ChainForceHistory(
        time_s=np.linspace(0.0, 1.0, _T),
        link_tension_n=np.ones((_T, _N)),
        max_tension_n=np.ones(_T),
        curvature_rad=np.zeros((_T, _N - 1)),
        max_curvature_rad=np.zeros(_T),
    )


class TestSwingPlots:
    def test_joint_torques(self, mock_ax, swing_history) -> None:
        plot_swing_joint_torques(mock_ax, swing_history)
        assert mock_ax.plot.call_count == _NJOINTS
        assert mock_ax.axhline.call_count == 1
        assert mock_ax.set_title.call_count == 1

    def test_joint_power_includes_total(self, mock_ax, swing_history) -> None:
        plot_swing_joint_power(mock_ax, swing_history)
        assert mock_ax.plot.call_count == _NJOINTS + 1  # joints + total

    def test_angle(self, mock_ax, swing_history) -> None:
        plot_swing_angle(mock_ax, swing_history)
        assert mock_ax.plot.call_count == 1
        assert mock_ax.axhline.call_count == 1

    def test_com_height(self, mock_ax, swing_history) -> None:
        plot_swing_com_height(mock_ax, swing_history)
        assert mock_ax.plot.call_count == 1

    def test_energy(self, mock_ax, swing_history) -> None:
        plot_swing_energy(mock_ax, swing_history)
        assert mock_ax.plot.call_count == 1

    def test_com_path_marks_start_and_end(self, mock_ax, swing_history) -> None:
        plot_swing_com_path(mock_ax, swing_history)
        assert mock_ax.plot.call_count == 3  # path + start + end


class TestChainPlots:
    def test_tension(self, mock_ax, chain_history) -> None:
        plot_chain_tension(mock_ax, chain_history)
        assert mock_ax.plot.call_count == 2  # max + mean
        assert mock_ax.set_title.call_count == 1

    def test_curvature(self, mock_ax, chain_history) -> None:
        plot_chain_curvature(mock_ax, chain_history)
        assert mock_ax.plot.call_count == 1

    def test_energy(self, mock_ax) -> None:
        plot_chain_energy(mock_ax, np.linspace(0, 1, _T), np.zeros(_T))
        assert mock_ax.plot.call_count == 1

    def test_tip_speed(self, mock_ax) -> None:
        plot_chain_tip_speed(mock_ax, np.linspace(0, 1, _T), np.zeros(_T))
        assert mock_ax.plot.call_count == 1


class TestMotionAnalysisPanel:
    def test_axes_keys(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["alpha", "beta", "gamma"], rows=2, cols=2)
        assert set(panel.axes) == {"alpha", "beta", "gamma"}
        assert panel.canvas is not None
        assert panel.toolbar is not None

    def test_clear_rebuilds_axes(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["a"], rows=1, cols=1)
        first = panel.axes["a"]
        panel.clear()
        assert set(panel.axes) == {"a"}
        assert panel.axes["a"] is not first

    def test_draw_runs(self, qapp, swing_history) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["torques"], rows=1, cols=1)
        plot_swing_joint_torques(panel.axes["torques"], swing_history)
        panel.draw()  # should not raise

    def test_rejects_empty_axis_names(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        with pytest.raises(ValueError, match="non-empty"):
            MotionAnalysisPanel([], rows=1, cols=1)

    def test_rejects_undersized_grid(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        with pytest.raises(ValueError, match=r"rows \* cols"):
            MotionAnalysisPanel(["a", "b", "c"], rows=1, cols=1)

    def test_rejects_nonpositive_dims(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        with pytest.raises(ValueError, match="positive"):
            MotionAnalysisPanel(["a"], rows=0, cols=1)

    def test_has_legends_reflects_axis_state(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["a", "b"], rows=1, cols=2)
        assert panel.has_legends() is False
        panel.axes["a"].plot([0, 1], [0, 1], label="series")
        panel.axes["a"].legend()
        assert panel.has_legends() is True

    def test_set_legends_visible_toggles_only_legend_bearing_axes(self, qapp) -> None:
        from movement_optimizer.gui.motion_analysis_panel import MotionAnalysisPanel

        panel = MotionAnalysisPanel(["a", "b"], rows=1, cols=2)
        panel.axes["a"].plot([0, 1], [0, 1], label="series")
        legend = panel.axes["a"].legend()

        panel.set_legends_visible(False)
        assert legend.get_visible() is False
        panel.set_legends_visible(True)
        assert legend.get_visible() is True
        # The legend-free axis is simply skipped (no error).
        assert panel.axes["b"].get_legend() is None
