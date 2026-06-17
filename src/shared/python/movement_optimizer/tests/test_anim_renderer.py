# Copyright (c) 2026 D-Sorganization. All rights reserved.
from unittest.mock import MagicMock

import numpy as np
import pytest

from movement_optimizer.gui.anim_renderer import draw_anim_frame
from movement_optimizer.models import BodyModel
from movement_optimizer.trajectory import OptimizationResult


@pytest.fixture
def mock_ax():
    return MagicMock()


@pytest.fixture
def mock_dynamics():
    dyn = MagicMock()
    dyn.forward_kinematics.return_value = {
        "ankle": np.array([0.0, 0.0]),
        "knee": np.array([0.0, 0.5]),
        "hip": np.array([0.0, 1.0]),
        "shoulder": np.array([0.0, 1.5]),
    }
    dyn.bar_position.return_value = np.array([0.0, 1.5])
    return dyn


@pytest.fixture
def dummy_result():
    t = np.linspace(0, 1, 10)
    q = np.zeros((10, 3))
    qd = np.zeros((10, 3))
    qdd = np.zeros((10, 3))
    torques = np.zeros((10, 3))
    power = np.zeros((10, 3))
    com = np.zeros((10, 2))
    bar = np.zeros((10, 2))
    return OptimizationResult(
        t=t,
        q=q,
        qd=qd,
        qdd=qdd,
        torques=torques,
        power=power,
        com=com,
        bar=bar,
        success=True,
        cost=0.0,
        com_horizontal_range_cm=0.0,
        elapsed_s=0.1,
        n_evals=100,
        n_joint_limit_violations=0,
    )


@pytest.fixture
def body():
    return BodyModel(body_mass=80.0, height=1.8)


class TestAnimRenderer:
    def test_draw_anim_frame_standing(self, mock_ax, mock_dynamics, dummy_result, body):
        draw_anim_frame(
            mock_ax,
            0,
            dummy_result,
            mock_dynamics,
            body,
            "Squat",
            "squat",
        )
        mock_ax.clear.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_ax.set_xlim.assert_called_once()
        mock_ax.set_ylim.assert_called_once()
        # BarbellRenderer, BodyRenderer should be called.
        # But we don't strictly assert internal methods of BodyRenderer unless patched.

    def test_draw_anim_frame_deadlift(self, mock_ax, mock_dynamics, dummy_result, body):
        draw_anim_frame(
            mock_ax,
            2,
            dummy_result,
            mock_dynamics,
            body,
            "Deadlift",
            "deadlift",
        )
        mock_ax.clear.assert_called_once()
        mock_ax.set_title.assert_called_once()

    def test_draw_anim_frame_bench_press(self, mock_ax, mock_dynamics, dummy_result, body):
        draw_anim_frame(
            mock_ax,
            5,
            dummy_result,
            mock_dynamics,
            body,
            "Bench Press",
            "bench_press",
        )
        mock_ax.clear.assert_called_once()
        mock_ax.set_title.assert_called_once()
        # Check bench elements
        assert mock_ax.fill_between.call_count >= 1
