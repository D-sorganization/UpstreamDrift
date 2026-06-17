# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Smoke tests for the 3D Bilateral renderer (issue #225)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import pytest

from movement_optimizer.gui.bilateral_3d_renderer import (
    draw_bilateral_3d_pose,
    is_3d_axis,
)
from movement_optimizer.models import (
    Bilateral3DModel,
    Bilateral3DPose,
    BodyModel,
)


@pytest.fixture()
def model() -> Bilateral3DModel:
    return Bilateral3DModel(BodyModel(75.0, 1.75))


def test_is_3d_axis_true_for_3d_subplot() -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert is_3d_axis(ax) is True
    plt.close(fig)


def test_is_3d_axis_false_for_2d_subplot() -> None:
    fig, ax = plt.subplots()
    assert is_3d_axis(ax) is False
    plt.close(fig)


def test_draw_t_pose_runs_without_error(model: Bilateral3DModel) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_bilateral_3d_pose(ax, model, model.t_pose(), title="T-pose test")
    # Axes limits should be nontrivial
    zlim = ax.get_zlim()
    assert zlim[1] > zlim[0]
    plt.close(fig)


def test_draw_rejects_non_model(model: Bilateral3DModel) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    with pytest.raises(TypeError, match="Bilateral3DModel"):
        draw_bilateral_3d_pose(ax, "nope", model.t_pose())  # type: ignore[arg-type]
    plt.close(fig)


def test_draw_rejects_non_pose(model: Bilateral3DModel) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    with pytest.raises(TypeError, match="Bilateral3DPose"):
        draw_bilateral_3d_pose(ax, model, (0.0, 0.0, 0.0))  # type: ignore[arg-type]
    plt.close(fig)


def test_draw_squat_pose_runs(model: Bilateral3DModel) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pose = Bilateral3DPose.from_sagittal((0.3, -1.3, 0.6))
    draw_bilateral_3d_pose(ax, model, pose)
    plt.close(fig)
