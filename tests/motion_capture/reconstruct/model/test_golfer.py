"""The scapula-capable golfer model (#9710): topology, symmetry, scapula observability."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.model import FitOptions, fit_trajectory
from src.motion_capture.reconstruct.model.golfer import (
    GOLFER_LANDMARK_MAP,
    GOLFER_SPEC,
    SIMSCAPE_NAMES,
    golfer_model,
    simscape_rows,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

pytestmark = pytest.mark.unit


def test_topology_follows_the_matlab_chain_plus_legs() -> None:
    model = golfer_model()
    parents = {j.name: j.parent for j in GOLFER_SPEC.joints}
    assert parents["spine"] == "pelvis" and parents["hub"] == "spine"
    assert (
        parents["left_scapula"] == "hub" and parents["left_shoulder"] == "left_scapula"
    )
    assert parents["left_elbow"] == "left_shoulder"
    assert (
        parents["left_forearm"] == "left_elbow"
        and parents["left_wrist"] == "left_forearm"
    )
    assert parents["left_knee"] == "left_hip" and parents["left_hip"] == "pelvis"
    by_name = {j.name: j for j in GOLFER_SPEC.joints}
    assert (
        by_name["left_scapula"].axes == "xy" and by_name["left_scapula"].length is None
    )
    assert (
        by_name["left_shoulder"].axes == "xyz"
        and by_name["left_shoulder"].length == "hub_to_shoulder"
    )
    assert by_name["hub"].axes == "y" and by_name["spine"].axes == "xz"
    assert by_name["left_elbow"].axes == "x" and by_name["left_forearm"].axes == "y"
    assert by_name["left_wrist"].axes == "xz"
    # 27 MATLAB DOFs are named; legs (2 x 4) and the head are the extension.
    assert len(SIMSCAPE_NAMES) == 27
    assert all(name in model.dof_names for name in SIMSCAPE_NAMES)
    assert model.n_dof == 27 + 2 * (3 + 1)
    # Every landmark the detectors observe maps to a reconstruct joint.
    observed = set(GOLFER_LANDMARK_MAP.to_reconstruct)
    assert observed == set(model.landmark_names)
    assert set(GOLFER_LANDMARK_MAP.to_reconstruct.values()) == set(JOINT_NAMES)


def test_rest_pose_is_symmetric_and_scapula_strut_has_matlab_length() -> None:
    model = golfer_model()
    q = np.zeros((1, model.n_dof))
    q[0, 1] = 1.0
    pos = model.forward(q)[0]
    idx = model.index
    left, right = pos[idx["left_shoulder"]], pos[idx["right_shoulder"]]
    np.testing.assert_allclose(left * [1, 1, -1], right, atol=1e-12)
    np.testing.assert_allclose(pos[idx["left_scapula"]], pos[idx["hub"]], atol=1e-12)
    assert np.linalg.norm(left - pos[idx["hub"]]) == pytest.approx(0.254)
    assert pos[idx["hub"]][1] > pos[idx["spine"]][1] > pos[idx["pelvis"]][1]
    assert (
        pos[idx["left_ankle"]][1] < pos[idx["left_knee"]][1] < pos[idx["left_hip"]][1]
    )
    np.testing.assert_allclose(
        pos[idx["left_forearm"]], pos[idx["left_elbow"]], atol=1e-12
    )


def test_scapula_elevation_is_recovered_from_shoulder_motion_alone() -> None:
    """Raise one shoulder relative to the hub: only the scapula can do that."""
    model = golfer_model()
    fps, frames = 60.0, 50
    t = np.arange(frames) / fps
    q_true = np.zeros((frames, model.n_dof))
    q_true[:, 1] = 1.0
    s = model.dof_slice("left_scapula")
    q_true[:, s.start] = 0.4 * np.sin(2 * np.pi * 0.5 * t)  # scapular elevation
    q_true[:, model.dof_slice("left_elbow")] = 0.6
    q_true[:, model.dof_slice("left_knee")] = 0.3
    q_true[:, model.dof_slice("right_knee")] = 0.3
    truth = model.landmarks(q_true)
    rng = np.random.default_rng(2)
    observed = truth + rng.normal(0, 0.002, truth.shape)
    q0 = np.zeros_like(q_true)
    q0[:, :3] = q_true[:, :3]
    fit = fit_trajectory(
        model,
        observed,
        fps,
        q0=q0,
        options=FitOptions(sigma_landmark_m=0.004, max_iterations=80),
    )
    assert fit.rms_m < 0.005
    got = fit.q[:, s.start]
    assert np.sqrt(np.mean((got - q_true[:, s.start]) ** 2)) < np.radians(4)
    assert np.abs(fit.q[:, model.dof_slice("right_scapula")]).max() < np.radians(6)
    assert fit.velocity_violations == 0


def test_simscape_rows_name_the_27_variables_in_degrees() -> None:
    model = golfer_model()
    q = np.zeros((2, model.n_dof))
    q[1, model.dof_slice("left_scapula").start] = np.radians(10.0)
    q[1, 0] = 0.5
    rows = simscape_rows(model.dof_names, q, 60.0)
    assert rows[1]["time_s"] == pytest.approx(1 / 60)
    assert rows[1]["LScapStartPositionX"] == pytest.approx(10.0)
    assert rows[1]["TranslationStartPositionX"] == pytest.approx(0.5)
    assert set(rows[0]) == {"time_s", *SIMSCAPE_NAMES.values()}
    with pytest.raises(ValueError, match="must be"):
        simscape_rows(model.dof_names, q[:, :5], 60.0)
