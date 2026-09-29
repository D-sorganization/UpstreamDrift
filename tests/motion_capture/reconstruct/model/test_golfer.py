"""The scapula-capable golfer (#9710, #9714): topology in Simscape frames,
geometry, scapula observability, export round trip, and replay of the
GolfSwing3D_Kinetic logs through the model."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.motion_capture.reconstruct.model import FitOptions, fit_trajectory
from src.motion_capture.reconstruct.model.golfer import (
    GOLFER_LANDMARK_MAP,
    GOLFER_SPEC,
    HUB_LENGTH_M,
    SIMSCAPE_LOG_COLUMNS,
    SIMSCAPE_NAMES,
    SIMSCAPE_SENSORS,
    WORLD_SIMSCAPE_FROM_OURS,
    golfer_model,
    simscape_angles_to_q,
    simscape_rows,
    simscape_variable_names,
)
from src.motion_capture.reconstruct.model.simscape import load_trials
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

pytestmark = pytest.mark.unit

DATASETS = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "Scripts"
    / "Dataset Generator"
)


def test_topology_follows_the_simscape_chain_plus_legs() -> None:
    model = golfer_model()
    parents = {j.name: j.parent for j in GOLFER_SPEC.joints}
    assert parents["torso"] == "pelvis" and parents["spine"] == "torso"
    assert parents["hub"] == "spine"
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
    assert by_name["torso"].axes == "z" and by_name["spine"].axes == "zx"
    assert by_name["hub"].axes == "" and by_name["left_scapula"].length is None
    assert (
        by_name["left_scapula"].axes == "xz" and by_name["right_scapula"].axes == "yz"
    )
    assert by_name["left_shoulder"].axes == "xyz"
    assert by_name["right_shoulder"].axes == "yzx"
    assert by_name["left_shoulder"].length == "hub_to_shoulder"
    assert by_name["left_elbow"].axes == "y" and by_name["left_forearm"].axes == "x"
    assert by_name["right_elbow"].axes == "x" and by_name["right_forearm"].axes == "y"
    assert by_name["left_wrist"].axes == "xz"
    # 27 MATLAB DOFs are named; legs (2 x 4) and the head are the extension.
    assert len(SIMSCAPE_NAMES) == 27
    assert all(name in model.dof_names for name in SIMSCAPE_NAMES)
    assert len({n for n, _ in SIMSCAPE_NAMES.values()}) == 27
    assert model.n_dof == 27 + 2 * (3 + 1)
    # Every landmark the detectors observe maps to a reconstruct joint.
    observed = set(GOLFER_LANDMARK_MAP.to_reconstruct)
    assert observed == set(model.landmark_names)
    assert set(GOLFER_LANDMARK_MAP.to_reconstruct.values()) == set(JOINT_NAMES)


def test_zero_pose_geometry_matches_the_simscape_offsets() -> None:
    model = golfer_model()
    q = np.zeros((1, model.n_dof))
    q[0, 1] = 1.0
    pos = model.forward(q)[0]
    idx = model.index

    def dist(a: str, b: str) -> float:
        return float(np.linalg.norm(pos[idx[a]] - pos[idx[b]]))

    np.testing.assert_allclose(pos[idx["torso"]], pos[idx["pelvis"]], atol=1e-12)
    assert dist("spine", "torso") == pytest.approx(0.061)
    assert dist("hub", "spine") == pytest.approx(HUB_LENGTH_M)
    assert pytest.approx(0.24904, abs=1e-5) == HUB_LENGTH_M
    for side in ("left", "right"):
        np.testing.assert_allclose(
            pos[idx[f"{side}_scapula"]], pos[idx["hub"]], atol=1e-12
        )
        assert dist(f"{side}_shoulder", "hub") == pytest.approx(0.254)
        assert dist(f"{side}_elbow", f"{side}_shoulder") == pytest.approx(0.3047)
        np.testing.assert_allclose(
            pos[idx[f"{side}_forearm"]], pos[idx[f"{side}_elbow"]], atol=1e-12
        )
    # Legs hang opposite to the trunk and hips sit symmetrically about the pelvis.
    trunk = pos[idx["spine"]] - pos[idx["pelvis"]]
    assert np.dot(pos[idx["left_knee"]] - pos[idx["left_hip"]], trunk) < 0
    np.testing.assert_allclose(
        pos[idx["left_hip"]] - pos[idx["pelvis"]],
        -(pos[idx["right_hip"]] - pos[idx["pelvis"]]),
        atol=1e-12,
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
    # Simscape scapula axes: 2 mm landmark noise on a 0.254 m strut plus the
    # smoothness prior leave about 4 degrees of RMS on a 23-degree swing.
    assert np.sqrt(np.mean((got - q_true[:, s.start]) ** 2)) < np.radians(5)
    # Trunk twist with both scapulae counter-rotating leaves every shoulder in
    # place, so only the weak rest prior settles that share; a few degrees of
    # right-scapula motion is that gauge, not a fit failure.
    assert np.abs(fit.q[:, model.dof_slice("right_scapula")]).max() < np.radians(8)
    assert fit.velocity_violations == 0


def test_simscape_rows_apply_signs_and_reaxe_translations() -> None:
    model = golfer_model()
    q = np.zeros((2, model.n_dof))
    col = {n: i for i, n in enumerate(model.dof_names)}
    q[1, col["left_scapula.rx"]] = np.radians(10.0)
    q[1, col["left_scapula.rz"]] = np.radians(10.0)  # sign -1 in Simscape
    q[1, 0], q[1, 1], q[1, 2] = 0.5, 0.7, 0.2  # ours: x target, y up, z right
    rows = simscape_rows(model.dof_names, q, 60.0)
    assert rows[1]["time_s"] == pytest.approx(1 / 60)
    assert rows[1]["LScapStartPositionX"] == pytest.approx(10.0)
    assert rows[1]["LScapStartPositionY"] == pytest.approx(-10.0)
    assert rows[1]["TranslationStartPositionX"] == pytest.approx(0.5)
    assert rows[1]["TranslationStartPositionY"] == pytest.approx(-0.2)
    assert rows[1]["TranslationStartPositionZ"] == pytest.approx(0.7)
    assert set(rows[0]) == {"time_s", *(n for n, _ in SIMSCAPE_NAMES.values())}
    assert simscape_variable_names()["left_scapula.rz"] == "LScapStartPositionY"
    with pytest.raises(ValueError, match="must be"):
        simscape_rows(model.dof_names, q[:, :5], 60.0)


def test_simscape_angles_round_trip_through_the_export() -> None:
    model = golfer_model()
    rng = np.random.default_rng(5)
    q = np.zeros((4, model.n_dof))
    col = {n: i for i, n in enumerate(model.dof_names)}
    for ours in SIMSCAPE_NAMES:
        q[:, col[ours]] = rng.uniform(-1.0, 1.0, 4)
    rows = simscape_rows(model.dof_names, q, 60.0)
    exported = {
        name: np.array([r[name] for r in rows]) for name in rows[0] if name != "time_s"
    }
    # Translations come back through the same world map.
    back = simscape_angles_to_q(model, exported)
    back[:, :3] = (
        np.column_stack([exported[f"TranslationStartPosition{a}"] for a in "XYZ"])
        @ WORLD_SIMSCAPE_FROM_OURS
    )
    for ours in SIMSCAPE_NAMES:
        np.testing.assert_allclose(back[:, col[ours]], q[:, col[ours]], atol=1e-12)


@pytest.mark.slow
@pytest.mark.skipif(
    not DATASETS.is_dir(), reason="dataset generator logs not checked out"
)
def test_logged_simscape_angles_replay_to_the_logged_sensor_frames() -> None:
    """Drive the golfer with GolfSwing3D_Kinetic's own angles: its body frames
    and joint positions must land on the logged sensors (#9714)."""
    files = sorted(DATASETS.glob("golf_swing_dataset_*/trial_*.csv"))[:3]
    frames = load_trials(files)
    rows: list[dict[str, str]] = []
    for f in files:
        with f.open(encoding="utf-8") as handle:
            rows += list(csv.DictReader(handle))
    angles = {
        name: np.array([float(r[column]) for r in rows])
        for name, column in SIMSCAPE_LOG_COLUMNS.items()
    }
    model = golfer_model()
    q = simscape_angles_to_q(model, angles)
    pos, body = model.forward_frames(q)
    world = WORLD_SIMSCAPE_FROM_OURS
    for ours, sensor in SIMSCAPE_SENSORS.items():
        ours_in_simscape = np.einsum("ij,tjk->tik", world, body[:, model.index[ours]])
        rel = np.einsum("tji,tjk->tik", ours_in_simscape, frames.rotation[sensor])
        # The logs are rounded (rows are orthonormal only to ~5e-3), so the
        # residual is measured after re-orthonormalisation.
        assert Rotation.from_matrix(rel).magnitude().max() < 3e-3, (ours, sensor)

    def relative(name: str) -> np.ndarray:
        return np.einsum(
            "ij,tj->ti",
            world,
            pos[:, model.index[name]] - pos[:, model.index["pelvis"]],
        )

    torso = frames.position["Torso"]
    for ours, sensor in (("spine", "Spine"), ("left_shoulder", "LS")):
        assert np.abs(relative(ours) - (frames.position[sensor] - torso)).max() < 2e-3
    elbow = frames.position["LF"] - 0.1778 * frames.rotation["LF"][:, :, 2] - torso
    assert np.abs(relative("left_elbow") - elbow).max() < 2e-3
