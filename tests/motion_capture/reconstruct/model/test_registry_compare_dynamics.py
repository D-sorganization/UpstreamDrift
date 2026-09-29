"""Model registry and pendulums (#9730), model comparison (#9731), kinetics (#9714)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.model import (
    ArticulatedModel,
    FitOptions,
    fit_trajectory,
)
from src.motion_capture.reconstruct.model.compare import (
    compare_models,
    dof_penalised_score,
    load_comparison,
)
from src.motion_capture.reconstruct.model.dynamics import (
    forward_replay,
    gravity_torque,
    inertia,
    inverse_dynamics,
    kinetics_report,
    segment_masses,
)
from src.motion_capture.reconstruct.model.registry import (
    DEFAULT_MODEL,
    RegisteredModel,
    get_model,
    model_names,
    register_model,
)
from src.motion_capture.reconstruct.model.session import LandmarkMap
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from tests.motion_capture.reconstruct.model.test_integration import _golfer_session

pytestmark = pytest.mark.unit


def test_registry_has_the_golfer_as_default_and_the_pendulums() -> None:
    assert DEFAULT_MODEL == "golfer"
    assert {"golfer", "double_pendulum", "triple_pendulum"} <= set(model_names())
    double = get_model("double_pendulum")
    model = ArticulatedModel(double.spec)
    assert model.n_dof == 3 + 3 and model.landmark_names == ("pivot", "hands")
    triple = ArticulatedModel(get_model("triple_pendulum").spec)
    assert triple.n_dof == 3 + 3 + 1
    with pytest.raises(Exception, match="unknown model"):
        get_model("quadruple")
    with pytest.raises(Exception, match="learnable"):
        RegisteredModel("x", double.spec, double.landmark_map, ("nope",), "")
    with pytest.raises(Exception, match="already"):
        register_model(double)


def test_midpoint_landmark_sources_average_both_sides() -> None:
    joints = np.zeros((2, len(JOINT_NAMES), 3))
    joints[:, JOINT_NAMES.index("left_wrist")] = [1.0, 0.0, 0.0]
    joints[:, JOINT_NAMES.index("right_wrist")] = [3.0, 0.0, 0.0]
    model = ArticulatedModel(get_model("double_pendulum").spec)
    observed = get_model("double_pendulum").landmark_map.observed(model, joints)
    np.testing.assert_allclose(observed[:, 1], [[2.0, 0.0, 0.0]] * 2)
    with pytest.raises(Exception, match="reconstruct joints"):
        LandmarkMap({"a": ("left_wrist", "sacrum")})


def test_pendulums_fit_a_planar_swing_and_learn_the_arm_length() -> None:
    reg = get_model("double_pendulum")
    model = ArticulatedModel(reg.spec)
    fps, frames = 60.0, 60
    t = np.arange(frames) / fps
    q_true = np.zeros((frames, model.n_dof))
    q_true[:, 1] = 1.4
    pivot = model.dof_slice("pivot")
    q_true[:, pivot.start] = 0.2  # a tilted swing plane
    q_true[:, pivot.stop - 1] = 1.2 * np.sin(2.0 * t)  # the arm angle in that plane
    truth = model.landmarks(q_true, {"arm": 0.66})
    fit = fit_trajectory(
        model,
        truth,
        fps,
        q0=q_true * 0.0 + q_true[:, :3].sum() * 0,
        lengths_m={"arm": 0.60},  # wrong prior length
        options=FitOptions(
            fit_lengths=("arm",),
            sigma_length_m=0.1,
            sigma_accel_rad_s2=500,
            max_iterations=80,
        ),
    )
    assert fit.rms_m < 0.005
    assert abs(fit.lengths_m["arm"] - 0.66) < 0.01
    arm = pivot.stop - 1
    assert np.sqrt(np.mean((fit.q[:, arm] - q_true[:, arm]) ** 2)) < 0.05


def test_compare_models_ranks_and_writes_reports(tmp_path: Path) -> None:
    root = _golfer_session(tmp_path)
    report = compare_models(
        root, ["golfer", "double_pendulum"], options=FitOptions(max_iterations=40)
    )
    names = [s.model for s in report.ranking]
    assert set(names) == {"golfer", "double_pendulum"}
    assert report.ranking[0].score <= report.ranking[1].score
    golfer = next(s for s in report.ranking if s.model == "golfer")
    assert golfer.landmarks == 15 and golfer.dof == 35
    assert (root / "model" / "golfer" / "fit_report.json").is_file()
    assert (root / "model" / "double_pendulum" / "joint_angles.json").is_file()
    assert (
        (root / "model" / "comparison.md")
        .read_text(encoding="utf-8")
        .startswith("| model")
    )
    loaded = load_comparison(root)
    assert loaded is not None and len(loaded["ranking"]) == 2
    assert dof_penalised_score(0.01, 10, 100, 5) < dof_penalised_score(0.01, 30, 100, 5)
    with pytest.raises(Exception, match="positive"):
        dof_penalised_score(0.0, 1, 1, 1)


def _pendulum_true_motion() -> tuple[ArticulatedModel, np.ndarray, float]:
    """A free double pendulum hanging under gravity, swung by a known torque history."""
    model = ArticulatedModel(get_model("triple_pendulum").spec)
    fps, frames = 120.0, 120
    t = np.arange(frames) / fps
    q = np.zeros((frames, model.n_dof))
    q[:, 1] = 1.4
    q[:, model.dof_slice("elbows")] = (0.5 * np.sin(3.0 * t))[:, None]
    q[:, model.dof_slice("hands")] = (0.3 * np.cos(2.0 * t))[:, None]
    return model, q, fps


def test_inverse_dynamics_matches_gravity_statics_and_replays_the_motion() -> None:
    model, q, fps = _pendulum_true_motion()
    seg = segment_masses(model, 80.0)
    assert sum(seg.mass_kg) == pytest.approx(80.0 * (0.027 + 0.022))
    # Static hanging pendulum: torque about the hinge equals the gravity moment.
    still = np.repeat(q[:1], 5, axis=0)
    still[:, model.dof_slice("elbows")] = np.pi / 2  # forearm horizontal
    g = gravity_torque(model, still, seg)
    hinge = model.dof_slice("elbows").start
    fore = get_model("triple_pendulum").spec.lengths_m["forearm"]
    expected = 9.81 * seg.mass_kg[1] * fore / 2  # the forearm's own moment
    assert abs(g[0, hinge]) == pytest.approx(expected, rel=1e-3)
    m_q = inertia(model, still, seg)
    assert m_q.shape == (5, model.n_dof, model.n_dof)
    np.testing.assert_allclose(m_q[0], m_q[0].T, atol=1e-9)
    parts = inverse_dynamics(model, q, fps, seg)
    assert parts["tau"].shape == q.shape
    replay = forward_replay(model, q, fps, seg, parts["tau"])
    assert replay["max_abs_error"] < 0.05  # radians over one second of motion
    with pytest.raises(Exception, match="q shape"):
        inverse_dynamics(model, q[:2], fps, seg)


def test_kinetics_report_is_json_ready(tmp_path: Path) -> None:
    model, q, fps = _pendulum_true_motion()
    report = kinetics_report(model, q, fps, body_mass_kg=80.0, names={"a": "b"})
    text = json.dumps(report)
    assert (
        report["replay_max_abs_error"] < 0.05
        and "point masses" in report["assumptions"]
    )
    assert set(report["peak_torque"]) == set(model.dof_names)
    (tmp_path / "k.json").write_text(text, encoding="utf-8")
