"""Articulated kinematics (#9711) and the continuous fit (#9712) on hand-built chains."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.model import (
    ArticulatedModel,
    FitOptions,
    Joint,
    ModelSpec,
    fit_to_dict,
    fit_trajectory,
    wrap_angles,
)

pytestmark = pytest.mark.unit

# A trunk with a scapula link between an upper pivot and the shoulder, then
# an arm: root(3 rot) -> pivot(hinge) -> scapula(2 dof) -> shoulder(3) -> elbow(1).
ARM = ModelSpec(
    name="test_arm",
    joints=(
        Joint("pelvis", None, axes="xyz"),
        Joint("pivot", "pelvis", (0.0, 1.0, 0.0), "torso", axes="z", landmark=False),
        Joint(
            "scapula",
            "pivot",
            (0.0, 0.0, 1.0),
            "clavicle",
            axes="xy",
            limits_rad=((-0.6, 0.6), (-0.6, 0.6)),
        ),
        Joint("shoulder", "scapula", (0.0, 0.0, 1.0), "scapula", axes="xyz"),
        Joint(
            "elbow",
            "shoulder",
            (0.0, -1.0, 0.0),
            "upper_arm",
            axes="x",
            limits_rad=((0.0, 2.6),),
        ),
        Joint("wrist", "elbow", (0.0, -1.0, 0.0), "forearm", axes=""),
        Joint("hip", "pelvis", (0.0, 0.0, 1.0), "hip_half", axes="xyz"),
        Joint(
            "knee",
            "hip",
            (0.0, -1.0, 0.0),
            "thigh",
            axes="x",
            limits_rad=((0.0, 2.5),),
        ),
        Joint("ankle", "knee", (0.0, -1.0, 0.0), "shank", axes=""),
    ),
    lengths_m={
        "torso": 0.5,
        "clavicle": 0.15,
        "scapula": 0.1,
        "upper_arm": 0.3,
        "forearm": 0.26,
        "hip_half": 0.1,
        "thigh": 0.44,
        "shank": 0.42,
    },
)


def test_spec_contracts() -> None:
    with pytest.raises(Exception, match="unit"):
        Joint("a", "root", (0.0, 2.0, 0.0), "l")
    with pytest.raises(Exception, match="distinct"):
        Joint("a", None, axes="xx")
    with pytest.raises(Exception, match="precede"):
        ModelSpec("m", (Joint("r", None), Joint("b", "a", (0, 1, 0), "l")), {"l": 1})
    with pytest.raises(Exception, match="length parameter"):
        ModelSpec("m", (Joint("r", None), Joint("a", "r", (0, 1, 0), "l")), {})


def test_forward_kinematics_keeps_lengths_and_maps_dofs() -> None:
    model = ArticulatedModel(ARM)
    assert model.n_dof == 3 + 3 + 1 + 2 + 3 + 1 + 0 + 3 + 1 + 0
    assert model.dof_names[:3] == ["pelvis.tx", "pelvis.ty", "pelvis.tz"]
    assert model.landmark_names == (
        "pelvis",
        "scapula",
        "shoulder",
        "elbow",
        "wrist",
        "hip",
        "knee",
        "ankle",
    )
    q = np.zeros((4, model.n_dof))
    q[:, 0] = np.arange(4)  # root x translation
    q[:, model.dof_slice("elbow")] = np.pi / 2  # elbow flexed
    pos = model.forward(q)
    assert pos.shape == (4, 9, 3)
    idx = model.index
    np.testing.assert_allclose(pos[:, idx["pivot"], 1], 0.5)  # torso up
    np.testing.assert_allclose(pos[:, idx["pelvis"], 0], np.arange(4))
    for child, parent, length in (
        ("scapula", "pivot", 0.15),
        ("shoulder", "scapula", 0.1),
        ("elbow", "shoulder", 0.3),
        ("wrist", "elbow", 0.26),
    ):
        d = np.linalg.norm(pos[:, idx[child]] - pos[:, idx[parent]], axis=1)
        np.testing.assert_allclose(d, length, atol=1e-12)
    # A 90 degree elbow flexion about x turns the forearm's -y offset toward -z.
    forearm = pos[0, idx["wrist"]] - pos[0, idx["elbow"]]
    np.testing.assert_allclose(forearm, [0.0, 0.0, -0.26], atol=1e-12)
    lo, hi = model.limits()
    assert (
        lo[model.dof_slice("elbow")][0] == 0.0
        and hi[model.dof_slice("elbow")][0] == 2.6
    )
    assert np.isinf(lo[0])


def test_jacobian_matches_finite_difference_of_landmarks() -> None:
    model = ArticulatedModel(ARM)
    rng = np.random.default_rng(0)
    q = rng.normal(0, 0.3, (3, model.n_dof))
    jac = model.jacobian(q)
    assert jac.shape == (3, 8 * 3, model.n_dof)
    d = model.dof_slice("shoulder").start
    q2 = q.copy()
    q2[:, d] += 1e-5
    fd = (model.landmarks(q2) - model.landmarks(q)).reshape(3, -1) / 1e-5
    np.testing.assert_allclose(jac[:, :, d], fd, atol=1e-5)
    lj = model.length_jacobian(q, dict(ARM.lengths_m), ["forearm"])
    wrist_rows = slice(4 * 3, 5 * 3)
    assert lj.shape == (3, 24, 1) and np.abs(lj[:, wrist_rows, 0]).max() > 0.5
    assert wrap_angles(np.array([[0, 0, 0, 4.0]]), model)[0, 3] == pytest.approx(
        4 - 2 * np.pi
    )


def _true_motion(model: ArticulatedModel, frames: int, fps: float) -> np.ndarray:
    t = np.arange(frames) / fps
    q = np.zeros((frames, model.n_dof))
    q[:, 1] = 1.0 + 0.02 * np.sin(2 * np.pi * 0.5 * t)  # root bobs
    q[:, model.dof_slice("pelvis")] = np.column_stack(
        [0.1 * np.sin(t), 0.8 * np.sin(0.7 * t), 0.05 * np.cos(t)]
    )
    q[:, model.dof_slice("pivot")] = (0.3 * np.sin(1.3 * t))[:, None]
    q[:, model.dof_slice("scapula")] = np.column_stack(
        [0.4 * np.sin(1.1 * t), 0.3 * np.cos(0.9 * t)]
    )
    q[:, model.dof_slice("shoulder")] = np.column_stack(
        [0.9 * np.sin(1.5 * t), 0.4 * np.sin(0.8 * t), 0.5 * np.cos(1.2 * t)]
    )
    q[:, model.dof_slice("elbow")] = (1.0 + 0.8 * np.sin(1.7 * t))[:, None]
    q[:, model.dof_slice("hip")] = np.column_stack(
        [0.3 * np.sin(0.9 * t), 0.2 * np.sin(1.1 * t), 0.1 * np.cos(0.6 * t)]
    )
    q[:, model.dof_slice("knee")] = (0.4 + 0.3 * np.sin(1.4 * t))[:, None]
    return q


def test_fit_recovers_the_motion_and_rejects_a_jump() -> None:
    model = ArticulatedModel(ARM)
    fps, frames = 60.0, 90
    q_true = _true_motion(model, frames, fps)
    truth = model.landmarks(q_true)
    rng = np.random.default_rng(1)
    observed = truth + rng.normal(0, 0.003, truth.shape)
    wrist = model.landmark_names.index("wrist")
    observed[40, wrist] += [0.0, 0.25, 0.0]  # an impossible 25 cm jump for one frame
    observed[50:55, 1] = np.nan  # scapula unobserved for five frames
    options = FitOptions(
        sigma_landmark_m=0.005, max_iterations=80, max_velocity_rad_s=20.0
    )
    fit = fit_trajectory(model, observed, fps, q0=q_true * 0.0, options=options)
    assert fit.q.shape == q_true.shape and fit.rms_m < 0.006
    # The jump was rejected, not followed: model wrist stays near the truth.
    assert any(r.frame == 40 and r.landmark == "wrist" for r in fit.rejected)
    assert fit.weights[40, wrist] == 0.0
    assert np.linalg.norm(fit.landmarks_m[40, wrist] - truth[40, wrist]) < 0.02
    # Continuity: no DOF moves much faster than the true motion's own peak.
    vel = np.abs(np.diff(fit.q, axis=0)) * fps
    true_vel = np.abs(np.diff(q_true, axis=0)) * fps
    assert vel[:, 3:].max() < 2.0 * true_vel[:, 3:].max() + 1.0
    assert fit.velocity_violations == 0
    # Determined DOFs are recovered; the elbow and knee hinges within 3 degrees.
    for joint in ("elbow", "knee"):
        s = model.dof_slice(joint)
        assert np.sqrt(np.mean((fit.q[:, s] - q_true[:, s]) ** 2)) < np.radians(3)
    # The unobserved scapula frames are bridged by the prior, not left at zero.
    assert np.linalg.norm(fit.landmarks_m[52, 1] - truth[52, 1]) < 0.03
    # Elbow stays inside its limits.
    e = fit.q[:, model.dof_slice("elbow")]
    assert e.min() > -0.05 and e.max() < 2.65
    payload = fit_to_dict(fit, fps)
    assert payload["dof_names"] == list(model.dof_names) and len(payload["q"]) == frames
    with pytest.raises(Exception, match="landmark order"):
        fit_trajectory(model, observed[:, :2], fps)


def test_fit_can_refine_a_measured_length() -> None:
    model = ArticulatedModel(ARM)
    fps, frames = 60.0, 40
    q_true = _true_motion(model, frames, fps)
    truth = model.landmarks(q_true)
    wrong = {**ARM.lengths_m, "forearm": 0.24}  # tape said 0.24, truth is 0.26
    options = FitOptions(
        fit_lengths=("forearm",), sigma_length_m=0.02, max_iterations=80
    )
    fit = fit_trajectory(
        model, truth, fps, q0=q_true * 0.0, lengths_m=wrong, options=options
    )
    assert abs(fit.lengths_m["forearm"] - 0.26) < 0.01


def test_constant_pre_and_post_frames_wrap_the_moving_primitives() -> None:
    """child = parent . R(pre) . R_axes(q) . R(post), and frames() exposes it."""
    from scipy.spatial.transform import Rotation

    pre, post = (0.3, -0.2, 1.1), (-1.0, 0.4, 0.2)
    spec = ModelSpec(
        name="frames/1.0",
        joints=(
            Joint("root", None, axes=""),
            Joint("hinge", "root", (0.0, 0.0, 1.0), "seg", "x", (), True, pre, post),
            Joint("tip", "hinge", (1.0, 0.0, 0.0), "seg", ""),
        ),
        lengths_m={"seg": 0.5},
    )
    model = ArticulatedModel(spec)
    q = np.zeros((2, model.n_dof))
    q[:, model.dof_slice("hinge")] = [[0.4], [-0.9]]
    pos, frames = model.forward_frames(q)
    expected = (
        Rotation.from_rotvec(pre)
        * Rotation.from_euler("X", q[:, model.dof_slice("hinge")])
        * Rotation.from_rotvec(post)
    ).as_matrix()
    np.testing.assert_allclose(frames[:, model.index["hinge"]], expected, atol=1e-12)
    np.testing.assert_allclose(model.frames(q), frames)
    tip = pos[:, model.index["hinge"]] + 0.5 * expected[:, :, 0]
    np.testing.assert_allclose(pos[:, model.index["tip"]], tip, atol=1e-12)
    assert (
        Joint("plain", "root", (0.0, 0.0, 1.0), "seg", "x")
        .constant_frames()[0]
        .tolist()
        == np.eye(3).tolist()
    )


@pytest.mark.parametrize(
    "axes,signs", [("x", (1,)), ("xz", (1, -1)), ("zxy", (-1, 1, 1))]
)
def test_decompose_primitives_round_trips_and_reports_the_remainder(
    axes: str, signs: tuple[int, ...]
) -> None:
    from src.motion_capture.reconstruct.model.kinematics import (
        _axis_rotations,
        decompose_primitives,
    )

    rng = np.random.default_rng(3)
    q = rng.uniform(-1.2, 1.2, (5, len(axes)))
    rot = _axis_rotations(q * np.asarray(signs), axes)
    got, remainder = decompose_primitives(rot, axes, signs)
    np.testing.assert_allclose(got, q, atol=1e-9)
    np.testing.assert_allclose(remainder, 0.0, atol=1e-9)
    # A rotation the primitives cannot express leaves a remainder.
    spoiled = _axis_rotations(np.full((5, 3), 0.3), "yzx")
    _, remainder = decompose_primitives(spoiled, axes[:1])
    assert remainder.min() > 0.1


def test_gate_keeps_everything_when_the_model_cannot_represent_the_motion() -> None:
    """A rigid 0.2 m link fitted to 0.5 m-apart landmarks: every observation
    is beyond the gate, so rejecting would leave nothing. The fit keeps all,
    reports the honest RMS, and rejects nothing."""
    spec = ModelSpec(
        name="stick/1.0",
        joints=(
            Joint("root", None, axes="xyz"),
            Joint("tip", "root", (0.0, -1.0, 0.0), "link", ""),
        ),
        lengths_m={"link": 0.2},
    )
    model = ArticulatedModel(spec)
    frames = 8
    observed = np.zeros((frames, 2, 3))
    observed[:, 1, 1] = -0.5
    fit = fit_trajectory(
        model, observed, 60.0, options=FitOptions(max_iterations=30, gate=5.0)
    )
    assert fit.rejected == ()
    assert np.isfinite(fit.rms_m) and 0.1 < fit.rms_m < 0.2
