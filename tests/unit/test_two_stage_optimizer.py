"""Unit tests for the dynamics-consistent two-stage trajectory optimizer.

Synthetic surrogates are used so the tests run on CPU in <1s and require no
trained checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

ML_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
)
sys.path.insert(0, str(ML_DIR))

from two_stage_optimizer import (  # noqa: E402
    run_two_stage,
    stage_a_optimize_kinematics,
    stage_b_optimize_torques,
)
from optimize_body_kinematics_for_club import optimize_body_kinematics  # noqa: E402
from optimize_torques_for_desired_kinematics import optimize_torques  # noqa: E402
from train_dynamics_surrogate import DynamicsMLP  # noqa: E402
from two_stage_runner import _build_surrogate_callable, parse_args  # noqa: E402

N = 30
J = 6


class _UnsafePayload:
    def __reduce__(self):
        return (eval, ("'unsafe'",))


def _linear_fk(weight: torch.Tensor):
    """Forward kinematics: clubface = q @ weight (linear; analytically invertible)."""

    def fk(q: torch.Tensor) -> torch.Tensor:
        return q @ weight

    return fk


def _write_surrogate_checkpoint(path: Path) -> None:
    model = DynamicsMLP(input_dim=1, output_dim=1, hidden_sizes=[4])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_columns": ["q0"],
            "target_columns": ["club_x"],
            "x_mean": [0.0],
            "x_std": [1.0],
            "y_mean": [0.0],
            "y_std": [1.0],
            "config": {"hidden_sizes": [4]},
        },
        path,
    )


def _linear_surrogate(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Surrogate: q_dot_next = q @ A + q_dot @ B + tau @ C."""

    def f(q: torch.Tensor, qd: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return q @ a + qd @ b + tau @ c

    return f


@pytest.fixture
def torch_seed() -> None:
    torch.manual_seed(0)


def test_stage_a_recovers_q_from_synthetic_club_trajectory(torch_seed: None) -> None:
    weight = torch.eye(J)[:, :3]  # club is first 3 joints
    fk = _linear_fk(weight)
    q_true = torch.randn(N, J) * 0.1
    target = fk(q_true)
    q_rest = torch.zeros(1, J)

    result = stage_a_optimize_kinematics(
        forward_kinematics=fk,
        clubface_target=target,
        q_rest=q_rest,
        motion_weight=1e-6,
        steps=400,
        learning_rate=5e-2,
    )
    residual = torch.mean((fk(result.q) - target) ** 2).item()
    assert residual < 1e-4
    assert result.final_loss < 1e-3


def test_stage_b_recovers_torques_from_known_q_qdot(torch_seed: None) -> None:
    a = torch.zeros(J, J)
    b = torch.zeros(J, J)
    c = torch.eye(J)  # q_dot_next == tau
    surrogate = _linear_surrogate(a, b, c)
    q = torch.randn(N, J)
    qd = torch.randn(N, J)
    tau_true = torch.randn(N, J) * 0.3
    qd_target = surrogate(q, qd, tau_true)

    result = stage_b_optimize_torques(
        surrogate=surrogate,
        q=q,
        q_dot=qd,
        q_dot_target=qd_target,
        smooth_weight=1e-8,
        steps=400,
        learning_rate=5e-2,
    )
    err = torch.mean((result.tau - tau_true) ** 2).item()
    assert err < 1e-2


def test_two_stage_pipeline_round_trip(torch_seed: None) -> None:
    weight = torch.eye(J)[:, :3]
    fk = _linear_fk(weight)
    a = torch.zeros(J, J)
    b = torch.zeros(J, J)
    c = torch.eye(J)
    surrogate = _linear_surrogate(a, b, c)

    q_true = torch.randn(N, J) * 0.05
    target = fk(q_true)
    q_rest = torch.zeros(1, J)

    out = run_two_stage(
        forward_kinematics=fk,
        surrogate=surrogate,
        clubface_target=target,
        q_rest=q_rest,
        dt=1.0 / 60.0,
        stage_a_steps=300,
        stage_b_steps=200,
        motion_weight=1e-5,
        smooth_weight=1e-6,
        learning_rate=5e-2,
    )
    tracking = torch.mean((fk(out.stage_a.q) - target) ** 2).item()
    assert tracking < 1e-3
    assert out.stage_b.tau.shape == (N, J)
    assert out.stage_b.final_loss < 1.0
    # Stage B residual should at least improve over the zero-torque baseline.
    history_first = out.stage_b.history[0]["tracking_loss"]
    history_last = out.stage_b.history[-1]["tracking_loss"]
    assert history_last <= history_first


def test_continuity_enforced_at_start(torch_seed: None) -> None:
    weight = torch.eye(J)[:, :3]
    fk = _linear_fk(weight)
    q_true = torch.randn(N, J) * 0.1
    target = fk(q_true)
    q_rest = torch.zeros(1, J)
    initial = torch.full((J,), 0.42)

    result = stage_a_optimize_kinematics(
        forward_kinematics=fk,
        clubface_target=target,
        q_rest=q_rest,
        initial_state=initial,
        motion_weight=1e-5,
        steps=50,
        learning_rate=1e-2,
    )
    assert torch.allclose(result.q[0], initial, atol=1e-6)

    # Stage B: tau initialized at zeros so tau[0] is continuous with tau_init[0].
    a = torch.zeros(J, J)
    b = torch.zeros(J, J)
    c = torch.eye(J)
    surrogate = _linear_surrogate(a, b, c)
    qd = torch.randn(N, J)
    qd_target = qd.clone()
    tau_init = torch.zeros(N, J)
    sb = stage_b_optimize_torques(
        surrogate=surrogate,
        q=torch.zeros(N, J),
        q_dot=qd,
        q_dot_target=qd_target,
        tau_init=tau_init,
        steps=20,
        learning_rate=1e-3,
    )
    # First-step torque should remain close to its initial (continuous start).
    assert torch.linalg.norm(sb.tau[0]) < 1.0


def test_runner_cli_args_parse() -> None:
    args = parse_args(
        [
            "--body-checkpoint",
            "body.pt",
            "--dynamics-checkpoint",
            "dyn.pt",
            "--desired-club-csv",
            "club.csv",
            "--reference-body-csv",
            "ref.csv",
            "--stage-a-steps",
            "10",
            "--stage-b-steps",
            "5",
            "--motion-weight",
            "0.001",
            "--smooth-weight",
            "0.0001",
            "--device",
            "cpu",
        ]
    )
    assert args.body_checkpoint == Path("body.pt")
    assert args.dynamics_checkpoint == Path("dyn.pt")
    assert args.stage_a_steps == 10
    assert args.stage_b_steps == 5
    assert args.motion_weight == pytest.approx(0.001)
    assert args.smooth_weight == pytest.approx(0.0001)
    assert args.device == "cpu"


@pytest.mark.unit
def test_two_stage_runner_loads_safe_surrogate_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "safe.pt"
    _write_surrogate_checkpoint(checkpoint)

    model, input_columns, target_columns = _build_surrogate_callable(checkpoint, "cpu")

    assert input_columns == ["q0"]
    assert target_columns == ["club_x"]
    with torch.no_grad():
        prediction = model(torch.zeros(1, 1))
    assert prediction.shape == (1, 1)


@pytest.mark.unit
def test_two_stage_runner_rejects_unsafe_checkpoint_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"payload": _UnsafePayload()}, checkpoint)

    with pytest.raises(ValueError, match="cannot be loaded safely"):
        _build_surrogate_callable(checkpoint, "cpu")


@pytest.mark.unit
def test_body_kinematics_optimizer_rejects_unsafe_checkpoint_payload(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"payload": _UnsafePayload()}, checkpoint)

    with pytest.raises(ValueError, match="cannot be loaded safely"):
        optimize_body_kinematics(
            checkpoint_path=checkpoint,
            reference_body_path=tmp_path / "reference.json",
            desired_club_path=tmp_path / "desired.json",
            output_path=tmp_path / "body_targets.json",
            steps=1,
            learning_rate=1e-2,
            motion_weight=1e-5,
            acceleration_weight=1e-8,
            device_name="cpu",
        )


@pytest.mark.unit
def test_torque_optimizer_rejects_unsafe_checkpoint_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"payload": _UnsafePayload()}, checkpoint)

    with pytest.raises(ValueError, match="cannot be loaded safely"):
        optimize_torques(
            checkpoint_path=checkpoint,
            current_state_path=tmp_path / "current.json",
            desired_state_path=tmp_path / "desired.json",
            output_path=tmp_path / "torques.json",
            initial_controls_path=None,
            steps=1,
            learning_rate=1e-2,
            effort_weight=1e-6,
            device_name="cpu",
        )


def test_stage_a_rejects_negative_motion_weight() -> None:
    with pytest.raises(ValueError, match="motion_weight"):
        stage_a_optimize_kinematics(
            forward_kinematics=lambda q: q,
            clubface_target=torch.zeros(N, J),
            q_rest=torch.zeros(1, J),
            motion_weight=-0.1,
        )


def test_stage_b_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must all match"):
        stage_b_optimize_torques(
            surrogate=lambda q, qd, tau: q,
            q=torch.zeros(N, J),
            q_dot=torch.zeros(N, J),
            q_dot_target=torch.zeros(N, J + 1),
        )
