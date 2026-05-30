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
from two_stage_runner import parse_args  # noqa: E402

N = 30
J = 6


def _linear_fk(weight: torch.Tensor):
    """Forward kinematics: clubface = q @ weight (linear; analytically invertible)."""

    def fk(q: torch.Tensor) -> torch.Tensor:
        return q @ weight

    return fk


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
