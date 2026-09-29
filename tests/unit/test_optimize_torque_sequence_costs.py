"""Unit tests for orientation + total-work cost terms (issue #4045).

Covers the new cost helpers in
``src/engines/.../MachineLearning/optimize_torque_sequence_for_club.py``:
- quaternion-geodesic orientation term (sign-invariant, zero on match)
- total-work regularizer (reduces torque magnitude vs unregularized baseline)
- ``--cost-mode`` CLI argument
- legacy ``cost-mode position`` parity with the original PR #3966 cost
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
)
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

otsc = importlib.import_module("optimize_torque_sequence_for_club")


# ---------------------------------------------------------------------------
# Orientation term
# ---------------------------------------------------------------------------


def test_orientation_term_zero_when_quaternion_matches() -> None:
    n = 20
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)).astype(np.float64)
    assert otsc.quaternion_orientation_term_numpy(q, q) == pytest.approx(0.0, abs=1e-12)

    q_t = torch.as_tensor(q, dtype=torch.float64)
    val = otsc.quaternion_orientation_term(q_t, q_t)
    assert float(val) == pytest.approx(0.0, abs=1e-12)


def test_orientation_term_zero_when_quat_equals_neg_quat() -> None:
    n = 20
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((n, 4))
    q = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    val_np = otsc.quaternion_orientation_term_numpy(q, -q)
    assert val_np == pytest.approx(0.0, abs=1e-10)

    q_t = torch.as_tensor(q, dtype=torch.float64)
    val_t = otsc.quaternion_orientation_term(q_t, -q_t)
    assert float(val_t) == pytest.approx(0.0, abs=1e-10)


def test_orientation_term_positive_for_distinct_quaternions() -> None:
    n = 4
    q1 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)).astype(np.float64)
    # 90 deg rotation about x-axis
    q2 = np.tile(
        np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]), (n, 1)
    ).astype(np.float64)
    val = otsc.quaternion_orientation_term_numpy(q1, q2)
    assert val == pytest.approx((np.pi / 2) ** 2, rel=1e-3)


# ---------------------------------------------------------------------------
# Total-work regularizer
# ---------------------------------------------------------------------------


def test_total_work_regularizer_reduces_torque_magnitude() -> None:
    """A time-varying target produces smaller torques when work-regularised."""
    n_steps, n_joints = 20, 4
    time = torch.linspace(0.0, 0.2, n_steps, dtype=torch.float64)
    # Target: a triangle-wave-ish tau(t) so the unregularised optimum has
    # nonzero omega and thus nonzero work. The regulariser should pull the
    # solution toward a flatter (and therefore lower-magnitude) profile.
    target = torch.zeros((n_steps, n_joints), dtype=torch.float64)
    target[:, 0] = torch.linspace(-1.0, 1.0, n_steps, dtype=torch.float64)
    target[:, 1] = torch.linspace(1.0, -1.0, n_steps, dtype=torch.float64)
    target[:, 2] = torch.sin(torch.linspace(0.0, 3.14, n_steps, dtype=torch.float64))
    target[:, 3] = torch.cos(torch.linspace(0.0, 3.14, n_steps, dtype=torch.float64))

    def run(lambda_: float) -> float:
        torch.manual_seed(0)
        tau = torch.nn.Parameter(torch.zeros((n_steps, n_joints), dtype=torch.float64))
        opt = torch.optim.Adam([tau], lr=5e-2)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            track = torch.mean((tau - target) ** 2)
            dt = (time[1:] - time[:-1]).clamp(min=1e-9)
            omega = (tau[1:] - tau[:-1]) / dt.unsqueeze(1)
            tau_mid = 0.5 * (tau[1:] + tau[:-1])
            t_mid = 0.5 * (time[1:] + time[:-1])
            work = otsc.total_work_regularizer(tau_mid, omega, t_mid)
            loss = track + lambda_ * work
            loss.backward()
            opt.step()
        return float(tau.detach().abs().mean())

    base = run(0.0)
    reg = run(0.5)
    assert reg < base, (
        f"expected regularised mean |tau| ({reg}) < unregularised ({base})"
    )


def test_total_work_numpy_matches_shared_helper() -> None:
    rng = np.random.default_rng(1)
    n = 20
    time = np.linspace(0.0, 0.2, n)
    tau = rng.standard_normal((n, 4))
    omega = rng.standard_normal((n, 4))
    expected = float(np.trapezoid(np.sum(np.abs(tau * omega), axis=1), time))
    got = otsc.total_work_numpy(tau, omega, time)
    assert got == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Cost-mode resolution
# ---------------------------------------------------------------------------


def test_cost_mode_position_only_matches_legacy() -> None:
    """`cost_mode=position` keeps the legacy effort+smoothness regularizer."""
    cfg = otsc.resolve_cost_config("position")
    assert cfg.mode == "position"
    assert cfg.regularizer_kind == "effort_smoothness"
    assert cfg.effort_weight == 1e-6
    assert cfg.smoothness_weight == 1e-4


def test_cost_mode_position_orientation_includes_orientation_term() -> None:
    cfg = otsc.resolve_cost_config("position_orientation")
    assert cfg.mode == "position_orientation"
    assert cfg.regularizer_kind == "effort_smoothness"
    assert cfg.orientation_weight == otsc.DEFAULT_ORIENTATION_WEIGHT


def test_cost_mode_full_uses_total_work_regularizer() -> None:
    cfg = otsc.resolve_cost_config("full")
    assert cfg.mode == "full"
    assert cfg.regularizer_kind == "total_work"
    assert cfg.lambda_ == otsc.DEFAULT_LAMBDA


def test_cost_mode_full_with_explicit_regularizer_override() -> None:
    cfg = otsc.resolve_cost_config("full", regularizer_kind="effort_smoothness")
    assert cfg.regularizer_kind == "effort_smoothness"


def test_cost_mode_invalid_raises() -> None:
    with pytest.raises(ValueError):
        otsc.resolve_cost_config("nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_argument_parses() -> None:
    parser = otsc.build_argument_parser()
    ns = parser.parse_args(
        [
            "--desired-club-csv",
            "a.csv",
            "--reference-body-csv",
            "b.csv",
            "--cost-mode",
            "full",
            "--orientation-weight",
            "0.25",
            "--lambda",
            "1e-3",
        ]
    )
    assert ns.cost_mode == "full"
    assert ns.orientation_weight == 0.25
    assert ns.lambda_ == 1e-3
    assert ns.regularizer_kind is None  # default-resolved later


def test_cli_default_cost_mode_is_position_for_backwards_compat() -> None:
    parser = otsc.build_argument_parser()
    ns = parser.parse_args(
        ["--desired-club-csv", "a.csv", "--reference-body-csv", "b.csv"]
    )
    assert ns.cost_mode == "position"
