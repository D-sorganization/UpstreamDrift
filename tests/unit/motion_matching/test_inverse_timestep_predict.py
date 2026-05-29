"""Coverage tests for :mod:`motion_matching.inverse_timestep.predict`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from src.shared.python.motion_matching.inverse_timestep.model import (
    TimestepInverseConfig,
    TimestepInverseDynamics,
)
from src.shared.python.motion_matching.inverse_timestep.predict import (
    load_with_stats,
    predict_torques,
)


def _model(
    n_joints: int = 5, hidden: int = 16, n_blocks: int = 1
) -> TimestepInverseDynamics:
    """Tiny model for fast tests."""
    cfg = TimestepInverseConfig(
        input_dim=3 * n_joints,
        output_dim=n_joints,
        hidden=hidden,
        n_blocks=n_blocks,
        dropout=0.0,
        use_missing_indicator=False,
    )
    return TimestepInverseDynamics(cfg)


def _stats(n_joints: int = 5):
    state = {
        "mean": np.zeros(3 * n_joints, dtype=np.float32),
        "std": np.ones(3 * n_joints, dtype=np.float32),
    }
    tau = {
        "mean": np.zeros(n_joints, dtype=np.float32),
        "std": np.ones(n_joints, dtype=np.float32),
    }
    return state, tau


def test_predict_torques_basic_shape() -> None:
    """Pin: predict returns ``(B, n_joints)`` float32."""
    n = 5
    model = _model(n)
    state, tau = _stats(n)
    rng = np.random.default_rng(42)
    q = rng.normal(size=(2, n))
    qd = rng.normal(size=(2, n))
    qdd = rng.normal(size=(2, n))
    out = predict_torques(model, q, qd, qdd, state_stats=state, tau_stats=tau)
    assert out.shape == (2, n)
    assert out.dtype == np.float32


def test_predict_torques_attached_stats() -> None:
    """Pin: stats attached to the model are used when kwargs are None."""
    n = 5
    model = _model(n)
    state, tau = _stats(n)
    model._state_stats = state
    model._tau_stats = tau
    rng = np.random.default_rng(0)
    q = rng.normal(size=(1, n))
    out = predict_torques(model, q, q, q)
    assert out.shape == (1, n)


def test_predict_torques_requires_model_type() -> None:
    """Pin: non-model arg rejected with TypeError."""
    with pytest.raises(TypeError, match="TimestepInverseDynamics"):
        predict_torques("nope", np.zeros((1, 5)), np.zeros((1, 5)), np.zeros((1, 5)))


def test_predict_torques_requires_stats() -> None:
    """Pin: missing stats raises ValueError."""
    n = 5
    model = _model(n)
    with pytest.raises(ValueError, match="standardisation stats"):
        predict_torques(model, np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)))


def test_predict_torques_njoints_mismatch() -> None:
    """Pin: trailing-dim mismatch is rejected."""
    n = 5
    model = _model(n)
    state, tau = _stats(n)
    bad = np.zeros((1, n + 1))
    with pytest.raises(ValueError, match=r"shape\[-1\]"):
        predict_torques(
            model,
            bad,
            np.zeros((1, n)),
            np.zeros((1, n)),
            state_stats=state,
            tau_stats=tau,
        )


def test_predict_torques_batch_mismatch() -> None:
    """Pin: differing batch sizes rejected."""
    n = 5
    model = _model(n)
    state, tau = _stats(n)
    with pytest.raises(ValueError, match="batch sizes must match"):
        predict_torques(
            model,
            np.zeros((2, n)),
            np.zeros((3, n)),
            np.zeros((2, n)),
            state_stats=state,
            tau_stats=tau,
        )


def test_predict_torques_1d_input_promoted_to_2d() -> None:
    """Pin: 1-D inputs are promoted to a single-batch (1, n)."""
    n = 5
    model = _model(n)
    state, tau = _stats(n)
    out = predict_torques(
        model,
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        state_stats=state,
        tau_stats=tau,
    )
    assert out.shape == (1, n)


def test_predict_torques_3d_rejected() -> None:
    """Pin: 3-D inputs rejected with shape error."""
    n = 5
    model = _model(n)
    state, tau = _stats(n)
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        predict_torques(
            model,
            np.zeros((1, 1, n)),
            np.zeros((1, n)),
            np.zeros((1, n)),
            state_stats=state,
            tau_stats=tau,
        )


def test_predict_zero_std_floored() -> None:
    """Pin: zero-std columns are floored so prediction does not explode."""
    n = 5
    model = _model(n)
    state, tau = _stats(n)
    state["std"] = np.zeros(3 * n, dtype=np.float32)  # degenerate
    out = predict_torques(
        model,
        np.zeros((1, n)),
        np.zeros((1, n)),
        np.zeros((1, n)),
        state_stats=state,
        tau_stats=tau,
    )
    assert np.all(np.isfinite(out))


def test_load_with_stats_missing_file(tmp_path: Path) -> None:
    """Pin: missing checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_with_stats(tmp_path / "nope.pt")


def test_load_with_stats_round_trip(tmp_path: Path) -> None:
    """Pin: load_with_stats attaches state_stats / tau_stats to model."""
    n = 5
    model = _model(n)
    state, tau = _stats(n)
    ckpt_path = tmp_path / "ts.pt"
    payload = model.state_payload()
    payload["state_stats"] = state
    payload["tau_stats"] = tau
    torch.save(payload, ckpt_path)
    loaded = load_with_stats(ckpt_path)
    assert hasattr(loaded, "_state_stats")
    assert hasattr(loaded, "_tau_stats")
