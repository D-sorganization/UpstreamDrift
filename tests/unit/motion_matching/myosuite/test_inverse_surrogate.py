"""Tests for MyoSuite inverse surrogate model and training infrastructure.

Covers:
- Model construction and config validation
- Forward pass shape + bounds correctness
- Dropout disabled during eval mode
- SurrogateDataset size/shape contract
- train_surrogate smoke test (CPU, mock data, 1 epoch)
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="PyTorch not installed")

import torch  # noqa: E402

from src.engines.physics_engines.myosuite.python.motion_matching.inverse_surrogate import (
    InverseSurrogateConfig,
    MyoSuiteInverseSurrogate,
)
from src.engines.physics_engines.myosuite.python.motion_matching.train_surrogate import (
    SurrogateDataset,
    train_surrogate,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_JOINTS = 22
N_MUSCLES = 290
SEQ_LEN = 30  # Short for speed
BATCH = 2


@pytest.fixture()
def small_cfg() -> InverseSurrogateConfig:
    """Minimal config for fast unit tests (small hidden + layers)."""
    return InverseSurrogateConfig(
        n_joints=N_JOINTS,
        n_muscles=N_MUSCLES,
        seq_len=SEQ_LEN,
        hidden_dim=32,
        n_layers=2,
        dropout=0.1,
    )


@pytest.fixture()
def model(small_cfg: InverseSurrogateConfig) -> MyoSuiteInverseSurrogate:
    return MyoSuiteInverseSurrogate(small_cfg)


# ── Config tests ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_config_is_frozen(small_cfg: InverseSurrogateConfig) -> None:
    """InverseSurrogateConfig must be immutable (frozen dataclass)."""
    with pytest.raises((AttributeError, TypeError)):
        small_cfg.n_joints = 99  # type: ignore[misc]


@pytest.mark.unit
def test_config_defaults() -> None:
    """Default hyperparameters should match documented values."""
    cfg = InverseSurrogateConfig(n_joints=22, n_muscles=290, seq_len=300)
    assert cfg.hidden_dim == 512
    assert cfg.n_layers == 4
    assert cfg.dropout == 0.1


# ── Forward pass tests ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_forward_output_shape(model: MyoSuiteInverseSurrogate) -> None:
    """Output tensor must be (B, T, n_muscles) for arbitrary batch/seq."""
    joint_q = torch.randn(BATCH, SEQ_LEN, N_JOINTS)
    joint_v = torch.randn(BATCH, SEQ_LEN, N_JOINTS)
    out = model(joint_q, joint_v)
    assert out.shape == (BATCH, SEQ_LEN, N_MUSCLES)


@pytest.mark.unit
def test_activations_strictly_in_unit_interval(model: MyoSuiteInverseSurrogate) -> None:
    """Muscle activations must be in the open interval (0, 1) — strictly."""
    joint_q = torch.randn(BATCH, SEQ_LEN, N_JOINTS)
    joint_v = torch.randn(BATCH, SEQ_LEN, N_JOINTS)
    with torch.no_grad():
        out = model(joint_q, joint_v)
    assert torch.all(out > 0.0), "Activations must be strictly positive"
    assert torch.all(out < 1.0), "Activations must be strictly below 1"


@pytest.mark.unit
def test_eval_mode_is_deterministic(model: MyoSuiteInverseSurrogate) -> None:
    """Repeated eval-mode forward passes on the same input must be identical."""
    model.eval()
    joint_q = torch.randn(1, SEQ_LEN, N_JOINTS)
    joint_v = torch.randn(1, SEQ_LEN, N_JOINTS)
    with torch.no_grad():
        out1 = model(joint_q, joint_v)
        out2 = model(joint_q, joint_v)
    assert torch.allclose(out1, out2), "Eval mode must produce deterministic results"


@pytest.mark.unit
def test_single_timestep_batch(model: MyoSuiteInverseSurrogate) -> None:
    """Model must handle T=1 (single-frame inference) without error."""
    joint_q = torch.randn(1, 1, N_JOINTS)
    joint_v = torch.randn(1, 1, N_JOINTS)
    out = model(joint_q, joint_v)
    assert out.shape == (1, 1, N_MUSCLES)


@pytest.mark.unit
def test_gradient_flows_through_model(small_cfg: InverseSurrogateConfig) -> None:
    """Loss.backward() must succeed — confirms model is differentiable."""
    model = MyoSuiteInverseSurrogate(small_cfg)
    model.train()
    joint_q = torch.randn(2, SEQ_LEN, N_JOINTS)
    joint_v = torch.randn(2, SEQ_LEN, N_JOINTS)
    target = torch.rand(2, SEQ_LEN, N_MUSCLES)
    pred = model(joint_q, joint_v)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    # All parameters should have gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Missing gradient for parameter: {name}"


# ── Dataset tests ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_surrogate_dataset_len() -> None:
    """SurrogateDataset must report a positive length."""
    ds = SurrogateDataset("mock", seq_len=SEQ_LEN, n_joints=N_JOINTS, n_muscles=N_MUSCLES)
    assert len(ds) > 0


@pytest.mark.unit
def test_surrogate_dataset_item_shapes() -> None:
    """Each dataset item must have correct (seq_len, n_joints/muscles) shapes."""
    ds = SurrogateDataset("mock", seq_len=SEQ_LEN, n_joints=N_JOINTS, n_muscles=N_MUSCLES)
    q, v, m = ds[0]
    assert q.shape == (SEQ_LEN, N_JOINTS)
    assert v.shape == (SEQ_LEN, N_JOINTS)
    assert m.shape == (SEQ_LEN, N_MUSCLES)


@pytest.mark.unit
def test_surrogate_dataset_activations_bounded() -> None:
    """Mock muscle activations returned by the dataset must be in [0, 1]."""
    ds = SurrogateDataset("mock", seq_len=SEQ_LEN, n_joints=N_JOINTS, n_muscles=N_MUSCLES)
    _, _, m = ds[0]
    assert torch.all(m >= 0.0)
    assert torch.all(m <= 1.0)


# ── Training smoke test ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_train_surrogate_smoke(tmp_path: pytest.FixturePath) -> None:  # type: ignore[name-defined]
    """Single-epoch training must complete and write a checkpoint."""
    train_surrogate(
        data_path="mock_dir",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        save_dir=tmp_path,
    )
    checkpoint = tmp_path / "surrogate_weights.pt"
    assert checkpoint.exists(), "Checkpoint file must be written after training"
    state = torch.load(checkpoint, map_location="cpu")
    assert isinstance(state, dict)
    assert len(state) > 0, "Checkpoint must contain model weights"
