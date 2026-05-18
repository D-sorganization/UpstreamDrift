"""Tests for Phase 1 PINNs hybrid architecture.

Covers MlpResidual, RigidCore, and HybridPINN. Tests are skipped
gracefully when optional dependencies (JAX, Equinox, Pinocchio) are
not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import pinocchio  # noqa: F401

    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

from src.shared.python.physics_informed.hybrid_model import HybridPINN
from src.shared.python.physics_informed.mlp_residual import MlpResidual
from src.shared.python.physics_informed.rigid_core import RigidCore

# ---------------------------------------------------------------------------
# MlpResidual tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_mlp_residual_output_shape() -> None:
    """MlpResidual forward pass returns correct output shape."""
    key = jax.random.PRNGKey(0)
    mlp = MlpResidual(input_dim=6, output_dim=3, hidden_dims=[16, 16], key=key)
    x = jnp.ones(6)
    out = mlp(x)
    assert out.shape == (3,)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_mlp_residual_finite() -> None:
    """MlpResidual outputs are all finite values (no NaN/Inf)."""
    key = jax.random.PRNGKey(42)
    mlp = MlpResidual(input_dim=4, output_dim=4, hidden_dims=[8], key=key)
    x = jnp.zeros(4)
    out = mlp(x)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_mlp_residual_single_hidden_layer() -> None:
    """MlpResidual with a single hidden layer produces correct shape."""
    key = jax.random.PRNGKey(7)
    mlp = MlpResidual(input_dim=3, output_dim=2, hidden_dims=[32], key=key)
    x = jnp.array([1.0, -0.5, 0.3])
    out = mlp(x)
    assert out.shape == (2,)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_mlp_residual_no_hidden_layers() -> None:
    """MlpResidual with empty hidden_dims is a single linear layer."""
    key = jax.random.PRNGKey(99)
    mlp = MlpResidual(input_dim=5, output_dim=5, hidden_dims=[], key=key)
    x = jnp.ones(5)
    out = mlp(x)
    assert out.shape == (5,)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_mlp_residual_different_keys_different_weights() -> None:
    """Two MlpResiduals initialized with different keys produce different outputs."""
    key1 = jax.random.PRNGKey(1)
    key2 = jax.random.PRNGKey(2)
    mlp1 = MlpResidual(input_dim=4, output_dim=2, hidden_dims=[8], key=key1)
    mlp2 = MlpResidual(input_dim=4, output_dim=2, hidden_dims=[8], key=key2)
    x = jnp.ones(4)
    out1 = mlp1(x)
    out2 = mlp2(x)
    # Different random init → outputs should differ
    assert not jnp.allclose(out1, out2)


def test_mlp_residual_raises_without_jax() -> None:
    """MlpResidual raises ImportError when JAX is unavailable."""
    if HAS_JAX:
        pytest.skip("JAX is installed; cannot test absence path")
    with pytest.raises(ImportError, match="jax"):
        MlpResidual(input_dim=3, output_dim=3, hidden_dims=[8], key=None)


# ---------------------------------------------------------------------------
# RigidCore tests
# ---------------------------------------------------------------------------


def test_rigid_core_invalid_path_raises() -> None:
    """RigidCore raises ValueError for a non-existent URDF path."""
    with pytest.raises((ValueError, ImportError)):
        RigidCore("/nonexistent/path/robot.urdf")


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio not installed")
def test_rigid_core_invalid_path_raises_value_error() -> None:
    """RigidCore raises ValueError for a non-existent URDF path with Pinocchio."""
    with pytest.raises(ValueError, match="urdf_path"):
        RigidCore("/nonexistent/path/robot.urdf")


# ---------------------------------------------------------------------------
# HybridPINN tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (HAS_JAX and HAS_PINOCCHIO),
    reason="JAX or Pinocchio not installed",
)
def test_hybrid_pinn_sum() -> None:
    """HybridPINN output = rigid + residual (stub rigid returning zeros)."""
    # Build a stub RigidCore that returns a zeros vector of length 3
    stub_rigid = MagicMock(spec=RigidCore)
    stub_rigid.nq = 3
    stub_rigid.nv = 3
    stub_rigid.compute_torques.return_value = np.zeros(3)

    key = jax.random.PRNGKey(0)
    # Input: concat(q, dq, ddq) = 9 dims; output: 3 dims (matches nv)
    mlp = MlpResidual(input_dim=9, output_dim=3, hidden_dims=[16], key=key)

    hybrid = HybridPINN(rigid_core=stub_rigid, mlp_residual=mlp)

    q = np.zeros(3)
    dq = np.zeros(3)
    ddq = np.zeros(3)
    result = hybrid.predict(q, dq, ddq)

    assert result.shape == (3,)
    stub_rigid.compute_torques.assert_called_once()


def test_hybrid_pinn_shape_mismatch_raises() -> None:
    """HybridPINN raises ValueError when rigid and MLP output dims mismatch."""
    # Stub rigid: returns 3-element torque
    stub_rigid = MagicMock(spec=RigidCore)
    stub_rigid.nq = 3
    stub_rigid.nv = 3
    stub_rigid.compute_torques.return_value = np.zeros(3)

    # Stub MLP: returns 5-element output (mismatch with rigid)
    stub_mlp = MagicMock(spec=MlpResidual)
    stub_mlp.output_dim = 5
    stub_mlp.return_value = np.zeros(5)

    hybrid = HybridPINN(rigid_core=stub_rigid, mlp_residual=stub_mlp)

    q = np.zeros(3)
    dq = np.zeros(3)
    ddq = np.zeros(3)

    with pytest.raises(ValueError, match="shape"):
        hybrid.predict(q, dq, ddq)


def test_hybrid_pinn_output_is_sum_of_rigid_and_residual() -> None:
    """HybridPINN returns elementwise sum of rigid and MLP residual torques."""
    rigid_torque = np.array([1.0, 2.0, 3.0])
    residual_torque = np.array([0.1, 0.2, 0.3])

    stub_rigid = MagicMock(spec=RigidCore)
    stub_rigid.nq = 3
    stub_rigid.nv = 3
    stub_rigid.compute_torques.return_value = rigid_torque

    stub_mlp = MagicMock(spec=MlpResidual)
    stub_mlp.output_dim = 3
    stub_mlp.return_value = residual_torque
    stub_mlp.__call__ = MagicMock(return_value=residual_torque)

    hybrid = HybridPINN(rigid_core=stub_rigid, mlp_residual=stub_mlp)

    q = np.zeros(3)
    dq = np.zeros(3)
    ddq = np.zeros(3)
    result = hybrid.predict(q, dq, ddq)

    expected = rigid_torque + residual_torque
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("q_shape", "dq_shape", "ddq_shape"),
    [
        ((4,), (3,), (3,)),  # q wrong
        ((3,), (4,), (3,)),  # dq wrong
        ((3,), (3,), (4,)),  # ddq wrong
    ],
)
def test_hybrid_pinn_mismatched_input_shapes_raise(
    q_shape: tuple[int, ...],
    dq_shape: tuple[int, ...],
    ddq_shape: tuple[int, ...],
) -> None:
    """HybridPINN raises ValueError when q/dq/ddq shapes are inconsistent."""
    stub_rigid = MagicMock(spec=RigidCore)
    stub_rigid.nq = 3
    stub_rigid.nv = 3
    stub_rigid.compute_torques.return_value = np.zeros(3)

    stub_mlp = MagicMock(spec=MlpResidual)
    stub_mlp.output_dim = 3

    hybrid = HybridPINN(rigid_core=stub_rigid, mlp_residual=stub_mlp)

    with pytest.raises(ValueError):
        hybrid.predict(
            np.zeros(q_shape),
            np.zeros(dq_shape),
            np.zeros(ddq_shape),
        )
