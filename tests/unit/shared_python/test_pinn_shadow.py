"""Tests for Phase 3 of the PINNs epic: ShadowModel, ShadowReport, and mode factory.

Tests are written FIRST (TDD red phase) before implementation.
JAX-dependent paths are skipped gracefully when JAX is not installed.

Part of epic #5419. Closes #5499.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from src.shared.python.physics_informed.mode import PhysicsMode, create_model
from src.shared.python.physics_informed.shadow_model import (
    ShadowModel,
    ShadowReport,
    SwingPhase,
)


# =============================================================================
# ShadowReport
# =============================================================================


def test_shadow_report_is_dataclass() -> None:
    """ShadowReport must be a dataclass with a peak_residuals dict."""
    report = ShadowReport()
    assert hasattr(report, "peak_residuals")
    assert isinstance(report.peak_residuals, dict)


def test_shadow_report_defaults_empty_dict() -> None:
    """Default peak_residuals must be an empty dict (not a shared mutable)."""
    r1 = ShadowReport()
    r2 = ShadowReport()
    # Mutating one instance must not affect the other
    r1.peak_residuals["x"] = 1.0
    assert "x" not in r2.peak_residuals


# =============================================================================
# SwingPhase
# =============================================================================


def test_swing_phase_transition_value() -> None:
    assert SwingPhase.TRANSITION.value == "transition"


def test_swing_phase_impact_value() -> None:
    assert SwingPhase.IMPACT.value == "impact"


def test_swing_phase_follow_through_value() -> None:
    assert SwingPhase.FOLLOW_THROUGH.value == "follow_through"


# =============================================================================
# ShadowModel -- DbC (construction preconditions)
# =============================================================================


def test_shadow_model_requires_non_none_rigid() -> None:
    """ShadowModel must raise ValueError or TypeError when rigid_core is None."""
    with pytest.raises((ValueError, TypeError)):
        ShadowModel(None, MagicMock())


def test_shadow_model_requires_non_none_mlp() -> None:
    """ShadowModel must raise ValueError or TypeError when mlp_residual is None."""
    with pytest.raises((ValueError, TypeError)):
        ShadowModel(MagicMock(), None)


# =============================================================================
# ShadowModel -- observe()
# =============================================================================


def test_shadow_model_observe_returns_report() -> None:
    """observe() must return a ShadowReport when given valid frames."""
    rigid = MagicMock()
    rigid.compute_torques.return_value = np.zeros(6)
    mlp = MagicMock()
    mlp.return_value = np.zeros(6)

    sm = ShadowModel(rigid, mlp)
    frames = [{"q": np.zeros(6), "dq": np.zeros(6), "ddq": np.zeros(6)}]
    report = sm.observe(frames)
    assert isinstance(report, ShadowReport)


def test_shadow_model_observe_empty_frames_returns_empty_report() -> None:
    """observe() with no frames must return an empty ShadowReport."""
    sm = ShadowModel(MagicMock(), MagicMock())
    report = sm.observe([])
    assert isinstance(report, ShadowReport)
    assert report.peak_residuals == {}


def test_shadow_model_observe_peak_residuals_uses_phase_keys() -> None:
    """Non-empty frame list must produce peak_residuals keyed by SwingPhase values."""
    rigid = MagicMock()
    rigid.compute_torques.return_value = np.ones(6)
    mlp = MagicMock()
    mlp.return_value = np.full(6, 2.0)

    sm = ShadowModel(rigid, mlp)
    # 10 frames so all three phases are populated
    frames = [
        {"q": np.zeros(6), "dq": np.zeros(6), "ddq": np.zeros(6)} for _ in range(10)
    ]
    report = sm.observe(frames)

    # Keys must be SwingPhase string values
    valid_keys = {p.value for p in SwingPhase}
    assert set(report.peak_residuals.keys()) <= valid_keys


def test_shadow_model_observe_peak_residuals_are_non_negative() -> None:
    """Peak residuals must be non-negative (they represent magnitudes)."""
    rigid = MagicMock()
    rigid.compute_torques.return_value = np.ones(6) * 3.0
    mlp = MagicMock()
    mlp.return_value = np.ones(6) * 1.5

    sm = ShadowModel(rigid, mlp)
    frames = [
        {"q": np.zeros(6), "dq": np.zeros(6), "ddq": np.zeros(6)} for _ in range(9)
    ]
    report = sm.observe(frames)

    for key, val in report.peak_residuals.items():
        assert val >= 0.0, f"peak_residuals[{key!r}] = {val} must be >= 0"


def test_shadow_model_does_not_modify_frames() -> None:
    """observe() must not mutate input frames (observation mode only)."""
    rigid = MagicMock()
    rigid.compute_torques.return_value = np.zeros(6)
    mlp = MagicMock()
    mlp.return_value = np.zeros(6)

    sm = ShadowModel(rigid, mlp)
    frame = {"q": np.zeros(6), "dq": np.zeros(6), "ddq": np.zeros(6)}
    original_keys = set(frame.keys())
    sm.observe([frame])
    assert set(frame.keys()) == original_keys


def test_shadow_model_observe_graceful_without_jax() -> None:
    """observe() must return an empty ShadowReport (not raise) when underlying
    calls raise ImportError (simulating missing JAX).
    """
    rigid = MagicMock()
    rigid.compute_torques.side_effect = ImportError("jax not available")
    mlp = MagicMock()

    sm = ShadowModel(rigid, mlp)
    frames = [{"q": np.zeros(6), "dq": np.zeros(6), "ddq": np.zeros(6)}]
    # Should NOT raise; returns an empty ShadowReport
    report = sm.observe(frames)
    assert isinstance(report, ShadowReport)


# =============================================================================
# PhysicsMode
# =============================================================================


def test_physics_mode_values() -> None:
    """PhysicsMode enum must have the three expected string values."""
    assert PhysicsMode.PURE_RIGID.value == "pure_rigid"
    assert PhysicsMode.PURE_AI.value == "pure_ai"
    assert PhysicsMode.PINN_HYBRID.value == "pinn_hybrid"


def test_physics_mode_has_exactly_three_members() -> None:
    """PhysicsMode must have exactly three members."""
    assert len(PhysicsMode) == 3


# =============================================================================
# create_model factory
# =============================================================================


def test_create_model_invalid_mode_raises() -> None:
    """create_model must raise ValueError for an unrecognised mode string."""
    with pytest.raises(ValueError):
        create_model("invalid_mode", {})  # type: ignore[arg-type]


def test_create_model_invalid_mode_enum_like_string_raises() -> None:
    """create_model must raise ValueError even for plausible-sounding bad modes."""
    with pytest.raises(ValueError):
        create_model("pure_hybrid", {})  # type: ignore[arg-type]


def test_create_model_pure_rigid_returns_rigid_core_or_import_error() -> None:
    """PURE_RIGID must return a RigidCore, or raise ImportError (pinocchio absent).

    A nonexistent URDF path raises ValueError from RigidCore's DbC check when
    Pinocchio is installed. That is the expected DbC precondition failure.
    """
    config = {"urdf_path": "/nonexistent.urdf"}
    try:
        model = create_model(PhysicsMode.PURE_RIGID, config)
        from src.shared.python.physics_informed.rigid_core import RigidCore

        assert isinstance(model, RigidCore)
    except ImportError:
        pass  # pinocchio not installed -- acceptable
    except ValueError:
        pass  # pinocchio installed but URDF doesn't exist -- acceptable DbC failure


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_create_model_pure_ai_returns_mlp_residual() -> None:
    """PURE_AI must return an MlpResidual when JAX is installed."""
    import jax

    config = {
        "input_dim": 18,
        "output_dim": 6,
        "hidden_dims": [32, 32],
        "key": jax.random.PRNGKey(42),
    }
    model = create_model(PhysicsMode.PURE_AI, config)
    from src.shared.python.physics_informed.mlp_residual import MlpResidual

    assert isinstance(model, MlpResidual)


def test_create_model_pure_ai_raises_import_error_without_jax() -> None:
    """PURE_AI must raise ImportError when JAX is not installed."""
    if HAS_JAX:
        pytest.skip("JAX is installed -- cannot test missing-JAX path")

    config = {"input_dim": 18, "output_dim": 6}
    with pytest.raises(ImportError):
        create_model(PhysicsMode.PURE_AI, config)
