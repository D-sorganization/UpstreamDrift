"""Unit tests for multi-engine counterfactual capability matrix and engine contracts (#10286, CF-4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import CapabilityLevel
from src.shared.python.motion_matching.counterfactual_matrix import (
    COUNTERFACTUAL_CAPABILITY_MATRIX,
    format_capability_matrix_markdown,
    get_counterfactual_capability_matrix,
    get_engine_counterfactual_capability,
)

pytestmark = pytest.mark.unit


def test_capability_matrix_contains_all_five_engines() -> None:
    """Matrix contains all 5 required engines with guaranteed state restoration."""
    matrix = get_counterfactual_capability_matrix()
    expected_keys = {"pinocchio", "mujoco", "drake", "opensim", "simscape"}
    assert set(matrix.keys()) == expected_keys

    for key, cap in matrix.items():
        assert cap.ztcf_status == CapabilityLevel.FULL, f"{key} ZTCF must be FULL"
        assert cap.zvcf_status == CapabilityLevel.FULL, f"{key} ZVCF must be FULL"
        assert cap.state_restoration_guarantee is True, (
            f"{key} must guarantee state restoration"
        )
        assert len(cap.qualification_evidence) > 0
        assert len(cap.actuation_mapping) > 0
        assert len(cap.constraint_dynamics_support) > 0


def test_get_engine_counterfactual_capability_lookup() -> None:
    """Lookup resolves by key or human-readable engine name."""
    pin_cap = get_engine_counterfactual_capability("pinocchio")
    assert pin_cap.engine_name == "Pinocchio"
    assert pin_cap.supports_closed_loop_reaction is True

    drake_cap = get_engine_counterfactual_capability("Drake")
    assert drake_cap.backend_key == "drake"

    simscape_cap = get_engine_counterfactual_capability("Simscape Multibody R2025b")
    assert simscape_cap.backend_key == "simscape"
    assert simscape_cap.supports_closed_loop_reaction is True

    with pytest.raises(KeyError, match="Unknown engine 'unknown_backend'"):
        get_engine_counterfactual_capability("unknown_backend")


def test_format_capability_matrix_markdown() -> None:
    """Markdown formatter produces table with all 5 engine rows."""
    table = format_capability_matrix_markdown()
    assert "| Engine | Backend Key |" in table
    assert "Pinocchio" in table
    assert "MuJoCo" in table
    assert "Drake" in table
    assert "OpenSim" in table
    assert "Simscape Multibody R2025b" in table


def test_opensim_state_restoration_under_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenSim guarantees state/control restoration and re-realization even on exception."""
    from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine
    from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
        OpenSimPhysicsEngine,
    )

    engine = OpenSimPhysicsEngine.__new__(OpenSimPhysicsEngine)
    BasePhysicsEngine.__init__(engine)
    mock_model = MagicMock()
    mock_state = MagicMock()

    mock_model.getNumControls.return_value = 2
    mock_model.getNumSpeeds.return_value = 2
    controls_mock = MagicMock()
    mock_model.updControls.return_value = controls_mock

    # Exception during realizeDynamics
    mock_model.realizeDynamics.side_effect = RuntimeError("Dynamics realization failed")

    engine._model = None
    engine._state = None
    monkeypatch.setattr(engine, "_model", mock_model, raising=False)
    monkeypatch.setattr(engine, "_state", mock_state, raising=False)
    monkeypatch.setattr(
        engine,
        "get_state",
        MagicMock(return_value=(np.array([1.0, 2.0]), np.array([0.5, 0.6]))),
        raising=False,
    )
    mock_set_state = MagicMock()
    monkeypatch.setattr(engine, "set_state", mock_set_state, raising=False)
    monkeypatch.setattr(engine, "set_control", MagicMock(), raising=False)

    q_test = np.array([0.1, 0.2])
    v_test = np.array([0.3, 0.4])

    with pytest.raises(RuntimeError, match="Dynamics realization failed"):
        engine.compute_ztcf(q_test, v_test)

    # State and controls must have been restored despite the exception
    assert mock_set_state.call_count == 2  # Once for test state, once in finally
    restored_args = mock_set_state.call_args[0]
    np.testing.assert_allclose(restored_args[0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(restored_args[1], np.array([0.5, 0.6]))
    controls_mock.update.assert_called_once()


def test_opensim_no_empty_success_fallback() -> None:
    """Uninitialized OpenSim raises an error instead of returning empty array."""
    from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
        OpenSimPhysicsEngine,
    )

    engine = OpenSimPhysicsEngine.__new__(OpenSimPhysicsEngine)
    engine._model = None
    engine._state = None

    with pytest.raises((RuntimeError, Exception)):
        engine.compute_ztcf(np.array([1.0]), np.array([0.0]))

    with pytest.raises((RuntimeError, Exception)):
        engine.compute_zvcf(np.array([1.0]))

    with pytest.raises((RuntimeError, Exception)):
        engine.compute_control_acceleration(np.array([1.0]))


def test_pinocchio_no_empty_success_fallback() -> None:
    """Uninitialized Pinocchio raises an error instead of returning empty array."""
    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
        PinocchioPhysicsEngine,
    )

    engine = PinocchioPhysicsEngine.__new__(PinocchioPhysicsEngine)
    engine.model = None
    engine.data = None

    with pytest.raises((RuntimeError, Exception)):
        engine.compute_ztcf(np.array([1.0]), np.array([0.0]))

    with pytest.raises((RuntimeError, Exception)):
        engine.compute_zvcf(np.array([1.0]))

    with pytest.raises((RuntimeError, Exception)):
        engine.compute_control_acceleration(np.array([1.0]))


def test_drake_actuator_mapping_and_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drake maps actuator command through actuation matrix when nu != nv."""
    from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine
    from src.engines.physics_engines.drake.python.drake_physics_engine import (
        DrakePhysicsEngine,
    )

    engine = DrakePhysicsEngine.__new__(DrakePhysicsEngine)
    BasePhysicsEngine.__init__(engine)
    engine.plant_context = None

    # Check uninitialized raises RuntimeError
    with pytest.raises(RuntimeError, match="Drake plant context is not initialized"):
        engine.compute_control_acceleration(np.array([1.0]))

    with pytest.raises(RuntimeError, match="Drake plant context is not initialized"):
        engine.compute_ztcf(np.array([1.0]), np.array([0.0]))

    with pytest.raises(RuntimeError, match="Drake plant context is not initialized"):
        engine.compute_zvcf(np.array([1.0]))

    # Test actuator matrix projection when nu (1) != nv (2)
    mock_plant = MagicMock()
    mock_plant.num_velocities.return_value = 2
    mock_plant.num_actuators.return_value = 1
    # B maps 1 actuator to joint 2: [[0.0], [1.0]]
    mock_plant.MakeActuationMatrix.return_value = np.array([[0.0], [1.0]])
    mock_plant.CalcMassMatrixViaInverseDynamics.return_value = np.eye(2)

    monkeypatch.setattr(engine, "plant", mock_plant, raising=False)
    monkeypatch.setattr(engine, "plant_context", MagicMock(), raising=False)

    u_actuator = np.array([5.0])
    a_ctrl = engine.compute_control_acceleration(u_actuator)

    # Should equal M^-1 * (B @ u) = eye(2) * [0, 5] = [0, 5]
    np.testing.assert_allclose(a_ctrl, np.array([0.0, 5.0]))


def test_simscape_counterfactual_doc_and_deferred_semantics() -> None:
    """Simscape adapter documents R2025b killswitch and replay bundle routes."""
    from src.engines.simscape.adapter import SimscapeAdapter

    doc_ztcf = SimscapeAdapter.compute_ztcf.__doc__
    doc_zvcf = SimscapeAdapter.compute_zvcf.__doc__
    assert doc_ztcf is not None
    assert doc_zvcf is not None
    assert "run_ztcf_simulation.m" in doc_ztcf
    assert "CounterfactualTrajectory" in doc_ztcf
    assert "run_ztcf_simulation.m" in doc_zvcf
    assert "CounterfactualTrajectory" in doc_zvcf
