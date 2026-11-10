"""MuJoCo, Pinocchio, and export stamping for #14943 follow-on."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.counterfactuals import (
    CounterfactualResult,
)
from src.shared.python.engine_core.counterfactual_contract import (
    COUNTERFACTUAL_DEFINITION_VERSION,
)
from src.shared.python.motion_matching.counterfactual import AccelerationDecomposition

pytestmark = pytest.mark.unit


def test_mujoco_counterfactual_result_stamps_definition_version() -> None:
    z = np.zeros(3)
    result = CounterfactualResult(
        type="zvcf",
        observed_acceleration=z,
        counterfactual_acceleration=z,
        delta_acceleration=z,
        observed_position=None,
        counterfactual_position=None,
        delta_position=None,
        torque_attributed_effect=z,
        velocity_attributed_effect=None,
    )
    assert result.definition_version == COUNTERFACTUAL_DEFINITION_VERSION


def test_acceleration_decomposition_export_stamp() -> None:
    a = np.array([0.0, 1.0])
    decomp = AccelerationDecomposition(a_grav=a, a_drift=a, a_ctrl=a)
    assert decomp.definition_version == COUNTERFACTUAL_DEFINITION_VERSION
    assert decomp.export_stamp() == {
        "counterfactual_definition_version": COUNTERFACTUAL_DEFINITION_VERSION,
    }


def test_pinocchio_compute_zvcf_emits_deprecation() -> None:
    pin = pytest.importorskip("pinocchio")
    if not hasattr(pin, "__version__"):
        pytest.skip("Pinocchio mocked")
    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
        PinocchioPhysicsEngine,
    )

    from pathlib import Path

    urdf = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "models"
        / "double_pendulum.urdf"
    )
    if not urdf.is_file():
        pytest.skip("double_pendulum.urdf fixture missing")
    engine = PinocchioPhysicsEngine()
    try:
        engine.load_from_path(str(urdf))
    except (ImportError, OSError, ValueError, RuntimeError):
        pytest.skip("Pinocchio model not loadable in this environment")
    q = np.zeros(engine.model.nq)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = engine.compute_zvcf(q)
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    gravity = engine.compute_gravity_only_acceleration(q)
    np.testing.assert_allclose(legacy, gravity)
