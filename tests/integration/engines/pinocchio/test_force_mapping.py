"""Native force duality uses the velocity constraint, never the log-pose chart."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

if isinstance(sys.modules.get("pinocchio"), Mock):
    pytest.skip(
        "Native Pinocchio required; run this file in isolation", allow_module_level=True
    )
pytest.importorskip("pinocchio")

from src.engines.physics_engines.pinocchio.python.native_model import (  # noqa: E402
    FullBodyPinocchioModel,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]


@pytest.fixture
def native():
    root = Path(__file__).resolve().parents[4]
    spec = json.loads(
        (root / "docs/development/full_body_models/full_body_spec_v1.json").read_bytes()
    )
    plant = FullBodyPinocchioModel(spec)
    names = plant.coordinate_order
    q = {name: float(0.08 * np.sin(i)) for i, name in enumerate(names)}
    v = {name: float(0.1 * np.cos(i)) for i, name in enumerate(names)}
    return plant, q, v


def test_force_map_reproduces_native_constraint_velocity(native):
    plant, q, v = native
    expected = plant.closure_trajectory_residuals(q, v, dict.fromkeys(q, 0.0)).rate
    mapping = plant.closure_force_jacobian(q)
    assert mapping.names == tuple(q)
    np.testing.assert_allclose(
        mapping.jacobian @ np.array(list(v.values())), expected, atol=1e-10
    )
    assert not mapping.jacobian.flags.writeable


def test_force_map_differs_from_log_pose_derivative_away_from_closure(native):
    plant, q, _ = native
    pose = plant.closure_position_linearization(q)
    force = plant.closure_force_jacobian(q)
    assert np.linalg.norm(pose.position) > 0.01
    assert np.linalg.norm(pose.jacobian - force.jacobian) > 0.01


def test_force_map_refreshes_state_and_order_without_dynamics(native, monkeypatch):
    plant, q, _ = native
    first = plant.closure_force_jacobian(q).jacobian.copy()
    q = dict(reversed(list(q.items())))
    q["REInput"] += 0.3

    def forbidden(*args, **kwargs):
        raise AssertionError("kinematic force map must not run constrained dynamics")

    monkeypatch.setattr(plant, "accelerations", forbidden)
    actual = plant.closure_force_jacobian(q)
    assert actual.names == tuple(q)
    assert np.linalg.norm(actual.jacobian[:, ::-1] - first) > 0.01


def test_force_map_rejects_nonfinite_coordinates(native):
    plant, q, _ = native
    q["SpineInputX"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plant.closure_force_jacobian(q)
