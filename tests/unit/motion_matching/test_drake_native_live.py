"""Live native Drake adapter contracts, using an explicit immutable bundle."""

import json
import os
from pathlib import Path
import numpy as np
import pytest
from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel


@pytest.fixture
def model():
    root = os.environ.get("DRAKE_NATIVE_ASSETS")
    if root is None:
        pytest.skip("Set DRAKE_NATIVE_ASSETS to the qualified raw bundle")
    root = Path(root)
    raw = (root / "native_geometry_spec_9967.json").read_bytes()
    engine = NativeDrakeModel(
        (root / "native-golf-9967-01.urdf").read_bytes(),
        (root / "native-golf-9967-01.sidecar.json").read_bytes(),
        raw,
    )
    doc = json.loads(
        (root / "native-root-force-9967-02/returned-candidate.json").read_text()
    )
    q = dict(zip(engine.names, doc["q0"], strict=True))
    v = dict(zip(engine.names, doc["qd0"], strict=True))
    return engine, q, v


def test_native_inventory_and_mass(model):
    engine, q, v = model
    frames = engine.frame_poses(q)
    assert len(engine.names) == 27
    assert len(frames) == 16
    assert engine.plant.num_positions() == 27
    assert engine.plant.num_velocities() == 27
    assert engine.plant.CalcTotalMass(engine.context) == pytest.approx(
        77.60581783574676, abs=1e-12
    )
    assert np.linalg.eigvalsh(engine.plant.CalcMassMatrix(engine.context)).min() > 0
    assert np.isneginf(engine.plant.GetPositionLowerLimits()).all()
    assert np.isposinf(engine.plant.GetPositionUpperLimits()).all()


def test_state_inventory_and_nonfinite_rejected(model):
    engine, q, v = model
    with pytest.raises(ValueError, match="inventory"):
        engine.frame_poses({})
    with pytest.raises(ValueError, match="finite"):
        engine.frame_poses({**q, engine.names[0]: np.nan})
    with pytest.raises(ValueError, match="inventory"):
        engine.accelerations(q, {}, v)


def test_closure_evidence_requires_dynamics_and_is_detached(model):
    engine, q, v = model
    with pytest.raises(ValueError, match="accelerations"):
        engine.closure_errors()
    engine.accelerations(q, v, dict.fromkeys(engine.names, 0.0))
    pose, rate = engine.closure_errors()
    assert np.max(np.abs(pose)) < 1e-10
    assert np.max(np.abs(rate)) < 1e-10
    pose[:] = 100
    assert np.max(np.abs(engine.closure_errors()[0])) < 1e-10
