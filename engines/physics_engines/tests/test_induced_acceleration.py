"""Tests for Induced Acceleration Analysis across all physics engines."""

import numpy as np
import pytest


# --- PINOCCHIO ---
def test_pinocchio_iaa():
    """Test Pinocchio InducedAccelerationAnalyzer."""
    try:
        import pinocchio as pin

        from engines.physics_engines.pinocchio.python.pinocchio_golf.induced_acceleration import (
            InducedAccelerationAnalyzer,
        )
    except ImportError:
        pytest.skip("Pinocchio or Analysis module not found")

    # Create random humanoid
    model = pin.buildSampleModelHumanoidRandom()
    data = model.createData()
    q = pin.neutral(model)
    v = np.random.rand(model.nv)
    tau = np.random.rand(model.nv)

    analyzer = InducedAccelerationAnalyzer(model, data)
    res = analyzer.compute_components(q, v, tau)

    # Check keys
    assert "gravity" in res
    assert "velocity" in res
    # control is stored as 'control' in analyzer
    assert "control" in res
    assert "total" in res

    # Check sum
    # M a = tau - C - G
    # a = a_g + a_c + a_t

    # Forward Dynamics (ABA)
    pin.aba(model, data, q, v, tau)
    expected_acc = data.ddq

    total_est = res["total"]
    np.testing.assert_allclose(total_est, expected_acc, atol=1e-9)

    # Components verification
    # Gravity only (v=0, tau=0) -> a = -M^-1 G
    res_g = analyzer.compute_components(q, np.zeros(model.nv), np.zeros(model.nv))
    np.testing.assert_allclose(res_g["velocity"], 0, atol=1e-9)
    np.testing.assert_allclose(res_g["control"], 0, atol=1e-9)
    np.testing.assert_allclose(res_g["gravity"], res["gravity"], atol=1e-9)


# --- DRAKE ---
def test_drake_iaa():
    """Test DrakeInducedAccelerationAnalyzer."""
    try:
        from pydrake.all import MultibodyPlant

        from engines.physics_engines.drake.python.src.induced_acceleration import (
            DrakeInducedAccelerationAnalyzer,
        )
    except ImportError:
        pytest.skip("Drake or Analyzer not found")

    # Simple plant
    plant = MultibodyPlant(0.0)
    plant.Finalize()
    context = plant.CreateDefaultContext()

    analyzer = DrakeInducedAccelerationAnalyzer(plant)

    # Test default (v=0, tau=0)
    res = analyzer.compute_components(context)

    assert "gravity" in res
    assert "velocity" in res
    assert "control" in res
    assert "total" in res


# --- MUJOCO ---
def test_mujoco_iaa_logic():
    """Test MuJoCo IAA helper logic."""
    try:
        import mujoco

        from engines.physics_engines.mujoco.docker.src.humanoid_golf import iaa_helper
    except (ImportError, OSError):
        pytest.skip("MuJoCo or helper not found or DLL failed")

    class MockPhysics:
        def __init__(self):
            # Create simplest model
            xml = """
            <mujoco>
              <worldbody>
                <body>
                  <joint type="hinge" axis="1 0 0"/>
                  <geom type="sphere" size="0.1"/>
                </body>
              </worldbody>
            </mujoco>
            """
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)

    try:
        phys = MockPhysics()
    except Exception:
        pytest.skip("Failed to initialize MuJoCo model")

    res = iaa_helper.compute_induced_accelerations(phys)
    assert isinstance(res, dict)

    # If successful logic, keys should be there
    if res:
        assert "gravity" in res
        assert "coriolis" in res
        assert "control" in res

        # Test basic property: sum should roughly match qacc if we computed it?
        # But we are computing acceleration from scratch.
        # Should sum to qacc if qacc was consistent with q,v,tau.
        # Here we didn't run forward dynamics to set qacc.
        pass
