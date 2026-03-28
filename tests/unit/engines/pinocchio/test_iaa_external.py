import pytest

pytest.importorskip("pinocchio", reason="pinocchio not installed")

import numpy as np
import pinocchio as pin

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.induced_acceleration import (
    InducedAccelerationAnalyzer,
)


def test_iaa_external_decomposition():
    """Test that IAA decomposition sums up correctly with external forces."""
    model = pin.buildSampleModelHumanoidRandom()
    data = model.createData()
    analyzer = InducedAccelerationAnalyzer(model, data)

    # Random state and torque
    q = pin.neutral(model)
    v = np.random.rand(model.nv)
    tau = np.random.rand(model.nv)

    # external forces
    f_ext = {1: np.random.rand(6)}

    components = analyzer.compute_components(q, v, tau, f_ext=f_ext)

    sum_components = (
        components["gravity"]
        + components["velocity"]
        + components["control"]
        + components["external"]
    )

    np.testing.assert_allclose(components["total"], sum_components, atol=1e-10)
