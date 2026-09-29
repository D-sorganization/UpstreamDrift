import numpy as np
import pytest

try:
    import pinocchio as pin

    if not hasattr(pin, "Model") or not hasattr(pin, "SE3"):
        pytest.skip(
            "pinocchio is a stub/mock, not the full library", allow_module_level=True
        )
except ImportError:
    pytest.skip("pinocchio not installed", allow_module_level=True)

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.induced_acceleration import (  # noqa: E402
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
