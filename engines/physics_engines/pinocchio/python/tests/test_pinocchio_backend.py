import typing

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

try:
    from dtack.backends.pinocchio_backend import PinocchioBackend
except ImportError:
    pytest.skip("dtack dependencies missing", allow_module_level=True)


class HelperPinocchioBackend(PinocchioBackend):  # type: ignore[misc, no-any-unimported]
    """Subclass to allow initialization with an existing model."""

    def __init__(self, model: typing.Any) -> None:  # noqa: ANN401
        # Skip super().__init__ which requires a file
        self.model = model
        self.data = model.createData()
        self.collision_model = pin.GeometryModel()
        self.visual_model = pin.GeometryModel()
        self.collision_data = self.collision_model.createData()
        self.visual_data = self.visual_model.createData()


def test_compute_bias_forces_correctness() -> None:
    """Verify that compute_bias_forces returns the correct NLE vector."""
    model = pin.buildSampleModelHumanoid()
    backend = HelperPinocchioBackend(model)

    q = pin.neutral(model)
    # Random velocity
    rng = np.random.default_rng(42)
    v = rng.standard_normal(model.nv)

    # Calculate expected value using the reference method
    # RNEA with a=0 is exactly the bias forces (C*v + g)
    # Note: The previous implementation of compute_bias_forces was BUGGY.
    # It returned self.data.nle, but pin.rnea does NOT update data.nle!
    # The correct ground truth is the return value of rnea(a=0).
    data_ref = model.createData()
    expected = pin.rnea(model, data_ref, q, v, np.zeros(model.nv))

    # Calculate actual value using the backend
    # This will test whatever implementation is currently in PinocchioBackend
    result = backend.compute_bias_forces(q, v)

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
        err_msg="Bias forces do not match reference RNEA calculation",
    )
