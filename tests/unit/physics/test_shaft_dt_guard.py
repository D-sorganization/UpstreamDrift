"""Tests for dt <= 0 guard in FiniteElementShaftModel.step().

Covers issue #3054: integrator must raise ValueError for non-positive dt
to prevent silent division-by-zero in the Newmark-beta formulation.
"""

from __future__ import annotations

import pytest

from src.shared.python.physics._shaft_fem import FiniteElementShaftModel
from src.shared.python.physics.flexible_shaft import create_standard_shaft

pytestmark = pytest.mark.unit

pytestmark = pytest.mark.unit


@pytest.fixture()
def initialized_fem() -> FiniteElementShaftModel:
    """Return a fully initialized FE shaft model ready to step."""
    model = FiniteElementShaftModel(n_elements=4)
    props = create_standard_shaft()
    model.initialize(props)
    return model


class TestStepDtGuard:
    """Verify that step() rejects non-positive dt values."""

    def test_step_dt_zero_raises(
        self, initialized_fem: FiniteElementShaftModel
    ) -> None:
        """step(dt=0) must raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            initialized_fem.step(dt=0)

    def test_step_dt_negative_raises(
        self, initialized_fem: FiniteElementShaftModel
    ) -> None:
        """step(dt=-1) must raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            initialized_fem.step(dt=-1)

    def test_step_dt_negative_float_raises(
        self, initialized_fem: FiniteElementShaftModel
    ) -> None:
        """step(dt=-1e-6) must raise ValueError (small negative)."""
        with pytest.raises(ValueError, match="dt must be positive"):
            initialized_fem.step(dt=-1e-6)

    def test_step_dt_positive_succeeds(
        self, initialized_fem: FiniteElementShaftModel
    ) -> None:
        """step(dt=0.001) must succeed and return a ShaftState."""
        state = initialized_fem.step(dt=0.001)
        assert state is not None
