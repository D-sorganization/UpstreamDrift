"""Wave 6 coverage: src.learning.sim2real._simscape_compat."""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.sim2real._simscape_compat import (
    SimscapeSystemIdCompat,
    wrap_for_system_identification,
)


class FakeAdapter:
    """Lightweight stand-in for SimscapeAdapter."""

    dof = 3

    def __init__(self) -> None:
        self.damping_set: list[np.ndarray] = []

    def get_link_masses(self) -> np.ndarray:
        return np.ones(self.dof)

    def some_passthrough(self) -> str:
        return "passed"


class TestCompat:
    def test_adapter_none_raises(self) -> None:
        with pytest.raises(ValueError):
            SimscapeSystemIdCompat(None)  # type: ignore[arg-type]

    def test_passthrough(self) -> None:
        c = SimscapeSystemIdCompat(FakeAdapter())
        assert c.some_passthrough() == "passed"
        np.testing.assert_array_equal(c.get_link_masses(), np.ones(3))

    def test_set_joint_damping_records(self) -> None:
        c = SimscapeSystemIdCompat(FakeAdapter())
        c.set_joint_damping(np.array([1.0, 2.0, 3.0]))
        c.set_joint_damping(np.array([4.0, 5.0, 6.0]))
        history = c.damping_history
        assert len(history) == 2
        np.testing.assert_array_equal(history[0], [1.0, 2.0, 3.0])

    def test_compat_dispatch_getter_returns_zeros(self) -> None:
        c = SimscapeSystemIdCompat(FakeAdapter())
        out = c.get_friction_coefficients()
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_compat_dispatch_setter_noop(self) -> None:
        c = SimscapeSystemIdCompat(FakeAdapter())
        # should not raise
        assert c.set_motor_strength(np.zeros(3)) is None

    def test_compat_dispatch_getter_no_dof(self) -> None:
        class AdapterNoDof:
            @property
            def dof(self) -> int:
                raise RuntimeError("no dof")

        c = SimscapeSystemIdCompat(AdapterNoDof())
        out = c.get_motor_strength()
        # falls back to length-0 array
        assert out.shape == (0,)

    def test_warn_once(self) -> None:
        c = SimscapeSystemIdCompat(FakeAdapter())
        # Trigger twice; second call should not re-add to _warned
        c.set_joint_damping(np.zeros(3))
        c.set_joint_damping(np.zeros(3))
        assert "set_joint_damping" in c._warned

    def test_wrap_helper(self) -> None:
        wrapped = wrap_for_system_identification(FakeAdapter())
        assert isinstance(wrapped, SimscapeSystemIdCompat)
