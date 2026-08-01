"""Integration bound-validation regression tests (#3182).

Restores strict bound validation in ``signal_toolkit`` integration: inverted,
out-of-range, and NaN bounds must raise ``ValueError`` (not clamp), and the
guards must survive ``python -O`` assert stripping. Split out of
``test_signal_toolkit.py`` so the fully annotated regression lives in its own
file (delta-CI mypy clean).
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_toolkit.calculus import Integrator, compute_integral
from signal_toolkit.core import Signal


class TestBoundValidation:
    """Bound validation for ``compute_integral`` / ``Integrator`` (#3182)."""

    def test_inverted_bounds_raise(self) -> None:
        """Inverted bounds (lower > upper) must raise ValueError (#3182)."""
        t = np.linspace(0, 10, 101)
        signal = Signal(t, np.ones(101) * 5.0)
        with pytest.raises(ValueError):
            compute_integral(signal, lower_bound=8, upper_bound=2)

    def test_out_of_range_bounds_raise(self) -> None:
        """Out-of-range bounds must raise ValueError instead of clamping (#3182)."""
        t = np.linspace(0, 10, 101)
        signal = Signal(t, np.ones(101) * 5.0)
        with pytest.raises(ValueError):
            compute_integral(signal, lower_bound=-5, upper_bound=999)

    def test_nan_bounds_raise(self) -> None:
        """NaN bounds must raise ValueError (#3182)."""
        t = np.linspace(0, 10, 101)
        signal = Signal(t, np.ones(101) * 5.0)
        with pytest.raises(ValueError):
            compute_integral(signal, lower_bound=float("nan"), upper_bound=10)

    def test_valid_bounds_unchanged(self) -> None:
        """Valid in-range bounds still integrate correctly (#3182)."""
        t = np.linspace(0, 10, 1001)
        signal = Signal(t, np.ones(1001) * 5.0)
        result = compute_integral(signal, lower_bound=2, upper_bound=8)
        # Integral of 5 from 2 to 8 = 30
        assert result.value == pytest.approx(30.0, rel=0.01)

    def test_bound_guards_survive_optimized_mode(self) -> None:
        """Guards must not be bare asserts; replicate ``python -O`` (#3182)."""
        # _validate_bounds uses explicit ValueError, not assert, so it
        # raises regardless of the ``-O`` assert-stripping flag.
        with pytest.raises(ValueError):
            Integrator._validate_bounds(8.0, 2.0, 0.0, 10.0)
