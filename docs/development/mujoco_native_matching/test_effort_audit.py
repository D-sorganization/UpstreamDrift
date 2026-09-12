"""Check polynomial extrema used by the read-only effort audit."""

import numpy as np
import pytest

from audit_effort_bounds import extrema


def test_interior_extremum_and_endpoints() -> None:
    result = extrema(np.array([-1.0, 1.0, 0.0]), 1.0)
    assert result["maximum"] == pytest.approx(0.25)
    assert result["maximum_time_s"] == pytest.approx(0.5)
    assert result["minimum"] == pytest.approx(0)


def test_constant_and_extrapolated_linear() -> None:
    assert extrema(np.array([2.0]), 0.8)["maximum"] == 2
    assert extrema(np.array([1.0, 0.0]), 0.85)["maximum"] == 0.85


def test_nonfinite_values_rejected() -> None:
    with pytest.raises(ValueError):
        extrema(np.array([np.nan, 1.0]), 1)
