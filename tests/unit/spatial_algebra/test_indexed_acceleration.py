"""Tests for src.shared.python.spatial_algebra.indexed_acceleration (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.spatial_algebra.indexed_acceleration import (
    AccelerationClosureError,
    IndexedAcceleration,
)


def _make_accel(
    gravity: float = 1.0,
    coriolis: float = 0.5,
    applied_torque: float = 0.3,
    constraint: float = 0.0,
    external: float = 0.0,
    n: int = 3,
    centrifugal: float | None = None,
) -> IndexedAcceleration:
    return IndexedAcceleration(
        gravity=np.full(n, gravity),
        coriolis=np.full(n, coriolis),
        applied_torque=np.full(n, applied_torque),
        constraint=np.full(n, constraint),
        external=np.full(n, external),
        centrifugal=np.full(n, centrifugal) if centrifugal is not None else None,
    )


class TestIndexedAccelerationTotal:
    def test_total_sums_components(self) -> None:
        ia = _make_accel(gravity=1.0, coriolis=0.5, applied_torque=0.3)
        expected = 1.0 + 0.5 + 0.3 + 0.0 + 0.0
        np.testing.assert_allclose(ia.total, np.full(3, expected), atol=1e-12)

    def test_total_with_centrifugal(self) -> None:
        ia = _make_accel(gravity=1.0, coriolis=0.5, applied_torque=0.3, centrifugal=0.2)
        expected = 1.0 + 0.5 + 0.3 + 0.0 + 0.0 + 0.2
        np.testing.assert_allclose(ia.total, np.full(3, expected), atol=1e-12)

    def test_total_without_centrifugal(self) -> None:
        ia = _make_accel(gravity=2.0, coriolis=0.0, applied_torque=0.0)
        np.testing.assert_allclose(ia.total, np.full(3, 2.0), atol=1e-12)

    def test_zero_total_when_all_zero(self) -> None:
        ia = _make_accel(0.0, 0.0, 0.0)
        np.testing.assert_allclose(ia.total, np.zeros(3), atol=1e-12)


class TestAssertClosure:
    def test_passes_when_closed(self) -> None:
        ia = _make_accel(gravity=1.0, coriolis=0.5, applied_torque=0.3)
        # This should not raise
        ia.assert_closure(ia.total)

    def test_raises_when_closure_fails(self) -> None:
        ia = _make_accel(gravity=1.0, coriolis=0.5, applied_torque=0.3)
        wrong_total = ia.total + 10.0  # Large residual
        with pytest.raises(AccelerationClosureError):
            ia.assert_closure(wrong_total)

    def test_passes_within_tolerance(self) -> None:
        ia = _make_accel(gravity=1.0)
        noisy_total = ia.total + 1e-8  # Within default tolerance
        ia.assert_closure(noisy_total)

    def test_fails_beyond_tolerance(self) -> None:
        ia = _make_accel(gravity=0.0, coriolis=0.0, applied_torque=0.0)
        # Small total magnitude → joint-space tolerance applies (1e-6)
        wrong_total = np.full(3, 1e-5)  # Just above tolerance
        with pytest.raises(AccelerationClosureError):
            ia.assert_closure(wrong_total)


class TestGetContributionPercentages:
    def test_indexed_acceleration_returns_dict(self) -> None:
        ia = _make_accel(gravity=1.0, coriolis=0.5, applied_torque=0.3)
        result = ia.get_contribution_percentages()
        assert isinstance(result, dict)

    def test_indexed_acceleration_has_expected_keys(self) -> None:
        ia = _make_accel(gravity=1.0)
        result = ia.get_contribution_percentages()
        for key in ["gravity", "coriolis", "applied_torque", "constraint", "external"]:
            assert key in result

    def test_zero_total_returns_all_zeros(self) -> None:
        ia = _make_accel(0.0, 0.0, 0.0)
        result = ia.get_contribution_percentages()
        for v in result.values():
            assert v == pytest.approx(0.0)

    def test_all_gravity_gives_100_percent(self) -> None:
        ia = _make_accel(gravity=5.0, coriolis=0.0, applied_torque=0.0)
        result = ia.get_contribution_percentages()
        assert result["gravity"] == pytest.approx(100.0, rel=1e-6)

    def test_percentages_are_non_negative(self) -> None:
        ia = _make_accel(gravity=1.0, coriolis=0.5, applied_torque=0.25)
        result = ia.get_contribution_percentages()
        for v in result.values():
            assert v >= 0.0


class TestAccelerationClosureError:
    def test_indexed_acceleration_is_exception(self) -> None:
        assert issubclass(AccelerationClosureError, Exception)

    def test_can_raise_and_catch(self) -> None:
        with pytest.raises(AccelerationClosureError):
            raise AccelerationClosureError("test closure failure")
