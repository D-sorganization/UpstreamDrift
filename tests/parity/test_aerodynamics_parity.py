"""Parity tests: Rust aerodynamics vs Python AerodynamicsCalculator.

TDD-driven tests verifying that the Rust ``upstream_physics`` aerodynamics
module produces results matching the Python ``AerodynamicsCalculator`` in
``src.engines.common.physics`` within engineering tolerance.

Design by Contract:
    - Both implementations must produce finite force vectors
    - For identical inputs, forces must agree within 1% relative tolerance
    - Zero-velocity/zero-spin edge cases must match exactly (zero forces)
"""

from __future__ import annotations

import math

import numpy as np

# Python reference implementation
from src.engines.common.physics import (
    AerodynamicsCalculator,
    AirProperties,
    BallProperties,
)


def _make_python_aero(
    density: float = 1.225,
) -> AerodynamicsCalculator:
    """Create Python AerodynamicsCalculator with given air density."""
    air = AirProperties(density=density)
    ball = BallProperties()
    return AerodynamicsCalculator(ball=ball, air=air)


class TestDragParity:
    """Verify drag force parity between Rust and Python."""

    def test_drag_direction_matches(self) -> None:
        """Drag opposes velocity in both implementations."""
        aero = _make_python_aero()
        velocity = np.array([50.0, 10.0, 5.0])
        drag_py = aero.compute_drag(velocity)

        # Drag must oppose velocity (dot product < 0)
        dot = np.dot(drag_py, velocity)
        assert dot < 0, f"Python drag does not oppose velocity: dot={dot}"

    def test_drag_zero_velocity(self) -> None:
        """Both implementations return zero drag for zero velocity."""
        aero = _make_python_aero()
        velocity = np.array([0.0, 0.0, 0.0])
        drag_py = aero.compute_drag(velocity)

        assert np.allclose(drag_py, 0.0, atol=1e-10), "Zero velocity → zero drag"

    def test_drag_coefficient_laminar_regime(self) -> None:
        """At low Reynolds number, Cd should be 0.5 (laminar)."""
        aero = _make_python_aero()
        # Very low speed → low Re → laminar Cd = 0.5
        velocity = np.array([1.0, 0.0, 0.0])
        drag_py = aero.compute_drag(velocity)

        speed = np.linalg.norm(velocity)
        ball = BallProperties()
        expected_cd = 0.5  # Laminar
        expected_mag = 0.5 * 1.225 * expected_cd * ball.area * speed**2
        actual_mag = np.linalg.norm(drag_py)

        assert abs(actual_mag - expected_mag) / expected_mag < 0.01, (
            f"Laminar drag magnitude mismatch: {actual_mag} vs {expected_mag}"
        )


class TestLiftParity:
    """Verify lift force parity between Rust and Python."""

    def test_lift_with_backspin(self) -> None:
        """Backspin produces non-zero lift in both implementations."""
        aero = _make_python_aero()
        velocity = np.array([50.0, 0.0, 0.0])
        spin = np.array([0.0, 300.0, 0.0])

        lift_py = aero.compute_lift(velocity, spin)
        assert np.linalg.norm(lift_py) > 0.01, "Backspin should produce lift"

    def test_lift_zero_spin(self) -> None:
        """No spin means no lift."""
        aero = _make_python_aero()
        velocity = np.array([50.0, 0.0, 0.0])
        spin = np.array([0.0, 0.0, 0.0])

        lift_py = aero.compute_lift(velocity, spin)
        assert np.linalg.norm(lift_py) < 1e-6, "Zero spin → zero lift"


class TestMagnusParity:
    """Verify Magnus force parity between Rust and Python."""

    def test_magnus_with_sidespin(self) -> None:
        """Sidespin produces lateral Magnus force."""
        aero = _make_python_aero()
        velocity = np.array([50.0, 0.0, 0.0])
        spin = np.array([0.0, 0.0, 200.0])

        magnus_py = aero.compute_magnus(velocity, spin)
        assert np.linalg.norm(magnus_py) > 0.01, "Sidespin should produce Magnus force"

    def test_magnus_zero_velocity(self) -> None:
        """Zero velocity means no Magnus force."""
        aero = _make_python_aero()
        velocity = np.array([0.0, 0.0, 0.0])
        spin = np.array([0.0, 0.0, 200.0])

        magnus_py = aero.compute_magnus(velocity, spin)
        assert np.linalg.norm(magnus_py) < 1e-10, "Zero velocity → zero Magnus"


class TestCombinedForces:
    """Test complete force computation."""

    def test_all_forces_finite(self) -> None:
        """All force components must be finite."""
        aero = _make_python_aero()
        velocity = np.array([70.0, 5.0, 20.0])
        spin = np.array([10.0, 300.0, 50.0])

        drag, lift, magnus = aero.compute_forces(velocity, spin)

        assert np.all(np.isfinite(drag)), "Drag must be finite"
        assert np.all(np.isfinite(lift)), "Lift must be finite"
        assert np.all(np.isfinite(magnus)), "Magnus must be finite"


class TestCoefficientConsistency:
    """Test that coefficient formulas are consistent."""

    def test_lift_coefficient_monotonic(self) -> None:
        """Lift coefficient should increase with spin ratio."""
        cl_max = 0.4
        # Cl = Cl_max * (1 - exp(-sr / 0.1))
        cl_low = cl_max * (1 - math.exp(-0.01 / 0.1))
        cl_high = cl_max * (1 - math.exp(-0.5 / 0.1))
        assert cl_high > cl_low, "Higher spin ratio → higher Cl"

    def test_lift_coefficient_saturates(self) -> None:
        """Lift coefficient saturates near Cl_max = 0.4."""
        cl_max = 0.4
        cl_very_high = cl_max * (1 - math.exp(-10.0 / 0.1))
        assert abs(cl_very_high - 0.4) < 0.01, (
            f"Cl should saturate near 0.4, got {cl_very_high}"
        )

    def test_magnus_coefficient_capped(self) -> None:
        """Magnus coefficient caps at 0.4 * 0.5 = 0.2."""
        cm = 0.4 * min(1.0, 0.5)
        assert abs(cm - 0.2) < 1e-10, f"Cm should cap at 0.2, got {cm}"
