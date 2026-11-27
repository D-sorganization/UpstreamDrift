"""Magnus transverse-spin and frame-sign acceptance for #15042."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.physics.aerodynamics import (
    AerodynamicsSpec,
    compute_total_force,
    is_rust_available,
)
from src.shared.python.physics.aerodynamics._rust_facade import _python_fallback_total

pytestmark = pytest.mark.unit


def _spec() -> AerodynamicsSpec:
    return AerodynamicsSpec(
        mass=0.04593,
        radius=0.02135,
        drag_coefficient=0.25,
        spin_decay_rate=0.05,
        air_density=1.225,
        drag_enabled=False,
        lift_enabled=True,
        magnus_enabled=True,
    )


class TestMagnusTransverseSpin15042:
    def test_rifle_spin_produces_zero_spin_force(self) -> None:
        v = np.array([50.0, 0.0, 0.0])
        spin = np.array([250.0, 0.0, 0.0])
        f = _python_fallback_total(_spec(), v, spin)
        assert math.hypot(*f) < 1e-9

    def test_backspin_negative_y_lifts_positive_z(self) -> None:
        v = np.array([50.0, 0.0, 0.0])
        spin = np.array([0.0, -300.0, 0.0])
        f = _python_fallback_total(_spec(), v, spin)
        assert f[2] > 0.01
        assert abs(f[0]) < 1e-6 and abs(f[1]) < 1e-6

    def test_near_rifle_spin_decreases_smoothly(self) -> None:
        v = np.array([50.0, 0.0, 0.0])
        spec = _spec()
        omega = 250.0
        epsilons = (5.0, 1.0, 0.1, 1e-2, 1e-4, 1e-6, 0.0)
        prev = math.hypot(
            *_python_fallback_total(spec, v, np.array([omega, epsilons[0], 0.0]))
        )
        for eps in epsilons[1:]:
            mag = math.hypot(
                *_python_fallback_total(spec, v, np.array([omega, eps, 0.0]))
            )
            assert mag <= prev + 1e-9
            prev = mag

    def test_python_fallback_matches_rust_kernel_on_grid(self) -> None:
        if not is_rust_available():
            pytest.skip("upstream_physics Rust kernel not installed")
        spec = _spec()
        v = np.array([62.0, -4.0, 18.0])
        spin = np.array([12.0, 265.0, -35.0])
        rust = compute_total_force(spec, v, spin)
        py = _python_fallback_total(spec, v, spin)
        assert np.allclose(rust, py, atol=1e-10)
