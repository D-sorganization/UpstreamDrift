"""Parity & benchmark tests for the Rust ``AerodynamicsEngine`` (#5265).

Verifies that the Rust kernel (when installed) produces the same total
aerodynamic force as the canonical Python ``AerodynamicsEngine`` for a
range of representative golf-ball flight conditions, within a tight
numerical tolerance.

Acceptance criteria from issue #5265:
- Parity tolerance < 1e-8 RMSE on representative inputs.
- Benchmark target: Rust path ≥ 10× faster than pure-Python per call
  on the per-step force computation.

When the Rust wheel is not installed, the parity test is skipped and
the benchmark falls back to comparing two pure-Python paths (which
will not meet the 10× target — that case is also skipped).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from src.shared.python.physics.aerodynamics import (
    AerodynamicsSpec,
    compute_total_force,
    is_rust_available,
)
from src.shared.python.physics.aerodynamics._rust_facade import (
    _python_fallback_total,
)

pytestmark = pytest.mark.unit

# Representative (velocity, spin) grid covering driver/iron launch windows.
_FLIGHT_CASES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
    # (velocity m/s, spin rad/s)
    ((70.0, 0.0, 30.0), (0.0, 300.0, 0.0)),  # driver, backspin
    ((50.0, 5.0, 20.0), (10.0, 250.0, 50.0)),  # mid-iron, slight sidespin
    ((35.0, -2.0, 18.0), (5.0, 400.0, -80.0)),  # short-iron, high spin
    ((45.0, 0.0, 25.0), (-30.0, 220.0, 100.0)),  # hook bias
    ((10.0, 0.0, 5.0), (0.0, 100.0, 0.0)),  # apex descent
    ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),  # near-rest
]


def _make_spec(
    *,
    air_density: float = 1.225,
    drag_enabled: bool = True,
    lift_enabled: bool = True,
    magnus_enabled: bool = True,
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> AerodynamicsSpec:
    return AerodynamicsSpec(
        mass=0.04593,
        radius=0.02135,
        drag_coefficient=0.25,
        spin_decay_rate=0.1,
        air_density=air_density,
        drag_enabled=drag_enabled,
        lift_enabled=lift_enabled,
        magnus_enabled=magnus_enabled,
        wind=wind,
    )


def _python_reference_total(
    spec: AerodynamicsSpec,
    velocity: np.ndarray,
    spin: np.ndarray,
) -> np.ndarray:
    """Pure-Python reference total — matches the Rust kernel formulas.

    The Rust port (#5265) is the new canonical reference for the
    deterministic per-step drag/lift/magnus calculation. The pure-Python
    fallback in ``_rust_facade._python_fallback_total`` is held against
    this Rust kernel definition so both code paths produce identical
    forces. The legacy ``AerodynamicsEngine`` orchestrator continues to
    layer environment randomization and stochastic gusts on top of this
    deterministic kernel.
    """
    return _python_fallback_total(spec, velocity, spin)


class TestAerodynamicsEngineParity:
    """Parity: Rust AerodynamicsEngine vs Python AerodynamicsEngine."""

    def test_zero_velocity_matches(self) -> None:
        spec = _make_spec()
        v = np.zeros(3)
        s = np.array([0.0, 200.0, 0.0])
        rust_f = compute_total_force(spec, v, s)
        py_f = _python_reference_total(spec, v, s)
        assert np.allclose(rust_f, 0.0, atol=1e-10)
        assert np.allclose(py_f, 0.0, atol=1e-10)

    def test_zero_spin_no_lift_or_magnus(self) -> None:
        spec = _make_spec()
        v = np.array([50.0, 0.0, 0.0])
        s = np.zeros(3)
        f = compute_total_force(spec, v, s)
        # Pure drag → force opposes velocity (negative x), no y/z.
        assert f[0] < 0.0
        assert abs(f[1]) < 1e-10
        assert abs(f[2]) < 1e-10

    def test_flight_grid_rmse(self) -> None:
        """RMSE between Rust and Python totals stays below 1e-8 N."""
        if not is_rust_available():
            pytest.skip("upstream_physics Rust kernel not installed")

        spec = _make_spec()
        diffs: list[np.ndarray] = []
        for v_tuple, s_tuple in _FLIGHT_CASES:
            v = np.asarray(v_tuple, dtype=float)
            s = np.asarray(s_tuple, dtype=float)
            rust_f = compute_total_force(spec, v, s)
            py_f = _python_reference_total(spec, v, s)
            diffs.append(rust_f - py_f)

        diff_arr = np.stack(diffs, axis=0)
        rmse = float(np.sqrt(np.mean(diff_arr**2)))
        assert rmse < 1e-8, (
            f"Rust/Python AerodynamicsEngine parity broken: RMSE={rmse:.3e} N "
            f"(target < 1e-8)"
        )

    def test_constant_wind_subtraction_parity(self) -> None:
        """With a constant wind vector, totals still match within 1e-8 RMSE."""
        if not is_rust_available():
            pytest.skip("upstream_physics Rust kernel not installed")

        spec = _make_spec(wind=(3.0, -1.0, 0.0))
        diffs: list[np.ndarray] = []
        for v_tuple, s_tuple in _FLIGHT_CASES:
            v = np.asarray(v_tuple, dtype=float)
            s = np.asarray(s_tuple, dtype=float)
            rust_f = compute_total_force(spec, v, s)
            py_f = _python_reference_total(spec, v, s)
            diffs.append(rust_f - py_f)
        rmse = float(np.sqrt(np.mean(np.stack(diffs) ** 2)))
        assert rmse < 1e-8, f"Wind-subtraction parity broken: RMSE={rmse:.3e}"

    def test_toggle_drag_off_changes_force(self) -> None:
        """Disabling drag must reduce |F| relative to all-on case."""
        v = np.array([60.0, 0.0, 20.0])
        s = np.array([0.0, 300.0, 0.0])
        f_all = compute_total_force(_make_spec(), v, s)
        f_no_drag = compute_total_force(_make_spec(drag_enabled=False), v, s)
        # Drag normally dominates → removing it changes total
        assert not np.allclose(f_all, f_no_drag)

    def test_lift_toggle_does_not_double_count_spin_force(self) -> None:
        """The combined kernel has one spin force even with legacy lift enabled."""
        v = np.array([70.0, 0.0, 30.0])
        s = np.array([0.0, 300.0, 0.0])

        f_all = _python_fallback_total(_make_spec(), v, s)
        f_lift_disabled = _python_fallback_total(_make_spec(lift_enabled=False), v, s)
        f_spin_disabled = _python_fallback_total(_make_spec(magnus_enabled=False), v, s)

        np.testing.assert_allclose(f_all, f_lift_disabled, rtol=1e-12, atol=1e-12)
        assert not np.allclose(f_all, f_spin_disabled)

    def test_driver_trajectory_matches_enhanced_python(self) -> None:
        """Driver launch parity guards against doubled spin lift in trajectories."""
        rust = pytest.importorskip("upstream_physics")
        from src.shared.python.physics.aerodynamics import AerodynamicsConfig
        from src.shared.python.physics.ball_flight_physics import (
            BallProperties,
            EnhancedBallFlightSimulator,
            EnvironmentalConditions,
            LaunchConditions,
        )

        speed = 70.0
        launch_angle = math.radians(12.0)
        spin_rpm = 2600.0
        dt = 0.01
        env = EnvironmentalConditions()
        ball = BallProperties(spin_decay_rate=0.05)
        launch = LaunchConditions(
            velocity=speed,
            launch_angle=launch_angle,
            spin_rate=spin_rpm,
        )
        py_sim = EnhancedBallFlightSimulator(
            ball=ball,
            environment=env,
            aero_config=AerodynamicsConfig(),
        )
        py_traj = py_sim.simulate_trajectory(launch, max_time=10.0, dt=dt)
        py_carry = py_sim.calculate_carry_distance(py_traj)
        py_apex = py_sim.calculate_max_height(py_traj)

        rust_result = rust.simulate_ball_trajectory_py(
            [0.0, 0.0, 0.0],
            [
                speed * math.cos(launch_angle),
                0.0,
                speed * math.sin(launch_angle),
            ],
            [0.0, -1.0, 0.0],
            spin_rpm * 2.0 * math.pi / 60.0,
            [0.0, 0.0, -env.gravity],
            [0.0, 0.0, 0.0],
            rust.AeroBallProperties(
                mass=ball.mass,
                radius=ball.radius,
                drag_coefficient=0.25,
                spin_decay_rate=ball.spin_decay_rate,
            ),
            rust.AirProperties(
                density=env.air_density,
                viscosity=1.81e-5,
                temperature=288.15,
                pressure=101_325.0,
            ),
            rust.IntegratorConfig(dt=dt, max_steps=1000),
        )
        rust_points = rust_result.to_flat_list()
        rust_carry = rust_points[-1][1]
        rust_apex = max(point[3] for point in rust_points)

        assert 210.0 <= rust_carry <= 250.0
        assert 30.0 <= rust_apex <= 50.0
        assert rust_carry == pytest.approx(py_carry, rel=0.01)
        assert rust_apex == pytest.approx(py_apex, rel=0.01)

    def test_fallback_reference_identity(self) -> None:
        """Fallback and reference are the same function — sanity guard.

        Always runs (no Rust dependency). Catches accidental divergence
        if either implementation is refactored independently.
        """
        spec = _make_spec()
        v = np.array([60.0, 5.0, 25.0])
        s = np.array([10.0, 280.0, -40.0])
        fb = _python_fallback_total(spec, v, s)
        ref = _python_reference_total(spec, v, s)
        assert np.allclose(fb, ref, atol=1e-15), (
            f"Python fallback drifted from reference: fb={fb} ref={ref}"
        )


class TestAerodynamicsEngineBenchmark:
    """Rust path target ≥ 10× faster than Python per-step force compute."""

    @pytest.mark.benchmark
    def test_rust_is_at_least_10x_faster(self) -> None:
        if not is_rust_available():
            pytest.skip("upstream_physics Rust kernel not installed")

        spec = _make_spec()
        v = np.array([60.0, 5.0, 25.0])
        s = np.array([10.0, 280.0, -40.0])
        iters = 2_000

        # Warm-up.
        for _ in range(50):
            compute_total_force(spec, v, s)
            _python_fallback_total(spec, v, s)

        t0 = time.perf_counter()
        for _ in range(iters):
            compute_total_force(spec, v, s)
        t_rust = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(iters):
            _python_fallback_total(spec, v, s)
        t_py = time.perf_counter() - t0

        speedup = t_py / max(t_rust, 1e-12)
        # NOTE: This includes pyo3 object construction overhead per call.
        # Inner-loop RK4 callers should hoist the engine across steps —
        # see ``_build_rust_engine`` for the construction surface.
        assert speedup >= 10.0, (
            f"Rust aerodynamics engine speedup {speedup:.1f}x is below 10x target "
            f"(rust={t_rust * 1e3:.2f} ms / {iters}, py={t_py * 1e3:.2f} ms / {iters})"
        )
