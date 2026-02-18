"""Performance benchmarks for core physics operations.

Establishes baseline metrics for spatial algebra and physics computations
that are critical to simulation performance. Tracks regressions over time.

Run with: pytest tests/benchmarks/test_physics_benchmarks.py --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark
class TestSpatialAlgebraBenchmarks:
    """Baseline performance benchmarks for spatial algebra operations."""

    def test_rotation_matrix_creation(self, benchmark: pytest.fixture) -> None:
        """Benchmark rotation matrix creation from euler angles."""
        try:
            from src.shared.python.spatial_algebra.pose6dof import (
                euler_to_rotation_matrix,
            )
        except ImportError:
            pytest.skip("spatial_algebra not available")

        euler = [0.1, 0.2, 0.3]
        result = benchmark(euler_to_rotation_matrix, euler)
        assert result.shape == (3, 3)

    def test_skew_matrix(self, benchmark: pytest.fixture) -> None:
        """Benchmark skew-symmetric matrix computation."""
        try:
            from src.shared.python.spatial_algebra.spatial_vectors import skew
        except ImportError:
            pytest.skip("spatial_vectors not available")

        v = np.array([1.0, 2.0, 3.0])
        result = benchmark(skew, v)
        assert result.shape == (3, 3)

    def test_numpy_matrix_multiply_6x6(self, benchmark: pytest.fixture) -> None:
        """Benchmark 6x6 matrix multiplication (common in spatial algebra)."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((6, 6))
        B = rng.standard_normal((6, 6))
        result = benchmark(np.matmul, A, B)
        assert result.shape == (6, 6)


@pytest.mark.benchmark
class TestPhysicsFunctionBenchmarks:
    """Baseline benchmarks for physics computation functions."""

    def test_gravity_on_slope(self, benchmark: pytest.fixture) -> None:
        """Benchmark gravity component computation on slope."""
        from src.shared.python.physics.terrain import compute_gravity_on_slope

        # Clear cache to benchmark raw computation, then test cached path
        compute_gravity_on_slope.cache_clear()
        result = benchmark(compute_gravity_on_slope, 15.0)
        g_par, g_perp = result
        assert g_par > 0
        assert g_perp > 0

    def test_gravity_on_slope_cached(self, benchmark: pytest.fixture) -> None:
        """Benchmark gravity computation with warm cache."""
        from src.shared.python.physics.terrain import compute_gravity_on_slope

        # Warm the cache first
        compute_gravity_on_slope.cache_clear()
        compute_gravity_on_slope(15.0)
        result = benchmark(compute_gravity_on_slope, 15.0)
        g_par, g_perp = result
        assert g_par > 0
        assert g_perp > 0

    def test_spin_decay(self, benchmark: pytest.fixture) -> None:
        """Benchmark spin decay computation."""
        from src.shared.python.physics.flight_model_options import compute_spin_decay

        compute_spin_decay.cache_clear()
        result = benchmark(compute_spin_decay, 300.0, 1.5, 0.05)
        assert result > 0
        assert result < 300.0

    def test_air_density_at_altitude(self, benchmark: pytest.fixture) -> None:
        """Benchmark air density altitude correction."""
        from src.shared.python.physics.flight_model_options import (
            compute_air_density_at_altitude,
        )

        compute_air_density_at_altitude.cache_clear()
        result = benchmark(compute_air_density_at_altitude, 1.225, 500.0)
        assert 0 < result < 1.225

    def test_section_inertia(self, benchmark: pytest.fixture) -> None:
        """Benchmark hollow section moment of inertia computation."""
        from src.shared.python.physics.flexible_shaft import compute_section_inertia

        compute_section_inertia.cache_clear()
        result = benchmark(compute_section_inertia, 0.012, 0.001)
        assert result > 0

    def test_section_area(self, benchmark: pytest.fixture) -> None:
        """Benchmark hollow section cross-sectional area computation."""
        from src.shared.python.physics.flexible_shaft import compute_section_area

        compute_section_area.cache_clear()
        result = benchmark(compute_section_area, 0.012, 0.001)
        assert result > 0

    def test_get_club_config(self, benchmark: pytest.fixture) -> None:
        """Benchmark club configuration lookup."""
        from src.shared.python.physics.equipment import get_club_config

        get_club_config.cache_clear()
        result = benchmark(get_club_config, "driver")
        assert "head_mass" in result


@pytest.mark.benchmark
class TestAerodynamicsBenchmarks:
    """Baseline benchmarks for aerodynamic force calculations."""

    def test_drag_force(self, benchmark: pytest.fixture) -> None:
        """Benchmark drag force calculation."""
        try:
            from src.shared.python.physics.aerodynamics import DragModel
        except ImportError:
            pytest.skip("aerodynamics module not available")

        model = DragModel()
        velocity = np.array([50.0, 10.0, 5.0])
        result = benchmark(model.calculate, velocity)
        assert result.shape == (3,)

    def test_lift_force(self, benchmark: pytest.fixture) -> None:
        """Benchmark lift force calculation."""
        try:
            from src.shared.python.physics.aerodynamics import LiftModel
        except ImportError:
            pytest.skip("aerodynamics module not available")

        model = LiftModel()
        velocity = np.array([50.0, 0.0, 5.0])
        spin = np.array([0.0, -300.0, 0.0])
        result = benchmark(model.calculate, velocity, spin)
        assert result.shape == (3,)

    def test_magnus_force(self, benchmark: pytest.fixture) -> None:
        """Benchmark Magnus force calculation."""
        try:
            from src.shared.python.physics.aerodynamics import MagnusModel
        except ImportError:
            pytest.skip("aerodynamics module not available")

        model = MagnusModel()
        velocity = np.array([50.0, 0.0, 5.0])
        spin = np.array([0.0, -300.0, 50.0])
        result = benchmark(model.calculate, velocity, spin)
        assert result.shape == (3,)

    def test_combined_aero_forces(self, benchmark: pytest.fixture) -> None:
        """Benchmark combined aerodynamic engine force computation."""
        try:
            from src.shared.python.physics.aerodynamics import AerodynamicsEngine
        except ImportError:
            pytest.skip("aerodynamics module not available")

        engine = AerodynamicsEngine()
        velocity = np.array([60.0, 5.0, 10.0])
        spin = np.array([10.0, -250.0, 30.0])
        result = benchmark(engine.compute_forces, velocity, spin)
        assert "total" in result
        assert result["total"].shape == (3,)


@pytest.mark.benchmark
class TestTerrainBenchmarks:
    """Baseline benchmarks for terrain query operations."""

    def test_elevation_query(self, benchmark: pytest.fixture) -> None:
        """Benchmark terrain elevation interpolation."""
        from src.shared.python.physics.terrain import ElevationMap

        elev = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=45.0,
        )
        result = benchmark(elev.get_elevation, 50.0, 50.0)
        assert isinstance(result, float)

    def test_surface_normal_query(self, benchmark: pytest.fixture) -> None:
        """Benchmark surface normal vector computation."""
        from src.shared.python.physics.terrain import ElevationMap

        elev = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=45.0,
        )
        result = benchmark(elev.get_normal, 50.0, 50.0)
        assert result.shape == (3,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_slope_angle_query(self, benchmark: pytest.fixture) -> None:
        """Benchmark slope angle computation."""
        from src.shared.python.physics.terrain import ElevationMap

        elev = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
        )
        result = benchmark(elev.get_slope_angle, 50.0, 50.0)
        assert 0 <= result <= 90
