"""Tests for src.shared.python.physics.topography module.

Covers:
- TopographyBounds dataclass properties
- TopographyData construction and loading
- Elevation queries (heightmap and contour point paths)
- Gradient queries (numerical accuracy)
- Normal vector computation
- Vectorized sample_uniform / to_heightmap
- Factory functions (create_flat_terrain, create_sloped_terrain, create_undulating_terrain)
- TopographyProvider Protocol structural subtyping
- save_to_file / from_file round-trip (npy, csv, json)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.physics.topography import (
    ElevationPoint,
    TopographyBounds,
    TopographyData,
    TopographyProvider,
    create_flat_terrain,
    create_sloped_terrain,
    create_undulating_terrain,
)

# ---------------------------------------------------------------------------
# TopographyBounds
# ---------------------------------------------------------------------------


class TestTopographyBounds:
    """Tests for TopographyBounds dataclass."""

    def test_defaults(self) -> None:
        b = TopographyBounds()
        assert b.min_x == 0.0
        assert b.max_x == 100.0
        assert b.min_y == 0.0
        assert b.max_y == 100.0

    def test_width_height(self) -> None:
        b = TopographyBounds(min_x=10.0, max_x=60.0, min_y=5.0, max_y=25.0)
        assert b.width == pytest.approx(50.0)
        assert b.height == pytest.approx(20.0)

    def test_elevation_range(self) -> None:
        b = TopographyBounds(min_z=-2.0, max_z=8.0)
        assert b.elevation_range == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TopographyData – unloaded state
# ---------------------------------------------------------------------------


class TestTopographyDataUnloaded:
    """Unloaded TopographyData should return safe defaults."""

    def test_not_loaded_initially(self) -> None:
        topo = TopographyData()
        assert not topo.is_loaded

    def test_get_elevation_returns_zero_when_unloaded(self) -> None:
        topo = TopographyData()
        assert topo.get_elevation_at(np.array([5.0, 5.0])) == 0.0

    def test_sample_uniform_returns_zeros_when_unloaded(self) -> None:
        topo = TopographyData()
        result = topo.sample_uniform(10, 10)
        assert result.shape == (10, 10)
        assert np.all(result == 0.0)

    def test_to_heightmap_returns_zeros_when_unloaded(self) -> None:
        topo = TopographyData()
        result = topo.to_heightmap(20)
        assert result.shape == (20, 20)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# TopographyData – heightmap path
# ---------------------------------------------------------------------------


class TestTopographyDataHeightmap:
    """Tests using set_heightmap (RegularGridInterpolator path)."""

    @pytest.fixture
    def flat_topo(self) -> TopographyData:
        """2m elevation everywhere."""
        topo = TopographyData(TopographyBounds(min_x=0, max_x=10, min_y=0, max_y=10))
        heightmap = np.full((10, 10), 2.0)
        topo.set_heightmap(heightmap, smooth=False)
        return topo

    @pytest.fixture
    def sloped_topo(self) -> TopographyData:
        """Linear slope: z = x/10 (0 at x=0, 1 at x=10)."""
        topo = TopographyData(TopographyBounds(min_x=0, max_x=10, min_y=0, max_y=10))
        ny, nx = 20, 20
        x = np.linspace(0, 10, nx)
        y = np.linspace(0, 10, ny)
        X, _ = np.meshgrid(x, y)
        heightmap = X / 10.0
        topo.set_heightmap(heightmap, smooth=False)
        return topo

    def test_is_loaded_after_set_heightmap(self, flat_topo: TopographyData) -> None:
        assert flat_topo.is_loaded

    def test_flat_elevation_at_center(self, flat_topo: TopographyData) -> None:
        z = flat_topo.get_elevation_at(np.array([5.0, 5.0]))
        assert z == pytest.approx(2.0, abs=0.01)

    def test_flat_elevation_at_corners(self, flat_topo: TopographyData) -> None:
        for x, y in [(0, 0), (0, 10), (10, 0), (10, 10)]:
            z = flat_topo.get_elevation_at(np.array([float(x), float(y)]))
            assert z == pytest.approx(2.0, abs=0.05), f"Corner ({x},{y}) failed: z={z}"

    def test_flat_gradient_is_near_zero(self, flat_topo: TopographyData) -> None:
        grad = flat_topo.get_gradient_at(np.array([5.0, 5.0]))
        assert grad.shape == (2,)
        assert abs(grad[0]) < 0.1  # nearly flat
        assert abs(grad[1]) < 0.1

    def test_sloped_elevation_increases_with_x(
        self, sloped_topo: TopographyData
    ) -> None:
        z0 = sloped_topo.get_elevation_at(np.array([0.0, 5.0]))
        z5 = sloped_topo.get_elevation_at(np.array([5.0, 5.0]))
        z10 = sloped_topo.get_elevation_at(np.array([10.0, 5.0]))
        assert z0 < z5 < z10

    def test_sloped_gradient_dzdx_positive(self, sloped_topo: TopographyData) -> None:
        # For z = x/10, dz/dx = 0.1, dz/dy = 0
        grad = sloped_topo.get_gradient_at(np.array([5.0, 5.0]))
        assert grad[0] == pytest.approx(0.1, abs=0.02)  # dz/dx
        assert abs(grad[1]) < 0.02  # dz/dy ~ 0

    def test_normal_vector_is_unit_length(self, sloped_topo: TopographyData) -> None:
        normal = sloped_topo.get_normal_at(np.array([5.0, 5.0]))
        assert normal.shape == (3,)
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-9)

    def test_normal_flat_points_up(self, flat_topo: TopographyData) -> None:
        normal = flat_topo.get_normal_at(np.array([5.0, 5.0]))
        assert normal[2] > 0.99  # mostly pointing up

    def test_out_of_bounds_position_is_clamped(self, flat_topo: TopographyData) -> None:
        # Position outside bounds should be clamped and return a finite value
        z = flat_topo.get_elevation_at(np.array([100.0, 100.0]))
        assert math.isfinite(z)

    def test_sample_uniform_shape(self, flat_topo: TopographyData) -> None:
        result = flat_topo.sample_uniform(5, 8)
        assert result.shape == (8, 5)

    def test_sample_uniform_values_flat(self, flat_topo: TopographyData) -> None:
        result = flat_topo.sample_uniform(10, 10)
        # Flat terrain: all values close to 2.0
        assert np.all(np.abs(result - 2.0) < 0.1)

    def test_to_heightmap_shape(self, flat_topo: TopographyData) -> None:
        result = flat_topo.to_heightmap(50)
        assert result.shape == (50, 50)

    def test_to_heightmap_is_consistent_with_sample_uniform(
        self, sloped_topo: TopographyData
    ) -> None:
        h1 = sloped_topo.to_heightmap(20)
        h2 = sloped_topo.sample_uniform(20, 20)
        np.testing.assert_array_almost_equal(h1, h2, decimal=10)

    def test_get_statistics_flat(self, flat_topo: TopographyData) -> None:
        stats = flat_topo.get_statistics()
        assert "min_elevation" in stats
        assert "max_elevation" in stats
        assert "mean_elevation" in stats
        assert stats["min_elevation"] == pytest.approx(2.0, abs=0.1)
        assert stats["max_elevation"] == pytest.approx(2.0, abs=0.1)

    def test_smoothing_does_not_crash(self) -> None:
        topo = TopographyData(TopographyBounds(min_x=0, max_x=10, min_y=0, max_y=10))
        heightmap = np.random.default_rng(42).standard_normal((20, 20))
        topo.set_heightmap(heightmap, smooth=True, smooth_sigma=1.5)
        assert topo.is_loaded


# ---------------------------------------------------------------------------
# TopographyData – contour point path
# ---------------------------------------------------------------------------


class TestTopographyDataContourPoints:
    """Tests using set_contour_points (RBF/LinearNDInterpolator path)."""

    @pytest.fixture
    def contour_topo(self) -> TopographyData:
        """Four corners at different elevations."""
        points = [
            ElevationPoint(x=0.0, y=0.0, z=0.0),
            ElevationPoint(x=10.0, y=0.0, z=1.0),
            ElevationPoint(x=0.0, y=10.0, z=2.0),
            ElevationPoint(x=10.0, y=10.0, z=3.0),
            ElevationPoint(x=5.0, y=5.0, z=1.5),  # center
        ]
        topo = TopographyData()
        topo.set_contour_points(points)
        return topo

    def test_is_loaded_after_set_contour_points(
        self, contour_topo: TopographyData
    ) -> None:
        assert contour_topo.is_loaded

    def test_elevation_at_known_point(self, contour_topo: TopographyData) -> None:
        # The center point is set at z=1.5
        z = contour_topo.get_elevation_at(np.array([5.0, 5.0]))
        assert math.isfinite(z)

    def test_bounds_set_from_points(self, contour_topo: TopographyData) -> None:
        b = contour_topo.bounds
        assert b.min_x == pytest.approx(0.0)
        assert b.max_x == pytest.approx(10.0)
        assert b.min_y == pytest.approx(0.0)
        assert b.max_y == pytest.approx(10.0)

    def test_sample_uniform_contour(self, contour_topo: TopographyData) -> None:
        result = contour_topo.sample_uniform(5, 5)
        assert result.shape == (5, 5)
        assert np.all(np.isfinite(result))

    def test_empty_contour_points_does_not_crash(self) -> None:
        topo = TopographyData()
        topo.set_contour_points([])
        # After setting empty list, is_loaded stays False
        assert not topo.is_loaded


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    """Tests for create_flat_terrain, create_sloped_terrain, create_undulating_terrain."""

    def test_create_flat_terrain_elevation(self) -> None:
        topo = create_flat_terrain(width=50.0, height=30.0, elevation=5.0)
        z = topo.get_elevation_at(np.array([25.0, 15.0]))
        assert z == pytest.approx(5.0, abs=0.1)

    def test_create_flat_terrain_bounds(self) -> None:
        topo = create_flat_terrain(width=50.0, height=30.0)
        assert topo.bounds.width == pytest.approx(50.0)
        assert topo.bounds.height == pytest.approx(30.0)

    def test_create_sloped_terrain_increases_downhill(self) -> None:
        # Default slope in +x direction
        topo = create_sloped_terrain(
            width=100.0,
            height=100.0,
            slope_direction=np.array([1.0, 0.0]),
            slope_magnitude=0.1,
            base_elevation=0.0,
        )
        z0 = topo.get_elevation_at(np.array([0.0, 50.0]))
        z50 = topo.get_elevation_at(np.array([50.0, 50.0]))
        # slope_magnitude=0.1 → slope goes downhill in +x, so z0 > z50
        assert z0 > z50

    def test_create_undulating_terrain_has_variation(self) -> None:
        topo = create_undulating_terrain(
            width=100.0, height=100.0, amplitude=2.0, wavelength=20.0
        )
        result = topo.sample_uniform(20, 20)
        # Undulating terrain should have some variance (not flat)
        assert np.std(result) > 0.1

    def test_all_factory_outputs_are_loaded(self) -> None:
        assert create_flat_terrain().is_loaded
        assert create_sloped_terrain().is_loaded
        assert create_undulating_terrain().is_loaded


# ---------------------------------------------------------------------------
# File I/O round-trip
# ---------------------------------------------------------------------------


class TestFileIO:
    """Round-trip tests for save_to_file / from_file."""

    def test_npy_round_trip(self, tmp_path: Path) -> None:
        topo = create_sloped_terrain(width=20.0, height=20.0, slope_magnitude=0.05)
        outfile = tmp_path / "terrain.npy"
        topo.save_to_file(outfile)
        loaded = TopographyData.from_file(outfile, width=20.0, height=20.0)
        assert loaded.is_loaded
        # Elevation at center should be approximately the same
        z_orig = topo.get_elevation_at(np.array([10.0, 10.0]))
        z_loaded = loaded.get_elevation_at(np.array([10.0, 10.0]))
        assert z_loaded == pytest.approx(z_orig, abs=0.05)

    def test_json_round_trip(self, tmp_path: Path) -> None:
        topo = create_flat_terrain(width=10.0, height=10.0, elevation=3.5)
        outfile = tmp_path / "terrain.json"
        topo.save_to_file(outfile)
        loaded = TopographyData.from_file(outfile)
        assert loaded.is_loaded
        z = loaded.get_elevation_at(np.array([5.0, 5.0]))
        assert z == pytest.approx(3.5, abs=0.1)

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        # Write CSV manually
        csv_file = tmp_path / "terrain.csv"
        csv_file.write_text("x,y,elevation\n0,0,1\n10,0,2\n0,10,3\n10,10,4\n5,5,2.5\n")
        loaded = TopographyData.from_file(csv_file)
        assert loaded.is_loaded
        z = loaded.get_elevation_at(np.array([5.0, 5.0]))
        assert math.isfinite(z)

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        topo = create_flat_terrain()
        with pytest.raises(ValueError, match="Unsupported"):
            topo.save_to_file(tmp_path / "terrain.xyz")

    def test_from_file_unsupported_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.xyz"
        bad_file.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            TopographyData.from_file(bad_file)


# ---------------------------------------------------------------------------
# Protocol structural subtyping
# ---------------------------------------------------------------------------


class TestTopographyProviderProtocol:
    """Verify TopographyData satisfies TopographyProvider Protocol."""

    def test_topography_data_is_provider(self) -> None:
        topo = create_flat_terrain()
        assert isinstance(topo, TopographyProvider)

    def test_protocol_methods_callable(self) -> None:
        topo = create_flat_terrain()
        # All Protocol methods must exist and be callable
        assert callable(topo.get_elevation_at)
        assert callable(topo.get_gradient_at)
        _ = topo.bounds  # property access

    def test_custom_provider_satisfies_protocol(self) -> None:
        """A class with correct method signatures should satisfy the protocol."""

        class MinimalProvider:
            def get_elevation_at(self, position: np.ndarray) -> float:
                return 0.0

            def get_gradient_at(self, position: np.ndarray) -> np.ndarray:
                return np.zeros(2)

            @property
            def bounds(self) -> TopographyBounds:
                return TopographyBounds()

        provider = MinimalProvider()
        assert isinstance(provider, TopographyProvider)
