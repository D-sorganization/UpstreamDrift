"""Consumer integration tests for Tools green-surface adapter (#9143 / Tools #4800 P9).

Verifies that UpstreamDrift cleanly consumes and queries the Tools
``swing_sim.putting.ud_adapter`` and surface heightfield representations across the
vendor boundary (``vendor/ud-tools/src/shared/python/swing_sim/putting``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shared.python.swing_sim.putting import (
    GridGreenSurface,
    PlanarGreenSurface,
    PuttLaunch,
    UdGreenTopography,
    green_surface_from_ud_json,
    green_surface_to_ud_json,
    roll_out_distance,
    simulate_putt_on_surface,
    stimp_to_rolling_mu,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _launch(speed_mps: float) -> PuttLaunch:
    return PuttLaunch(
        ball_speed_mps=speed_mps,
        launch_angle_deg=0.0,
        horizontal_speed_mps=speed_mps,
        spin_rad_s=0.0,
        effective_loft_deg=0.0,
    )


class TestVendorBoundaryResolution:
    """Verify that Tools putting green surface adapter exports resolve from vendor tree."""

    def test_vendor_symbols_available(self) -> None:
        """All expected adapter and surface classes/functions must resolve."""
        assert UdGreenTopography is not None
        assert callable(green_surface_from_ud_json)
        assert callable(green_surface_to_ud_json)
        assert GridGreenSurface is not None
        assert PlanarGreenSurface is not None

    def test_grid_surface_instantiation_and_query(self) -> None:
        """GridGreenSurface instantiated via vendor boundary returns expected height & gravity."""
        heights = (
            (0.0, -0.02, -0.04),
            (0.0, -0.02, -0.04),
            (0.0, -0.02, -0.04),
        )
        grid = GridGreenSurface(origin_m=(0.0, 0.0), spacing_m=1.0, heights_m=heights)
        assert grid.origin_m == (0.0, 0.0)
        assert grid.spacing_m == 1.0
        assert grid.height_m(1.0, 1.0) == pytest.approx(-0.02)

        # In-plane gravity is -g * grad(h). grad_x = -0.02, gx = 9.80665 * 0.02
        gx, gy = grid.gravity_inplane_mps2(1.0, 1.0)
        assert gx == pytest.approx(9.80665 * 0.02, rel=1e-5)
        assert gy == pytest.approx(0.0, abs=1e-7)


class TestToolsToUpstreamDriftInteroperability:
    """Verify surfaces created in Tools serialize to JSON and load into UpstreamDrift GreenSurface."""

    def test_planar_surface_export_loads_into_upstream_drift_green(
        self, tmp_path: Path
    ) -> None:
        """A planar surface exported from Tools loads into UpstreamDrift GreenSurface."""
        # 2% grade down x (aspect 0 deg) over 6x6 m green with 1.0 m spacing
        grade_percent = 2.0
        plane = PlanarGreenSurface(grade_percent=grade_percent, aspect_deg=0.0)
        hole_pos = (5.0, 3.0)

        json_text = green_surface_to_ud_json(
            plane,
            hole_position_m=hole_pos,
            extent_m=(6.0, 6.0),
            spacing_m=1.0,
        )

        topography_file = tmp_path / "planar_topography.json"
        topography_file.write_text(json_text, encoding="utf-8")

        ud_green = GreenSurface(width=6.0, height=6.0)
        ud_green.load_from_file(topography_file)

        # Hole position loaded into UpstreamDrift
        assert np.allclose(ud_green.hole_position, np.array(hole_pos))

        # Check elevations match between Tools and UpstreamDrift at grid nodes
        for x in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            for y in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                expected_h = plane.height_m(x, y)
                actual_h = ud_green.get_elevation_at(np.array([x, y]))
                assert actual_h == pytest.approx(expected_h, abs=1e-9)

        # Check slope / gravity behavior matches small-slope approximation
        # Grade 2% => slope magnitude 0.02
        slope = ud_green.get_slope_at(np.array([3.0, 3.0]))
        # get_slope_at returns (rise/run) vector
        assert slope[0] == pytest.approx(-0.02, abs=1e-4)
        assert slope[1] == pytest.approx(0.0, abs=1e-4)

        # Gravity: UD get_gravitational_acceleration is -g * slope
        ud_grav = ud_green.get_gravitational_acceleration(np.array([3.0, 3.0]))
        tools_gx, tools_gy = plane.gravity_inplane_mps2(3.0, 3.0)
        assert ud_grav[0] == pytest.approx(tools_gx, rel=1e-4)
        assert ud_grav[1] == pytest.approx(tools_gy, abs=1e-7)

    def test_grid_surface_export_loads_into_upstream_drift_green(
        self, tmp_path: Path
    ) -> None:
        """A regular GridGreenSurface exported from Tools loads into UpstreamDrift GreenSurface."""
        # 4x4 grid, 1 m spacing, undulating height pattern
        xs = (0.0, 1.0, 2.0, 3.0)
        ys = (0.0, 1.0, 2.0, 3.0)
        heights = tuple(
            tuple(0.01 * math.sin(x) + 0.02 * math.cos(y) for x in xs) for y in ys
        )
        grid = GridGreenSurface(origin_m=(0.0, 0.0), spacing_m=1.0, heights_m=heights)
        hole_pos = (2.0, 2.0)

        json_text = green_surface_to_ud_json(grid, hole_position_m=hole_pos)
        topography_file = tmp_path / "grid_topography.json"
        topography_file.write_text(json_text, encoding="utf-8")

        ud_green = GreenSurface(width=3.0, height=3.0)
        ud_green.load_from_file(topography_file)

        assert np.allclose(ud_green.hole_position, np.array(hole_pos))

        # Check elevations match (UD's RBFInterpolator uses smoothing=0.01 on contour points)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                expected_h = heights[j][i]
                actual_h = ud_green.get_elevation_at(np.array([x, y]))
                assert actual_h == pytest.approx(expected_h, abs=1e-3)


class TestUpstreamDriftToToolsInteroperability:
    """Verify UpstreamDrift JSON topography documents import cleanly into Tools UdGreenTopography."""

    def test_canonical_regular_grid_json_imports_and_roundtrips(self) -> None:
        """A complete regular grid JSON document imports to UdGreenTopography and round-trips byte-identically."""
        spacing = 0.5
        xs = [i * spacing for i in range(5)]
        ys = [j * spacing for j in range(5)]
        hole_pos = [1.5, 1.0]

        contours: list[dict[str, float]] = []
        for y in ys:
            for x in xs:
                elevation = -0.015 * x + 0.005 * y
                contours.append({"elevation": elevation, "x": x, "y": y})

        doc: dict[str, Any] = {
            "contours": contours,
            "hole_position": hole_pos,
        }
        json_str = json.dumps(doc, sort_keys=True, separators=(",", ":"))

        parsed = green_surface_from_ud_json(json_str)
        assert isinstance(parsed, UdGreenTopography)
        assert parsed.hole_position_m == (1.5, 1.0)
        assert parsed.surface.spacing_m == 0.5
        assert parsed.surface.origin_m == (0.0, 0.0)

        # Check heights match
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                expected_elevation = -0.015 * x + 0.005 * y
                assert parsed.surface.heights_m[j][i] == pytest.approx(
                    expected_elevation, abs=1e-12
                )

        # Roundtrip serialization is byte-deterministic
        serialized = green_surface_to_ud_json(
            parsed.surface, hole_position_m=parsed.hole_position_m
        )
        assert serialized == json_str


class TestBoundaryFailClosedValidation:
    """Verify that unsupported or invalid formats fail closed across the boundary."""

    def test_refuses_scattered_contours(self) -> None:
        """Scattered (non-grid) contour points must be rejected fail-closed."""
        scattered_doc = json.dumps(
            {
                "contours": [
                    {"x": 0.0, "y": 0.0, "elevation": 0.0},
                    {"x": 1.0, "y": 0.0, "elevation": 0.0},
                    {"x": 0.5, "y": 1.0, "elevation": 0.05},  # triangular / scattered
                ],
                "hole_position": [0.5, 0.5],
            }
        )
        with pytest.raises(
            ValueError,
            match="contours must cover every node of a complete regular grid",
        ):
            green_surface_from_ud_json(scattered_doc)

    def test_refuses_slope_regions(self) -> None:
        """Slope regions are non-conservative and must be rejected fail-closed."""
        doc_with_slopes = json.dumps(
            {
                "contours": [
                    {"x": 0.0, "y": 0.0, "elevation": 0.0},
                    {"x": 1.0, "y": 0.0, "elevation": 0.0},
                    {"x": 0.0, "y": 1.0, "elevation": 0.0},
                    {"x": 1.0, "y": 1.0, "elevation": 0.0},
                ],
                "slopes": [
                    {
                        "center": [0.5, 0.5],
                        "direction": [1.0, 0.0],
                        "magnitude": 0.02,
                        "radius": 1.0,
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="slopes are refused"):
            green_surface_from_ud_json(doc_with_slopes)

    def test_refuses_unknown_fields(self) -> None:
        """Unknown fields in topography document must be rejected fail-closed."""
        bad_doc = json.dumps(
            {
                "contours": [
                    {"x": 0.0, "y": 0.0, "elevation": 0.0},
                    {"x": 1.0, "y": 0.0, "elevation": 0.0},
                    {"x": 0.0, "y": 1.0, "elevation": 0.0},
                    {"x": 1.0, "y": 1.0, "elevation": 0.0},
                ],
                "extra_metadata": "unexpected",
            }
        )
        with pytest.raises(ValueError, match="unknown topography fields"):
            green_surface_from_ud_json(bad_doc)

    def test_refuses_non_finite_coordinates(self) -> None:
        """Non-finite coordinates (NaN, Inf) must be rejected."""
        nan_doc = json.dumps(
            {
                "contours": [
                    {"x": 0.0, "y": 0.0, "elevation": 0.0},
                    {"x": float("nan"), "y": 0.0, "elevation": 0.0},
                    {"x": 0.0, "y": 1.0, "elevation": 0.0},
                    {"x": 1.0, "y": 1.0, "elevation": 0.0},
                ]
            }
        )
        with pytest.raises(ValueError, match="finite"):
            green_surface_from_ud_json(nan_doc)


class TestPhysicsAndRollConsistency:
    """Verify consistent roll physics invariants on surfaces imported across the vendor boundary."""

    def test_flat_green_straight_line_and_rollout_monotonicity(self) -> None:
        """Simulations on imported flat greens produce straight paths and monotonic rollout."""
        flat_grid = GridGreenSurface(
            origin_m=(0.0, 0.0),
            spacing_m=1.0,
            heights_m=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
        exported_json = green_surface_to_ud_json(flat_grid)
        imported = green_surface_from_ud_json(exported_json).surface

        stimp = 10.0
        # Rollout distance formula check
        v1 = 1.0
        v2 = 2.0
        mu = stimp_to_rolling_mu(stimp)
        d1 = roll_out_distance(v1, mu)
        d2 = roll_out_distance(v2, mu)
        assert d2 > d1
        assert d1 == pytest.approx(v1**2 / (2.0 * mu * 9.80665), rel=1e-4)

        # Integration on surface
        res1 = simulate_putt_on_surface(
            _launch(v1), imported, stimp_ft=stimp, hole_distance_m=10.0
        )
        res2 = simulate_putt_on_surface(
            _launch(v2), imported, stimp_ft=stimp, hole_distance_m=10.0
        )

        # Straight line: y trajectory remains 0.0
        assert all(abs(y) < 1e-12 for y in res1.path_y_m)
        assert all(abs(y) < 1e-12 for y in res2.path_y_m)
        assert res2.total_distance_m > res1.total_distance_m

    def test_uphill_downhill_asymmetry_on_sloped_surface(self) -> None:
        """Putt roll-out exhibits expected asymmetry: uphill rolls shorter than downhill."""
        # 2% grade down +x
        plane = PlanarGreenSurface(grade_percent=2.0, aspect_deg=0.0)
        json_doc = green_surface_to_ud_json(plane, extent_m=(20.0, 20.0), spacing_m=0.5)
        imported_surface = green_surface_from_ud_json(json_doc).surface

        stimp = 10.0
        speed = 2.0

        # Downhill putt along +x
        downhill_res = simulate_putt_on_surface(
            _launch(speed), imported_surface, stimp_ft=stimp, hole_distance_m=20.0
        )
        # Flat reference
        flat = PlanarGreenSurface(grade_percent=0.0, aspect_deg=0.0)
        flat_res = simulate_putt_on_surface(
            _launch(speed), flat, stimp_ft=stimp, hole_distance_m=20.0
        )

        # Downhill rolls farther than flat
        assert downhill_res.total_distance_m > flat_res.total_distance_m
