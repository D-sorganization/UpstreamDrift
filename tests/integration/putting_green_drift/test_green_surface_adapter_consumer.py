"""ADR-0045 F4: UD-side consumer-contract test for the green-surface adapter.

Completes the #9143 rider (issue #9346). #9143 shipped the format-level
consumer test (``tests/unit/putting/test_putting_green_consumer.py``):
geometry and gravity agree across the vendor boundary. This module adds
the *physics-engine* half the ADR-0045 Validation section calls for: the
same green and the same launch driven through UpstreamDrift's shared
putting core (:mod:`src.shared.python.putting_dynamics`, not a
reimplementation of it) versus the vendored Tools integrator
``shared.python.swing_sim.putting.simulate_putt_on_surface``. It gates the
flat-green straight line, the break sign under a cross slope and
monotonicity in stimp. Since ADR-0058 retired ``ud-legacy-roll/1``, both
sides run the USGA Stimpmeter law, so the old ~2.854 mu-ratio pin
(Tools#4819) becomes a roll-out agreement gate (ratio 1 within 5 %;
measured 0.9986 on 2026-11-01, #14141 step 10c).

Coordinate framing
-------------------
Tools' ``simulate_putt_on_surface`` always launches from local ``(0, 0)``
with the hole on the ``+x`` axis (module docstring of
``shared.python.swing_sim.putting.green``). The shared core uses the same
centered frame, so the physics gates author each green once from the same
height field ``f(x, y_rel)``: on the shared core directly and, for Tools,
as UD-shaped JSON fed to the adapter. UpstreamDrift's ``GreenSurface``
(first quadrant, ``0 <= y <= height``) appears only in the round-trip and
refusal tests.

Grain is turned off (``grain_strength=0.0``) in the shared-physics gates:
Tools has no grain concept, and UD's default bent-grass grain would add a
same-direction friction discount that has nothing to do with the roll
law this module compares.

CI posture: mirrors ``tests/integration/launch_monitor_drift`` (skip
locally when ``vendor/ud-tools`` is not materialised; every CI job that
runs this file fetches the pinned Tools checkout via
``.github/actions/fetch-pinned-tools``, so it always executes for real
there). The helper is imported, not reimplemented (DRY).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python._surface_data import (
    ContourPoint,
)
from src.engines.physics_engines.putting_green.python.green_surface import GreenSurface
from src.shared.python.core.physics_constants import (
    GOLF_BALL_RADIUS_M as UD_GOLF_BALL_RADIUS_M,
)
from src.shared.python.core.physics_constants import GRAVITY_M_S2 as UD_GRAVITY_M_S2
from src.shared.python.putting_dynamics import (
    BallState,
    FrictionField,
    HeightField,
    PuttResult,
    SurfaceSpec,
    TurfSpec,
    simulate_ball,
)
from src.shared.python.putting_dynamics import (
    stimp_to_rolling_mu as shared_stimp_to_rolling_mu,
)
from tests.integration.launch_monitor_drift.conftest import (
    require_vendored_tools_stack,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

require_vendored_tools_stack()

from shared.python.swing_sim.impact import (  # noqa: E402
    GOLF_BALL_RADIUS_M as TOOLS_GOLF_BALL_RADIUS_M,
)
from shared.python.swing_sim.putting import (  # noqa: E402
    PuttLaunch,
    green_surface_from_ud_json,
    green_surface_to_ud_json,
    simulate_putt_on_surface,
    stimp_to_rolling_mu,
)

# --- Shared-physics-gate grid: coarse but planar-exact (RBF and bilinear
# both reproduce a plane exactly regardless of node density), so a sparse
# grid keeps the RBF fit cheap without losing precision. ---
_PHYS_SPACING_M = 2.0
_PHYS_X_MAX_M = 20.0
_PHYS_Y_HALF_SPAN_M = 8.0
_STIMP_FT = 10.0


def _phys_nodes() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(xs, y_rel)`` node coordinates shared by the UD and wire grids."""
    n_x = int(_PHYS_X_MAX_M / _PHYS_SPACING_M) + 1
    n_y = int(2.0 * _PHYS_Y_HALF_SPAN_M / _PHYS_SPACING_M) + 1
    xs = tuple(i * _PHYS_SPACING_M for i in range(n_x))
    y_rel = tuple(-_PHYS_Y_HALF_SPAN_M + j * _PHYS_SPACING_M for j in range(n_y))
    return xs, y_rel


def _wire_json_for(height_fn: Callable[[float, float], float]) -> str:
    """UD-shaped topography JSON in Tools' own (centered-at-origin) frame."""
    xs, y_rel = _phys_nodes()
    contours = [
        {"x": x, "y": y, "elevation": float(height_fn(x, y))} for y in y_rel for x in xs
    ]
    return json.dumps({"contours": contours})


def _shared_surface(
    height_fn: Callable[[float, float], float], stimp_ft: float = _STIMP_FT
) -> SurfaceSpec:
    """The same height field on the shared core, in Tools' centered frame.

    Grain is off (module docstring); the turf is the shared USGA law.
    """
    xs, y_rel = _phys_nodes()
    heights = np.array([[float(height_fn(x, y)) for x in xs] for y in y_rel])
    origin = (xs[0], y_rel[0])
    ones = np.ones_like(heights)
    return SurfaceSpec(
        height=HeightField(heights, spacing_m=_PHYS_SPACING_M, origin_m=origin),
        friction_field=FrictionField(
            ones, ones.copy(), spacing_m=_PHYS_SPACING_M, origin_m=origin
        ),
        friction=TurfSpec(stimp_ft=stimp_ft, grain_strength=0.0).to_friction_params(),
    )


def _shared_putt(
    surface: SurfaceSpec, speed_mps: float, *, rolling: bool = False
) -> PuttResult:
    """A putt along +x from the origin; ``rolling`` locks spin to ``v / r``
    so the whole path is roll-out (isolates the rolling mu)."""
    spin = speed_mps / UD_GOLF_BALL_RADIUS_M if rolling else 0.0
    ball = BallState(x_m=2.0, y_m=0.0, vx_mps=speed_mps, spin_rad_s=spin)
    return simulate_ball(ball, surface)


def _tools_launch(speed_mps: float, *, spin_rad_s: float = 0.0) -> PuttLaunch:
    return PuttLaunch(
        ball_speed_mps=speed_mps,
        launch_angle_deg=0.0,
        horizontal_speed_mps=speed_mps,
        spin_rad_s=spin_rad_s,
        effective_loft_deg=0.0,
    )


@pytest.fixture(scope="module")
def flat_wire_surface():
    return green_surface_from_ud_json(_wire_json_for(lambda x, y: 0.0)).surface


@pytest.fixture(scope="module")
def cross_left_wire_surface():
    return green_surface_from_ud_json(_wire_json_for(lambda x, y: -0.02 * y)).surface


@pytest.fixture(scope="module")
def cross_right_wire_surface():
    return green_surface_from_ud_json(_wire_json_for(lambda x, y: 0.02 * y)).surface


class TestRoundTripPreservesGeometryAtGridNodes:
    """Item 1: a green authored with UD's own code round-trips through the
    adapter and back, geometry preserved at grid nodes.

    UD ships no topography exporter (only ``_load_json_topography``), so
    the "exported UD topography JSON" is synthesized field-for-field from
    the same :class:`ContourPoint` list used to author the real UD
    ``GreenSurface`` -- the same posture the vendored adapter's own fixture
    uses (its module docstring: "UD ships no canned topography JSON to
    copy"). Both legs of the round trip run through real UD load code.
    """

    @staticmethod
    def _height(x: float, y: float) -> float:
        return -0.015 * x + 0.01 * y

    def test_round_trip_preserves_geometry_at_grid_nodes(self, tmp_path) -> None:
        xs = (0.0, 1.0, 2.0, 3.0, 4.0)
        ys = (0.0, 1.0, 2.0, 3.0, 4.0)
        hole_position = (3.0, 2.0)
        points = [
            ContourPoint(x=x, y=y, elevation=self._height(x, y)) for y in ys for x in xs
        ]

        # Author the green with UD's own code.
        ud_green = GreenSurface(width=4.0, height=4.0)
        ud_green.set_contour_points(points)
        ud_green.set_hole_position(np.array(hole_position))
        for x in xs:
            for y in ys:
                assert ud_green.get_elevation_at(np.array([x, y])) == pytest.approx(
                    self._height(x, y), abs=1e-9
                )

        # "Export": synthesized field-for-field from the same points UD
        # authored (see class docstring), then proven UD-loadable.
        exported_text = json.dumps(
            {
                "contours": [
                    {"x": p.x, "y": p.y, "elevation": p.elevation} for p in points
                ],
                "hole_position": list(hole_position),
            },
            sort_keys=True,
        )
        exported_path = tmp_path / "ud_authored_topography.json"
        exported_path.write_text(exported_text, encoding="utf-8")
        reloaded_ud_green = GreenSurface(width=4.0, height=4.0)
        reloaded_ud_green.load_from_file(exported_path)
        assert np.allclose(reloaded_ud_green.hole_position, np.array(hole_position))

        # Into the vendored adapter.
        parsed = green_surface_from_ud_json(exported_text)
        assert parsed.hole_position_m == hole_position
        assert parsed.surface.spacing_m == pytest.approx(1.0)
        assert parsed.surface.origin_m == (0.0, 0.0)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                assert parsed.surface.heights_m[j][i] == pytest.approx(
                    self._height(x, y), abs=1e-12
                )
                assert parsed.surface.height_m(x, y) == pytest.approx(
                    self._height(x, y), abs=1e-12
                )

        # And back into a fresh UD GreenSurface -- geometry preserved at
        # every grid node, full circle.
        back_text = green_surface_to_ud_json(
            parsed.surface, hole_position_m=parsed.hole_position_m
        )
        back_path = tmp_path / "adapter_exported_topography.json"
        back_path.write_text(back_text, encoding="utf-8")
        final_ud_green = GreenSurface(width=4.0, height=4.0)
        final_ud_green.load_from_file(back_path)
        assert np.allclose(final_ud_green.hole_position, np.array(hole_position))
        for x in xs:
            for y in ys:
                assert final_ud_green.get_elevation_at(
                    np.array([x, y])
                ) == pytest.approx(self._height(x, y), abs=1e-9)


class TestSharedPhysicsGates:
    """Item 2: the same green and launch through UD's shared putting core
    (ADR-0058) and the vendored ``simulate_putt_on_surface``. Both run the
    USGA Stimpmeter law, so beyond the straight line, break sign and stimp
    monotonicity, their roll-out must agree (it pinned a 2.854 ratio while
    UD ran the retired ``ud-legacy-roll/1``)."""

    def test_flat_green_rolls_straight_on_both_sides(self, flat_wire_surface) -> None:
        shared = _shared_putt(_shared_surface(lambda x, y: 0.0), 1.5)
        assert shared.rest_y_m == pytest.approx(0.0, abs=1e-9)
        assert all(s.y_m == pytest.approx(0.0, abs=1e-9) for s in shared.samples)

        tools_result = simulate_putt_on_surface(
            _tools_launch(1.5),
            flat_wire_surface,
            stimp_ft=_STIMP_FT,
            hole_distance_m=30.0,
        )
        assert tools_result.break_m == 0.0
        assert all(y == 0.0 for y in tools_result.path_y_m)

    @pytest.mark.parametrize("side", [1.0, -1.0], ids=["downhill-left", "right"])
    def test_break_sign_matches_the_cross_slope(
        self, side: float, cross_left_wire_surface, cross_right_wire_surface
    ) -> None:
        surface = _shared_surface(lambda x, y: -0.02 * side * y)
        assert side * _shared_putt(surface, 1.5).rest_y_m > 0.0

        wire = cross_left_wire_surface if side > 0 else cross_right_wire_surface
        tools_result = simulate_putt_on_surface(
            _tools_launch(1.5), wire, stimp_ft=_STIMP_FT, hole_distance_m=30.0
        )
        assert side * tools_result.break_m > 0.0

    def test_rollout_monotone_in_stimp_on_both_sides(self, flat_wire_surface) -> None:
        shared, tools = [], []
        for stimp in (8.0, 10.0, 12.0):
            surface = _shared_surface(lambda x, y: 0.0, stimp)
            shared.append(_shared_putt(surface, 1.5).total_distance_m)
            tools_result = simulate_putt_on_surface(
                _tools_launch(1.5),
                flat_wire_surface,
                stimp_ft=stimp,
                hole_distance_m=30.0,
            )
            tools.append(tools_result.total_distance_m)
        assert shared[0] < shared[1] < shared[2]
        assert tools[0] < tools[1] < tools[2]

    def test_both_sides_run_the_same_usga_law(self, flat_wire_surface) -> None:
        """Both derive mu from the USGA Stimpmeter geometry; they differ only
        in how g and the release speed are rounded (about 0.1 %). Starting in
        pure roll makes the total distance the roll-out, isolating mu from
        each side's skid model, so the roll-out ratio is 1 within the 5 %
        the 2.854 pin allowed."""
        mu_ratio = shared_stimp_to_rolling_mu(_STIMP_FT) / stimp_to_rolling_mu(
            _STIMP_FT
        )
        assert mu_ratio == pytest.approx(1.0, rel=1e-2)

        speed = 2.0
        shared = _shared_putt(_shared_surface(lambda x, y: 0.0), speed, rolling=True)
        tools_result = simulate_putt_on_surface(
            _tools_launch(speed, spin_rad_s=speed / TOOLS_GOLF_BALL_RADIUS_M),
            flat_wire_surface,
            stimp_ft=_STIMP_FT,
            hole_distance_m=30.0,
        )
        ratio = shared.total_distance_m / tools_result.total_distance_m
        assert ratio == pytest.approx(1.0, rel=0.05)


class TestRefusesUdWeightedSlopeField:
    """Item 3: a weighted-slope UD field is genuinely UD-loadable, and the
    adapter refuses it with its documented, named reason (not just any
    error)."""

    def test_weighted_slope_field_is_ud_loadable_but_adapter_refused(
        self, tmp_path
    ) -> None:
        document = {
            "contours": [
                {"x": x, "y": y, "elevation": 0.0}
                for y in (0.0, 1.0, 2.0, 3.0, 4.0)
                for x in (0.0, 1.0, 2.0, 3.0, 4.0)
            ],
            "slopes": [
                {
                    "center": [2.0, 2.0],
                    "radius": 3.0,
                    "direction": [1.0, 0.0],
                    "magnitude": 0.02,
                }
            ],
        }
        text = json.dumps(document)

        # Prove this is a genuine UD-consumable field: UD's own loader
        # accepts it and the weighted slope is live in the gravity query.
        path = tmp_path / "weighted_slope_topography.json"
        path.write_text(text, encoding="utf-8")
        ud_green = GreenSurface(width=5.0, height=5.0)
        ud_green.load_from_file(path)
        gx, gy = ud_green.get_gravitational_acceleration(np.array([2.0, 2.0]))
        assert gx == pytest.approx(-UD_GRAVITY_M_S2 * 0.02, rel=1e-6)
        assert gy == pytest.approx(0.0, abs=1e-9)

        # The adapter refuses it with the documented, named reason: the
        # weighted slope field is non-conservative, not just "unsupported".
        with pytest.raises(ValueError, match="non-conservative"):
            green_surface_from_ud_json(text)
