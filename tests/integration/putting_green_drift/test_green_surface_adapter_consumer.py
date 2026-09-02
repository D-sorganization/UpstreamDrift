"""ADR-0045 F4: UD-side consumer-contract test for the green-surface adapter.

Completes the #9143 rider (issue #9346). #9143 shipped the format-level
consumer test (``tests/unit/putting/test_putting_green_consumer.py``):
geometry and gravity agree across the vendor boundary. This module adds
the *physics-engine* half the ADR-0045 Validation section calls for: the
same green and the same launch driven through UpstreamDrift's own
:mod:`~src.engines.physics_engines.putting_green.python.ball_roll_physics`
(not a reimplementation of it) versus the vendored Tools integrator
``shared.python.swing_sim.putting.simulate_putt_on_surface``, gating only
what ADR-0045 says both roll models share: the flat-green straight line,
the break sign under a cross slope, monotonicity in stimp, and the
documented ~2.854 mu-ratio pin (Tools#4819). Magnitudes are otherwise
**not** forced to agree -- the divergent mu laws are the named-models
contract of ADR-0045, not a bug (see the ``ud_adapter`` module docstring
and docs/adr/0045-putting-integration-one-experience-two-preserved-stacks.md).

Coordinate framing
-------------------
Tools' ``simulate_putt_on_surface`` always launches from local ``(0, 0)``
with the hole on the ``+x`` axis (module docstring of
``shared.python.swing_sim.putting.green``). UpstreamDrift's ``GreenSurface``
instead occupies the first quadrant (``is_on_green`` requires
``0 <= y <= height``), so the same green is authored twice from the same
height field ``f(x, y_rel)``: once in Tools' centered frame
(``y = y_rel``, fed to the adapter as UD-shaped JSON) and once in UD's own
frame, translated by ``_PHYS_Y_CENTER`` so the putt line sits away from the
green edge (``y = y_rel + _PHYS_Y_CENTER``). This is a coordinate label,
not a different green -- exactly the "placing the surface into the putt
frame is the caller's job" note in the ``ud_adapter`` module docstring.

Grain is turned off (``grain_strength=0.0``) in the shared-physics gates:
Tools has no grain concept, and UD's default bent-grass grain would add a
same-direction friction discount that has nothing to do with the mu-law
divergence this module gates.

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
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
)
from src.engines.physics_engines.putting_green.python.green_surface import GreenSurface
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)
from src.shared.python.core.physics_constants import (
    GOLF_BALL_RADIUS_M as UD_GOLF_BALL_RADIUS_M,
)
from src.shared.python.core.physics_constants import GRAVITY_M_S2 as UD_GRAVITY_M_S2
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
_PHYS_Y_CENTER_M = 8.0  # UD-frame y of the putt line (green is 16 m tall)
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


def _make_ud_green(height_fn: Callable[[float, float], float]) -> GreenSurface:
    """The same height field, authored with UD's own contour-point code.

    Turf is left at the constructor default; every test passes its own
    :class:`TurfProperties` explicitly to :class:`BallRollPhysics` (never
    mutating the fixture), so this green is safely module-scoped.
    """
    xs, y_rel = _phys_nodes()
    points = [
        ContourPoint(x=x, y=y + _PHYS_Y_CENTER_M, elevation=float(height_fn(x, y)))
        for y in y_rel
        for x in xs
    ]
    green = GreenSurface(width=_PHYS_X_MAX_M, height=2.0 * _PHYS_Y_HALF_SPAN_M)
    green.set_contour_points(points)
    # Hole capture is out of scope for these gates (module docstring); park
    # it far away so it can never interfere.
    green.set_hole_position(np.array([-1000.0, -1000.0]))
    return green


def _turf(stimp_ft: float) -> TurfProperties:
    """UD turf at a given stimp with grain disabled (see module docstring)."""
    return TurfProperties(stimp_rating=stimp_ft, grain_strength=0.0)


def _ud_struck_state(speed_mps: float) -> BallState:
    """A struck putt: zero initial spin, matching Tools' own ``_launch``."""
    return BallState(
        position=np.array([0.0, _PHYS_Y_CENTER_M]),
        velocity=np.array([speed_mps, 0.0]),
        spin=np.zeros(3),
    )


def _ud_rolling_state(speed_mps: float) -> BallState:
    """A putt already in pure roll (spin locked to ``v / r``).

    Used only by the ratio-pin gate, to isolate the rolling-resistance mu
    from each engine's own (differently modeled) skid phase.
    """
    return BallState(
        position=np.array([0.0, _PHYS_Y_CENTER_M]),
        velocity=np.array([speed_mps, 0.0]),
        spin=np.array([0.0, -speed_mps / UD_GOLF_BALL_RADIUS_M, 0.0]),
    )


def _arc_length(positions: np.ndarray) -> float:
    """Path length of a UD trajectory, matching Tools' ``total_distance_m``
    (both are a sum of per-step displacement magnitudes, not net
    displacement)."""
    diffs = np.diff(np.asarray(positions), axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def _tools_launch(speed_mps: float, *, spin_rad_s: float = 0.0) -> PuttLaunch:
    return PuttLaunch(
        ball_speed_mps=speed_mps,
        launch_angle_deg=0.0,
        horizontal_speed_mps=speed_mps,
        spin_rad_s=spin_rad_s,
        effective_loft_deg=0.0,
    )


@pytest.fixture(scope="module")
def flat_ud_green() -> GreenSurface:
    return _make_ud_green(lambda x, y: 0.0)


@pytest.fixture(scope="module")
def flat_wire_surface():
    return green_surface_from_ud_json(_wire_json_for(lambda x, y: 0.0)).surface


@pytest.fixture(scope="module")
def cross_left_ud_green() -> GreenSurface:
    return _make_ud_green(lambda x, y: -0.02 * y)


@pytest.fixture(scope="module")
def cross_left_wire_surface():
    return green_surface_from_ud_json(_wire_json_for(lambda x, y: -0.02 * y)).surface


@pytest.fixture(scope="module")
def cross_right_ud_green() -> GreenSurface:
    return _make_ud_green(lambda x, y: 0.02 * y)


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
    """Item 2: same green + same launch through UD's real
    ``BallRollPhysics`` vs the vendored ``simulate_putt_on_surface``. Only
    what both roll models share is gated (module docstring)."""

    def test_flat_green_rolls_straight_on_both_sides(
        self, flat_ud_green, flat_wire_surface
    ) -> None:
        speed = 1.5

        ud_physics = BallRollPhysics(
            turf=_turf(_STIMP_FT), green=flat_ud_green, integrator="rk4"
        )
        ud_result = ud_physics.simulate_putt(
            _ud_struck_state(speed), max_time=30.0, dt=0.002
        )
        assert ud_result["final_position"][1] == pytest.approx(
            _PHYS_Y_CENTER_M, abs=1e-9
        )
        assert all(
            pos[1] == pytest.approx(_PHYS_Y_CENTER_M, abs=1e-9)
            for pos in ud_result["positions"]
        )

        tools_result = simulate_putt_on_surface(
            _tools_launch(speed),
            flat_wire_surface,
            stimp_ft=_STIMP_FT,
            hole_distance_m=30.0,
        )
        assert tools_result.break_m == 0.0
        assert all(y == 0.0 for y in tools_result.path_y_m)

    def test_break_sign_matches_cross_slope_downhill_left(
        self, cross_left_ud_green, cross_left_wire_surface
    ) -> None:
        speed = 1.5

        ud_physics = BallRollPhysics(
            turf=_turf(_STIMP_FT), green=cross_left_ud_green, integrator="rk4"
        )
        ud_result = ud_physics.simulate_putt(
            _ud_struck_state(speed), max_time=30.0, dt=0.002
        )
        ud_break_m = float(ud_result["final_position"][1]) - _PHYS_Y_CENTER_M
        assert ud_break_m > 0.0

        tools_result = simulate_putt_on_surface(
            _tools_launch(speed),
            cross_left_wire_surface,
            stimp_ft=_STIMP_FT,
            hole_distance_m=30.0,
        )
        assert tools_result.break_m > 0.0

    def test_break_sign_matches_cross_slope_downhill_right(
        self, cross_right_ud_green, cross_right_wire_surface
    ) -> None:
        speed = 1.5

        ud_physics = BallRollPhysics(
            turf=_turf(_STIMP_FT), green=cross_right_ud_green, integrator="rk4"
        )
        ud_result = ud_physics.simulate_putt(
            _ud_struck_state(speed), max_time=30.0, dt=0.002
        )
        ud_break_m = float(ud_result["final_position"][1]) - _PHYS_Y_CENTER_M
        assert ud_break_m < 0.0

        tools_result = simulate_putt_on_surface(
            _tools_launch(speed),
            cross_right_wire_surface,
            stimp_ft=_STIMP_FT,
            hole_distance_m=30.0,
        )
        assert tools_result.break_m < 0.0

    def test_rollout_monotone_in_stimp_on_both_sides(
        self, flat_ud_green, flat_wire_surface
    ) -> None:
        speed = 1.5
        ud_distances = []
        tools_distances = []
        for stimp in (8.0, 10.0, 12.0):
            ud_physics = BallRollPhysics(
                turf=_turf(stimp), green=flat_ud_green, integrator="rk4"
            )
            ud_result = ud_physics.simulate_putt(
                _ud_struck_state(speed), max_time=30.0, dt=0.002
            )
            ud_distances.append(_arc_length(ud_result["positions"]))

            tools_result = simulate_putt_on_surface(
                _tools_launch(speed),
                flat_wire_surface,
                stimp_ft=stimp,
                hole_distance_m=30.0,
            )
            tools_distances.append(tools_result.total_distance_m)

        assert ud_distances[0] < ud_distances[1] < ud_distances[2]
        assert tools_distances[0] < tools_distances[1] < tools_distances[2]

    def test_roll_out_ratio_matches_the_documented_constant(
        self, flat_ud_green, flat_wire_surface
    ) -> None:
        """Tools#4819 / ADR-0045 Validation: mu_tools / mu_ud is a
        stimp-independent constant ~2.854 (UD rolls ~2.85x farther at the
        same stimp) -- the divergence is the named-models contract of
        ADR-0045, not a bug. Both simulators start already in pure roll
        (spin locked to v/r) so total distance IS the roll-out distance,
        isolating mu without either engine's own skid model."""
        speed = 2.0

        ud_physics = BallRollPhysics(
            turf=_turf(_STIMP_FT), green=flat_ud_green, integrator="rk4"
        )
        ud_result = ud_physics.simulate_putt(
            _ud_rolling_state(speed), max_time=30.0, dt=0.002
        )
        ud_distance_m = _arc_length(ud_result["positions"])

        tools_spin_rad_s = speed / TOOLS_GOLF_BALL_RADIUS_M
        tools_result = simulate_putt_on_surface(
            _tools_launch(speed, spin_rad_s=tools_spin_rad_s),
            flat_wire_surface,
            stimp_ft=_STIMP_FT,
            hole_distance_m=30.0,
        )
        tools_distance_m = tools_result.total_distance_m

        # The documented constant, from each engine's own real mu law (not
        # a copy of Tools' reproduction of UD's formula).
        mu_ud = _turf(_STIMP_FT).effective_friction
        mu_tools = stimp_to_rolling_mu(_STIMP_FT)
        mu_ratio = mu_tools / mu_ud
        assert mu_ratio == pytest.approx(2.854, abs=2e-3)

        empirical_ratio = ud_distance_m / tools_distance_m
        assert empirical_ratio == pytest.approx(2.854, rel=0.05)
        assert empirical_ratio == pytest.approx(mu_ratio, rel=0.05)


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
