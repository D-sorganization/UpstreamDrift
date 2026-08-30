"""More marketed bounce must never dig deeper (issue #9247).

Bounce exists so the sole skids instead of digging.  It is the primary
design variable of a wedge and the whole reason this tool compares soles,
and until #9247 the model had its sign backwards: over a 24 mm sole at
-14 deg of attack, 18 deg more bounce bought 4.9 mm *more* depth and 40 g
*more* sand, monotone across the range.

That survived a suite of 2,000-odd tests because **nothing asserted the
physical ordering**.  Every test pinned a shape, a conservation identity
or a scalar against itself; none of them said which way round the answer
should come out.  This module says it.

Two regimes, and the boundary between them
------------------------------------------

The ordering has to hold in both, which is why a test that pinned only
the shallow case would be half a guard -- and this defect is exactly what
half-guards let through:

* **Non-burying.**  The sole planes, comes back out, and cuts a divot.
  More bounce is shallower and moves less sand.
* **Burying.**  The presentation angle goes negative, the leading edge
  leads into the bed and the head goes three to four times deeper.  More
  bounce is *still* shallower here -- the two regimes differ in magnitude,
  not in sign.  Prior work read a reversal here, but that reading was
  against the inverted shallow regime: it was the shallow numbers that
  were backwards, not these.
* **The boundary** sits near zero presentation bounce and moves a little
  with the delivery, so it is bracketed rather than pinned: at a steep
  attack the sweep must contain both a burying and a non-burying design.

The heads are lofted once per module at a coarse resolution -- the
ordering is a property of the sole's sign, not of the discretisation --
so the whole file costs a few seconds.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.typing import NDArray

from bunkershot3d.geometry import (
    TRAVEL_AXIS_BODY,
    CamberFit,
    WedgeGeometry,
    build_wedge_mesh,
    compute_mass_properties,
    delivered_rotation,
    effective_loft_deg,
    entry_velocity_m_s,
    loft_wedge,
    shaft_axis,
)
from bunkershot3d.geometry.bounce import MarketedBounce, geometric_from_marketed
from bunkershot3d.geometry.presets import get_preset
from bunkershot3d.metrics import HeadModel, StrikeScene, StrikeTrace, divot_metrics
from bunkershot3d.sand import PlayingCondition, SandState, playing_condition
from bunkershot3d.solvers import (
    HeadKinematics,
    ShotSettings,
    SurfaceElements,
    simulate_shot,
)
from bunkershot3d.solvers.coefficients import MaterialResponse
from bunkershot3d.solvers.drft import DRFTSolver

pytestmark = pytest.mark.unit

#: Marketed bounce, degrees.  Starts at 8: a 5 deg sole is not a
#: constructible 24 mm profile at this resolution, and the point here is
#: the ordering across a span, not the extreme.
BOUNCES_DEG = (8.0, 14.0, 20.0, 26.0)

#: Attack angles at which every design in ``BOUNCES_DEG`` planes and
#: exits.  Squarely inside the non-burying regime.
SHALLOW_ATTACKS_DEG = (-2.0, -4.0, -6.0)

#: An attack steep enough that the low-bounce end of the sweep buries and
#: the high-bounce end does not, so one sweep spans both regimes.
STEEP_ATTACK_DEG = -14.0

#: Delivery speed.  Greenside is 20-27 m/s.
SPEED_M_S = 25.0

#: A window long enough that even a buried head comes back out, so the
#: two regimes are comparable on one depth axis instead of one of them
#: being a truncation.
_SETTINGS = ShotSettings(time_step_s=2.5e-4, max_time_s=0.400, require_exit=False)

_N_PROFILE_POINTS = 16
_N_STATIONS = 7

#: How much deeper a buried head goes than a planing one.  Measured
#: ratios are 3.9-4.7; 2.0 asks only that the two regimes are separated
#: by a wide margin rather than pinning the model's magnitude.
_BURYING_DEPTH_RATIO = 2.0


@pytest.fixture(scope="module")
def sand() -> SandState:
    """A firm USGA bed, the condition the defect was reported in."""
    return playing_condition(PlayingCondition.FIRM)


@pytest.fixture(scope="module")
def solver(sand: SandState) -> DRFTSolver:
    """The F0 solver for that bed."""
    return DRFTSolver(
        material=MaterialResponse.from_sand_state(sand), dynamic_terms_active=True
    )


class _Head:
    """A lofted head plus the two things a shot needs from it."""

    def __init__(self, geometry: WedgeGeometry) -> None:
        lofted = loft_wedge(
            geometry,
            n_profile_points=_N_PROFILE_POINTS,
            n_stations=_N_STATIONS,
            camber_fit=CamberFit.NEAREST,
        )
        self.geometry = geometry
        self.elements = SurfaceElements.from_mesh(lofted.mesh)
        sole = self.elements.centroids_m[self.elements.normals[:, 2] < 0.0]
        self.sole_reference_m: NDArray[np.float64] = np.asarray(
            sole[int(np.argmin(sole[:, 2]))], dtype=np.float64
        )


def _design(marketed_bounce_deg: float) -> WedgeGeometry:
    """A 24 mm sole at the given marketed bounce, everything else held."""
    base = get_preset("sm9_58_m").geometry
    return dataclasses.replace(
        base,
        sole_width_m=0.024,
        geometric_bounce=geometric_from_marketed(
            MarketedBounce(float(marketed_bounce_deg)),
            sole_width_m=0.024,
            entry_height_m=base.entry_height_m,
            datum_offset_m=base.datum_offset_m,
        ),
    )


@pytest.fixture(scope="module")
def heads() -> dict[float, _Head]:
    """One lofted head per marketed bounce, built once for the module."""
    return {bounce: _Head(_design(bounce)) for bounce in BOUNCES_DEG}


def _shoot(
    solver: DRFTSolver,
    head: _Head,
    attack_deg: float,
    *,
    face_open_deg: float = 0.0,
    shaft_lean_deg: float = 0.0,
) -> tuple[float, float | None]:
    """March one head and return ``(max sole depth, displaced mass)``.

    The mass is ``None`` when the strike has no measurable divot, which is
    what a buried head produces: it stops, or reverses out of its own
    crater, and a prismatic divot is then undefined.  Depth is always
    available, which is why the ordering is asserted on it in both
    regimes and on mass only where a divot exists.
    """
    rotation = delivered_rotation(
        lie_deg=head.geometry.lie_deg,
        face_open_deg=face_open_deg,
        shaft_lean_deg=shaft_lean_deg,
    )
    result = simulate_shot(
        solver,
        head.elements,
        head_mass_kg=head.geometry.head_mass_kg,
        kinematics=HeadKinematics(
            velocity_m_s=entry_velocity_m_s(
                speed_m_s=SPEED_M_S, attack_angle_deg=attack_deg
            ),
            position_m=np.zeros(3, dtype=np.float64),
            orientation=rotation,
        ),
        settings=_SETTINGS,
        sole_reference_body_m=head.sole_reference_m,
    )
    depth_m = result.max_sole_depth_m
    if result.n_steps < 3:
        return depth_m, None
    scene = StrikeScene(
        sand_surface_height_m=0.0,
        ball_position_m=np.zeros(3, dtype=np.float64),
        travel_axis=np.asarray(TRAVEL_AXIS_BODY, dtype=np.float64),
    )
    try:
        divot = divot_metrics(
            StrikeTrace.from_shot(result),
            _head_model(head),
            scene,
            width_m=head.geometry.sole_width_m,
            bulk_density_kg_m3=1600.0,
        )
    except ValueError:
        return depth_m, None
    return depth_m, divot.mass_kg


def _head_model(head: _Head) -> HeadModel:
    """The metrics package's view of a head, built from the mesh."""
    mesh = build_wedge_mesh(
        head.geometry,
        n_profile_points=_N_PROFILE_POINTS,
        n_stations=_N_STATIONS,
        camber_fit=CamberFit.NEAREST,
    )
    mass = compute_mass_properties(mesh, mass_kg=head.geometry.head_mass_kg)
    _, axis = shaft_axis(head.geometry)
    return HeadModel(
        mass_kg=head.geometry.head_mass_kg,
        centre_of_mass_body_m=mass.centroid_m,
        sole_reference_body_m=head.sole_reference_m,
        shaft_axis_body=np.asarray(axis, dtype=np.float64),
        inertia_body_kg_m2=mass.inertia_kg_m2,
    )


class TestTheFrameTheOrderingRestsOn:
    """Cheap guards that catch a re-mirror without marching a shot."""

    def test_the_head_travels_leading_edge_first(
        self, heads: dict[float, _Head]
    ) -> None:
        """The sole descends away from the direction of travel.

        A wedge strikes leading edge first.  The mesh puts the leading
        edge forward of the sole's lowest point, so the travel axis has
        to point from the low point toward the leading edge -- otherwise
        the sole is a ramp descending into the bed and bounce digs.
        """
        for bounce, head in heads.items():
            lowest = head.sole_reference_m
            leading_x = head.geometry.face_progression_m
            rearward_m = (float(lowest[0]) - leading_x) * -TRAVEL_AXIS_BODY[0]
            assert rearward_m > 0.0, (
                f"at {bounce} deg of bounce the sole's lowest point is "
                f"{rearward_m * 1e3:.4g} mm rearward of the leading edge, so "
                "the head would be driven trailing edge first"
            )

    def test_travel_axis_is_horizontal_and_unit(self) -> None:
        axis = np.asarray(TRAVEL_AXIS_BODY, dtype=np.float64)
        assert float(np.linalg.norm(axis)) == pytest.approx(1.0, abs=1e-12)
        assert float(axis[2]) == 0.0

    def test_the_mesh_face_looks_along_the_travel_direction(
        self, heads: dict[float, _Head]
    ) -> None:
        """A wedge's face points where the ball is going.

        Measured on the head's own face, which is the only large planar
        surface it has and so its one unambiguous witness.  This is what
        makes the frame question answerable at all: whichever way ``x``
        is labelled, the face and the travel direction have to agree.
        """
        for bounce, head in heads.items():
            normal = _largest_planar_normal(head.elements)
            along = float(normal @ np.asarray(TRAVEL_AXIS_BODY, dtype=np.float64))
            assert along > 0.0, (
                f"at {bounce} deg of bounce the face normal {np.round(normal, 4)} "
                f"points away from the travel axis {TRAVEL_AXIS_BODY}: the head "
                "would strike with its back"
            )

    @pytest.mark.parametrize("shaft_lean_deg", [4.0, 8.0, 14.0])
    def test_shaft_lean_de_lofts_the_mesh(
        self, heads: dict[float, _Head], shaft_lean_deg: float
    ) -> None:
        """Forward lean must take loft off the *mesh*, degree for degree.

        Anchored on the published rate rather than on
        :func:`delivered_rotation`, because a test that measures a
        rotation against itself passes in either frame -- which is how a
        mirror survives.  Before #9247 the workbench's copy of this
        composition *added* loft here: at 6 degrees of forward lean the
        sand saw 65 degrees while the report said 52.
        """
        head = heads[BOUNCES_DEG[1]]
        square = _mesh_loft_deg(head, face_open_deg=0.0, shaft_lean_deg=0.0)
        leaned = _mesh_loft_deg(head, face_open_deg=0.0, shaft_lean_deg=shaft_lean_deg)
        assert leaned == pytest.approx(square - shaft_lean_deg, abs=1e-6)

    @pytest.mark.parametrize("face_open_deg", [10.0, 20.0])
    def test_opening_the_face_adds_loft_to_the_mesh(
        self, heads: dict[float, _Head], face_open_deg: float
    ) -> None:
        """Opening adds loft at ``Omega cos(lie)``, on the mesh.

        The other half of the rotation, anchored the same way: a mirrored
        composition takes loft off instead.  The band is the one the
        first-order rate is quoted to.
        """
        head = heads[BOUNCES_DEG[1]]
        square = _mesh_loft_deg(head, face_open_deg=0.0, shaft_lean_deg=0.0)
        opened = _mesh_loft_deg(head, face_open_deg=face_open_deg, shaft_lean_deg=0.0)
        expected = face_open_deg * math.cos(math.radians(head.geometry.lie_deg))
        assert opened - square == pytest.approx(expected, abs=0.6)

    def test_the_library_angles_describe_the_mesh_that_was_marched(
        self, heads: dict[float, _Head]
    ) -> None:
        """The quoted delivered loft is the one the sand sees.

        The mesh reads a little more loft than the design declares -- a
        discretisation offset -- but that offset must not depend on the
        delivery.  Before #9247 it moved by ``2 * (Omega cos(lie) -
        lean)``, from -7.8 to +28.9 degrees over ordinary deliveries.
        """
        head = heads[BOUNCES_DEG[1]]
        square_offset = (
            _mesh_loft_deg(head, face_open_deg=0.0, shaft_lean_deg=0.0)
            - head.geometry.loft_deg
        )
        for face_open_deg, shaft_lean_deg in (
            (0.0, 6.0),
            (0.0, 14.0),
            (10.0, 0.0),
            (10.0, 6.0),
            (20.0, 8.0),
        ):
            from_mesh = _mesh_loft_deg(
                head, face_open_deg=face_open_deg, shaft_lean_deg=shaft_lean_deg
            )
            from_library = effective_loft_deg(
                loft_deg=head.geometry.loft_deg,
                lie_deg=head.geometry.lie_deg,
                face_open_deg=face_open_deg,
                shaft_lean_deg=shaft_lean_deg,
            )
            assert from_mesh - from_library == pytest.approx(square_offset, abs=0.01)


def _mesh_loft_deg(
    head: _Head, *, face_open_deg: float, shaft_lean_deg: float
) -> float:
    """Elevation of the head's face after the delivery, read off the mesh."""
    rotation = delivered_rotation(
        lie_deg=head.geometry.lie_deg,
        face_open_deg=face_open_deg,
        shaft_lean_deg=shaft_lean_deg,
    )
    world = rotation @ _largest_planar_normal(head.elements)
    return math.degrees(math.asin(float(np.clip(world[2], -1.0, 1.0))))


def _largest_planar_normal(elements: SurfaceElements) -> NDArray[np.float64]:
    """Area-weighted normal of the head's largest near-coplanar cluster.

    Found without assuming a direction, so it cannot smuggle the answer
    in: every element's 2-degree neighbourhood is scored by area and the
    heaviest wins.  On a wedge that is the face.
    """
    normals = elements.normals
    areas = elements.areas_m2
    threshold = math.cos(math.radians(2.0))
    best_area = -1.0
    best = np.zeros(len(normals), dtype=np.bool_)
    for index in range(len(normals)):
        close = normals @ normals[index] > threshold
        area = float(areas[close].sum())
        if area > best_area:
            best_area, best = area, close
    weighted = (normals[best] * areas[best, None]).sum(axis=0)
    return np.asarray(weighted / np.linalg.norm(weighted), dtype=np.float64)


class TestNonBuryingRegime:
    """The invariant that was missing: more bounce is never deeper."""

    @pytest.mark.parametrize("attack_deg", SHALLOW_ATTACKS_DEG)
    def test_more_bounce_never_digs_deeper(
        self, solver: DRFTSolver, heads: dict[float, _Head], attack_deg: float
    ) -> None:
        depths = [_shoot(solver, heads[b], attack_deg)[0] for b in BOUNCES_DEG]
        for low, high, shallow, deep in zip(
            BOUNCES_DEG, BOUNCES_DEG[1:], depths, depths[1:], strict=False
        ):
            assert deep <= shallow + 1e-6, (
                f"at {attack_deg} deg of attack, {high} deg of bounce reached "
                f"{deep * 1e3:.3f} mm against {shallow * 1e3:.3f} mm at {low} "
                "deg: more bounce dug deeper"
            )
        assert depths[-1] < depths[0], (
            "bounce must actually buy something: the sweep is flat, from "
            f"{depths[0] * 1e3:.3f} mm to {depths[-1] * 1e3:.3f} mm"
        )

    @pytest.mark.parametrize("attack_deg", SHALLOW_ATTACKS_DEG)
    def test_more_bounce_never_moves_more_sand(
        self, solver: DRFTSolver, heads: dict[float, _Head], attack_deg: float
    ) -> None:
        masses = [_shoot(solver, heads[b], attack_deg)[1] for b in BOUNCES_DEG]
        assert all(mass is not None for mass in masses), (
            "every design in the non-burying regime must cut a measurable "
            f"divot; got {masses}"
        )
        for low, high, light, heavy in zip(
            BOUNCES_DEG, BOUNCES_DEG[1:], masses, masses[1:], strict=False
        ):
            assert heavy is not None and light is not None
            assert heavy <= light + 1e-9, (
                f"at {attack_deg} deg of attack, {high} deg of bounce moved "
                f"{heavy * 1e3:.2f} g against {light * 1e3:.2f} g at {low} deg: "
                "more bounce moved more sand"
            )

    def test_the_ordering_survives_an_open_leaning_delivery(
        self, solver: DRFTSolver, heads: dict[float, _Head]
    ) -> None:
        """A square delivery leaves the rotation at identity.

        So it cannot see a mirrored face-open or shaft-lean sense at all.
        This one can: 10 degrees open and 6 of forward lean is the
        workbench's shipped delivery, and it is the composition that
        carried the mirror.
        """
        depths = [
            _shoot(solver, heads[b], -4.0, face_open_deg=10.0, shaft_lean_deg=6.0)[0]
            for b in BOUNCES_DEG
        ]
        assert depths == sorted(depths, reverse=True), (
            f"open, leaning delivery inverted the ordering: {depths}"
        )


class TestBuryingRegime:
    """Where the head buries, the sign is the same -- only the size differs."""

    def test_the_sweep_spans_both_regimes(
        self, solver: DRFTSolver, heads: dict[float, _Head]
    ) -> None:
        """A steep attack must bury the low-bounce end and not the high.

        This is what makes the burying assertions meaningful rather than
        vacuous, and it brackets the boundary: it lies strictly inside
        the swept range at this attack angle.
        """
        depths = [_shoot(solver, heads[b], STEEP_ATTACK_DEG)[0] for b in BOUNCES_DEG]
        assert depths[0] > _BURYING_DEPTH_RATIO * depths[-1], (
            f"at {STEEP_ATTACK_DEG} deg of attack the sweep does not reach the "
            f"burying regime: {[round(d * 1e3, 2) for d in depths]} mm"
        )

    def test_more_bounce_never_digs_deeper_where_the_head_buries(
        self, solver: DRFTSolver, heads: dict[float, _Head]
    ) -> None:
        """The ordering does not reverse across the boundary.

        Prior work read a reversal here.  That reading was taken against
        an inverted shallow regime: the shallow numbers were the ones
        that were backwards.  With the frame un-mirrored the sign is the
        same on both sides of the boundary.
        """
        depths = [_shoot(solver, heads[b], STEEP_ATTACK_DEG)[0] for b in BOUNCES_DEG]
        for low, high, shallow, deep in zip(
            BOUNCES_DEG, BOUNCES_DEG[1:], depths, depths[1:], strict=False
        ):
            assert deep <= shallow + 1e-6, (
                f"in the burying regime {high} deg of bounce reached "
                f"{deep * 1e3:.3f} mm against {shallow * 1e3:.3f} mm at {low} "
                "deg: more bounce dug deeper"
            )

    def test_a_steeper_attack_never_makes_a_design_shallower(
        self, solver: DRFTSolver, heads: dict[float, _Head]
    ) -> None:
        """The other lever, checked for the same class of sign error.

        Depth against attack angle is the ordering nobody doubts, and it
        held even while bounce was inverted -- which is why it could not
        have caught this on its own.  Kept because a fix that inverted
        *this* instead would otherwise pass the file.
        """
        for bounce in BOUNCES_DEG:
            depths = [
                _shoot(solver, heads[bounce], attack)[0]
                for attack in (-2.0, -4.0, -6.0)
            ]
            assert depths == sorted(depths), (
                f"at {bounce} deg of bounce a steeper attack came out "
                f"shallower: {[round(d * 1e3, 3) for d in depths]} mm"
            )
