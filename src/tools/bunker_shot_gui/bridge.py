"""Plumbing between the F0 solver and the W7 metrics (issue #8618).

:mod:`bunkershot3d.solvers` emits a shot trace; :mod:`bunkershot3d.metrics`
consumes a :class:`~bunkershot3d.metrics.trace.StrikeTrace` and a
:class:`~bunkershot3d.metrics.bounce_map.SoleLoadTrace`. Neither package knows
about the other's shapes, which is deliberate -- the metrics are defined on the
*result artifact* so they mean the same thing at every fidelity tier -- so
something has to carry a result from one to the other. That is this module.

It is the only place in the workbench that reaches past the two packages'
headline APIs, and everything it does is stated rather than assumed:

* the lofted head is **cached**, because solving the camber segment per station
  costs about a second and a design sweep re-uses one geometry many times;
* the head is **placed** so its sole reference enters where the designer asked;
* the per-element sole load is **replayed**, because a shot records the
  resultant and the bounce-utilisation map needs the distribution.

Two things this module used to do have moved into the library, because they
were never workbench concerns: it started the head in free flight itself, and
it coasted the recorded path out ballistically at zero wrench when the solver
stopped with the sole still geometrically in the divot. Both were the
undocumented ritual of issue #8702 -- necessary to get a trace the metrics
would accept, and invisible to anyone outside this application.
:func:`~bunkershot3d.solvers.shot.simulate_shot` now records the whole strike,
and :meth:`~bunkershot3d.metrics.trace.StrikeTrace.from_shot` converts it, so
the workbench composes the two packages the same way any other caller does.

No Qt, no arithmetic that is not either the solver's or the metric's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from bunkershot3d.geometry import (
    CamberFit,
    LoftedWedge,
    MassProperties,
    WedgeGeometry,
    compute_mass_properties,
    loft_wedge,
    shaft_axis,
)
from bunkershot3d.metrics import (
    BounceUtilisation,
    HeadModel,
    SoleLoadTrace,
    StrikeTrace,
)
from bunkershot3d.solvers import (
    DRFTSolver,
    HeadKinematics,
    IntrusionState,
    ShotResult,
    SurfaceElements,
)
from src.shared.python.core.contracts import require

from .design import SwingSetup

__all__ = [
    "MAP_BINS",
    "HeadBuild",
    "SoleLoadMap",
    "build_head",
    "delivery_rotation",
    "entry_kinematics",
    "sole_load_map",
    "sole_load_trace",
    "strike_trace",
]

MAP_BINS = 12
"""Bins per axis of the sole map. Twelve resolves a 20 mm sole to ~1.7 mm."""

_SOLE_NORMAL_CEILING = 0.0
"""An element belongs to the sole when its outward normal points downward."""


@dataclass(frozen=True)
class HeadBuild:
    """The derived, reusable half of a design: mesh, elements, mass.

    Attributes:
        geometry: The design vector this build came from.
        elements_body: Surface discretisation in body axes.
        mass: Volume, centroid and inertia of the lofted head.
        sole_mask: ``(m,)`` elements whose outward normal points downward.
        sole_reference_body_m: ``(3,)`` lowest sole element at address -- the
            point whose depth defines the divot.
        shaft_axis_body: ``(3,)`` unit shaft axis, head toward grip.
        loft: The lofting record, carrying the camber area the head was
            actually built with. The workbench lofts with
            :attr:`~bunkershot3d.geometry.CamberFit.NEAREST` because a
            designer dragging a bounce slider must keep getting a head to
            look at -- but the substitution is then *reported*, never
            assumed away (issue #8698).
    """

    geometry: WedgeGeometry
    elements_body: SurfaceElements
    mass: MassProperties
    sole_mask: NDArray[np.bool_]
    sole_reference_body_m: NDArray[np.float64]
    shaft_axis_body: NDArray[np.float64]
    loft: LoftedWedge

    @property
    def effective_camber_area_m2(self) -> float:
        """The camber area the lofted head actually carries."""
        return self.loft.effective_camber_area_m2

    @property
    def camber_was_clamped(self) -> bool:
        """Whether the declared camber area had to be substituted."""
        return self.loft.camber_was_clamped

    @property
    def head_model(self) -> HeadModel:
        """The metrics package's view of this head."""
        return HeadModel(
            mass_kg=self.geometry.head_mass_kg,
            centre_of_mass_body_m=self.mass.centroid_m,
            sole_reference_body_m=self.sole_reference_body_m,
            shaft_axis_body=self.shaft_axis_body,
            inertia_body_kg_m2=self.mass.inertia_kg_m2,
        )


@dataclass(frozen=True)
class SoleLoadMap:
    """Where the sole carried the strike, resolved spatially.

    Attributes:
        utilisation: The W7 bounce-utilisation metrics.
        density_pa_s: ``(MAP_BINS, MAP_BINS)`` impulse density per bin, with
            NaN in bins holding no sole element. Rows run leading edge to
            trailing edge (body ``x``), columns heel to toe (body ``y``).
        along_edges_m: Bin edges along body ``x``.
        across_edges_m: Bin edges along body ``y``.
    """

    utilisation: BounceUtilisation
    density_pa_s: NDArray[np.float64]
    along_edges_m: NDArray[np.float64]
    across_edges_m: NDArray[np.float64]

    @property
    def peak_density_pa_s(self) -> float:
        """Largest binned impulse density, or 0.0 when the map is empty."""
        if not np.isfinite(self.density_pa_s).any():
            return 0.0
        return float(np.nanmax(self.density_pa_s))


@lru_cache(maxsize=16)
def build_head(
    geometry: WedgeGeometry, n_profile_points: int, n_stations: int
) -> HeadBuild:
    """Loft a head and derive everything the solver and metrics need.

    Cached: the camber segment is solved by root-finding per station, which
    costs about a second, and a design sweep re-uses one geometry many times.

    Args:
        geometry: The design vector.
        n_profile_points: Sole samples per cross-section.
        n_stations: Cross-sections from heel to toe.

    Returns:
        The reusable build.

    Raises:
        ValueError: If the sole cannot be lofted at this resolution.
    """
    lofted = loft_wedge(
        geometry,
        n_profile_points=n_profile_points,
        n_stations=n_stations,
        camber_fit=CamberFit.NEAREST,
    )
    mesh = lofted.mesh
    elements = SurfaceElements.from_mesh(mesh)
    sole_mask = elements.normals[:, 2] < _SOLE_NORMAL_CEILING
    sole_centroids = elements.centroids_m[sole_mask]
    reference = sole_centroids[int(np.argmin(sole_centroids[:, 2]))]
    _, axis = shaft_axis(geometry)
    return HeadBuild(
        geometry=geometry,
        elements_body=elements,
        mass=compute_mass_properties(mesh, mass_kg=geometry.head_mass_kg),
        sole_mask=sole_mask,
        sole_reference_body_m=np.asarray(reference, dtype=np.float64),
        shaft_axis_body=np.asarray(axis, dtype=np.float64),
        loft=lofted,
    )


def delivery_rotation(
    geometry: WedgeGeometry, swing: SwingSetup
) -> NDArray[np.float64]:
    """Return the body-to-world rotation for a delivered head.

    Opening the face is a rotation about the shaft axis; shaft lean is a pure
    pitch applied after it. This is the same composition
    :mod:`bunkershot3d.geometry.delivery` uses for its closed-form angles,
    built here on the public shaft axis so the mesh and those angles cannot
    disagree.

    Args:
        geometry: The design vector, supplying the lie angle.
        swing: The delivery.

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    _, axis = shaft_axis(geometry)
    opening = Rotation.from_rotvec(np.asarray(axis) * math.radians(swing.face_open_deg))
    lean = Rotation.from_euler("y", math.radians(swing.shaft_lean_deg))
    return np.asarray((lean * opening).as_matrix(), dtype=np.float64)


def entry_kinematics(build: HeadBuild, swing: SwingSetup) -> HeadKinematics:
    """Place the head so its sole reference enters where the designer asked.

    Only the along-track station is set here. The height is the solver's:
    ``ShotSettings.start_at_first_contact`` drops the head so the sole
    reference reaches the free surface after the recorded free-flight lead-in,
    which is the approach the dig-versus-skid discriminator measures the
    delivered path slope across. This module used to compute that clearance
    itself; doing it once, in the solver, is what makes the workbench's trace
    the same trace any other caller gets.

    Args:
        build: The lofted head.
        swing: The delivery.

    Returns:
        The entry pose and velocity.
    """
    orientation = delivery_rotation(build.geometry, swing)
    speed = float(swing.clubhead_speed_mps)
    attack_rad = math.radians(swing.attack_angle_deg)
    velocity = speed * np.array(
        [math.cos(attack_rad), 0.0, math.sin(attack_rad)], dtype=np.float64
    )
    require(
        -float(velocity[2]) > 0.0,
        "the head must be descending to enter the sand",
        value=-float(velocity[2]),
    )
    reference_world = orientation @ build.sole_reference_body_m
    position = np.array(
        [
            -swing.entry_distance_behind_ball_m - float(reference_world[0]),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return HeadKinematics(
        velocity_m_s=velocity, position_m=position, orientation=orientation
    )


def strike_trace(result: ShotResult) -> StrikeTrace | None:
    """Convert a shot trace into the metrics package's input contract.

    A thin wrapper on
    :meth:`~bunkershot3d.metrics.trace.StrikeTrace.from_shot`, kept only so
    that a shot too short to differentiate reads as "no trace" here rather
    than as an exception in the middle of assembling an outcome. It adds no
    samples and no arithmetic of its own.

    Args:
        result: The shot.

    Returns:
        The trace, or ``None`` when it is too short to differentiate.
    """
    if result.n_steps < 3:
        return None
    return StrikeTrace.from_shot(result)


def sole_load_trace(
    solver: DRFTSolver,
    build: HeadBuild,
    result: ShotResult,
    orientation: NDArray[np.float64],
) -> SoleLoadTrace | None:
    """Replay the trace to recover the compressive load on each sole element.

    ``simulate_shot`` records the resultant, not the distribution, so the
    per-element response is re-derived from the recorded poses. The head does
    not rotate during the shot, so replaying the recorded position at the
    recorded velocity reproduces the states the march actually solved.

    Args:
        solver: The solver the shot was run with.
        build: The lofted head.
        result: The shot trace.
        orientation: The constant body-to-world rotation.

    Returns:
        The per-element sole loading, or ``None`` when the trace is too short
        for an impulse.
    """
    if result.n_steps < 2:
        return None
    oriented = build.elements_body.transformed(rotation=orientation)
    sole_index = np.flatnonzero(build.sole_mask)
    slot = np.full(build.elements_body.n_elements, -1, dtype=np.int64)
    slot[sole_index] = np.arange(sole_index.size)
    loads = np.zeros((result.n_steps, sole_index.size), dtype=np.float64)
    for step in range(result.n_steps):
        position = result.positions_m[step]
        world = oriented.translated(position)
        response = solver.element_response(
            IntrusionState(
                world, result.velocities_m_s[step], reference_point_m=position
            )
        )
        if response.index.size == 0:
            continue
        keep = slot[response.index] >= 0
        if not keep.any():
            continue
        active = response.index[keep]
        traction = (
            response.depth_traction_pa[keep] + response.inertial_traction_pa[keep]
        )
        # Compression only: the projection onto the inward normal. Sand cannot
        # pull on a sole, and the depth term's tangential part can point
        # outward on a steeply raked element without meaning tension.
        normal_load = (
            np.maximum(-np.einsum("ij,ij->i", traction, world.normals[active]), 0.0)
            * world.areas_m2[active]
        )
        loads[step, slot[active]] = normal_load
    return SoleLoadTrace(
        time_s=result.times_s,
        element_centroid_body_m=build.elements_body.centroids_m[sole_index],
        element_area_m2=build.elements_body.areas_m2[sole_index],
        element_normal_force_N=loads,
    )


def sole_load_map(load: SoleLoadTrace, utilisation: BounceUtilisation) -> SoleLoadMap:
    """Bin the per-element impulse density onto a rectangular sole grid.

    Args:
        load: The per-element loading.
        utilisation: Its bounce-utilisation metrics.

    Returns:
        The map, with NaN in bins holding no sole element.
    """
    centroids = load.element_centroid_body_m
    along_edges = _bin_edges(centroids[:, 0], MAP_BINS)
    across_edges = _bin_edges(centroids[:, 1], MAP_BINS)
    along = np.clip(np.digitize(centroids[:, 0], along_edges[1:-1]), 0, MAP_BINS - 1)
    across = np.clip(np.digitize(centroids[:, 1], across_edges[1:-1]), 0, MAP_BINS - 1)
    impulse = np.zeros((MAP_BINS, MAP_BINS), dtype=np.float64)
    area = np.zeros((MAP_BINS, MAP_BINS), dtype=np.float64)
    np.add.at(impulse, (along, across), utilisation.element_impulse_Ns)
    np.add.at(area, (along, across), load.element_area_m2)
    density = np.full((MAP_BINS, MAP_BINS), np.nan, dtype=np.float64)
    occupied = area > 0.0
    density[occupied] = impulse[occupied] / area[occupied]
    return SoleLoadMap(
        utilisation=utilisation,
        density_pa_s=density,
        along_edges_m=along_edges,
        across_edges_m=across_edges,
    )


def _bin_edges(stations: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    """Return ``n_bins + 1`` equal-width edges spanning ``stations``."""
    low, high = float(stations.min()), float(stations.max())
    if high <= low:
        high = low + 1.0
    return np.linspace(low, high, n_bins + 1, dtype=np.float64)
