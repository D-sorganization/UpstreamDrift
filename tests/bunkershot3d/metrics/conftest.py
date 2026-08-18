"""Synthetic strike traces with hand-computable metrics (issue #8614, W7).

Every builder here produces a trace whose metrics can be worked out on paper,
because the point of the must-pass tests is to check the *arithmetic*, not to
check the code against itself. Two traces carry most of the suite:

**The vee trace** -- a sole travelling at a constant 20 m/s along +x whose depth
is piecewise linear in x, entering at 120 mm behind the ball on a 0.25 slope,
bottoming 20 mm deep at 40 mm behind the ball, and leaving on a 0.20 slope
60 mm past it. Every breakpoint sits on a sample, so trapezoidal integration of
the depth profile is exact and the divot section area is the triangle
``0.5 * 0.180 m * 0.020 m = 1.8e-3 m^2``.

**The decelerating trace** -- a head whose centre of mass follows
``x(t) = 25 t - 500 t^2`` for 10 ms, so its speed falls exactly 25 -> 15 m/s at a
constant 1000 m/s^2, held back by a constant -300 N. ``m a = 0.3 * 1000 = 300 N``
so the force and the motion are consistent, and the work the head does on the
sand is exactly its kinetic-energy loss: ``0.5 * 0.3 * (25^2 - 15^2) = 60 J``.

Position is differentiated with ``edge_order=2``, which is exact for a quadratic
at both ends, so those endpoint values are not quadrature-limited.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import (
    HeadModel,
    SoleLoadTrace,
    StrikeScene,
    StrikeTrace,
)

#: Sample interval shared by the builders [s].
DT_S = 1.0e-4

#: Along-track speed of the vee trace [m/s].
VEE_SPEED_MPS = 20.0

#: Sample spacing of the vee trace along the travel axis [m].
VEE_DX_M = VEE_SPEED_MPS * DT_S

#: Vee-trace breakpoints: (along-track station [m], sole depth [m]).
VEE_ENTRY_X_M = -0.120
VEE_APEX_X_M = -0.040
VEE_APEX_DEPTH_M = 0.020
VEE_EXIT_X_M = 0.060
VEE_DESCENT_SLOPE = 0.25
VEE_ASCENT_SLOPE = 0.20

#: Measured bunker-sand bulk density (Covia Signature 500, research addendum).
SAND_BULK_DENSITY_KG_M3 = 1550.0

#: Effective cutting width used by the divot tests [m].
DIVOT_WIDTH_M = 0.020


def reference_head() -> HeadModel:
    """Return the head every builder here is expressed for.

    A 0.300 kg wedge with its CG 10 mm above the recorded origin, its sole
    reference 20 mm below it, the shaft axis along body +z, and a diagonal
    inertia tensor whose shaft-axis term is 4e-4 kg.m^2 (of the order of
    ``m r^2`` for a 35 mm radius of gyration).

    Returns:
        The head model.
    """
    return HeadModel(
        mass_kg=0.300,
        centre_of_mass_body_m=np.array([0.0, 0.0, 0.010]),
        sole_reference_body_m=np.array([0.0, 0.0, -0.020]),
        shaft_axis_body=np.array([0.0, 0.0, 1.0]),
        inertia_body_kg_m2=np.diag([2.0e-4, 3.0e-4, 4.0e-4]),
    )


def reference_scene() -> StrikeScene:
    """Return the scene: flat sand at z = 0, ball at the origin, travel along +x."""
    return StrikeScene(
        sand_surface_height_m=0.0,
        ball_position_m=np.array([0.0, 0.0, 0.0]),
        travel_axis=np.array([1.0, 0.0, 0.0]),
    )


def vee_depth_m(station_m: np.ndarray) -> np.ndarray:
    """Return the vee trace's sole depth at an along-track station.

    Args:
        station_m: Along-track stations [m], measured from the ball.

    Returns:
        Depth below the undisturbed surface [m], positive downward.
    """
    station = np.asarray(station_m, dtype=float)
    descending = VEE_DESCENT_SLOPE * (station - VEE_ENTRY_X_M)
    ascending = VEE_APEX_DEPTH_M - VEE_ASCENT_SLOPE * (station - VEE_APEX_X_M)
    return np.where(station <= VEE_APEX_X_M, descending, ascending)


def build_trace(
    *,
    sole_path_m: np.ndarray,
    time_s: np.ndarray,
    head: HeadModel,
    force_N: np.ndarray | None = None,
    moment_Nm: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
) -> StrikeTrace:
    """Build a trace from a prescribed **sole** path.

    Args:
        sole_path_m: ``(T, 3)`` world path of the sole reference point.
        time_s: ``(T,)`` sample times [s].
        head: Head whose sole offset relates the sole path to the origin.
        force_N: ``(T, 3)`` or ``(3,)`` sand force; defaults to zero.
        moment_Nm: ``(T, 3)`` or ``(3,)`` sand moment; defaults to zero.
        quaternions: ``(T, 4)`` orientations; defaults to the identity, in which
            case the head origin is simply the sole path minus the sole offset.

    Returns:
        The strike trace.
    """
    count = time_s.size
    if quaternions is None:
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
        head_position = sole_path_m - head.sole_reference_body_m
    else:
        from bunkershot3d.metrics import rotate_body_to_world

        head_position = sole_path_m - rotate_body_to_world(
            quaternions, head.sole_reference_body_m
        )
    return StrikeTrace(
        time_s=time_s,
        head_position_m=head_position,
        head_orientation_quat=quaternions,
        sand_force_N=np.broadcast_to(
            np.zeros(3) if force_N is None else np.asarray(force_N, dtype=float),
            (count, 3),
        ).copy(),
        sand_moment_Nm=np.broadcast_to(
            np.zeros(3) if moment_Nm is None else np.asarray(moment_Nm, dtype=float),
            (count, 3),
        ).copy(),
    )


def build_vee_trace(
    *,
    head: HeadModel | None = None,
    force_N: np.ndarray | None = None,
    moment_Nm: np.ndarray | None = None,
    start_x_m: float = -0.200,
    end_x_m: float = 0.100,
) -> StrikeTrace:
    """Build the vee trace described in this module's docstring.

    Args:
        head: Head to express the path for; defaults to :func:`reference_head`.
        force_N: Constant sand force [N]; defaults to zero.
        moment_Nm: Constant sand moment [N.m]; defaults to zero.
        start_x_m: First along-track station [m].
        end_x_m: Last along-track station [m].

    Returns:
        The trace, sampled every 2 mm of travel at 20 m/s.
    """
    head = reference_head() if head is None else head
    count = int(round((end_x_m - start_x_m) / VEE_DX_M)) + 1
    station = start_x_m + VEE_DX_M * np.arange(count)
    sole_path = np.column_stack([station, np.zeros(count), -vee_depth_m(station)])
    time_s = DT_S * np.arange(count)
    return build_trace(
        sole_path_m=sole_path,
        time_s=time_s,
        head=head,
        force_N=force_N,
        moment_Nm=moment_Nm,
    )


def build_piecewise_trace(
    breakpoints: list[tuple[float, float]],
    *,
    head: HeadModel | None = None,
    force_N: np.ndarray | None = None,
    moment_Nm: np.ndarray | None = None,
) -> StrikeTrace:
    """Build a trace whose sole depth is piecewise linear in along-track station.

    Args:
        breakpoints: ``(station_m, depth_m)`` pairs in increasing station order.
            The first and last set the extent of the trace, so the depth is
            never extrapolated.
        head: Head; defaults to :func:`reference_head`.
        force_N: Constant sand force [N]; defaults to zero.
        moment_Nm: Constant sand moment [N.m]; defaults to zero.

    Returns:
        The trace, sampled every 2 mm of travel at 20 m/s.
    """
    head = reference_head() if head is None else head
    stations = np.array([point[0] for point in breakpoints], dtype=float)
    depths = np.array([point[1] for point in breakpoints], dtype=float)
    count = int(round((stations[-1] - stations[0]) / VEE_DX_M)) + 1
    station = stations[0] + VEE_DX_M * np.arange(count)
    depth = np.interp(station, stations, depths)
    return build_trace(
        sole_path_m=np.column_stack([station, np.zeros(count), -depth]),
        time_s=DT_S * np.arange(count),
        head=head,
        force_N=force_N,
        moment_Nm=moment_Nm,
    )


def build_decelerating_trace(
    *,
    head: HeadModel | None = None,
    entry_speed_mps: float = 25.0,
    exit_speed_mps: float = 15.0,
    duration_s: float = 0.010,
    n_samples: int = 101,
    depth_m: float = 0.030,
    moment_Nm: np.ndarray | None = None,
) -> StrikeTrace:
    """Build a uniformly decelerating trace with a consistent constant force.

    The sand force is set to ``-m a`` exactly, so the work the head does on the
    sand equals its kinetic-energy loss and the energy partition closes with a
    zero residual.

    Args:
        head: Head; defaults to :func:`reference_head`.
        entry_speed_mps: Speed at the first sample.
        exit_speed_mps: Speed at the last sample.
        duration_s: Window length [s].
        n_samples: Number of samples.
        depth_m: Constant sole depth below the surface [m], so the trace is
            entirely inside the sand.
        moment_Nm: Constant sand moment [N.m]; defaults to zero.

    Returns:
        The trace.
    """
    head = reference_head() if head is None else head
    time_s = np.linspace(0.0, duration_s, n_samples)
    deceleration = (entry_speed_mps - exit_speed_mps) / duration_s
    station = entry_speed_mps * time_s - 0.5 * deceleration * time_s**2
    sole_path = np.column_stack(
        [station, np.zeros(n_samples), np.full(n_samples, -depth_m)]
    )
    return build_trace(
        sole_path_m=sole_path,
        time_s=time_s,
        head=head,
        force_N=np.array([-head.mass_kg * deceleration, 0.0, 0.0]),
        moment_Nm=moment_Nm,
    )


def build_sole_load_trace(
    *,
    forces_N: np.ndarray | None = None,
    areas_m2: np.ndarray | None = None,
    duration_s: float = 0.010,
) -> SoleLoadTrace:
    """Build a four-element sole load trace with constant per-element loads.

    Defaults: areas ``[1, 2, 1, 2] cm^2`` and loads ``[100, 0, 50, 0] N`` held
    for 10 ms, so the impulses are ``[1.0, 0, 0.5, 0] N.s`` and the impulse
    densities ``[1e4, 0, 5e3, 0] Pa.s``.

    Args:
        forces_N: ``(E,)`` constant per-element load [N].
        areas_m2: ``(E,)`` element areas [m^2].
        duration_s: Window length [s].

    Returns:
        The load trace.
    """
    forces = (
        np.array([100.0, 0.0, 50.0, 0.0])
        if forces_N is None
        else np.asarray(forces_N, dtype=float)
    )
    areas = (
        np.array([1.0e-4, 2.0e-4, 1.0e-4, 2.0e-4])
        if areas_m2 is None
        else np.asarray(areas_m2, dtype=float)
    )
    count = forces.size
    centroids = np.column_stack(
        [
            np.linspace(-0.010, 0.010, count),
            np.linspace(-0.030, 0.030, count),
            np.zeros(count),
        ]
    )
    time_s = np.linspace(0.0, duration_s, 3)
    return SoleLoadTrace(
        time_s=time_s,
        element_centroid_body_m=centroids,
        element_area_m2=areas,
        element_normal_force_N=np.tile(forces, (time_s.size, 1)),
    )


@pytest.fixture
def head() -> HeadModel:
    """The reference head."""
    return reference_head()


@pytest.fixture
def scene() -> StrikeScene:
    """The reference scene."""
    return reference_scene()


@pytest.fixture
def vee_trace() -> StrikeTrace:
    """The vee trace with no sand wrench."""
    return build_vee_trace()


@pytest.fixture
def decelerating_trace() -> StrikeTrace:
    """The uniformly decelerating trace with a consistent constant force."""
    return build_decelerating_trace()
