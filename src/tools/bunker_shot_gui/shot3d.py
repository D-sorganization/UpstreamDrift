"""The 3-D shot scene: the head, the free surface and the divot (issue #8706).

The workbench can already say which part of the sole carried load and when.
This module carries the thing a designer actually pictures: the head moving
through the sand. It computes; it draws nothing. No Qt, no matplotlib, no
display, in keeping with the split issue #8618 established -- the drawing is
:mod:`src.tools.bunker_shot_gui.render3d`, and the *choice* of renderer is
the ADR-0027 viewport layer's, reached through
:func:`~.render.viewport_fallback`.

What a scene is allowed to say
------------------------------

F0 is a constitutive shortcut: it integrates an empirical resistive stress
over the head's wetted surface and never solves the sand's motion. Two
consequences are load-bearing here, and both are enforced by the value
objects rather than left to whatever draws them.

**The sand surface is a height, not a bed.** :class:`SandSurface` is one
number -- the world ``z`` of the undisturbed free surface, the same
``free_surface_height_m`` the solver judged every element depth against.
There are no grains at F0, so :attr:`SandSurface.resolves_grains` is
``False``; a renderer that stippled a grain bed would be inventing a field
the model has none of.

A grain-resolving tier is different, and issue #8729 made room for it
without loosening any of the above. :attr:`ShotScene.sand` carries a
solved field when there is one, and the scene refuses a field whose tier
disagrees with its own: an F1 grain bed animated over an F0 trajectory
looks exactly like an F1 shot and is a claim neither run made. The flags
that decide every caption -- ``resolves_grains`` on both the surface and
the divot -- are stored as data and validated against the field's
presence, so a scene cannot draw moving sand under a sentence denying it.

**The divot is the head's swept envelope, not transported sand.** F0 never
moves a grain, so the only honest divot at this tier is a statement about
where the head has *been*: :class:`DivotSection` is the running minimum of
the head's own lower surface at each along-track station, clipped at the
free surface. It only ever deepens, because a swept envelope cannot
un-sweep -- a divot that filled back in would be a claim about sand
transport, which is exactly the claim F0 cannot make.

The free surface is *recovered from the trace* rather than passed in
alongside it. ``ShotResult`` records ``sole_depths_m`` and the pose, and
those two fix the surface by definition:
``free_surface = sole_depth + z(sole reference)``. Recovering it means the
drawn surface and the solver's own depths cannot drift apart.

Cameras
-------

:class:`CameraPreset` is data, not a matplotlib call. Each preset states an
elevation and azimuth in degrees *and* a backend-neutral unit
:attr:`~CameraPreset.eye_direction`, so the same three views survive a move
to MeshCat, Rerun or VTK when one of them is installed. The three are the
ones a shot is actually discussed in: down the line, face on, and sole level
sighting along the leading edge -- the last being the view in which the
divot section is a cross-section rather than a foreshortened smear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.geometry import TriangleMesh
from bunkershot3d.solvers import (
    EnvelopeStatus,
    FidelityTier,
    ShotResult,
    ValidityVerdict,
)
from src.shared.python.visualization.viewport import ViewportOverlayPayload

from .bridge import HeadBuild, free_surface_height_m
from .sandvolume import SandVolume

__all__ = [
    "DIVOT_STATIONS",
    "CameraPreset",
    "DivotSection",
    "SandSurface",
    "ShotScene",
    "shot_scene",
    "viewport_payload",
]

DIVOT_STATIONS = 96
"""Along-track stations the swept envelope is resolved on.

Ninety-six spans a ~150 mm divot at about 1.6 mm, which is finer than the
12-bin sole map and coarse enough that the running minimum over a whole shot
stays a millisecond of work.
"""

_MARKER_NAMES: tuple[str, ...] = (
    "sole_reference",
    "leading_edge",
    "trailing_edge",
    "heel",
    "toe",
)
"""The named points a backend-neutral payload carries, in payload order."""


class CameraPreset(str, Enum):
    """A named view of the shot, in the terms the shot is discussed in.

    Attributes:
        DOWN_THE_LINE: From behind the entry, sighting along the target
            line. The view in which entry point and divot length read.
        FACE_ON: Square to the target line, the broadcast view. The view in
            which shaft lean and attack angle read.
        SOLE_LEVEL: In the free-surface plane, sighting along the leading
            edge. The view in which the divot is a true cross-section.
    """

    DOWN_THE_LINE = "down_the_line"
    FACE_ON = "face_on"
    SOLE_LEVEL = "sole_level"

    @classmethod
    def _missing_(cls, value: object) -> CameraPreset:
        """Name the valid views when coercion fails.

        Args:
            value: Whatever was offered.

        Returns:
            Never; this always raises.

        Raises:
            ValueError: Always. A view name arrives from a combo box or a
                saved layout, and the stock enum message does not say what
                the alternatives are.
        """
        valid = ", ".join(item.value for item in cls)
        raise ValueError(f"unknown camera preset {value!r}; valid: {valid}")

    @property
    def label(self) -> str:
        """A short heading for this view."""
        return _CAMERA_LABEL[self]

    @property
    def description(self) -> str:
        """One line saying what the view is for."""
        return _CAMERA_DESCRIPTION[self]

    @property
    def elevation_deg(self) -> float:
        """Eye elevation above the free-surface plane [deg]."""
        return _CAMERA_ANGLES[self][0]

    @property
    def azimuth_deg(self) -> float:
        """Eye azimuth about world ``z``, measured from ``+x`` [deg]."""
        return _CAMERA_ANGLES[self][1]

    @property
    def eye_direction(self) -> NDArray[np.float64]:
        """Unit vector from the subject toward the eye, world axes.

        Backend-neutral: matplotlib takes the two angles, and a provider
        that wants a camera position takes this and a radius. Deriving one
        from the other keeps the fallback and a future MeshCat view showing
        the same thing.
        """
        elevation = math.radians(self.elevation_deg)
        azimuth = math.radians(self.azimuth_deg)
        return np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float64,
        )


_CAMERA_ANGLES: dict[CameraPreset, tuple[float, float]] = {
    # Behind the entry on the target line: azimuth 180 deg puts the eye at
    # -x, so the view direction is +x, the way the head travels.
    CameraPreset.DOWN_THE_LINE: (12.0, 180.0),
    # Square to the target line, a little above it.
    CameraPreset.FACE_ON: (14.0, -90.0),
    # Level with the free surface, sighting along the leading edge, which
    # runs heel to toe. Zero elevation exactly: "sole level" is a claim
    # about where the eye is, and a few degrees of tilt would foreshorten
    # the very section this view exists to show.
    CameraPreset.SOLE_LEVEL: (0.0, -90.0),
}

_CAMERA_LABEL: dict[CameraPreset, str] = {
    CameraPreset.DOWN_THE_LINE: "Down the line",
    CameraPreset.FACE_ON: "Face on",
    CameraPreset.SOLE_LEVEL: "Sole level, along the leading edge",
}

_CAMERA_DESCRIPTION: dict[CameraPreset, str] = {
    CameraPreset.DOWN_THE_LINE: (
        "from behind the entry along the target line; entry point and divot "
        "length read here"
    ),
    CameraPreset.FACE_ON: (
        "square to the target line; shaft lean and attack angle read here"
    ),
    CameraPreset.SOLE_LEVEL: (
        "in the free-surface plane, sighting along the leading edge; the divot "
        "is a true section here rather than a foreshortened smear"
    ),
}


def _check_extent(name: str, extent: tuple[float, float]) -> tuple[float, float]:
    """Validate a drawn extent.

    Args:
        name: Which extent, for the message.
        extent: ``(low, high)`` bounds [m].

    Returns:
        The extent as floats.

    Raises:
        ValueError: If a bound is not finite or the pair does not increase.
            A ``raise`` rather than an ``assert``: an inverted extent would
            silently draw the surface inside out under ``python -O``.
    """
    low, high = float(extent[0]), float(extent[1])
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"{name} must be finite, got {extent!r}")
    if not low < high:
        raise ValueError(f"{name} must increase, got {low} to {high}")
    return (low, high)


@dataclass(frozen=True)
class SandSurface:
    """The undisturbed free surface: one height, and no grains.

    This is the whole of what F0 knows about the sand's geometry. The solver
    judges every element depth against a single ``free_surface_height_m``,
    so that number *is* the sand as far as this tier is concerned.

    Attributes:
        height_m: World ``z`` of the undisturbed free surface [m].
        along_extent_m: ``(low, high)`` world ``x`` the surface is drawn
            over [m].
        across_extent_m: ``(low, high)`` world ``y`` the surface is drawn
            over [m].
        resolves_grains: Whether the tier behind this surface solves a
            grain bed under it. Stored as data rather than decided by a
            constant, the same way :mod:`bunkershot3d.fields` stores tier
            and validity: a renderer that consults it gets the answer for
            *this* shot rather than for whichever tier was current when
            the property was written.
        tier: Which rung of the ADR-0032 ladder the surface came from,
            so the caption can name it rather than hard-coding one.
    """

    height_m: float
    along_extent_m: tuple[float, float]
    across_extent_m: tuple[float, float]
    resolves_grains: bool = False
    tier: FidelityTier = FidelityTier.F0

    def __post_init__(self) -> None:
        """Validate the surface.

        Raises:
            ValueError: If the height is not finite or an extent does not
                increase.
        """
        height = float(self.height_m)
        if not math.isfinite(height):
            raise ValueError(f"height_m must be finite, got {self.height_m!r}")
        object.__setattr__(self, "height_m", height)
        object.__setattr__(
            self, "along_extent_m", _check_extent("along_extent_m", self.along_extent_m)
        )
        object.__setattr__(
            self,
            "across_extent_m",
            _check_extent("across_extent_m", self.across_extent_m),
        )

    def describe(self) -> str:
        """One line stating what the drawn surface is, and is not.

        The plane is an *input* at every tier: the solver is set up with an
        undisturbed free-surface height and judges depths against it. What
        changes with the tier is whether anything is solved beneath it, so
        that is the half of the sentence this branches on.

        Returns:
            The sentence drawn beside, or inside, any 3-D frame.
        """
        height = f"sand: model free-surface height at z = {self.height_m * 1e3:.1f} mm"
        if not self.resolves_grains:
            return (
                f"{height}; {self.tier.value} resolves no grains, so this plane "
                "is a boundary condition, not a grain bed"
            )
        return (
            f"{height}; the undisturbed level {self.tier.value} was set up with, "
            "not a result -- the solved grains are the field drawn under it"
        )


@dataclass(frozen=True)
class DivotSection:
    """The section the head sweeps out under the free surface, over time.

    Not transported sand. F0 integrates a stress over the head and never
    solves the bed's motion, so the only divot it can honestly report is the
    running lower envelope of the head itself: at each along-track station,
    the lowest the head has reached at or before this sample, clipped at the
    free surface.

    Attributes:
        station_m: ``(S,)`` strictly increasing along-track stations, world
            ``x`` [m].
        floor_m: ``(T, S)`` world ``z`` of the swept envelope [m]. Equal to
            the free surface where the head has not been.
        surface_height_m: The free surface the envelope is clipped at [m].
        resolves_grains: Whether the tier behind it solved a grain bed.
            The envelope is the head's own geometry at every tier, but
            what it may be *contrasted with* changes: at F0 there is no
            transported sand to distinguish it from, and at a
            grain-resolving tier there is.
        tier: Which rung of the ADR-0032 ladder produced it, so the
            sentence names the tier it came from rather than a fixed one.
    """

    station_m: NDArray[np.float64]
    floor_m: NDArray[np.float64]
    surface_height_m: float
    resolves_grains: bool = False
    tier: FidelityTier = FidelityTier.F0

    def __post_init__(self) -> None:
        """Validate the section.

        Raises:
            ValueError: If shapes disagree, a value is not finite, the
                stations are not increasing, the floor rises above the free
                surface, or the envelope fills back in. The last is the
                important one: an envelope that un-sweeps is a claim about
                sand moving, which is the claim this tier cannot make.
        """
        stations = np.asarray(self.station_m, dtype=np.float64).reshape(-1)
        floor = np.asarray(self.floor_m, dtype=np.float64)
        height = float(self.surface_height_m)
        if not math.isfinite(height):
            raise ValueError(f"surface_height_m must be finite, got {height!r}")
        if stations.size < 2:
            raise ValueError(
                f"a divot section needs at least 2 stations, got {stations.size}"
            )
        if np.any(np.diff(stations) <= 0.0):
            raise ValueError("station_m must be strictly increasing")
        if floor.ndim != 2 or floor.shape[1] != stations.size:
            raise ValueError(
                f"floor_m must have shape (T, {stations.size}), got {floor.shape}"
            )
        for name, array in (("station_m", stations), ("floor_m", floor)):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite; found NaN or inf")
        if float(floor.max()) > height + _FLOOR_TOLERANCE_M:
            raise ValueError(
                "a divot floor cannot sit above the free surface it was cut "
                f"into; peak {float(floor.max())} m against a surface at {height} m"
            )
        if floor.shape[0] > 1 and float(np.diff(floor, axis=0).max()) > (
            _FLOOR_TOLERANCE_M
        ):
            raise ValueError(
                "a swept envelope may only ever deepen: F0 transports no sand, "
                "so a floor that rises between samples would be a claim about "
                "sand moving that this tier cannot make"
            )
        object.__setattr__(self, "station_m", stations)
        object.__setattr__(self, "floor_m", floor)
        object.__setattr__(self, "surface_height_m", height)

    @property
    def is_swept_envelope(self) -> bool:
        """Whether this section is the head's envelope rather than sand flow."""
        return True

    @property
    def n_frames(self) -> int:
        """Number of samples the envelope was accumulated over."""
        return int(self.floor_m.shape[0])

    @property
    def n_stations(self) -> int:
        """Number of along-track stations resolved."""
        return int(self.station_m.size)

    @property
    def depth_m(self) -> NDArray[np.float64]:
        """``(T, S)`` depth below the free surface [m]; never negative."""
        return np.maximum(self.surface_height_m - self.floor_m, 0.0)

    @property
    def max_depth_m(self) -> float:
        """Deepest the envelope ever reached [m]."""
        return float(self.depth_m.max())

    @property
    def section_area_m2(self) -> NDArray[np.float64]:
        """``(T,)`` cut area of the section at each sample [m^2].

        The trapezoidal integral of depth along the track, which is the same
        quantity :attr:`~bunkershot3d.metrics.divot.DivotMetrics.section_area_m2`
        reports for the whole shot -- reported here per sample so it can be
        scrubbed. Non-decreasing, because the envelope is.
        """
        return np.asarray(
            np.trapezoid(self.depth_m, x=self.station_m, axis=1), dtype=np.float64
        )

    def describe(self) -> str:
        """One line stating what the section is, and is not.

        Returns:
            The sentence drawn inside any frame showing the divot.
        """
        envelope = (
            "divot: the swept lower envelope of the head below the free "
            f"surface ({self.max_depth_m * 1e3:.1f} mm deepest)"
        )
        if not self.resolves_grains:
            return (
                f"{envelope}. {self.tier.value} moves no sand, so this is "
                "where the head has been, not where sand has gone"
            )
        return (
            f"{envelope}. Still the head's own envelope, not a sand surface: "
            f"{self.tier.value} does solve the bed, and where the sand went "
            "is the field, not this line"
        )


_FLOOR_TOLERANCE_M = 1e-12
"""Slack on the envelope invariants, for the running minimum's own rounding."""


@dataclass(frozen=True)
class ShotScene:
    """The head, the surface and the divot, resolved in time (issue #8706).

    Attributes:
        time_s: ``(T,)`` strictly increasing sample times [s].
        position_m: ``(T, 3)`` body-origin position, world axes [m].
        orientation: ``(T, 3, 3)`` body-to-world rotations.
        head_point_body_m: ``(P, 3)`` head surface points in body axes [m];
            the solver's own element centroids, so the drawn head is the
            head that was integrated over.
        head_mesh_body: The lofted, watertight head, in body axes -- the
            same :class:`~bunkershot3d.geometry.TriangleMesh` the solver
            discretised into ``head_point_body_m``, kept whole so a renderer
            can draw the head as a solid rather than as its element cloud
            (issue #8706 defect 1). Posed per frame by
            :meth:`head_mesh_world_m`, the same pose that places
            ``head_point_body_m``.
        sole_index: ``(K,)`` indices into ``head_point_body_m`` of the sole
            elements -- the ones whose outward normal points downward.
        sole_reference_body_m: ``(3,)`` the point the sole depth is measured
            at, in body axes.
        sole_depth_m: ``(T,)`` depth of that point below the free surface
            [m], positive downward, straight from the solver.
        surface: The free surface.
        divot: The swept section.
        verdict: The validity statement the whole scene must be read under.
        fidelity_tier: Which rung of the ADR-0032 ladder produced it.
        sand: The solved sand field, when the tier resolved one (issue
            #8729). ``None`` at F0, which moves no sand at all. It carries
            its own tier, and :meth:`__post_init__` refuses one that
            disagrees with this scene's: an F1 field animated over an F0
            shot is entirely plausible to look at and is a claim neither
            run made.
    """

    time_s: NDArray[np.float64]
    position_m: NDArray[np.float64]
    orientation: NDArray[np.float64]
    head_point_body_m: NDArray[np.float64]
    head_mesh_body: TriangleMesh
    sole_index: NDArray[np.int64]
    sole_reference_body_m: NDArray[np.float64]
    sole_depth_m: NDArray[np.float64]
    surface: SandSurface
    divot: DivotSection
    verdict: ValidityVerdict
    fidelity_tier: FidelityTier
    sand: SandVolume | None = None

    def __post_init__(self) -> None:
        """Validate the scene.

        Raises:
            ValueError: If shapes disagree, a value is not finite, time is
                not strictly increasing, an orientation is not a rotation,
                or the divot does not describe this shot. ``raise`` rather
                than ``assert``: ``python -O`` strips asserts, and a scene
                that failed these would be *drawn* rather than rejected.
        """
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        if times.size < 2:
            raise ValueError(f"a shot scene needs at least 2 samples, got {times.size}")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        positions = np.asarray(self.position_m, dtype=np.float64)
        if positions.shape != (times.size, 3):
            raise ValueError(
                f"position_m must have shape {(times.size, 3)}, got {positions.shape}"
            )
        rotations = np.asarray(self.orientation, dtype=np.float64)
        if rotations.shape != (times.size, 3, 3):
            raise ValueError(
                f"orientation must have shape {(times.size, 3, 3)}, "
                f"got {rotations.shape}"
            )
        points = np.asarray(self.head_point_body_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"head_point_body_m must have shape (P, 3), got {points.shape}"
            )
        if points.shape[0] == 0:
            raise ValueError("a shot scene needs at least one head point to draw")
        if not isinstance(self.head_mesh_body, TriangleMesh):
            raise ValueError(
                "head_mesh_body must be the lofted TriangleMesh, not the "
                f"element cloud; got {type(self.head_mesh_body).__name__}"
            )
        if self.head_mesh_body.n_faces == 0:
            raise ValueError("a shot scene needs a head mesh with at least one face")
        sole = np.asarray(self.sole_index, dtype=np.int64).reshape(-1)
        if sole.size == 0:
            raise ValueError(
                "a shot scene needs at least one sole point: the sole is the "
                "surface the whole workbench is about"
            )
        if int(sole.min()) < 0 or int(sole.max()) >= points.shape[0]:
            raise ValueError(
                f"sole_index must index head_point_body_m, got range "
                f"{int(sole.min())}..{int(sole.max())} for {points.shape[0]} points"
            )
        reference = np.asarray(self.sole_reference_body_m, dtype=np.float64).reshape(-1)
        if reference.shape != (3,):
            raise ValueError(
                f"sole_reference_body_m must have shape (3,), got {reference.shape}"
            )
        depths = np.asarray(self.sole_depth_m, dtype=np.float64).reshape(-1)
        if depths.shape != (times.size,):
            raise ValueError(
                f"sole_depth_m must have shape {(times.size,)}, got {depths.shape}"
            )
        for name, array in (
            ("time_s", times),
            ("position_m", positions),
            ("orientation", rotations),
            ("head_point_body_m", points),
            ("sole_reference_body_m", reference),
            ("sole_depth_m", depths),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite; found NaN or inf")
        if not isinstance(self.verdict, ValidityVerdict):
            raise ValueError(
                "a shot scene travels with the verdict it must be read under; "
                "an animation drawn without its validity statement is the most "
                "persuasive unlabelled picture this tool can produce"
            )
        if self.divot.n_frames != times.size:
            raise ValueError(
                "the divot section and the pose must come from one shot; got "
                f"{self.divot.n_frames} envelope samples against {times.size} poses"
            )
        self._check_sand()
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "position_m", positions)
        object.__setattr__(self, "orientation", rotations)
        object.__setattr__(self, "head_point_body_m", points)
        object.__setattr__(self, "sole_index", sole)
        object.__setattr__(self, "sole_reference_body_m", reference)
        object.__setattr__(self, "sole_depth_m", depths)

    def _check_sand(self) -> None:
        """Refuse a sand field that did not come from this shot's tier.

        The substitution this catches is the most persuasive unlabelled
        picture the workbench can produce: an F1 grain field animated over
        an F0 trajectory looks exactly like an F1 shot and is a claim
        neither run made. The surface's own ``resolves_grains`` has to
        agree too, because that flag is what every caption and every
        renderer branches on -- a scene whose caption said "resolves no
        grains" over a box of moving sand would be worse than either half
        alone.

        Raises:
            ValueError: If the tiers disagree, or the surface and the
                field disagree about whether grains were resolved.
        """
        sand = self.sand
        if sand is not None and sand.fidelity_tier is not self.fidelity_tier:
            raise ValueError(
                f"this scene was solved at {self.fidelity_tier.value} and the "
                f"sand field at {sand.fidelity_tier.value}; drawing one tier's "
                "grains over another tier's trajectory is entirely plausible "
                "to look at and is a claim neither run made"
            )
        if self.divot.resolves_grains != (sand is not None):
            raise ValueError(
                "the divot's resolves_grains and the presence of a sand field "
                f"must agree; got resolves_grains={self.divot.resolves_grains} "
                f"with {'a' if sand is not None else 'no'} field"
            )
        if self.surface.resolves_grains != (sand is not None):
            raise ValueError(
                "the surface's resolves_grains and the presence of a sand "
                f"field must agree; got resolves_grains="
                f"{self.surface.resolves_grains} with "
                f"{'a' if sand is not None else 'no'} field. Every caption in "
                "the 3-D view branches on that flag, so a scene that disagrees "
                "with itself draws a denial over solved sand, or promises "
                "grains it has none of"
            )

    # ---------------------------------------------------------------- extent

    @property
    def resolves_grains(self) -> bool:
        """Whether this scene carries a solved grain bed."""
        return self.sand is not None

    def sand_note(self) -> tuple[str, ...]:
        """The sentences a frame must draw about what its sand is.

        One place, so the matplotlib fallback and the VTK backend cannot
        drift into qualifying the same picture differently.

        Returns:
            The free-surface line, the extrusion line when there is a
            solved field, and the divot line -- in the order they are
            drawn.
        """
        lines = [self.surface.describe()]
        if self.sand is not None:
            lines.append(self.sand.describe())
        lines.append(self.divot.describe())
        return tuple(lines)

    @property
    def n_frames(self) -> int:
        """Number of samples in the scene."""
        return int(self.time_s.size)

    @property
    def n_head_points(self) -> int:
        """Number of head surface points drawn."""
        return int(self.head_point_body_m.shape[0])

    @property
    def n_head_mesh_faces(self) -> int:
        """Number of triangles in the lofted head mesh."""
        return self.head_mesh_body.n_faces

    @property
    def status(self) -> EnvelopeStatus:
        """How much of this scene may be believed."""
        return self.verdict.status

    def _check_frame(self, frame: int) -> int:
        """Validate a frame index.

        Args:
            frame: The requested sample index.

        Returns:
            The index.

        Raises:
            ValueError: If it is outside the recorded shot. A wrapped index
                would draw a different moment from the one the transport
                says it is showing.
        """
        if not 0 <= int(frame) < self.n_frames:
            raise ValueError(
                f"frame {frame} is outside the recorded shot, which has "
                f"{self.n_frames} samples"
            )
        return int(frame)

    # ----------------------------------------------------------------- poses

    def head_world_m(self, frame: int) -> NDArray[np.float64]:
        """Return the head's surface points at one sample, world axes.

        Args:
            frame: The sample index.

        Returns:
            ``(P, 3)`` world points, ``p + R c``.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        index = self._check_frame(frame)
        rotated = self.head_point_body_m @ self.orientation[index].T
        return np.asarray(rotated + self.position_m[index], dtype=np.float64)

    def head_mesh_world_m(self, frame: int) -> NDArray[np.float64]:
        """Return the lofted head mesh's vertices at one sample, world axes.

        Posed the same way :meth:`head_world_m` poses the element cloud
        (``v R^T + p``), so the solid a renderer draws and the centroids the
        solver integrated over never disagree about where the head is. The
        face topology (``head_mesh_body.faces``) does not depend on the
        frame -- only these vertex positions do.

        Args:
            frame: The sample index.

        Returns:
            ``(V, 3)`` world vertices, in ``head_mesh_body.faces`` order.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        index = self._check_frame(frame)
        rotated = self.head_mesh_body.vertices @ self.orientation[index].T
        return np.asarray(rotated + self.position_m[index], dtype=np.float64)

    def sole_world_m(self, frame: int) -> NDArray[np.float64]:
        """Return the sole elements' world points at one sample.

        Args:
            frame: The sample index.

        Returns:
            ``(K, 3)`` world points.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        return self.head_world_m(frame)[self.sole_index]

    @property
    def sole_reference_world_m(self) -> NDArray[np.float64]:
        """``(T, 3)`` world path of the sole reference point [m]."""
        rotated = np.einsum("tij,j->ti", self.orientation, self.sole_reference_body_m)
        return np.asarray(rotated + self.position_m, dtype=np.float64)

    @property
    def path_world_m(self) -> NDArray[np.float64]:
        """``(T, 3)`` the path the scene is framed on -- the sole reference."""
        return self.sole_reference_world_m

    @property
    def speed_m_s(self) -> NDArray[np.float64]:
        """``(T,)`` speed of the sole reference point [m/s].

        Differentiated from the recorded path rather than read from the
        solver's velocities, so it is the speed of the point the rest of
        this scene is framed on even when the head is rotating.
        """
        gradient = np.gradient(self.sole_reference_world_m, self.time_s, axis=0)
        return np.asarray(
            np.sqrt(np.einsum("ij,ij->i", gradient, gradient)), dtype=np.float64
        )

    def named_markers_world_m(self, frame: int) -> NDArray[np.float64]:
        """Return the five named head points at one sample, world axes.

        The points a backend-neutral payload can label: the sole reference,
        the leading and trailing edges, and the heel and toe extremes of the
        sole. Enough to orient a view without shipping the whole cloud.

        Args:
            frame: The sample index.

        Returns:
            ``(5, 3)`` world points, in :data:`_MARKER_NAMES` order.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        world = self.head_world_m(frame)
        sole = world[self.sole_index]
        body_sole = self.head_point_body_m[self.sole_index]
        reference = (
            self.orientation[self._check_frame(frame)] @ self.sole_reference_body_m
            + self.position_m[frame]
        )
        return np.asarray(
            [
                reference,
                sole[int(np.argmin(body_sole[:, 0]))],
                sole[int(np.argmax(body_sole[:, 0]))],
                sole[int(np.argmin(body_sole[:, 1]))],
                sole[int(np.argmax(body_sole[:, 1]))],
            ],
            dtype=np.float64,
        )


def _swept_envelope(
    scene_points_z: NDArray[np.float64],
    scene_points_x: NDArray[np.float64],
    stations: NDArray[np.float64],
    surface_height_m: float,
) -> NDArray[np.float64]:
    """Accumulate the running lower envelope of a moving point cloud.

    Args:
        scene_points_z: ``(T, K)`` world ``z`` of the sole points.
        scene_points_x: ``(T, K)`` world ``x`` of the same.
        stations: ``(S,)`` along-track stations to resolve on.
        surface_height_m: The free surface to clip at.

    Returns:
        ``(T, S)`` the envelope, non-increasing down the time axis.
    """
    n_frames = scene_points_z.shape[0]
    n_stations = stations.size
    floor = np.full((n_frames, n_stations), surface_height_m, dtype=np.float64)
    # Stations are uniform, so the bin of each point is arithmetic rather
    # than a search: a shot is ~200 samples x ~500 sole points, and
    # digitize over that costs more than the whole rest of the scene.
    width = float(stations[1] - stations[0])
    running = np.full(n_stations, surface_height_m, dtype=np.float64)
    for step in range(n_frames):
        below = scene_points_z[step] < surface_height_m
        if below.any():
            slot = np.clip(
                np.rint((scene_points_x[step][below] - stations[0]) / width).astype(
                    np.int64
                ),
                0,
                n_stations - 1,
            )
            np.minimum.at(running, slot, scene_points_z[step][below])
        floor[step] = running
    return floor


def shot_scene(
    build: HeadBuild,
    result: ShotResult,
    *,
    n_stations: int = DIVOT_STATIONS,
    sand: SandVolume | None = None,
) -> ShotScene | None:
    """Build the 3-D scene of one recorded shot.

    No solving and no re-simulation: the pose is the pose the march
    recorded, the surface is recovered from the same trace, and the divot is
    accumulated from where the head's own sole points went. The scene
    carries both the solver's own element centroids -- so a marker overlay
    stays the head that was integrated over -- and the lofted watertight
    mesh a renderer draws as a solid (issue #8706 defect 1); ``build`` has
    already lofted it, so this never re-lofts.

    Args:
        build: The lofted head.
        result: The shot trace.
        n_stations: Along-track stations to resolve the divot on.
        sand: The solved sand field for this shot, when the tier resolved
            one (issue #8729). It must have come from the same tier as
            ``result``; :class:`ShotScene` refuses the mismatch rather
            than drawing one tier's grains over another's trajectory.

    Returns:
        The scene, or ``None`` when the trace is too short to animate.

    Raises:
        ValueError: If ``n_stations`` is below two, or the trace's poses and
            sole depths disagree about where the free surface is.
    """
    if int(n_stations) < 2:
        raise ValueError(f"a divot section needs at least 2 stations, got {n_stations}")
    if result.n_steps < 2:
        return None

    points_body = np.asarray(build.elements_body.centroids_m, dtype=np.float64)
    sole_index = np.flatnonzero(np.asarray(build.sole_mask, dtype=bool))
    reference_body = np.asarray(build.sole_reference_body_m, dtype=np.float64)
    positions = np.asarray(result.positions_m, dtype=np.float64)
    rotations = np.asarray(result.orientations, dtype=np.float64)

    surface_height = free_surface_height_m(result)

    sole_body = points_body[sole_index]
    sole_world = np.einsum("tij,kj->tki", rotations, sole_body) + positions[:, None, :]
    sole_x, sole_z = sole_world[:, :, 0], sole_world[:, :, 2]

    # Frame the divot on the sole's own travel rather than on the body
    # origin: the envelope is cut by the sole, so anything outside its
    # along-track span is untouched surface by construction.
    stations = np.linspace(
        float(sole_x.min()), float(sole_x.max()), int(n_stations), dtype=np.float64
    )
    floor = _swept_envelope(sole_z, sole_x, stations, surface_height)

    margin = max(float(stations[-1] - stations[0]), 1e-3) * 0.15
    across = sole_world[:, :, 1]
    across_margin = max(float(across.max() - across.min()), 1e-3) * 0.5
    return ShotScene(
        time_s=result.times_s,
        position_m=positions,
        orientation=rotations,
        head_point_body_m=points_body,
        head_mesh_body=build.loft.mesh,
        sole_index=sole_index,
        sole_reference_body_m=reference_body,
        sole_depth_m=result.sole_depths_m,
        surface=SandSurface(
            height_m=surface_height,
            along_extent_m=(
                float(stations[0]) - margin,
                float(stations[-1]) + margin,
            ),
            across_extent_m=_across_extent(across, across_margin, sand),
            # Not a judgement about the tier in the abstract: this shot
            # either came with a solved field or it did not.
            resolves_grains=sand is not None,
            tier=result.fidelity_tier,
        ),
        divot=DivotSection(
            station_m=stations,
            floor_m=floor,
            surface_height_m=surface_height,
            resolves_grains=sand is not None,
            tier=result.fidelity_tier,
        ),
        verdict=result.verdict,
        fidelity_tier=result.fidelity_tier,
        sand=sand,
    )


def _across_extent(
    across: NDArray[np.float64], margin: float, sand: SandVolume | None
) -> tuple[float, float]:
    """The world ``y`` span the surface is drawn over.

    Widened to hold the extruded sheets when there are any: a surface
    plane narrower than the sand under it would crop the volume at its
    own edge and make the extrusion look like a solved, bounded slab.
    """
    low = float(across.min()) - margin
    high = float(across.max()) + margin
    if sand is None:
        return (low, high)
    return (
        min(low, float(sand.across_m.min())),
        max(high, float(sand.across_m.max())),
    )


def viewport_payload(
    scene: ShotScene, *, wrench: NDArray[np.float64] | None = None
) -> ViewportOverlayPayload:
    """Express a scene in the ADR-0027 backend-neutral overlay schema.

    This is the seam the epic asks for: BunkerShot3D renders *through* the
    canonical viewport layer rather than around it. The payload is what a
    MeshCat, Rerun or VTK provider would consume when one is installed, and
    it is the same object the matplotlib fallback in
    :mod:`~.render3d` draws when none is -- so the fallback cannot drift
    away from what a real 3-D backend would show.

    The validity status, the fidelity tier and the flat statement that no
    grains are resolved travel in ``meta``, so a backend that never imports
    this package can still stamp its frames.

    Args:
        scene: The scene to express.
        wrench: Optional ``(T, 6)`` sand wrench on the head [N, N.m], for a
            backend that draws force arrows.

    Returns:
        The payload.

    Raises:
        ValueError: If the wrench does not have one row per sample; the
            payload validates this itself.
    """
    markers = np.asarray(
        [scene.named_markers_world_m(frame) for frame in range(scene.n_frames)],
        dtype=np.float64,
    )
    return ViewportOverlayPayload(
        time_s=scene.time_s,
        trajectory_xyz=scene.sole_reference_world_m,
        markers_xyz=markers,
        marker_names=_MARKER_NAMES,
        wrench=None if wrench is None else np.asarray(wrench, dtype=np.float64),
        meta={
            "envelope_status": scene.status.value,
            "fidelity_tier": scene.fidelity_tier.value,
            "resolves_grains": scene.surface.resolves_grains,
            "free_surface_height_m": scene.surface.height_m,
            "sand_note": "\n".join(scene.sand_note()),
            "divot_note": scene.divot.describe(),
            **_sand_meta(scene),
        },
    )


def _sand_meta(scene: ShotScene) -> dict[str, str]:
    """What a backend needs to stamp a solved sand field, or nothing.

    Kept out of the payload literal so the F0 case adds no empty keys a
    provider would have to interpret: a missing ``sand_fidelity`` means
    there is no sand, which is the same thing ``resolves_grains`` already
    says, rather than a sand field of unknown standing.
    """
    sand = scene.sand
    if sand is None:
        return {}
    return {
        "sand_fidelity": sand.fidelity.value,
        "sand_tier": sand.fidelity_tier.value,
        "sand_digest": sand.source_digest,
        "sand_kinematics": sand.kinematics,
        "sand_extrusion_note": sand.describe(),
    }
