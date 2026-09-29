"""Extruding a plane-strain sand field into the 3-D scene (issue #8729, epic #8699).

Headless: numpy only, no matplotlib and no Qt, so the same volume feeds the
matplotlib fallback, the VTK/PyVista upgrade and a test.

Nothing here computes new physics
---------------------------------

Every number in a :class:`SandVolume` came out of
:mod:`bunkershot3d.fields`, which read it off the solver's own transfer
operators (issue #8710). This module reshapes the stored flat lattice back
into its two axes, decimates it to something a renderer can draw at
interactive rates, and repeats it across the declared effective width. It
forms no quantity the solve did not already have.

The one thing this module must not let anybody believe
------------------------------------------------------

**F1 solves a plane-strain section, so a volume built from it is an
extrusion, not a solved volume.** The section is swept across
``effective_width_m`` -- a *declared* width, itself an assumption rather
than a result -- and every sheet is bit-identical because plane strain has
no heel-to-toe direction to vary along. That is not a rendering shortcut
this module took; it is the model's own content. A 3-D picture of a 2-D
solve is the single most misleading thing this epic could ship, so the
volume carries :class:`~.slices.SliceFidelity.EXTRUDED` as data,
:meth:`SandVolume.describe` says the word in prose, and the renderers draw
the sheets as discrete, separated sheets rather than blending them into a
continuum that would look solved.

That vocabulary is deliberately the one issue #8711 already established for
2-D cuts. A viewer who has learned what "extruded from the solved plane"
means on a cross-section reads the same words here and does not have to
learn a second scheme.

Velocity is a direction, not a blob
-----------------------------------

Sand pushed ahead of the sole and sand riding up the face reach the same
speed. A volume that stored only a magnitude could not tell them apart, so
the in-plane components are kept and :meth:`SandVolume.arrows` hands a
renderer a coarse lattice of them to draw. The magnitude is derived from
the components rather than stored beside them, so the two can never drift.

Nothing auto-scales
-------------------

Issue #8728 fixed a real defect in the sole load field: two grinds each
normalised to their own peak looked identical however far apart they were.
:class:`SandVolumeScale` is fixed over every frame of every compared field
and injected, exactly as :class:`~.field.LoadScale` and
:class:`~.slices.SliceScale` are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.fields.schema import (
    FieldLayout,
    FieldQuantity,
    SandFieldSeries,
    series_digest,
)
from bunkershot3d.solvers.envelope import EnvelopeStatus
from bunkershot3d.solvers.protocol import FidelityTier

from .slices import DENSITY_COLORMAP, SPEED_COLORMAP, SliceFidelity

__all__ = [
    "DEFAULT_MAX_CELLS",
    "DEFAULT_SHEETS",
    "ArrowLattice",
    "SandVolume",
    "SandVolumeScale",
    "sand_volume",
    "sand_volume_scale",
]

DEFAULT_SHEETS = 5
"""Sheets drawn across the effective width.

Enough that the repetition is visible -- which is the honest reading of a
plane-strain extrusion -- and few enough that a matplotlib 3-D panel still
redraws inside the workbench's frame budget."""

DEFAULT_MAX_CELLS = 2400
"""Lattice cells kept per sheet, per frame.

A captured field is a few hundred by a few hundred nodes. Drawing every one
of them as a quad in a matplotlib 3-D panel costs seconds per frame, so the
lattice is strided down to this budget. Striding rather than averaging: an
averaged cell is a number the solve never held, and this module forms no
new quantities."""

_MIN_SHEETS = 2
_DIMENSION = 2
_MIN_LATTICE = 2
EDGE_ON_COSINE = 0.5
"""How square to the solved plane an eye must be to see the section.

The sheets span the swing plane, so an eye sighting *along* the target
line looks straight down them and sees a row of lines rather than a row
of sections. That is the extrusion seen end-on and is worth saying,
because a viewer who does not know it is looking at an edge-on plane
reads the stripes as across-width structure -- the one thing a
plane-strain solve has none of."""

_ARROWS_ALONG = 16
_ARROWS_UP = 10
_EPSILON = 1e-12


def _check_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    """Return a finite, increasing range or refuse it."""
    low, high = float(bounds[0]), float(bounds[1])
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"{name} must be finite, got {bounds!r}")
    if not low < high:
        raise ValueError(f"{name} must increase, got {low} to {high}")
    return (low, high)


@dataclass(frozen=True)
class ArrowLattice:
    """A coarse lattice of in-plane flow directions, ready for a quiver.

    Attributes:
        along_m: ``(na,)`` world ``x`` of each arrow column [m].
        up_m: ``(nu,)`` world ``z`` of each arrow row [m].
        velocity_along_m_s: ``(na, nu)`` along-track component [m/s].
        velocity_up_m_s: ``(na, nu)`` vertical component [m/s].
        occupied: ``(na, nu)`` whether each arrow sits in sand.
    """

    along_m: NDArray[np.float64]
    up_m: NDArray[np.float64]
    velocity_along_m_s: NDArray[np.float64]
    velocity_up_m_s: NDArray[np.float64]
    occupied: NDArray[np.bool_]

    @property
    def speed_m_s(self) -> NDArray[np.float64]:
        """``(na, nu)`` arrow lengths, zero where there is no sand."""
        return np.hypot(self.velocity_along_m_s, self.velocity_up_m_s)


@dataclass(frozen=True)
class SandVolume:
    """A plane-strain sand field swept across the declared effective width.

    Attributes:
        time_s: ``(T,)`` frame times [s].
        along_m: ``(nx,)`` world ``x`` of each lattice column [m].
        up_m: ``(nz,)`` world ``z`` of each lattice row [m].
        across_m: ``(ny,)`` world ``y`` of each drawn sheet [m]. The
            section is identical on all of them; see the module docstring.
        velocity_along_m_s: ``(T, nx, nz)`` along-track flow [m/s].
        velocity_up_m_s: ``(T, nx, nz)`` vertical flow [m/s].
        density_kg_m3: ``(T, nx, nz)`` bulk density [kg/m^3].
        occupied: ``(T, nx, nz)`` whether a cell holds sand at all.
        body_outline_m: ``(T, V, 2)`` intruder section per frame [m], or
            ``None``. A flow field with no body in it cannot answer the
            question it was computed for.
        fidelity: What a 3-D reading of this is. Always
            :attr:`~.slices.SliceFidelity.EXTRUDED` for a plane-strain
            tier, because that is what it is.
        fidelity_tier: Which rung of the ADR-0032 ladder resolved the sand.
        envelope_status: The validity this field must be read under.
        source_digest: The SHA-256 of the series this came from, so a drawn
            volume is traceable to the arrays behind it.
        kinematics: How the body's motion was supplied, in words.
        speed_headline: Where this shot sits against the published corpus.
        effective_width_m: The declared width the section is swept across.
        decimation_note: What striding the lattice threw away.
    """

    time_s: NDArray[np.float64]
    along_m: NDArray[np.float64]
    up_m: NDArray[np.float64]
    across_m: NDArray[np.float64]
    velocity_along_m_s: NDArray[np.float64]
    velocity_up_m_s: NDArray[np.float64]
    density_kg_m3: NDArray[np.float64]
    occupied: NDArray[np.bool_]
    body_outline_m: NDArray[np.float64] | None
    fidelity: SliceFidelity
    fidelity_tier: FidelityTier
    envelope_status: EnvelopeStatus
    source_digest: str
    kinematics: str
    speed_headline: str
    effective_width_m: float
    decimation_note: str

    def __post_init__(self) -> None:
        """Validate the volume.

        Raises:
            ValueError: If the lattice, the sheets and the frames do not
                describe one field. ``raise`` rather than ``assert``:
                ``python -O`` strips asserts, and a volume that failed
                these would be *drawn* rather than rejected.
        """
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        if times.size < 1:
            raise ValueError("a sand volume needs at least one frame")
        along = np.asarray(self.along_m, dtype=np.float64).reshape(-1)
        up = np.asarray(self.up_m, dtype=np.float64).reshape(-1)
        across = np.asarray(self.across_m, dtype=np.float64).reshape(-1)
        if across.size < _MIN_SHEETS:
            raise ValueError(
                f"a sand volume needs at least {_MIN_SHEETS} sheets to read as a "
                f"volume, got {across.size}; one sheet is a cross-section, and "
                "the 2-D slice view already draws that honestly"
            )
        expected = (times.size, along.size, up.size)
        for name in (
            "velocity_along_m_s",
            "velocity_up_m_s",
            "density_kg_m3",
            "occupied",
        ):
            array = np.asarray(getattr(self, name))
            if array.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {array.shape}"
                )
        if self.fidelity is SliceFidelity.SOLVED:
            raise ValueError(
                "a volume built from a plane-strain section is an extrusion, "
                "never a solved volume; labelling one SOLVED is the single "
                "most misleading thing this view could claim"
            )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "along_m", along)
        object.__setattr__(self, "up_m", up)
        object.__setattr__(self, "across_m", across)
        object.__setattr__(
            self,
            "velocity_along_m_s",
            np.asarray(self.velocity_along_m_s, dtype=np.float64),
        )
        object.__setattr__(
            self, "velocity_up_m_s", np.asarray(self.velocity_up_m_s, dtype=np.float64)
        )
        object.__setattr__(
            self, "density_kg_m3", np.asarray(self.density_kg_m3, dtype=np.float64)
        )
        object.__setattr__(self, "occupied", np.asarray(self.occupied, dtype=np.bool_))

    # ---------------------------------------------------------------- extent

    @property
    def n_frames(self) -> int:
        """Frames in the record."""
        return int(self.time_s.size)

    @property
    def n_along(self) -> int:
        """Lattice columns along the track."""
        return int(self.along_m.size)

    @property
    def n_up(self) -> int:
        """Lattice rows up the section."""
        return int(self.up_m.size)

    @property
    def n_sheets(self) -> int:
        """Sheets drawn across the width."""
        return int(self.across_m.size)

    @property
    def speed_m_s(self) -> NDArray[np.float64]:
        """``(T, nx, nz)`` speed, ``nan`` where there is no sand.

        Derived from the components rather than stored beside them, so a
        magnitude and its direction can never drift apart. ``nan`` rather
        than zero where the lattice is empty: zero would assert that the
        sand there is still, and there is no sand there at all.
        """
        magnitude = np.hypot(self.velocity_along_m_s, self.velocity_up_m_s)
        return np.where(self.occupied, magnitude, np.nan)

    @property
    def masked_density_kg_m3(self) -> NDArray[np.float64]:
        """``(T, nx, nz)`` density, ``nan`` where there is no sand."""
        return np.where(self.occupied, self.density_kg_m3, np.nan)

    @property
    def peak_speed_m_s(self) -> float:
        """The fastest sand anywhere in the record."""
        speed = self.speed_m_s
        if not np.any(np.isfinite(speed)):
            return 0.0
        return float(np.nanmax(speed))

    def _check_frame(self, frame: int) -> int:
        """Validate a frame index against the record.

        Args:
            frame: The requested frame.

        Returns:
            The index.

        Raises:
            ValueError: If it is outside the record.
        """
        if not 0 <= int(frame) < self.n_frames:
            raise ValueError(
                f"frame {frame} is outside the sand field, which has "
                f"{self.n_frames} frames"
            )
        return int(frame)

    # ----------------------------------------------------------- the volume

    def sheet_speed_m_s(self, frame: int) -> NDArray[np.float64]:
        """``(ny, nx, nz)`` speed on every sheet at one frame.

        Every sheet is the same section. Materialising the repetition is
        what lets a renderer hand one array to a volume mapper, and the
        identity across the first axis is asserted in the tests rather
        than left implicit.

        Args:
            frame: The frame.

        Returns:
            The repeated section.

        Raises:
            ValueError: If the frame is outside the record.
        """
        index = self._check_frame(frame)
        return np.repeat(self.speed_m_s[index][None, ...], self.n_sheets, axis=0)

    def sheet_density_kg_m3(self, frame: int) -> NDArray[np.float64]:
        """``(ny, nx, nz)`` density on every sheet at one frame.

        Args:
            frame: The frame.

        Returns:
            The repeated section.

        Raises:
            ValueError: If the frame is outside the record.
        """
        index = self._check_frame(frame)
        return np.repeat(
            self.masked_density_kg_m3[index][None, ...], self.n_sheets, axis=0
        )

    def channel(self, quantity: FieldQuantity, frame: int) -> NDArray[np.float64]:
        """``(nx, nz)`` of one drawable quantity at one frame.

        Args:
            quantity: Velocity magnitude or density.
            frame: The frame.

        Returns:
            The section, ``nan`` where there is no sand.

        Raises:
            ValueError: If the quantity is not one this view paints, or
                the frame is outside the record.
        """
        index = self._check_frame(frame)
        if quantity is FieldQuantity.VELOCITY:
            return self.speed_m_s[index]
        if quantity is FieldQuantity.DENSITY:
            return self.masked_density_kg_m3[index]
        raise ValueError(
            f"{quantity.value} is not a volume channel; the shear rate is a "
            "cross-section quantity and the 2-D slice view paints it"
        )

    def arrows(
        self, frame: int, *, n_along: int = _ARROWS_ALONG, n_up: int = _ARROWS_UP
    ) -> ArrowLattice:
        """A coarse lattice of flow directions at one frame.

        Coarser than the colour lattice on purpose: an arrow per painted
        cell is a grey smear that reads as texture rather than direction.

        Unoccupied cells carry a zero vector rather than ``nan``, because a
        single ``nan`` makes a matplotlib quiver refuse to update at all --
        the same accommodation :mod:`.render_slice` already makes. The
        ``occupied`` mask says which zeros mean "no sand" rather than
        "still sand", so nothing is lost by it.

        Args:
            frame: The frame.
            n_along: Arrow columns wanted.
            n_up: Arrow rows wanted.

        Returns:
            The lattice.

        Raises:
            ValueError: If the frame is outside the record.
        """
        index = self._check_frame(frame)
        columns = _stride_index(self.n_along, n_along)
        rows = _stride_index(self.n_up, n_up)
        picked = np.ix_(columns, rows)
        occupied = self.occupied[index][picked]
        along = np.where(occupied, self.velocity_along_m_s[index][picked], 0.0)
        up = np.where(occupied, self.velocity_up_m_s[index][picked], 0.0)
        return ArrowLattice(
            along_m=self.along_m[columns],
            up_m=self.up_m[rows],
            velocity_along_m_s=along,
            velocity_up_m_s=up,
            occupied=occupied,
        )

    def outline_world_m(self, frame: int) -> NDArray[np.float64] | None:
        """``(V, 2)`` intruder section at one frame, in ``(x, z)`` [m].

        Args:
            frame: The frame.

        Returns:
            The outline, or ``None`` when the field carried none.

        Raises:
            ValueError: If the frame is outside the record.
        """
        index = self._check_frame(frame)
        if self.body_outline_m is None:
            return None
        return np.asarray(self.body_outline_m[index], dtype=np.float64)

    # ---------------------------------------------------------------- words

    def viewing_note(self, eye_direction: NDArray[np.float64]) -> str:
        """How one camera is cutting this extrusion, in words.

        Shared by both backends rather than composed in each: a fallback
        and an upgrade that qualified the same picture differently would
        be worse than either qualifying it alone.

        Args:
            eye_direction: Unit vector from the subject toward the eye,
                world axes. :attr:`~.shot3d.CameraPreset.eye_direction`
                supplies it backend-neutrally, so no matplotlib or VTK
                camera maths is needed here.

        Returns:
            One line for the caption.
        """
        squareness = abs(float(np.asarray(eye_direction, dtype=np.float64)[1]))
        if squareness >= EDGE_ON_COSINE:
            return (
                "view: square to the solved plane, so this is the section "
                "itself; the sheets behind it are copies of it"
            )
        return (
            "view: sighting along the solved plane, so the sheets are "
            "edge-on -- the stripes are one section seen end-on, repeated "
            f"across {self.n_sheets} sheets, not across-width structure"
        )

    def describe(self) -> str:
        """The sentence drawn in the frame beside the volume.

        Returns:
            What the drawn sand is, and -- more to the point -- what it is
            not. A viewer looking at a box of moving material will read
            heel-to-toe structure into it unless told there is none.
        """
        return (
            f"sand: {self.fidelity_tier.value} resolves grains, and this is the "
            f"solved plane-strain section {self.fidelity.label}, swept "
            f"{self.effective_width_m * 1e3:.1f} mm across a declared width. "
            "Every sheet is the same solve repeated: plane strain has no "
            "heel-to-toe flow, so any across-width structure you see is the "
            "extrusion, not the sand"
        )

    def summary(self) -> str:
        """One line of what was kept, for the stamp under the verdict.

        Returns:
            Frame count, lattice size and what decimation dropped.
        """
        return (
            f"{self.n_frames} field frames, {self.n_along} x {self.n_up} lattice "
            f"on {self.n_sheets} sheets; {self.decimation_note}; {self.kinematics}"
        )


@dataclass(frozen=True)
class SandVolumeScale:
    """Fixed colour limits, shared across frames and across designs.

    Attributes:
        speed_m_s: ``(low, high)`` the speed ramp covers [m/s].
        density_kg_m3: ``(low, high)`` the density ramp covers [kg/m^3].
    """

    speed_m_s: tuple[float, float]
    density_kg_m3: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate the scale.

        Raises:
            ValueError: If either ramp is not finite and increasing. A
                degenerate ramp paints every cell the same colour, which
                looks like uniform sand rather than like a broken scale.
        """
        for name in ("speed_m_s", "density_kg_m3"):
            object.__setattr__(self, name, _check_range(name, getattr(self, name)))

    def limits(self, quantity: FieldQuantity) -> tuple[float, float]:
        """The ramp range for one channel.

        Args:
            quantity: Velocity magnitude or density.

        Returns:
            ``(low, high)``.

        Raises:
            ValueError: If the quantity is not one this view paints.
        """
        if quantity is FieldQuantity.VELOCITY:
            return self.speed_m_s
        if quantity is FieldQuantity.DENSITY:
            return self.density_kg_m3
        raise ValueError(
            f"{quantity.value} is not a volume channel; the shear rate is a "
            "cross-section quantity and the 2-D slice view paints it"
        )

    def colormap_name(self, quantity: FieldQuantity) -> str:
        """The ramp one channel is painted on.

        The same two ramps :mod:`.slices` already established for 2-D
        cuts, so a speed reads the same colour in the cross-section and in
        the volume.

        Args:
            quantity: Velocity magnitude or density.

        Returns:
            A matplotlib colormap name.

        Raises:
            ValueError: If the quantity is not one this view paints.
        """
        if quantity is FieldQuantity.VELOCITY:
            return SPEED_COLORMAP
        if quantity is FieldQuantity.DENSITY:
            return DENSITY_COLORMAP
        raise ValueError(
            f"{quantity.value} is not a volume channel; the shear rate is a "
            "cross-section quantity and the 2-D slice view paints it"
        )

    def unit(self, quantity: FieldQuantity) -> str:
        """The unit one channel is quoted in.

        Args:
            quantity: Velocity magnitude or density.

        Returns:
            The unit string.

        Raises:
            ValueError: If the quantity is not one this view paints.
        """
        self.limits(quantity)
        return quantity.unit

    def normalise(
        self, quantity: FieldQuantity, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map values onto ``[0, 1]`` against the fixed ramp.

        ``nan`` survives as ``nan`` so an empty cell stays transparent
        rather than clamping to the ramp's floor, which would paint air
        as still sand.

        Args:
            quantity: Velocity magnitude or density.
            values: Any shape.

        Returns:
            The same shape, in ``[0, 1]``.

        Raises:
            ValueError: If the quantity is not one this view paints.
        """
        low, high = self.limits(quantity)
        span = high - low
        with np.errstate(invalid="ignore"):
            scaled = (np.asarray(values, dtype=np.float64) - low) / max(span, _EPSILON)
            return np.clip(scaled, 0.0, 1.0)

    def merged(self, other: SandVolumeScale) -> SandVolumeScale:
        """Return the scale covering both this one and ``other``.

        Args:
            other: The scale to merge with.

        Returns:
            The covering scale, which is what makes two designs directly
            comparable rather than each normalised to its own peak.
        """
        return SandVolumeScale(
            speed_m_s=(
                min(self.speed_m_s[0], other.speed_m_s[0]),
                max(self.speed_m_s[1], other.speed_m_s[1]),
            ),
            density_kg_m3=(
                min(self.density_kg_m3[0], other.density_kg_m3[0]),
                max(self.density_kg_m3[1], other.density_kg_m3[1]),
            ),
        )


def sand_volume_scale(volumes: tuple[SandVolume, ...]) -> SandVolumeScale:
    """Build the one colour scale two or more designs are painted on.

    Args:
        volumes: Every volume that will be drawn on this scale. Passing
            both halves of an A/B comparison is what makes the two views
            readable against each other; passing one gives a ramp fixed
            across its own frames.

    Returns:
        The covering scale.

    Raises:
        ValueError: If no volume was supplied. Issue #8728: two designs
            each normalised to their own peak look identical however far
            apart they are, and an empty comparison silently scaled to
            nothing is the same defect one step earlier.
    """
    if not volumes:
        raise ValueError(
            "a shared sand scale needs at least one volume to cover; painting "
            "two designs each normalised to its own peak is what this prevents"
        )
    merged = _scale_for(volumes[0])
    for volume in volumes[1:]:
        merged = merged.merged(_scale_for(volume))
    return merged


def _scale_for(volume: SandVolume) -> SandVolumeScale:
    """The ramps one volume needs over its whole record."""
    speed = volume.speed_m_s
    density = volume.masked_density_kg_m3
    peak_speed = float(np.nanmax(speed)) if np.any(np.isfinite(speed)) else 0.0
    peak_density = float(np.nanmax(density)) if np.any(np.isfinite(density)) else 0.0
    return SandVolumeScale(
        # Both ramps start at zero so still sand reads as still and empty
        # bulk reads as empty, rather than each frame's own floor becoming
        # the colour of "nothing happening".
        speed_m_s=(0.0, max(peak_speed, 1e-3)),
        density_kg_m3=(0.0, max(peak_density, 1.0)),
    )


def _occupied_lines(live: NDArray[np.bool_], *, axis: int) -> NDArray[np.intp]:
    """Lattice indices on one axis that hold sand in at least one frame.

    Returned as a contiguous span from the first to the last such line
    rather than as the sparse set of them, because a decimated lattice
    has to stay uniform: a ragged one would force every renderer into
    scattered interpolation for no saving worth having.
    """
    other = 1 if axis == 2 else 2
    anywhere = np.any(live, axis=(0, other))
    found = np.flatnonzero(anywhere)
    if found.size < _MIN_LATTICE:
        # A field with no sand anywhere is a real answer, not a crash, and
        # the whole lattice is as good a frame for nothing as any other.
        return np.arange(live.shape[axis], dtype=np.intp)
    return np.arange(int(found[0]), int(found[-1]) + 1, dtype=np.intp)


def _stride_index(count: int, wanted: int) -> NDArray[np.intp]:
    """Evenly strided indices into an axis, never more than it holds."""
    if wanted < 1:
        raise ValueError(f"a lattice axis needs at least one sample, got {wanted}")
    if count <= wanted:
        return np.arange(count, dtype=np.intp)
    stride = int(math.ceil(count / wanted))
    return np.arange(0, count, stride, dtype=np.intp)


def _lattice_strides(shape: tuple[int, ...], max_cells: int) -> tuple[int, int]:
    """Strides that bring a lattice inside the cell budget.

    Strided rather than averaged: an averaged cell holds a number the
    solve never had, and nothing in this module forms a new quantity.
    """
    count_x, count_z = int(shape[0]), int(shape[1])
    if max_cells < 1:
        raise ValueError(f"max_cells must be at least 1, got {max_cells}")
    stride_x = stride_z = 1
    while math.ceil(count_x / stride_x) * math.ceil(count_z / stride_z) > max_cells:
        if math.ceil(count_x / stride_x) >= math.ceil(count_z / stride_z):
            stride_x += 1
        else:
            stride_z += 1
    return stride_x, stride_z


def _require_grains(series: SandFieldSeries) -> None:
    """Refuse a field from a tier that resolves no grains."""
    tier = series.provenance.fidelity_tier
    if tier is FidelityTier.F0:
        raise ValueError(
            "F0 resolves no grains, so there is no sand field to make a volume "
            "from; its flat plane is a boundary condition and the 3-D scene "
            "already says so. Drawing an F1 field over an F0 shot -- or the "
            "reverse -- is the one substitution this view must never make"
        )
    if series.layout is not FieldLayout.GRID:
        raise ValueError(
            f"a sand volume is built from a lattice, got a {series.layout.value} "
            "field; a particle field would need scattering onto a grid first, "
            "which would form numbers the solve never held"
        )
    if series.geometry is None or series.geometry.dimension != _DIMENSION:
        raise ValueError(
            "a sand volume extrudes a plane-strain section, so it needs a 2-D "
            f"lattice geometry; got {series.geometry!r}"
        )


def sand_volume(
    series: SandFieldSeries,
    *,
    n_sheets: int = DEFAULT_SHEETS,
    max_cells: int = DEFAULT_MAX_CELLS,
) -> SandVolume:
    """Extrude a stored plane-strain sand field into a drawable volume.

    Args:
        series: The captured field, from :mod:`bunkershot3d.fields`.
        n_sheets: Sheets drawn across the declared effective width. At
            least two, because one sheet is a cross-section rather than a
            volume and :mod:`.render_slice` already draws that honestly.
        max_cells: Lattice cells kept per sheet per frame. Lattice lines
            empty for the whole record are dropped first -- an F1 bed's
            run-in and ejecta headroom are mostly air -- and what remains
            is strided down to fit.

    Returns:
        The volume, labelled ``EXTRUDED`` and carrying its own tier,
        validity and source digest.

    Raises:
        ValueError: If the field resolves no grains, is not a 2-D
            lattice, or too few sheets were asked for.
    """
    _require_grains(series)
    geometry = series.geometry
    if geometry is None:  # pragma: no cover - guarded by _require_grains
        raise ValueError("a sand volume needs a lattice geometry")
    if n_sheets < _MIN_SHEETS:
        raise ValueError(
            f"a sand volume needs at least {_MIN_SHEETS} sheets to read as a "
            f"volume, got {n_sheets}; one sheet is a cross-section, and the "
            "2-D slice view already draws that honestly"
        )
    width = float(series.provenance.setting("effective_width_m", 0.0))
    if not (math.isfinite(width) and width > 0.0):
        raise ValueError(
            "this field declares no effective width, so there is nothing to "
            "extrude the solved section across; the width is a stated "
            "assumption of the plane-strain tier, not a result"
        )

    count_x, count_z = int(geometry.shape[0]), int(geometry.shape[1])
    frames = series.n_frames
    velocity = series.velocity_m_s.reshape(frames, count_x, count_z, _DIMENSION)
    density = series.density_kg_m3.reshape(frames, count_x, count_z)
    live = series.occupied().reshape(frames, count_x, count_z)

    # An F1 bed carries a long run-in and a tall ejecta headroom that hold
    # no sand in any frame. Framing the scene around them shrinks the
    # impact zone to a smudge, so lattice lines that are empty for the
    # whole record are dropped first. This hides nothing: a line with no
    # sand in it at any time carries no information at all, and the note
    # below reports exactly how many went.
    keep_x = _occupied_lines(live, axis=1)
    keep_z = _occupied_lines(live, axis=2)
    stride_x, stride_z = _lattice_strides((keep_x.size, keep_z.size), max_cells)
    columns = keep_x[::stride_x]
    rows = keep_z[::stride_z]

    picked = np.ix_(np.arange(frames, dtype=np.intp), columns, rows)
    occupied = live[picked]

    outline = (
        None if series.body_outline_m is None else np.asarray(series.body_outline_m)
    )
    kept = columns.size * rows.size
    note = (
        f"kept {kept} of {count_x * count_z} lattice cells "
        f"(every {stride_x} along, every {stride_z} up, over the "
        f"{keep_x.size} x {keep_z.size} lines that hold sand at some point)"
    )
    return SandVolume(
        time_s=series.time_s,
        along_m=geometry.axis_coordinates_m(0)[columns],
        up_m=geometry.axis_coordinates_m(1)[rows],
        across_m=np.linspace(-0.5 * width, 0.5 * width, n_sheets, dtype=np.float64),
        velocity_along_m_s=velocity[..., 0][picked],
        velocity_up_m_s=velocity[..., 1][picked],
        density_kg_m3=density[picked],
        occupied=occupied,
        body_outline_m=outline,
        # Not a choice this function makes: a plane-strain solve extruded
        # across a declared width *is* an extrusion, whatever it is drawn
        # with.
        fidelity=SliceFidelity.EXTRUDED,
        fidelity_tier=series.provenance.fidelity_tier,
        envelope_status=series.provenance.envelope_status,
        source_digest=series_digest(series),
        kinematics=series.provenance.kinematics,
        speed_headline=series.provenance.speed_headline(),
        effective_width_m=width,
        decimation_note=note,
    )
