"""The tier-neutral sand-field schema (issue #8710, epic #8699).

What a stored field has to survive
----------------------------------

A sand field is the only product in this package that a reader can look
at and *believe* without reading a number.  A velocity picture of the
impact zone is persuasive in a way a wrench table is not, so the file it
comes out of has to make three claims un-loseable:

1. **Which tier produced it.**  ADR-0033 chose F1, a 2-D plane-strain
   continuum.  A future grain tier would produce the same *shape* of
   data and a completely different standing of it.
2. **How valid it is.**  Every F1 verdict is
   :attr:`~bunkershot3d.solvers.envelope.EnvelopeStatus.BEYOND_VALIDATION`
   at best, because issue #8616 found no published measurement of any
   quantity F1 produces.
3. **What was thrown away to fit it on disk.**  A field is per-frame
   arrays over a whole grid; a 1 mm run is gigabytes.  Something is
   always dropped, and the honest move is to record what.

Tier and status are **data**, not filename
------------------------------------------

:class:`FieldProvenance` travels inside the file, and
:func:`series_digest` covers the provenance *and* the arrays with one
SHA-256.  Renaming ``illustrative.h5`` to ``predictive.h5`` changes
nothing a reader consults; editing the stored tier attribute breaks the
digest and :mod:`bunkershot3d.fields.store` refuses the file.  That is
the difference between a convention and a guarantee.

Representing more than one tier
-------------------------------

:class:`FieldLayout` is the seam.  A continuum tier writes ``GRID``: the
sample points are implied by an origin, a spacing and a shape, so they
cost nothing per frame.  A grain tier writes ``PARTICLE``: the sample
points move, so they are stored per frame.  Both carry the same
quantities in the same units, so a view written against this schema does
not care which tier it is looking at -- which is the point, because
switching tiers must not invalidate stored results.

Dimension is likewise stored rather than assumed: F1 is 2-D in the swing
plane, and a 3-D tier would write ``D = 3`` into the same containers.

What is *not* here
------------------

Everything a field **claims** -- its tier, its validity status, where it
says there is sand, and what was dropped to store it -- lives in
:mod:`bunkershot3d.fields.standing`.  A field's contents and a field's
standing change for different reasons, so they are separate modules.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BunkerShot3DValueError
from ..provenance.hashing import canonical_json
from .standing import (
    FIELD_SCHEMA_VERSION,
    FieldProvenance,
    OccupancyRule,
    RetentionRecord,
)
from .units import DENSITY_UNIT, SHEAR_RATE_UNIT, TIME_UNIT, VELOCITY_UNIT

__all__ = [
    "FieldLayout",
    "FieldQuantity",
    "GridGeometry",
    "SandFieldFrame",
    "SandFieldSeries",
    "series_digest",
]

_MIN_DIMENSION = 2
_MAX_DIMENSION = 3


class FieldLayout(StrEnum):
    """Where a tier's samples live, and therefore how they are stored."""

    GRID = "grid"
    """Fixed sample points on a uniform lattice.

    The positions are implied by :class:`GridGeometry` and are not stored
    per frame, which is most of the saving on a continuum run."""

    PARTICLE = "particle"
    """Sample points that move with the material.

    Positions are stored per frame because they are the state.  A grain
    tier writes this; so would a particle-space dump of a continuum."""


class FieldQuantity(StrEnum):
    """What a field carries, named so a view can ask for it by name."""

    VELOCITY = "velocity"
    """Vector, :data:`VELOCITY_UNIT`. Always present."""

    DENSITY = "density"
    """Scalar, :data:`DENSITY_UNIT`. Always present."""

    SHEAR_RATE = "shear_rate"
    """Scalar, :data:`SHEAR_RATE_UNIT`. Optional.

    The second invariant of the strain-rate tensor,
    ``sqrt(2 D : D)`` with ``D = sym(grad v)``.  Optional because a tier
    that only reports positions and velocities -- a grain tier reading
    back a dump, say -- has no velocity gradient to form it from, and
    fabricating one by differencing scattered points would invent a
    number rather than measure one."""

    @property
    def unit(self) -> str:
        """SI unit string for this quantity."""
        return _QUANTITY_UNITS[self]

    @property
    def label(self) -> str:
        """Axis/colour-bar label including the unit."""
        return f"{_QUANTITY_LABELS[self]} [{self.unit}]"


_QUANTITY_UNITS: Mapping[FieldQuantity, str] = MappingProxyType(
    {
        FieldQuantity.VELOCITY: VELOCITY_UNIT,
        FieldQuantity.DENSITY: DENSITY_UNIT,
        FieldQuantity.SHEAR_RATE: SHEAR_RATE_UNIT,
    }
)

_QUANTITY_LABELS: Mapping[FieldQuantity, str] = MappingProxyType(
    {
        FieldQuantity.VELOCITY: "sand speed",
        FieldQuantity.DENSITY: "sand density",
        FieldQuantity.SHEAR_RATE: "shear rate",
    }
)


@dataclass(frozen=True)
class GridGeometry:
    """A uniform lattice of sample points, stored instead of the points.

    Attributes:
        origin_m: ``(d,)`` position of sample ``(0, ..., 0)``.
        cell_size_m: Uniform spacing on every axis.
        shape: ``(d,)`` sample counts per axis, in the C order the flat
            sample axis is raveled with.
        axis_names: One name per axis, so a view can label an axis
            without knowing which tier wrote it. F1 writes
            ``("x", "z")`` -- along-path and up -- because plane strain
            has no ``y``.
    """

    origin_m: NDArray[np.float64]
    cell_size_m: float
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_m, dtype=np.float64).reshape(-1)
        shape = tuple(int(value) for value in self.shape)
        names = tuple(str(name) for name in self.axis_names)
        if not _MIN_DIMENSION <= len(shape) <= _MAX_DIMENSION:
            raise BunkerShot3DValueError(
                f"a field grid must be {_MIN_DIMENSION}-D or {_MAX_DIMENSION}-D, "
                f"got shape {self.shape!r}"
            )
        if origin.shape != (len(shape),):
            raise BunkerShot3DValueError(
                f"origin_m must have one entry per axis ({len(shape)}), got "
                f"shape {origin.shape}"
            )
        if len(names) != len(shape):
            raise BunkerShot3DValueError(
                f"axis_names must have one entry per axis ({len(shape)}), got {names!r}"
            )
        if any(count < 1 for count in shape):
            raise BunkerShot3DValueError(f"every axis needs a sample, got {shape!r}")
        size = float(self.cell_size_m)
        if not math.isfinite(size) or size <= 0.0:
            raise BunkerShot3DValueError(
                f"cell_size_m must be positive, got {self.cell_size_m!r}"
            )
        if not bool(np.all(np.isfinite(origin))):
            raise BunkerShot3DValueError(f"origin_m must be finite, got {origin!r}")
        origin.flags.writeable = False
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "cell_size_m", size)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axis_names", names)

    @property
    def dimension(self) -> int:
        """Number of spatial axes."""
        return len(self.shape)

    @property
    def n_samples(self) -> int:
        """Total sample count, the length of the flat sample axis."""
        return int(np.prod(self.shape))

    def axis_coordinates_m(self, axis: int) -> NDArray[np.float64]:
        """``(shape[axis],)`` sample coordinates along one axis.

        Args:
            axis: Axis index.

        Returns:
            The coordinates in metres.

        Raises:
            BunkerShot3DValueError: If ``axis`` is out of range.
        """
        if not 0 <= int(axis) < self.dimension:
            raise BunkerShot3DValueError(
                f"axis {axis!r} is outside a {self.dimension}-D grid"
            )
        index = int(axis)
        return self.origin_m[index] + np.arange(self.shape[index]) * self.cell_size_m

    def bounds_m(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(lower, upper)`` corners of the sampled region."""
        span = (np.asarray(self.shape, dtype=np.float64) - 1.0) * self.cell_size_m
        return self.origin_m, self.origin_m + span

    def sample_positions_m(self) -> NDArray[np.float64]:
        """``(n_samples, d)`` positions in flat-sample order.

        Materialised on request rather than stored, which is the whole
        reason ``GRID`` is cheaper than ``PARTICLE`` on disk.
        """
        axes = [self.axis_coordinates_m(axis) for axis in range(self.dimension)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([component.ravel() for component in mesh], axis=1)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping, for the digest and the sidecar."""
        return {
            "origin_m": [float(value) for value in self.origin_m],
            "cell_size_m": float(self.cell_size_m),
            "shape": [int(value) for value in self.shape],
            "axis_names": list(self.axis_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GridGeometry:
        """Rebuild from :meth:`to_dict`."""
        return cls(
            origin_m=np.asarray(payload["origin_m"], dtype=np.float64),
            cell_size_m=float(payload["cell_size_m"]),
            shape=tuple(int(value) for value in payload["shape"]),
            axis_names=tuple(str(name) for name in payload["axis_names"]),
        )


@dataclass(frozen=True, slots=True)
class SandFieldFrame:
    """One stored instant, as a view onto its series.

    Attributes:
        index: Frame index within the series.
        time_s: Simulation time of the frame.
        positions_m: ``(n, d)`` sample positions.
        velocity_m_s: ``(n, d)`` sample velocities.
        density_kg_m3: ``(n,)`` sample densities.
        shear_rate_1_s: ``(n,)`` shear rates, or ``None`` when the tier
            did not produce them.
    """

    index: int
    time_s: float
    positions_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    density_kg_m3: NDArray[np.float64]
    shear_rate_1_s: NDArray[np.float64] | None

    @property
    def speed_m_s(self) -> NDArray[np.float64]:
        """``(n,)`` velocity magnitudes."""
        return np.linalg.norm(self.velocity_m_s, axis=1)


@dataclass(frozen=True)
class SandFieldSeries:
    """A whole run's sand field, with its standing attached.

    Arrays are stacked over time rather than held as a list of frames:
    a colour scale has to see every frame at once (issue #8728), and a
    per-frame object graph would make that a loop over thousands of
    small allocations.

    Attributes:
        time_s: ``(T,)`` frame times.
        velocity_m_s: ``(T, N, D)`` sample velocities.
        density_kg_m3: ``(T, N)`` sample densities.
        shear_rate_1_s: ``(T, N)`` shear rates, or ``None``.
        positions_m: ``(T, N, D)`` sample positions, required for
            ``PARTICLE`` and ``None`` for ``GRID`` where the geometry
            implies them.
        layout: Which of the two the samples are.
        geometry: The lattice, required for ``GRID``.
        provenance: Tier, status and settings. Never optional.
        retention: What was dropped to get here. Never optional.
        occupancy: Where this field says there is sand. Never optional,
            and never a view's own choice -- see :class:`OccupancyRule`.
        body_outline_m: ``(T, V, D)`` intruder cross-section outline per
            frame, in the field's own coordinates, or ``None`` when the
            run had no intruder. A few dozen numbers a frame, and
            without them a velocity picture cannot distinguish sand
            pushed ahead of the sole from sand riding up the face --
            which is the question the whole view exists to answer.
    """

    time_s: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    density_kg_m3: NDArray[np.float64]
    shear_rate_1_s: NDArray[np.float64] | None
    positions_m: NDArray[np.float64] | None
    layout: FieldLayout
    geometry: GridGeometry | None
    provenance: FieldProvenance
    retention: RetentionRecord
    occupancy: OccupancyRule
    body_outline_m: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        velocity = np.asarray(self.velocity_m_s, dtype=np.float64)
        density = np.asarray(self.density_kg_m3, dtype=np.float64)
        layout = FieldLayout(str(self.layout))
        if times.size == 0:
            raise BunkerShot3DValueError(
                "a field series with no frames has nothing to show; an empty "
                "animation reads as a shot in which the sand never moved"
            )
        if not bool(np.all(np.isfinite(times))):
            raise BunkerShot3DValueError("frame times must all be finite")
        if bool(np.any(np.diff(times) < 0.0)):
            raise BunkerShot3DValueError(
                "frame times must be non-decreasing; an out-of-order field "
                "animates backwards through the impact"
            )
        if velocity.ndim != 3:
            raise BunkerShot3DValueError(
                f"velocity_m_s must be (T, N, D), got shape {velocity.shape}"
            )
        n_frames, n_samples, dimension = velocity.shape
        if n_frames != times.size:
            raise BunkerShot3DValueError(
                f"velocity_m_s has {n_frames} frames but there are {times.size} times"
            )
        if not _MIN_DIMENSION <= dimension <= _MAX_DIMENSION:
            raise BunkerShot3DValueError(
                f"a field must be {_MIN_DIMENSION}-D or {_MAX_DIMENSION}-D, got "
                f"{dimension} velocity components"
            )
        if density.shape != (n_frames, n_samples):
            raise BunkerShot3DValueError(
                f"density_kg_m3 must have shape {(n_frames, n_samples)}, got "
                f"{density.shape}"
            )
        shear = self._checked_optional(
            self.shear_rate_1_s, (n_frames, n_samples), "shear_rate_1_s"
        )
        positions = self._checked_optional(
            self.positions_m, (n_frames, n_samples, dimension), "positions_m"
        )
        if layout is FieldLayout.GRID:
            if self.geometry is None:
                raise BunkerShot3DValueError(
                    "a GRID series must carry its geometry: without it the sample "
                    "positions are unknowable and the field cannot be sliced"
                )
            if self.geometry.n_samples != n_samples:
                raise BunkerShot3DValueError(
                    f"the geometry describes {self.geometry.n_samples} samples but "
                    f"the arrays carry {n_samples}"
                )
            if self.geometry.dimension != dimension:
                raise BunkerShot3DValueError(
                    f"the geometry is {self.geometry.dimension}-D but the velocity "
                    f"has {dimension} components"
                )
        elif positions is None:
            raise BunkerShot3DValueError(
                "a PARTICLE series must carry per-frame positions: its samples "
                "move, so the geometry cannot imply them"
            )
        outline = self.body_outline_m
        if outline is not None:
            outline = np.asarray(outline, dtype=np.float64)
            if (
                outline.ndim != 3
                or outline.shape[0] != n_frames
                or outline.shape[2] != dimension
            ):
                raise BunkerShot3DValueError(
                    f"body_outline_m must have shape (T, V, {dimension}) with "
                    f"T = {n_frames}, got {outline.shape}"
                )
            if outline.shape[1] < 3:
                raise BunkerShot3DValueError(
                    f"a body outline needs at least 3 vertices to be a section, "
                    f"got {outline.shape[1]}"
                )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "shear_rate_1_s", shear)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "body_outline_m", outline)

    @staticmethod
    def _checked_optional(
        value: NDArray[np.float64] | None, expected: tuple[int, ...], name: str
    ) -> NDArray[np.float64] | None:
        """Coerce an optional array and check its shape."""
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.shape != expected:
            raise BunkerShot3DValueError(
                f"{name} must have shape {expected}, got {array.shape}"
            )
        return array

    @property
    def n_frames(self) -> int:
        """Number of stored frames."""
        return int(self.time_s.size)

    @property
    def n_samples(self) -> int:
        """Samples per frame."""
        return int(self.velocity_m_s.shape[1])

    @property
    def dimension(self) -> int:
        """Spatial dimension of the field."""
        return int(self.velocity_m_s.shape[2])

    @property
    def quantities(self) -> tuple[FieldQuantity, ...]:
        """Which quantities this series actually carries."""
        present = [FieldQuantity.VELOCITY, FieldQuantity.DENSITY]
        if self.shear_rate_1_s is not None:
            present.append(FieldQuantity.SHEAR_RATE)
        return tuple(present)

    @property
    def duration_s(self) -> float:
        """Span of the stored frames."""
        return float(self.time_s[-1] - self.time_s[0])

    def speed_m_s(self) -> NDArray[np.float64]:
        """``(T, N)`` velocity magnitudes over the whole run.

        Unmasked, so a caller who wants the raw transfer can have it.
        Anything that reports or draws a speed wants
        :meth:`occupied_speed_m_s` instead: a nodal velocity is momentum
        over mass, and at the tail of a stencil that mass is parts per
        million of a cell's sand, so the largest numbers in this array
        are round-off rather than flow.
        """
        return np.linalg.norm(self.velocity_m_s, axis=2)

    def occupied(self) -> NDArray[np.bool_]:
        """``(T, N)`` mask of the samples holding reportable sand."""
        return self.occupancy.occupied(self.density_kg_m3)

    def occupied_speed_m_s(self) -> NDArray[np.float64]:
        """``(T, N)`` speeds, ``nan`` where there is no reportable sand.

        ``nan`` rather than zero for the same reason the shear rate uses
        it: an empty cell is not a cell of stationary sand, and a
        ``nanmax`` over this array is the peak sand speed while a ``max``
        over :meth:`speed_m_s` is the peak numerical artefact.
        """
        return np.where(self.occupied(), self.speed_m_s(), np.nan)

    def peak_speed_m_s(self) -> float:
        """The fastest reportable sand in the run, or 0 if none moved."""
        speeds = self.occupied_speed_m_s()
        if not bool(np.isfinite(speeds).any()):
            return 0.0
        return float(np.nanmax(speeds))

    def sample_positions_m(self, frame: int) -> NDArray[np.float64]:
        """``(N, d)`` sample positions for one frame.

        Args:
            frame: Frame index.

        Returns:
            The positions, from the geometry for ``GRID`` and from the
            stored array for ``PARTICLE``.

        Raises:
            BunkerShot3DValueError: If ``frame`` is outside the series.
        """
        self._require_frame(frame)
        if self.positions_m is not None:
            return self.positions_m[int(frame)]
        if self.geometry is None:  # pragma: no cover - forbidden by __post_init__
            raise BunkerShot3DValueError("a GRID series must carry its geometry")
        return self.geometry.sample_positions_m()

    def frame(self, index: int) -> SandFieldFrame:
        """One frame as a :class:`SandFieldFrame`.

        Args:
            index: Frame index.

        Returns:
            The frame.

        Raises:
            BunkerShot3DValueError: If ``index`` is outside the series.
        """
        self._require_frame(index)
        position = int(index)
        return SandFieldFrame(
            index=position,
            time_s=float(self.time_s[position]),
            positions_m=self.sample_positions_m(position),
            velocity_m_s=self.velocity_m_s[position],
            density_kg_m3=self.density_kg_m3[position],
            shear_rate_1_s=(
                None if self.shear_rate_1_s is None else self.shear_rate_1_s[position]
            ),
        )

    def require_frame(self, index: int) -> None:
        """Refuse a frame index that is outside the stored series.

        Public because every view that scrubs this field needs the same
        refusal in the same words, and a view reaching for a private
        check would be free to invent a different one -- or, worse, to
        clamp to an end and show the wrong instant without saying so.

        Args:
            index: Frame index.

        Raises:
            BunkerShot3DValueError: If the index is outside the series.
        """
        self._require_frame(index)

    def _require_frame(self, index: int) -> None:
        """Precondition: ``index`` addresses a stored frame."""
        if not 0 <= int(index) < self.n_frames:
            raise BunkerShot3DValueError(
                f"frame {index} is outside the field, which has {self.n_frames} frames"
            )

    def metadata(self) -> dict[str, Any]:
        """Everything about the series except the bulk arrays.

        This is what :func:`series_digest` hashes alongside the arrays,
        and what :mod:`bunkershot3d.fields.store` writes as attributes.
        """
        return {
            "field_schema_version": int(FIELD_SCHEMA_VERSION),
            "layout": self.layout.value,
            "dimension": int(self.dimension),
            "n_frames": int(self.n_frames),
            "n_samples": int(self.n_samples),
            "has_shear_rate": self.shear_rate_1_s is not None,
            "has_positions": self.positions_m is not None,
            "has_body_outline": self.body_outline_m is not None,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "provenance": self.provenance.to_dict(),
            "retention": self.retention.to_dict(),
            "occupancy": self.occupancy.to_dict(),
            "units": {
                "time": TIME_UNIT,
                "velocity": VELOCITY_UNIT,
                "density": DENSITY_UNIT,
                "shear_rate": SHEAR_RATE_UNIT,
                "length": "m",
            },
        }


def series_digest(series: SandFieldSeries) -> str:
    """SHA-256 over a series' declared standing **and** its arrays.

    The two are hashed together on purpose.  Hashing only the arrays
    would let the tier be edited; hashing only the metadata would let
    the arrays be swapped.  Covering both is what makes
    "this field is F1 and BEYOND_VALIDATION" a checkable statement
    rather than a label.

    Arrays are hashed at ``float64`` in C order, so the digest is
    independent of how the series was laid out in memory.

    Args:
        series: The series to digest.

    Returns:
        A 64-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    digest.update(canonical_json(series.metadata()).encode("utf-8"))
    for name, array in _digest_arrays(series):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C"))
    return digest.hexdigest()


def _digest_arrays(
    series: SandFieldSeries,
) -> Sequence[tuple[str, NDArray[np.float64]]]:
    """The arrays covered by :func:`series_digest`, in a fixed order."""
    arrays: list[tuple[str, NDArray[np.float64]]] = [
        ("time_s", series.time_s),
        ("velocity_m_s", series.velocity_m_s),
        ("density_kg_m3", series.density_kg_m3),
    ]
    if series.shear_rate_1_s is not None:
        arrays.append(("shear_rate_1_s", series.shear_rate_1_s))
    if series.positions_m is not None:
        arrays.append(("positions_m", series.positions_m))
    if series.body_outline_m is not None:
        arrays.append(("body_outline_m", series.body_outline_m))
    return arrays
