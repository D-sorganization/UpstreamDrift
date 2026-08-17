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
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BunkerShot3DValueError
from ..provenance.hashing import canonical_json
from ..provenance.rng import SeedRecord
from ..solvers.envelope import MAX_VALIDATED_SPEED_M_S, EnvelopeStatus
from ..solvers.protocol import FidelityTier

__all__ = [
    "DEFAULT_OCCUPANCY_FLOOR_FRACTION",
    "DENSITY_UNIT",
    "FIELD_SCHEMA_VERSION",
    "SHEAR_RATE_UNIT",
    "TIME_UNIT",
    "VELOCITY_UNIT",
    "FieldIntegrityError",
    "FieldLayout",
    "FieldProvenance",
    "FieldQuantity",
    "GridGeometry",
    "OccupancyRule",
    "RetentionPolicy",
    "RetentionRecord",
    "SandFieldFrame",
    "SandFieldSeries",
    "series_digest",
]

FIELD_SCHEMA_VERSION = 1
"""Version of the sand-field payload itself.

Separate from :data:`bunkershot3d.io.SCHEMA_VERSION`, which versions the
*container*.  The two move independently: adding a quantity to a field
is not the same event as changing how a clubhead trace is laid out."""

VELOCITY_UNIT = "m/s"
DENSITY_UNIT = "kg/m^3"
SHEAR_RATE_UNIT = "1/s"
TIME_UNIT = "s"

_MIN_DIMENSION = 2
_MAX_DIMENSION = 3


class FieldIntegrityError(BunkerShot3DValueError):
    """A stored field's declared standing does not match its contents.

    Raised when the recomputed digest disagrees with the stored one,
    which is what happens if the tier or validity attribute is edited
    after the fact.  This is the failure mode issue #8710 calls
    non-negotiable, so it is a distinct exception rather than a generic
    value error a caller might already be swallowing.
    """


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


DEFAULT_OCCUPANCY_FLOOR_FRACTION = 0.10
"""Density below which a sample is reported as carrying no sand, as a
fraction of the bulk density.

**Measured, not chosen.**  A grid velocity is nodal momentum divided by
nodal mass, and at the outer tail of a B-spline stencil that mass is a
few parts per million of a cell's worth of sand.  Dividing round-off by
it produces enormous velocities that are numerics, not flow.  On the
2 mm reference capture of a 25 m/s shot the reported peak sand speed
runs 46.7 m/s with no floor, 32.2 m/s at 1 %, 29.0 m/s at 10 % and
28.3 m/s at 50 %: it stops moving at 10 %, and the 46.7 m/s "peak" sits
on a node holding 7.5e-6 of the bulk density.

The same number falls out of the physics, which is why it is this one
and not a tuning knob.  At ``dx = 2 mm`` and ``d50 = 0.458 mm`` a single
grain's cross-section is about 4 % of a cell, so a 10 % floor is
"fewer than about two and a half grains in this cell" -- below which a
continuum density is not a measurement of anything, for the same reason
:data:`~bunkershot3d.solvers.mpm.envelope.MIN_CELLS_PER_GRAIN` refuses a
sub-grain grid."""


@dataclass(frozen=True)
class OccupancyRule:
    """Where a field says there is sand, declared rather than assumed.

    Every view masks on this, and it travels inside the file, so two
    views of the same field cannot disagree about where the sand is and
    a masked picture cannot be re-thresholded into a different claim by
    a downstream widget.

    Attributes:
        reference_density_kg_m3: The material's bulk density, which the
            floor is a fraction of.
        floor_fraction: See :data:`DEFAULT_OCCUPANCY_FLOOR_FRACTION`.
        max_admissible_density_kg_m3: The densest bulk density this sand
            can actually reach, or ``None`` when the tier did not state
            one. A ceiling, not a clip: samples above it are kept and
            **counted**, because a nodal density is a scatter of
            particle masses onto a node and nothing in the transfer
            bounds it by the packing limit the constitutive model
            enforces on the particles. Sand denser than its own densest
            packing is a reporting artefact, and a colour bar running
            past that limit without saying so states something
            impossible.
    """

    reference_density_kg_m3: float
    floor_fraction: float = DEFAULT_OCCUPANCY_FLOOR_FRACTION
    max_admissible_density_kg_m3: float | None = None

    def __post_init__(self) -> None:
        density = float(self.reference_density_kg_m3)
        fraction = float(self.floor_fraction)
        if not math.isfinite(density) or density <= 0.0:
            raise BunkerShot3DValueError(
                f"reference_density_kg_m3 must be positive, got "
                f"{self.reference_density_kg_m3!r}"
            )
        if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise BunkerShot3DValueError(
                f"floor_fraction must lie in [0, 1), got {self.floor_fraction!r}"
            )
        ceiling = self.max_admissible_density_kg_m3
        if ceiling is not None:
            ceiling = float(ceiling)
            if not math.isfinite(ceiling) or ceiling < density:
                raise BunkerShot3DValueError(
                    f"max_admissible_density_kg_m3 must be at least the bulk "
                    f"density {density:.6g}, got "
                    f"{self.max_admissible_density_kg_m3!r}"
                )
        object.__setattr__(self, "reference_density_kg_m3", density)
        object.__setattr__(self, "floor_fraction", fraction)
        object.__setattr__(self, "max_admissible_density_kg_m3", ceiling)

    @property
    def floor_kg_m3(self) -> float:
        """The absolute density floor."""
        return self.reference_density_kg_m3 * self.floor_fraction

    def occupied(self, density_kg_m3: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Boolean mask of the samples that hold reportable sand.

        Args:
            density_kg_m3: Any-shaped density array.

        Returns:
            A mask of the same shape.
        """
        return np.asarray(density_kg_m3) >= self.floor_kg_m3

    def over_packing_limit(
        self, density_kg_m3: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Mask of samples denser than this sand can physically pack.

        Args:
            density_kg_m3: Any-shaped density array.

        Returns:
            A mask of the same shape; all ``False`` when no limit was
            stated.
        """
        values = np.asarray(density_kg_m3)
        if self.max_admissible_density_kg_m3 is None:
            return np.zeros(values.shape, dtype=bool)
        return values > self.max_admissible_density_kg_m3

    def packing_note(self, density_kg_m3: NDArray[np.float64]) -> str:
        """How much of a density array is above the packing limit.

        Empty when nothing is, so a caller can append it unconditionally
        and a clean field carries no apology it does not owe.
        """
        if self.max_admissible_density_kg_m3 is None:
            return ""
        over = self.over_packing_limit(density_kg_m3)
        count = int(over.sum())
        if count == 0:
            return ""
        values = np.asarray(density_kg_m3)
        return (
            f"{count} of {values.size} samples ({count / values.size * 100:.3g}%) "
            f"exceed the densest packing this sand admits "
            f"({self.max_admissible_density_kg_m3:.4g} {DENSITY_UNIT}, peak "
            f"{float(values[over].max()):.4g}); nodal density is a mass scatter "
            "and is not bounded by it, so that is a transfer artefact"
        )

    def describe(self) -> str:
        """One line naming the floor, and the ceiling where there is one."""
        line = (
            f"sand where density >= {self.floor_fraction * 100:.3g}% of "
            f"{self.reference_density_kg_m3:.4g} {DENSITY_UNIT} "
            f"({self.floor_kg_m3:.4g} {DENSITY_UNIT})"
        )
        if self.max_admissible_density_kg_m3 is None:
            return line
        return (
            f"{line}; densest admissible packing "
            f"{self.max_admissible_density_kg_m3:.4g} {DENSITY_UNIT}"
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping."""
        return {
            "reference_density_kg_m3": float(self.reference_density_kg_m3),
            "floor_fraction": float(self.floor_fraction),
            "max_admissible_density_kg_m3": (
                None
                if self.max_admissible_density_kg_m3 is None
                else float(self.max_admissible_density_kg_m3)
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OccupancyRule:
        """Rebuild from :meth:`to_dict`."""
        ceiling = payload.get("max_admissible_density_kg_m3")
        return cls(
            reference_density_kg_m3=float(payload["reference_density_kg_m3"]),
            floor_fraction=float(payload["floor_fraction"]),
            max_admissible_density_kg_m3=(None if ceiling is None else float(ceiling)),
        )


@dataclass(frozen=True)
class RetentionPolicy:
    """What a caller is willing to keep, decided before the run.

    Every entry here is a deliberate choice about what to lose.  The
    alternative -- writing everything and letting the filesystem decide
    -- is how a 1 mm F1 run turns into a gigabyte nobody meant to make.

    Attributes:
        target_frames: How many frames to keep. The stride is derived
            from this and the marched step count, so a longer run gets a
            coarser stride rather than a truncated tail. **Truncation is
            never the answer**: cutting the end off a shot removes
            exactly the part the question is about.
        store_dtype: On-disk element type. ``float32`` halves the file
            and costs ~7 significant decimal digits, which is far below
            the discretisation error of any field here; ``float64``
            keeps the solver's own precision.
        compression: HDF5 filter name, or ``""`` for none.
        compression_level: Filter level, where the filter takes one.
        region_m: Optional ``(lower, upper)`` crop of the sampled
            region, in metres, per axis. The run-in and run-out of the
            bed are far from the impact zone and carry nothing; cropping
            them is the largest single saving after the stride.
        include_shear_rate: Whether to form and keep the shear rate.
    """

    target_frames: int = 120
    store_dtype: str = "float32"
    compression: str = "gzip"
    compression_level: int = 4
    region_m: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    include_shear_rate: bool = True

    def __post_init__(self) -> None:
        if int(self.target_frames) < 1:
            raise BunkerShot3DValueError(
                f"target_frames must be at least 1, got {self.target_frames!r}"
            )
        if self.store_dtype not in _ALLOWED_DTYPES:
            raise BunkerShot3DValueError(
                f"store_dtype must be one of {sorted(_ALLOWED_DTYPES)}, got "
                f"{self.store_dtype!r}"
            )
        if not 0 <= int(self.compression_level) <= 9:
            raise BunkerShot3DValueError(
                f"compression_level must lie in [0, 9], got {self.compression_level!r}"
            )
        if self.region_m is not None:
            lower, upper = self.region_m
            if len(lower) != len(upper):
                raise BunkerShot3DValueError(
                    f"region_m corners must have the same length, got {self.region_m!r}"
                )
            if any(hi <= lo for lo, hi in zip(lower, upper, strict=True)):
                raise BunkerShot3DValueError(
                    f"region_m must be increasing on every axis, got {self.region_m!r}"
                )
            object.__setattr__(
                self,
                "region_m",
                (
                    tuple(float(value) for value in lower),
                    tuple(float(value) for value in upper),
                ),
            )
        object.__setattr__(self, "target_frames", int(self.target_frames))
        object.__setattr__(self, "compression_level", int(self.compression_level))

    @property
    def relative_precision(self) -> float:
        """Machine epsilon of :attr:`store_dtype`, the quantisation kept."""
        return float(np.finfo(np.dtype(self.store_dtype)).eps)

    def stride_for(self, n_steps: int) -> int:
        """Temporal stride that lands on or under :attr:`target_frames`.

        Args:
            n_steps: Steps the march will take.

        Returns:
            The stride, at least one.

        Raises:
            BunkerShot3DValueError: If ``n_steps`` is not positive.
        """
        steps = int(n_steps)
        if steps < 1:
            raise BunkerShot3DValueError(f"n_steps must be positive, got {n_steps!r}")
        return max(1, math.ceil(steps / self.target_frames))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping."""
        return {
            "target_frames": int(self.target_frames),
            "store_dtype": str(self.store_dtype),
            "compression": str(self.compression),
            "compression_level": int(self.compression_level),
            "region_m": (
                None
                if self.region_m is None
                else [list(self.region_m[0]), list(self.region_m[1])]
            ),
            "include_shear_rate": bool(self.include_shear_rate),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetentionPolicy:
        """Rebuild from :meth:`to_dict`."""
        region = payload.get("region_m")
        return cls(
            target_frames=int(payload["target_frames"]),
            store_dtype=str(payload["store_dtype"]),
            compression=str(payload["compression"]),
            compression_level=int(payload["compression_level"]),
            region_m=(
                None
                if region is None
                else (
                    tuple(float(value) for value in region[0]),
                    tuple(float(value) for value in region[1]),
                )
            ),
            include_shear_rate=bool(payload["include_shear_rate"]),
        )


_ALLOWED_DTYPES = frozenset({"float32", "float64"})


@dataclass(frozen=True)
class RetentionRecord:
    """What the policy actually cost, measured rather than intended.

    The policy says what was asked for; this says what happened.  They
    differ whenever a run is shorter than the target frame count, or a
    crop lands on a cell boundary, and the difference is exactly the
    thing a reader needs in order to know whether a feature is missing
    or was never there.

    Attributes:
        policy: The policy that produced this record.
        steps_marched: Steps the solver took.
        time_stride: Steps between kept frames.
        frames_kept: Frames actually stored.
        time_step_s: The solver's step, so a reader can recover the
            marched time base from the stride.
        samples_in_domain: Samples the solver carried per frame.
        samples_kept: Samples stored per frame after any crop.
        dropped: One line per thing that was dropped, in plain words.
            Empty only when genuinely nothing was.
    """

    policy: RetentionPolicy
    steps_marched: int
    time_stride: int
    frames_kept: int
    time_step_s: float
    samples_in_domain: int
    samples_kept: int
    dropped: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = {
            "steps_marched": self.steps_marched,
            "time_stride": self.time_stride,
            "frames_kept": self.frames_kept,
            "samples_in_domain": self.samples_in_domain,
            "samples_kept": self.samples_kept,
        }
        for name, value in counts.items():
            if int(value) < 1:
                raise BunkerShot3DValueError(f"{name} must be positive, got {value!r}")
        if int(self.samples_kept) > int(self.samples_in_domain):
            raise BunkerShot3DValueError(
                f"samples_kept ({self.samples_kept}) cannot exceed the "
                f"{self.samples_in_domain} the solver carried"
            )
        if not math.isfinite(self.time_step_s) or self.time_step_s <= 0.0:
            raise BunkerShot3DValueError(
                f"time_step_s must be positive, got {self.time_step_s!r}"
            )
        object.__setattr__(self, "dropped", tuple(str(line) for line in self.dropped))

    @property
    def temporal_fraction_kept(self) -> float:
        """Fraction of marched steps that survive as frames."""
        return float(self.frames_kept) / float(self.steps_marched)

    @property
    def spatial_fraction_kept(self) -> float:
        """Fraction of the solver's samples that survive the crop."""
        return float(self.samples_kept) / float(self.samples_in_domain)

    @property
    def sample_interval_s(self) -> float:
        """Wall time between stored frames."""
        return self.time_step_s * float(self.time_stride)

    def describe(self) -> str:
        """One line naming the stride, the crop and the precision."""
        parts = [
            f"{self.frames_kept} frames of {self.steps_marched} steps "
            f"(every {self.time_stride}, {self.sample_interval_s * 1e6:.3g} us)",
            f"{self.samples_kept} of {self.samples_in_domain} samples",
            f"stored {self.policy.store_dtype}",
        ]
        if self.policy.compression:
            parts.append(f"{self.policy.compression}-{self.policy.compression_level}")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping."""
        return {
            "policy": self.policy.to_dict(),
            "steps_marched": int(self.steps_marched),
            "time_stride": int(self.time_stride),
            "frames_kept": int(self.frames_kept),
            "time_step_s": float(self.time_step_s),
            "samples_in_domain": int(self.samples_in_domain),
            "samples_kept": int(self.samples_kept),
            "dropped": list(self.dropped),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetentionRecord:
        """Rebuild from :meth:`to_dict`."""
        return cls(
            policy=RetentionPolicy.from_dict(payload["policy"]),
            steps_marched=int(payload["steps_marched"]),
            time_stride=int(payload["time_stride"]),
            frames_kept=int(payload["frames_kept"]),
            time_step_s=float(payload["time_step_s"]),
            samples_in_domain=int(payload["samples_in_domain"]),
            samples_kept=int(payload["samples_kept"]),
            dropped=tuple(str(line) for line in payload.get("dropped", ())),
        )


@dataclass(frozen=True)
class FieldProvenance:
    """Which tier produced a field, how far outside its envelope, and why.

    This is the object issue #8710 calls non-negotiable.  It is stored
    inside the file, covered by :func:`series_digest`, and read by every
    downstream view -- so the tier and the validity status of a picture
    are properties of the picture, not of the path it was loaded from.

    Attributes:
        fidelity_tier: The tier that solved it.
        envelope_status: Worst status over the run.
        solver_name: Fully-qualified class name of the solver.
        kinematics: How the body's motion was supplied. F1 declares a
            straight-line constant-velocity approach rather than
            marching a whole shot (deferred to issue #8733), and a field
            animated from that approach must say so or it reads as a
            swing.
        peak_speed_m_s: Fastest body speed in the query.
        caveats: Caveat names carried by the verdict.
        reasons: The verdict's own reason lines.
        refused: Quantities this tier refuses to be quoted for.
        settings: Solver settings, as scalars and strings, so a run can
            be regenerated from the file alone.
        seeds: RNG seed records. Empty is legal and means the producing
            tier drew no random numbers -- F1 is deterministic -- but
            the emptiness is recorded rather than implied.
        field_schema_version: :data:`FIELD_SCHEMA_VERSION` at write time.
    """

    fidelity_tier: FidelityTier
    envelope_status: EnvelopeStatus
    solver_name: str
    kinematics: str
    peak_speed_m_s: float
    caveats: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()
    settings: Mapping[str, float | int | str] = field(default_factory=dict)
    seeds: tuple[SeedRecord, ...] = ()
    field_schema_version: int = FIELD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.fidelity_tier, FidelityTier):
            object.__setattr__(
                self, "fidelity_tier", FidelityTier(str(self.fidelity_tier))
            )
        if not isinstance(self.envelope_status, EnvelopeStatus):
            object.__setattr__(
                self, "envelope_status", EnvelopeStatus(str(self.envelope_status))
            )
        if not str(self.solver_name).strip():
            raise BunkerShot3DValueError(
                "solver_name must name the solver that produced the field; an "
                "anonymous field cannot be regenerated"
            )
        if not str(self.kinematics).strip():
            raise BunkerShot3DValueError(
                "kinematics must state how the body's motion was supplied: a "
                "declared approach and a marched shot are different claims and "
                "an animation of either looks the same"
            )
        speed = float(self.peak_speed_m_s)
        if not math.isfinite(speed) or speed < 0.0:
            raise BunkerShot3DValueError(
                f"peak_speed_m_s must be finite and non-negative, got "
                f"{self.peak_speed_m_s!r}"
            )
        object.__setattr__(self, "peak_speed_m_s", speed)
        for name in ("caveats", "reasons", "refused"):
            object.__setattr__(
                self, name, tuple(str(item) for item in getattr(self, name))
            )
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))
        object.__setattr__(self, "seeds", tuple(self.seeds))

    @property
    def speed_ratio(self) -> float:
        """Peak speed as a multiple of :data:`MAX_VALIDATED_SPEED_M_S`.

        1.44 m/s is the fastest intrusion anywhere in the published
        RFT/DRFT corpus, so a bunker shot is outside it from its first
        sample and this ratio is never below one in practice.
        """
        return self.peak_speed_m_s / MAX_VALIDATED_SPEED_M_S

    @property
    def is_within_published_speed(self) -> bool:
        """Whether the query stays inside the published speed corpus."""
        return self.peak_speed_m_s <= MAX_VALIDATED_SPEED_M_S

    def speed_headline(self) -> str:
        """The speed caveat, in words, for an in-frame stamp."""
        if self.is_within_published_speed:
            return (
                f"{self.peak_speed_m_s:.3g} m/s, within the "
                f"{MAX_VALIDATED_SPEED_M_S:.2f} m/s published corpus"
            )
        return (
            f"{self.peak_speed_m_s:.3g} m/s = {self.speed_ratio:.0f}x the "
            f"{MAX_VALIDATED_SPEED_M_S:.2f} m/s published corpus limit"
        )

    @property
    def status_label(self) -> str:
        """The envelope status as display text, composed in one place.

        Every view that quotes it -- the frame stamp, the workbench
        readout, a report -- takes it from here, so a designer reading
        two of them is not reading two vocabularies.
        """
        return str(self.envelope_status.value).replace("_", " ").upper()

    def headline(self) -> str:
        """Tier, status and speed standing on one line."""
        return (
            f"{self.status_label} - "
            f"{self.fidelity_tier.value} sand field; {self.speed_headline()}"
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe mapping, the input to :func:`series_digest`."""
        return {
            "fidelity_tier": self.fidelity_tier.value,
            "envelope_status": self.envelope_status.value,
            "solver_name": str(self.solver_name),
            "kinematics": str(self.kinematics),
            "peak_speed_m_s": float(self.peak_speed_m_s),
            "caveats": list(self.caveats),
            "reasons": list(self.reasons),
            "refused": list(self.refused),
            "settings": {
                str(key): value for key, value in sorted(self.settings.items())
            },
            "seeds": [record.to_dict() for record in self.seeds],
            "field_schema_version": int(self.field_schema_version),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FieldProvenance:
        """Rebuild from :meth:`to_dict`."""
        return cls(
            fidelity_tier=FidelityTier(str(payload["fidelity_tier"])),
            envelope_status=EnvelopeStatus(str(payload["envelope_status"])),
            solver_name=str(payload["solver_name"]),
            kinematics=str(payload["kinematics"]),
            peak_speed_m_s=float(payload["peak_speed_m_s"]),
            caveats=tuple(str(item) for item in payload.get("caveats", ())),
            reasons=tuple(str(item) for item in payload.get("reasons", ())),
            refused=tuple(str(item) for item in payload.get("refused", ())),
            settings=dict(payload.get("settings", {})),
            seeds=tuple(
                SeedRecord.from_dict(record) for record in payload.get("seeds", ())
            ),
            field_schema_version=int(payload.get("field_schema_version", 1)),
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
