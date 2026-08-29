"""What a stored sand field *claims* about itself (issue #8710).

:mod:`bunkershot3d.fields.schema` holds what a field **contains** -- the
arrays, where their samples are, and how the whole thing hashes.  This
module holds what it **claims**: which tier produced it, how valid that
tier's answer is, where it says there is sand at all, and what was thrown
away to fit it on disk.

The split is by concern rather than by size.  A field's contents and a
field's standing change for different reasons: adding a quantity is a
change to the container, while a new tier or a new validity rule is a
change to the claims.  Keeping them apart means neither edit has to open
the other file.

Everything here is JSON-round-trippable, because all of it is covered by
:func:`~bunkershot3d.fields.schema.series_digest` and therefore has to be
canonicalised the same way every time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BunkerShot3DValueError
from ..provenance.rng import SeedRecord
from ..solvers.envelope import MAX_VALIDATED_SPEED_M_S, EnvelopeStatus
from ..solvers.protocol import FidelityTier
from .units import DENSITY_UNIT

__all__ = [
    "DEFAULT_OCCUPANCY_FLOOR_FRACTION",
    "FIELD_SCHEMA_VERSION",
    "FieldIntegrityError",
    "FieldProvenance",
    "OccupancyRule",
    "RetentionPolicy",
    "RetentionRecord",
]

FIELD_SCHEMA_VERSION = 1
"""Version of the sand-field payload itself.

Separate from :data:`bunkershot3d.io.SCHEMA_VERSION`, which versions the
*container*.  The two move independently: adding a quantity to a field
is not the same event as changing how a clubhead trace is laid out.

It lives here rather than beside the arrays because it is a claim the
file makes about itself, and because
:class:`FieldProvenance` records it -- putting it with the container
would make the container import the standing and the standing import the
container."""


class FieldIntegrityError(BunkerShot3DValueError):
    """A stored field's declared standing does not match its contents.

    Raised when the recomputed digest disagrees with the stored one,
    which is what happens if the tier or validity attribute is edited
    after the fact.  This is the failure mode issue #8710 calls
    non-negotiable, so it is a distinct exception rather than a generic
    value error a caller might already be swallowing.
    """


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

    def setting(self, name: str, default: float | int | str = 0.0) -> float | int | str:
        """Return one recorded solver setting, or ``default`` if absent.

        Callers want a single stated assumption -- an effective width, a
        timestep -- not the whole mapping, and reaching through
        ``provenance.settings`` to ask makes every one of them depend on
        the settings being a mapping at all.

        Args:
            name: The setting's key.
            default: Returned when the run recorded no such setting.

        Returns:
            The recorded value, or ``default``.
        """
        return self.settings.get(name, default)

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
