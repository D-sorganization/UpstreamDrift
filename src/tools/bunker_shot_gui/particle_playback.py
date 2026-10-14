"""Playing a 3-D particle trajectory back, headless (issue #9545, #9693).

The viewer half of the slice-6 particle-trajectory artifact
(:mod:`bunkershot3d.fields.trajectory`). This module is pure Python and
NumPy: it decides *what* a frame shows, and :mod:`.particle_render_vtk`
decides only *how* to draw it, so every rule below is testable without a
display and shared by any front end that draws the same artifact.

The rules it enforces:

* **Identity.** Particles are served in roster order with their stable
  ids, straight from the verified reader; nothing is re-identified by
  position, so grains that cross cannot swap.
* **Time, not frame index.** Scrubbing maps a fraction onto simulation
  time, and club and ball are placed at the particle frame's own time by
  interpolating *their* samples -- never extrapolated past either end and
  never bridged across a missing (NaN) sample.
* **Display decimation is display only.** A bounded, deterministic subset
  of the roster is drawn when the glyph budget is exceeded. The values of
  the drawn particles are the undecimated values, and the stamp always
  carries the drawn/total count.
* **The stamp tells the truth.** Tier, verdict, producer and what a
  particle physically is (a grain, a parcel, a continuum material point)
  are composed into every frame.

No production solver emits the artifact yet (Chrono, the only
grain-resolving 3-D tier, needs ``pychrono``, which is not a declared
dependency). This module plays whatever a verified reader serves and
invents nothing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.fields.schema import GrainRoster, ParticleKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bunkershot3d.fields.standing import FieldProvenance
    from bunkershot3d.fields.trajectory import FrameState

__all__ = [
    "DEFAULT_MAX_GLYPHS",
    "BodyPose",
    "BodySource",
    "BodyTrack",
    "DisplayDecimation",
    "FrameRateReport",
    "ParticleFrameView",
    "ParticlePlayback",
    "TrajectorySource",
    "measure_frame_rate",
]

DEFAULT_MAX_GLYPHS = 20_000
"""Glyph budget before display decimation starts. A drawing budget, not a
physics setting: nothing a metric reads depends on it."""

_P95 = 0.95
_MS_PER_S = 1e3

_KIND_LINES: dict[ParticleKind, str] = {
    ParticleKind.DISCRETE_GRAIN: "particles: discrete grains (radius is the grain's)",
    ParticleKind.COARSE_PARTICLE: (
        "particles: coarse-grained parcels, not individual grains"
    ),
    ParticleKind.MATERIAL_POINT: (
        "particles: continuum material points, not solved grains"
    ),
}


class TrajectorySource(Protocol):
    """What playback needs from a verified trajectory.

    :class:`~bunkershot3d.fields.trajectory_reader.ParticleTrajectoryReader`
    satisfies it; depending on this protocol rather than on the reader keeps
    playback off the HDF5 layer.
    """

    @property
    def roster(self) -> GrainRoster: ...

    @property
    def time_s(self) -> NDArray[np.float64]: ...

    @property
    def provenance(self) -> FieldProvenance: ...

    def frame(self, index: int) -> FrameState: ...


_Element = TypeVar("_Element", bound=np.generic)


def _read_only(array: NDArray[_Element]) -> NDArray[_Element]:
    """An owned, write-protected copy of ``array``."""
    owned: NDArray[_Element] = np.array(array, copy=True)
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True)
class DisplayDecimation:
    """Which roster rows are drawn, and the sentence that says so.

    Attributes:
        total: Particles on the roster.
        stride: Every ``stride``-th roster row is drawn; 1 means all.
        rows: ``(K,)`` drawn roster rows, ascending.
    """

    total: int
    stride: int
    rows: NDArray[np.int64]

    @classmethod
    def for_count(cls, total: int, *, max_glyphs: int) -> DisplayDecimation:
        """A deterministic roster-order stride that fits ``max_glyphs``.

        Raises:
            ValueError: If ``total`` or ``max_glyphs`` is not positive.
        """
        if int(max_glyphs) < 1:
            raise ValueError(f"max_glyphs must be positive, got {max_glyphs!r}")
        if int(total) < 1:
            raise ValueError(f"a roster needs at least one particle, got {total!r}")
        stride = max(1, math.ceil(int(total) / int(max_glyphs)))
        rows = np.arange(0, int(total), stride, dtype=np.int64)
        return cls(int(total), stride, _read_only(rows))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DisplayDecimation):
            return NotImplemented
        return (self.total, self.stride) == (other.total, other.stride)

    def __hash__(self) -> int:
        return hash((self.total, self.stride))

    @property
    def retained(self) -> int:
        """How many particles are drawn."""
        return int(self.rows.size)

    @property
    def active(self) -> bool:
        """Whether any particle is left undrawn."""
        return self.stride > 1

    def label(self) -> str:
        """The drawn/total line every frame carries."""
        if not self.active:
            return f"drawing {self.retained} of {self.total} particles"
        return (
            f"display decimation: drawing {self.retained} of {self.total} "
            f"particles (every {self.stride}th in roster order); values and "
            "metrics use all of them"
        )


class BodySource(StrEnum):
    """Where a rigid body's path came from, stamped beside it."""

    SIMULATED = "simulated path"
    HELD = "held body, not a simulated path"
    EMPIRICAL = "empirical flight estimate, not simulated"


@dataclass(frozen=True)
class BodyTrack:
    """A rigid body's sampled position on its own clock.

    A NaN row marks a missing sample; it is never interpolated across.

    Attributes:
        name: Unique name within one playback, e.g. ``"club"``.
        time_s: ``(N,)`` strictly increasing finite sample times [s].
        position_m: ``(N, 3)`` world positions [m]; NaN rows are missing.
        source: Where the path came from.
    """

    name: str
    time_s: NDArray[np.float64]
    position_m: NDArray[np.float64]
    source: BodySource

    def __post_init__(self) -> None:
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        positions = np.asarray(self.position_m, dtype=np.float64)
        if not self.name:
            raise ValueError("a body track needs a name")
        if times.size == 0 or not bool(np.all(np.isfinite(times))):
            raise ValueError("body sample times must be finite and non-empty")
        if bool(np.any(np.diff(times) <= 0.0)):
            raise ValueError("body sample times must be strictly increasing")
        if positions.shape != (times.size, 3):
            raise ValueError(
                f"position_m must have shape {(times.size, 3)}, got {positions.shape}"
            )
        object.__setattr__(self, "time_s", _read_only(times))
        object.__setattr__(self, "position_m", _read_only(positions))
        object.__setattr__(self, "source", BodySource(self.source))

    def position_at(self, time_s: float) -> NDArray[np.float64] | None:
        """Linear position at ``time_s``, or ``None`` where there is no data.

        Postcondition: ``None`` outside the sampled interval and inside any
        interval bounded by a missing sample; never an extrapolation.
        """
        stamp = float(time_s)
        times = self.time_s
        if not math.isfinite(stamp) or stamp < times[0] or stamp > times[-1]:
            return None
        right = int(np.searchsorted(times, stamp, side="left"))
        if times[right] == stamp:
            row = self.position_m[right]
            return np.array(row) if bool(np.all(np.isfinite(row))) else None
        start, end = self.position_m[right - 1], self.position_m[right]
        if not bool(np.all(np.isfinite(start) & np.isfinite(end))):
            return None
        fraction = (stamp - times[right - 1]) / (times[right] - times[right - 1])
        return np.asarray(start + fraction * (end - start), dtype=np.float64)

    def pose_at(self, time_s: float) -> BodyPose:
        """The body as one frame shows it."""
        position = self.position_at(time_s)
        status = (
            self.source.value
            if position is not None
            else f"no data at t = {time_s * _MS_PER_S:.3f} ms"
        )
        return BodyPose(self.name, position, status)


@dataclass(frozen=True)
class BodyPose:
    """One body in one frame; ``position_m`` is ``None`` when it has no data."""

    name: str
    position_m: NDArray[np.float64] | None
    status: str


@dataclass(frozen=True)
class ParticleFrameView:
    """Everything one frame draws, in drawn-roster order, read-only.

    Attributes:
        frame_index: Retained frame shown.
        time_s: That frame's simulation time [s].
        grain_ids: ``(K,)`` stable ids of the drawn particles.
        radius_m: ``(K,)`` their radii [m].
        positions_m: ``(K, 3)`` world positions [m], NaN when inactive.
        active: ``(K,)`` whether each drawn particle is in the domain.
        bodies: Club, ball and any other bodies at ``time_s``.
        decimation: Which roster rows are drawn.
        stamp_lines: Tier, verdict, producer, particle kind, decimation and
            body status -- the text every rendering must carry.
    """

    frame_index: int
    time_s: float
    grain_ids: NDArray[np.int64]
    radius_m: NDArray[np.float64]
    positions_m: NDArray[np.float64]
    active: NDArray[np.bool_]
    bodies: tuple[BodyPose, ...]
    decimation: DisplayDecimation
    stamp_lines: tuple[str, ...]

    def visible_ids(self) -> NDArray[np.int64]:
        """Ids of the drawn particles that are in the domain this frame."""
        return np.asarray(self.grain_ids[self.active], dtype=np.int64)

    def body(self, name: str) -> BodyPose:
        """The named body.

        Raises:
            KeyError: If no body of that name is in this playback.
        """
        for pose in self.bodies:
            if pose.name == name:
                return pose
        raise KeyError(name)


class ParticlePlayback:
    """Transport over a verified trajectory: seek, step, scrub, view.

    Example:
        >>> with ParticleTrajectoryReader(path) as reader:
        ...     playback = ParticlePlayback(reader, bodies=(club,))
        ...     playback.scrub(0.5)
        ...     view = playback.view()
    """

    def __init__(
        self,
        source: TrajectorySource,
        *,
        bodies: Sequence[BodyTrack] = (),
        max_glyphs: int = DEFAULT_MAX_GLYPHS,
    ) -> None:
        """Bind a trajectory and the bodies drawn with it.

        Raises:
            ValueError: If two bodies share a name, the glyph budget is not
                positive, or the trajectory has no frames.
        """
        names = [track.name for track in bodies]
        if len(set(names)) != len(names):
            raise ValueError(f"body names must be unique, got {names}")
        roster = source.roster
        self._source = source
        self._roster = roster
        self._times = np.asarray(source.time_s, dtype=np.float64)
        if self._times.size == 0:
            raise ValueError("a trajectory with no frames has nothing to play")
        self._bodies = tuple(bodies)
        self._decimation = DisplayDecimation.for_count(
            roster.n_grains, max_glyphs=max_glyphs
        )
        self._header = self._header_lines(source.provenance, roster.kind)
        self._index = 0

    @staticmethod
    def _header_lines(provenance: FieldProvenance, kind: ParticleKind) -> list[str]:
        # Deferred: ``.report`` reaches the workbench model; keep this module's
        # import cost to the artifact schema alone.
        from .report import validity_stamp

        stamp = validity_stamp(provenance.envelope_status, provenance.fidelity_tier)
        return [
            *stamp.splitlines(),
            f"source: {provenance.solver_name}",
            _KIND_LINES[kind],
        ]

    @property
    def n_frames(self) -> int:
        """Retained frames in the trajectory."""
        return int(self._times.size)

    @property
    def frame_index(self) -> int:
        """The frame currently shown."""
        return self._index

    @property
    def time_s(self) -> float:
        """Simulation time of the frame currently shown [s]."""
        return float(self._times[self._index])

    @property
    def time_span_s(self) -> tuple[float, float]:
        """First and last retained frame times [s]; equal for one frame."""
        return float(self._times[0]), float(self._times[-1])

    def time_fraction(self) -> float:
        """Where the current frame sits in the run's *time* span, in ``[0, 1]``.

        The inverse of :meth:`scrub`; ``0.0`` for a single-frame run.
        """
        start, end = self.time_span_s
        return 0.0 if end <= start else (self.time_s - start) / (end - start)

    @property
    def decimation(self) -> DisplayDecimation:
        """Which roster rows are drawn."""
        return self._decimation

    def seek_frame(self, index: int) -> int:
        """Show retained frame ``index``.

        Raises:
            ValueError: If ``index`` is outside the trajectory.
        """
        position = int(index)
        if not 0 <= position < self.n_frames:
            raise ValueError(f"frame {index} is outside 0..{self.n_frames - 1}")
        self._index = position
        return position

    def step(self, delta: int = 1) -> int:
        """Move ``delta`` retained frames, stopping at either end."""
        return self.seek_frame(min(max(self._index + int(delta), 0), self.n_frames - 1))

    def seek_time(self, time_s: float) -> int:
        """Show the retained frame nearest ``time_s``, clamped to the run.

        Raises:
            ValueError: If ``time_s`` is not finite.
        """
        stamp = float(time_s)
        if not math.isfinite(stamp):
            raise ValueError(f"seek time must be finite, got {time_s!r}")
        return self.seek_frame(int(np.argmin(np.abs(self._times - stamp))))

    def scrub(self, fraction: float) -> int:
        """Show the frame at ``fraction`` of the run's *time* span.

        Raises:
            ValueError: If ``fraction`` is not in ``[0, 1]``.
        """
        share = float(fraction)
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"scrub fraction must be in [0, 1], got {fraction!r}")
        start, end = self.time_span_s
        return self.seek_time(start + share * (end - start))

    def view(self) -> ParticleFrameView:
        """The current frame, decimated for display and stamped."""
        state = self._source.frame(self._index)
        rows = self._decimation.rows
        moment = self.time_s
        poses = tuple(track.pose_at(moment) for track in self._bodies)
        stamp = (
            *self._header,
            f"t = {moment * _MS_PER_S:.3f} ms (frame {self._index + 1} of "
            f"{self.n_frames})",
            self._decimation.label(),
            *(f"{pose.name}: {pose.status}" for pose in poses),
        )
        return ParticleFrameView(
            frame_index=self._index,
            time_s=moment,
            grain_ids=_read_only(self._roster.grain_ids[rows]),
            radius_m=_read_only(self._roster.radius_m[rows]),
            positions_m=_read_only(state.positions_m[rows]),
            active=_read_only(state.active[rows]),
            bodies=poses,
            decimation=self._decimation,
            stamp_lines=stamp,
        )


@dataclass(frozen=True)
class FrameRateReport:
    """Measured per-frame wall time of a playback run.

    Attributes:
        n_frames: Frames timed.
        median_frame_s: Median frame time [s].
        p95_frame_s: 95th-percentile (nearest-rank) frame time [s].
        max_frame_s: Slowest frame [s].
    """

    n_frames: int
    median_frame_s: float
    p95_frame_s: float
    max_frame_s: float

    @property
    def median_fps(self) -> float:
        """Frames per second at the median frame time."""
        return math.inf if self.median_frame_s == 0.0 else 1.0 / self.median_frame_s


def measure_frame_rate(
    render_frame: Callable[[int], object],
    frames: Iterable[int],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> FrameRateReport:
    """Time ``render_frame`` once per frame and summarise median and tail.

    The hook a benchmark wraps around a real viewport (for example
    ``ParticleViewport.show_frame``); nothing is estimated.

    Raises:
        ValueError: If ``frames`` is empty or the clock runs backwards.
    """
    samples: list[float] = []
    for frame in frames:
        started = clock()
        render_frame(frame)
        samples.append(clock() - started)
    if not samples:
        raise ValueError("measure_frame_rate needs at least one frame")
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    if float(ordered[0]) < 0.0:
        raise ValueError("the clock ran backwards during measurement")
    tail = ordered[max(0, math.ceil(_P95 * ordered.size) - 1)]
    return FrameRateReport(
        n_frames=int(ordered.size),
        median_frame_s=float(np.median(ordered)),
        p95_frame_s=float(tail),
        max_frame_s=float(ordered[-1]),
    )
