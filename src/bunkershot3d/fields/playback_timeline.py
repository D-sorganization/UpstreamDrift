"""Read-only shared 3-D shot playback timeline contract (WP-3, Refs #15805)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BunkerShot3DValueError
from ..provenance.hashing import canonical_json
from ..solvers.envelope import EnvelopeStatus
from ..solvers.protocol import FidelityTier
from .schema import ParticleKind

TIMELINE_SCHEMA_VERSION = 1
BALL_DIAMETER_M = 0.04267
DEFAULT_ADDRESS_BURIAL_DEPTH_M = 0.005
MAX_HANDOFF_JUMP_M = 1e-3
_ROT_TOL, _SYNC_TOL_S, _HEX_LEN = 1e-5, 1e-9, 64
Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


class ShotPhase(StrEnum):
    """Canonical phase along one continuous bunker-to-green shot."""

    ADDRESS = "address"
    SAND_IMPACT = "sand_impact"
    SAND_EXIT = "sand_exit"
    FLIGHT = "flight"
    GREEN_ROLLOUT = "green_rollout"


_SAND = frozenset({ShotPhase.ADDRESS, ShotPhase.SAND_IMPACT, ShotPhase.SAND_EXIT})


def _freeze_vec3s(obj: object) -> None:
    """Validate and freeze position, linear velocity, and angular velocity 3-tuples."""
    for attr in ("position_m", "velocity_m_s", "angular_velocity_rad_s"):
        val = getattr(obj, attr)
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        if arr.shape != (3,) or not bool(np.all(np.isfinite(arr))):
            raise BunkerShot3DValueError(f"{attr} must be finite 3-vector")
        object.__setattr__(obj, attr, (float(arr[0]), float(arr[1]), float(arr[2])))


def _rot3(value: Any) -> Mat3:
    """Validate and freeze a 3x3 proper right-handed rotation matrix."""
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (3, 3) or not bool(np.all(np.isfinite(m))):
        raise BunkerShot3DValueError("orientation_mat must be finite (3, 3)")
    if not math.isclose(
        float(np.linalg.det(m)), 1.0, abs_tol=_ROT_TOL
    ) or not np.allclose(m.T @ m, np.eye(3), atol=_ROT_TOL):
        raise BunkerShot3DValueError("proper orthonormal rotation required (det=+1)")
    return tuple((float(r[0]), float(r[1]), float(r[2])) for r in m)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class BallState:
    """World-space 3-D golf ball state at one timeline instant."""

    position_m: Vec3
    velocity_m_s: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity_rad_s: Vec3 = (0.0, 0.0, 0.0)
    diameter_m: float = BALL_DIAMETER_M
    burial_depth_m: float = 0.0

    def __post_init__(self) -> None:
        """Validate finite ball kinematics, regulation diameter, and burial depth."""
        d, depth = float(self.diameter_m), float(self.burial_depth_m)
        if not math.isfinite(d) or d <= 0.0 or not math.isfinite(depth) or depth < 0.0:
            raise BunkerShot3DValueError("invalid diameter_m or burial_depth_m")
        _freeze_vec3s(self)
        object.__setattr__(self, "diameter_m", d)
        object.__setattr__(self, "burial_depth_m", depth)

    @property
    def is_visible_above_sand(self) -> bool:
        """True when the top of the ball sits above ``z = 0``."""
        return self.position_m[2] + self.diameter_m / 2.0 > 0.0


@dataclass(frozen=True, slots=True)
class ClubheadPose:
    """World-space rigid clubhead pose and velocity without axis inversions."""

    position_m: Vec3
    orientation_mat: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    velocity_m_s: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity_rad_s: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate world vectors and proper right-handed orientation."""
        _freeze_vec3s(self)
        object.__setattr__(self, "orientation_mat", _rot3(self.orientation_mat))


@dataclass(frozen=True, slots=True)
class ParticleSnapshot:
    """Synchronized read-only 3-D particle snapshot from a particle-resolving solve."""

    time_s: float
    positions_m: NDArray[np.float64]
    velocities_m_s: NDArray[np.float64]
    particle_kind: ParticleKind = ParticleKind.MATERIAL_POINT

    def __post_init__(self) -> None:
        """Validate matching non-empty particle arrays and lock writeability."""
        stamp = float(self.time_s)
        if not math.isfinite(stamp) or stamp < 0.0:
            raise BunkerShot3DValueError(f"invalid particle time_s={stamp!r}")
        for attr in ("positions_m", "velocities_m_s"):
            arr = np.array(getattr(self, attr), dtype=np.float64, copy=True)
            if (
                arr.ndim != 2
                or arr.shape[1] != 3
                or arr.shape[0] < 1
                or not bool(np.all(np.isfinite(arr)))
            ):
                raise BunkerShot3DValueError(
                    f"{attr} must be finite with shape (N>=1, 3)"
                )
            arr.flags.writeable = False
            object.__setattr__(self, attr, arr)
        if self.positions_m.shape != self.velocities_m_s.shape:
            raise BunkerShot3DValueError("positions_m and velocities_m_s shapes differ")
        object.__setattr__(self, "time_s", stamp)
        object.__setattr__(self, "particle_kind", ParticleKind(str(self.particle_kind)))

    def __eq__(self, other: object) -> bool:
        """Compare particle snapshots by timestamp, kind, and exact array contents."""
        if not isinstance(other, ParticleSnapshot):
            return NotImplemented
        return (
            self.time_s == other.time_s
            and self.particle_kind is other.particle_kind
            and np.array_equal(self.positions_m, other.positions_m)
            and np.array_equal(self.velocities_m_s, other.velocities_m_s)
        )


@dataclass(frozen=True, slots=True)
class TimelineFrame:
    """One synchronized read-only playback instant."""

    index: int
    time_s: float
    phase: ShotPhase
    ball: BallState
    clubhead: ClubheadPose
    particles: ParticleSnapshot | None = None

    def __post_init__(self) -> None:
        """Validate frame timestamp, phase, and particle timestamp synchronization."""
        idx, stamp = int(self.index), float(self.time_s)
        if idx < 0 or not math.isfinite(stamp) or stamp < 0.0:
            raise BunkerShot3DValueError(f"invalid index={idx} or time_s={stamp!r}")
        if not isinstance(self.ball, BallState) or not isinstance(
            self.clubhead, ClubheadPose
        ):
            raise BunkerShot3DValueError("invalid ball or clubhead state")
        p = self.particles
        if p is not None and (
            not isinstance(p, ParticleSnapshot)
            or not math.isclose(p.time_s, stamp, abs_tol=_SYNC_TOL_S)
        ):
            raise BunkerShot3DValueError("invalid or desynchronized particle snapshot")
        object.__setattr__(self, "index", idx)
        object.__setattr__(self, "time_s", stamp)
        object.__setattr__(self, "phase", ShotPhase(str(self.phase)))


def verify_handoff_continuity(
    outgoing: TimelineFrame | BallState,
    incoming: TimelineFrame | BallState,
    *,
    max_jump_m: float = MAX_HANDOFF_JUMP_M,
) -> float:
    """Enforce ``< max_jump_m`` position continuity across a phase handoff."""
    limit = float(max_jump_m)
    if not math.isfinite(limit) or limit <= 0.0:
        raise BunkerShot3DValueError(f"max_jump_m must be positive, got {limit!r}")
    if isinstance(outgoing, BallState) and isinstance(incoming, BallState):
        jump_m = math.dist(outgoing.position_m, incoming.position_m)
    elif isinstance(outgoing, TimelineFrame) and isinstance(incoming, TimelineFrame):
        dt = incoming.time_s - outgoing.time_s
        if dt < 0.0:
            raise BunkerShot3DValueError("handoff time moved backwards")
        op, ov = outgoing.ball.position_m, outgoing.ball.velocity_m_s
        ip, iv = incoming.ball.position_m, incoming.ball.velocity_m_s
        pred = [op[i] + 0.5 * (ov[i] + iv[i]) * dt for i in range(3)]
        jump_m = math.dist(pred, ip)
    else:
        raise BunkerShot3DValueError("outgoing and incoming must share type")
    if jump_m >= limit:
        raise BunkerShot3DValueError(
            f"handoff jump {jump_m * 1e3:.3f}mm >= {limit * 1e3:.2f}mm"
        )
    return jump_m


def _validate_frames(
    frames: tuple[TimelineFrame, ...], tier: FidelityTier, tol: float
) -> None:
    """Verify monotonic timestamps, tier honesty, and impact-to-flight continuity."""
    if not frames:
        raise BunkerShot3DValueError("timeline must contain at least one frame")
    for idx, f in enumerate(frames):
        if f.index != idx or (idx > 0 and f.time_s <= frames[idx - 1].time_s):
            raise BunkerShot3DValueError(f"invalid index or time ordering at {idx}")
        if tier is FidelityTier.F0 and f.particles is not None:
            raise BunkerShot3DValueError("F0 forbids cosmetic particle spray")
        if tier is FidelityTier.F1 and f.phase in _SAND and f.particles is None:
            raise BunkerShot3DValueError(f"F1 requires particles at sand frame {idx}")
        if idx > 0 and frames[idx - 1].phase in _SAND and f.phase is ShotPhase.FLIGHT:
            verify_handoff_continuity(frames[idx - 1], f, max_jump_m=tol)
    if tier is FidelityTier.F1 and not any(f.particles is not None for f in frames):
        raise BunkerShot3DValueError("F1 requires synchronized ParticleSnapshot arrays")


@dataclass(frozen=True)
class ShotPlaybackTimeline:
    """Read-only, tier-honest 3-D shot playback timeline with SHA-256 provenance."""

    frames: tuple[TimelineFrame, ...]
    fidelity_tier: FidelityTier
    verdict_status: EnvelopeStatus
    verdict_summary: str
    free_surface_height_m: float = 0.0
    handoff_tolerance_m: float = MAX_HANDOFF_JUMP_M
    schema_version: int = TIMELINE_SCHEMA_VERSION
    source_digest: str = ""

    def __post_init__(self) -> None:
        """Validate schema, tier honesty, continuity, and provenance digest."""
        if int(self.schema_version) != TIMELINE_SCHEMA_VERSION:
            raise BunkerShot3DValueError(f"unsupported schema {self.schema_version!r}")
        tier = FidelityTier(str(self.fidelity_tier))
        status = EnvelopeStatus(str(self.verdict_status))
        summary = str(self.verdict_summary).strip()
        surf_z, tol = float(self.free_surface_height_m), float(self.handoff_tolerance_m)
        if not summary or not math.isfinite(surf_z) or not (0.0 < tol < math.inf):
            raise BunkerShot3DValueError("invalid summary, surface, or tolerance")
        frames = tuple(self.frames)
        _validate_frames(frames, tier, tol)
        v: dict[str, Any] = {"frames": frames, "fidelity_tier": tier}
        v["verdict_status"] = status
        v["verdict_summary"] = summary
        v["free_surface_height_m"] = surf_z
        v["handoff_tolerance_m"] = tol
        for k, val in v.items():
            object.__setattr__(self, k, val)
        body = asdict(self)
        body.pop("source_digest", None)
        digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        if not self.source_digest:
            object.__setattr__(self, "source_digest", digest)
        elif self.source_digest != digest or len(self.source_digest) != _HEX_LEN:
            raise BunkerShot3DValueError("timeline source_digest mismatch")

    def frame(self, index: int) -> TimelineFrame:
        """Return the frame at ``index``, failing closed when out of range."""
        idx = int(index)
        if not 0 <= idx < len(self.frames):
            raise BunkerShot3DValueError(f"frame {idx} out of range")
        return self.frames[idx]

    def ball_positions_m(self) -> NDArray[np.float64]:
        """Return a read-only ``(T, 3)`` array of world ball center positions [m]."""
        arr = np.asarray([f.ball.position_m for f in self.frames], dtype=np.float64)
        arr.flags.writeable = False
        return arr

    def to_json(self) -> str:
        """Serialize the timeline to canonical JSON text."""
        return canonical_json(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShotPlaybackTimeline:
        """Rebuild and digest-verify a timeline from a JSON-decoded mapping."""
        if len(str(payload.get("source_digest", ""))) != _HEX_LEN:
            raise BunkerShot3DValueError("missing or malformed source_digest")
        B, C, P, T = BallState, ClubheadPose, ParticleSnapshot, TimelineFrame
        frames = tuple(
            T(
                d["index"],
                d["time_s"],
                d["phase"],
                B(**d["ball"]),
                C(**d["clubhead"]),
                P(**d["particles"]) if d.get("particles") else None,
            )
            for d in payload["frames"]
        )
        return cls(**{**payload, "frames": frames})

    @classmethod
    def from_json(cls, raw: str) -> ShotPlaybackTimeline:
        """Rebuild and digest-verify a timeline from JSON text."""
        return cls.from_dict(json.loads(raw))


def build_shot_playback_timeline(
    *,
    impact_frames: Sequence[TimelineFrame],
    flight_frames: Sequence[TimelineFrame] = (),
    rollout_frames: Sequence[TimelineFrame] = (),
    fidelity_tier: FidelityTier | str,
    verdict_status: EnvelopeStatus | str,
    verdict_summary: str,
    free_surface_height_m: float = 0.0,
    handoff_tolerance_m: float = MAX_HANDOFF_JUMP_M,
) -> ShotPlaybackTimeline:
    """Assemble and verify a continuous read-only 3-D shot playback timeline."""
    if not impact_frames:
        raise BunkerShot3DValueError("impact_frames must be non-empty")
    f0, tol, sz = impact_frames[0], handoff_tolerance_m, free_surface_height_m
    if f0.phase is ShotPhase.ADDRESS:
        b = f0.ball
        exp_z = sz + b.diameter_m / 2.0 - b.burial_depth_m
        if not math.isclose(b.position_m[2], exp_z, abs_tol=1e-9):
            raise BunkerShot3DValueError(f"address z != {exp_z:.6f}")
    combined = list(impact_frames)
    for seg, pos_only in ((flight_frames, False), (rollout_frames, True)):
        if not seg:
            continue
        prev = combined[-1]
        out_o: Any = prev.ball if pos_only else prev
        in_o: Any = seg[0].ball if pos_only else seg[0]
        verify_handoff_continuity(out_o, in_o, max_jump_m=tol)
        s = 1 if math.isclose(seg[0].time_s, prev.time_s, abs_tol=1e-12) else 0
        combined.extend(seg[s:])
    reindexed = tuple(replace(f, index=i) for i, f in enumerate(combined))
    t, st = FidelityTier(str(fidelity_tier)), EnvelopeStatus(str(verdict_status))
    return ShotPlaybackTimeline(reindexed, t, st, verdict_summary, sz, tol)
