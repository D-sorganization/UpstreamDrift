"""F0/F1 sand-ball momentum coupling, realism targets, and 4-frame strip (#15805)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.core.contracts import ensure, require

from ..exceptions import BunkerShot3DValueError
from ..solvers.envelope import EnvelopeStatus
from ..solvers.protocol import FidelityTier

__all__ = [
    "BunkerRealismEvaluation",
    "FourFrameVerificationStrip",
    "MomentumCouplingReport",
    "REQUIRED_STRIP_PHASES",
    "VerificationStripFrame",
    "build_four_frame_verification_strip",
    "check_sand_ball_momentum_coupling",
    "evaluate_bunker_realism_targets",
]

REQUIRED_STRIP_PHASES = ("address", "sand_entry", "release", "landing")
_VERDICT_RANK = {"within": 0, "extrapolated": 1, "beyond_validation": 2, "refused": 3}
_E = TypeVar("_E", bound=StrEnum)


def _norm_enum(enum_cls: type[_E], value: _E | str, label: str) -> _E:
    """Convert ``value`` into a validated ``enum_cls`` member."""
    try:
        return value if isinstance(value, enum_cls) else enum_cls(value)
    except ValueError as exc:
        raise BunkerShot3DValueError(f"unknown {label}: {value!r}") from exc


def _finite(name: str, value: float, *, minimum: float | None = None) -> float:
    """Return finite float ``value``, rejecting non-finite or below-minimum."""
    scalar = float(value)
    if not math.isfinite(scalar):
        raise BunkerShot3DValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and scalar < minimum:
        raise BunkerShot3DValueError(f"{name} must be >= {minimum}, got {scalar!r}")
    return scalar


def _vec(name: str, value: ArrayLike) -> NDArray[np.float64]:
    """Coerce ``value`` into a finite non-empty 1-D float64 vector."""
    arr = np.asarray(value, dtype=np.float64)
    arr = arr.reshape(1) if arr.ndim == 0 else arr
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise BunkerShot3DValueError(f"{name} must be a finite non-empty 1-D vector")
    return arr


def _resolve_impulses(
    dim: int,
    boundary: ArrayLike | None,
    particles: ArrayLike | None,
    force_n: ArrayLike | None,
    time_s: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Resolve primary integrated impulse and optional cross-check impulse."""
    trace_imp: NDArray[np.float64] | None = None
    if force_n is not None or time_s is not None:
        if force_n is None or time_s is None:
            raise BunkerShot3DValueError("boundary_force_N and time_s must be paired")
        t = _vec("time_s", time_s)
        f = np.asarray(force_n, dtype=np.float64)
        f = f[:, None] if f.ndim == 1 and dim == 1 else f
        bad_f = f.shape != (t.size, dim) or not np.all(np.isfinite(f))
        if t.size < 2 or np.any(np.diff(t) <= 0.0) or bad_f:
            raise BunkerShot3DValueError("invalid boundary_force_N / time_s trace")
        trace_imp = np.asarray(np.trapezoid(f, t, axis=0), dtype=np.float64)
    part_sum: NDArray[np.float64] | None = None
    if particles is not None:
        p = np.asarray(particles, dtype=np.float64)
        if p.ndim == 1:
            part_sum = _vec("sand_particle_impulses", p)
            if part_sum.size != dim:
                raise BunkerShot3DValueError(f"impulse dim {part_sum.size} != {dim}")
        elif (
            p.ndim == 2
            and p.shape[0] > 0
            and p.shape[1] == dim
            and np.all(np.isfinite(p))
        ):
            part_sum = np.asarray(np.sum(p, axis=0), dtype=np.float64)
        else:
            raise BunkerShot3DValueError(f"sand_particle_impulses invalid for {dim}")
    bound = None if boundary is None else _vec("boundary_impulse", boundary)
    if bound is not None and bound.size != dim:
        raise BunkerShot3DValueError(f"boundary impulse dim {bound.size} != {dim}")
    prim = bound if bound is not None else trace_imp
    prim = part_sum if prim is None else prim
    if prim is None:
        raise BunkerShot3DValueError("no sand/boundary impulse source was supplied")
    return prim, (part_sum if bound is not None else trace_imp)


@dataclass(frozen=True)
class MomentumCouplingReport:
    """Audit report comparing F1 ball momentum gain against sand boundary impulse."""

    p_ball_gain_kg_mps: tuple[float, ...]
    integrated_impulse_Ns: tuple[float, ...]
    imbalance_Ns: tuple[float, ...]
    imbalance_magnitude_Ns: float
    reference_magnitude_Ns: float
    relative_imbalance: float
    tolerance_Ns: float
    within_tolerance: bool
    fidelity_tier: FidelityTier = FidelityTier.F1
    verdict_status: EnvelopeStatus = EnvelopeStatus.BEYOND_VALIDATION

    def __post_init__(self) -> None:
        """Validate momentum vector lengths and non-negative scalar metrics."""
        dim = len(self.p_ball_gain_kg_mps)
        bad = len(self.integrated_impulse_Ns) != dim or len(self.imbalance_Ns) != dim
        if dim == 0 or bad:
            raise BunkerShot3DValueError("momentum vectors must share non-zero length")
        for key in ("imbalance_magnitude_Ns", "reference_magnitude_Ns"):
            _finite(key, getattr(self, key), minimum=0.0)
        _finite("relative_imbalance", self.relative_imbalance, minimum=0.0)
        _finite("tolerance_Ns", self.tolerance_Ns, minimum=0.0)


def check_sand_ball_momentum_coupling(
    p_ball_gain: ArrayLike,
    integrated_boundary_impulse: ArrayLike | None = None,
    *,
    sand_particle_impulses: ArrayLike | None = None,
    boundary_force_N: ArrayLike | None = None,
    time_s: ArrayLike | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    fail_closed: bool = True,
) -> MomentumCouplingReport:
    """Audit F1 sand-on-ball momentum coupling against integrated impulse."""
    rt, at = _finite("rtol", rtol, minimum=0.0), _finite("atol", atol, minimum=0.0)
    require(rt >= 0.0 and at >= 0.0, "tolerances must be non-negative")
    ball = _vec("p_ball_gain", p_ball_gain)
    prim, sec = _resolve_impulses(
        ball.size,
        integrated_boundary_impulse,
        sand_particle_impulses,
        boundary_force_N,
        time_s,
    )
    if sec is not None:
        diff = float(np.linalg.norm(prim - sec))
        scale = max(float(np.linalg.norm(prim)), float(np.linalg.norm(sec)))
        if diff > at + rt * scale:
            raise BunkerShot3DValueError(f"sand particle/trace mismatch {diff:.6g} N.s")
    imb = ball - prim
    imb_mag = float(np.linalg.norm(imb))
    ref_mag = max(float(np.linalg.norm(ball)), float(np.linalg.norm(prim)))
    tol_ns = at + rt * ref_mag
    within = imb_mag <= tol_ns
    if fail_closed and not within:
        raise BunkerShot3DValueError(f"momentum imbalance {imb_mag:.6g} > {tol_ns:.6g}")
    rel = imb_mag / ref_mag if ref_mag > 0.0 else 0.0
    b_t, p_t, i_t = (
        tuple(float(v) for v in ball),
        tuple(float(v) for v in prim),
        tuple(float(v) for v in imb),
    )
    report = MomentumCouplingReport(
        b_t, p_t, i_t, imb_mag, ref_mag, rel, tol_ns, within
    )
    ensure(report.imbalance_magnitude_Ns >= 0.0, "imbalance must be non-negative")
    return report


@dataclass(frozen=True)
class BunkerRealismEvaluation:
    """Greenside bunker realism evaluation preserving ADR-0032/0033 status."""

    launch_angle_deg: float
    backspin_rpm: float
    carry_m: float
    min_carry_m: float
    launch_angle_ok: bool
    backspin_ok: bool
    carry_ok: bool
    passed: bool
    verdict_status: EnvelopeStatus
    fidelity_tier: FidelityTier
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate numeric realism fields."""
        _finite("launch_angle_deg", self.launch_angle_deg)
        _finite("backspin_rpm", self.backspin_rpm)
        _finite("carry_m", self.carry_m, minimum=0.0)
        _finite("min_carry_m", self.min_carry_m, minimum=0.0)


def _bounds(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    """Validate an inclusive ``(low, high)`` numeric interval."""
    low, high = _finite(f"{name}[0]", bounds[0]), _finite(f"{name}[1]", bounds[1])
    if low > high:
        raise BunkerShot3DValueError(f"{name} requires low <= high, got {bounds!r}")
    return low, high


def evaluate_bunker_realism_targets(
    launch_angle_deg: float,
    backspin_rpm: float,
    carry_m: float,
    verdict_status: EnvelopeStatus | str,
    *,
    fidelity_tier: FidelityTier | str = FidelityTier.F0,
    min_carry_m: float = 5.0,
    launch_angle_bounds_deg: tuple[float, float] = (30.0, 45.0),
    backspin_bounds_rpm: tuple[float, float] = (4000.0, 8000.0),
) -> BunkerRealismEvaluation:
    """Evaluate angle, backspin, and carry targets while preserving verdict_status."""
    angle = _finite("launch_angle_deg", launch_angle_deg)
    spin = _finite("backspin_rpm", backspin_rpm)
    carry = _finite("carry_m", carry_m, minimum=0.0)
    min_c = _finite("min_carry_m", min_carry_m, minimum=1e-12)
    require(min_c > 0.0, "min_carry_m must be positive")
    ab = _bounds("launch_angle_bounds_deg", launch_angle_bounds_deg)
    sb = _bounds("backspin_bounds_rpm", backspin_bounds_rpm)
    status = _norm_enum(EnvelopeStatus, verdict_status, "verdict_status")
    tier = _norm_enum(FidelityTier, fidelity_tier, "fidelity_tier")
    a_ok, s_ok, c_ok = ab[0] <= angle <= ab[1], sb[0] <= spin <= sb[1], carry >= min_c
    not_ref = status is not EnvelopeStatus.REFUSED
    checks = (
        (a_ok, f"launch_angle_deg {angle:.3g} outside {ab!r}"),
        (s_ok, f"backspin_rpm {spin:.4g} outside {sb!r}"),
        (c_ok, f"carry_m {carry:.3g} < min_carry_m {min_c:.3g} m"),
        (not_ref, "verdict_status is refused"),
    )
    fails = tuple(msg for ok, msg in checks if not ok)
    passed = a_ok and s_ok and c_ok and not_ref
    return BunkerRealismEvaluation(
        angle, spin, carry, min_c, a_ok, s_ok, c_ok, passed, status, tier, fails
    )


@dataclass(frozen=True)
class VerificationStripFrame:
    """Single phase frame with camera extrinsics and tier/verdict stamps."""

    phase: str
    time_s: float
    elevation_deg: float
    azimuth_deg: float
    fidelity_tier: FidelityTier | str
    verdict_status: EnvelopeStatus | str
    frame_uri: str = ""

    def __post_init__(self) -> None:
        """Validate phase identity, timestamp, camera angles, and enums."""
        if self.phase not in REQUIRED_STRIP_PHASES:
            raise BunkerShot3DValueError(f"invalid phase: {self.phase!r}")
        elev = _finite("elevation_deg", self.elevation_deg)
        if not -90.0 <= elev <= 90.0:
            raise BunkerShot3DValueError(f"elevation_deg out of [-90, 90]: {elev!r}")
        object.__setattr__(self, "time_s", _finite("time_s", self.time_s, minimum=0.0))
        object.__setattr__(self, "elevation_deg", elev)
        object.__setattr__(
            self, "azimuth_deg", _finite("azimuth_deg", self.azimuth_deg)
        )
        t_val = _norm_enum(FidelityTier, self.fidelity_tier, "tier")
        v_val = _norm_enum(EnvelopeStatus, self.verdict_status, "verdict")
        object.__setattr__(self, "fidelity_tier", t_val)
        object.__setattr__(self, "verdict_status", v_val)


@dataclass(frozen=True)
class FourFrameVerificationStrip:
    """Matched-camera 4-phase verification strip comparing baseline and candidate."""

    phases: tuple[str, ...]
    baseline_frames: tuple[VerificationStripFrame, ...]
    candidate_frames: tuple[VerificationStripFrame, ...]
    baseline_tier: FidelityTier
    candidate_tier: FidelityTier
    baseline_verdict_status: EnvelopeStatus
    candidate_verdict_status: EnvelopeStatus
    overall_verdict_status: EnvelopeStatus
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Verify canonical 4-phase invariant on constructed strip."""
        bad_len = len(self.baseline_frames) != 4 or len(self.candidate_frames) != 4
        if self.phases != REQUIRED_STRIP_PHASES or bad_len:
            raise BunkerShot3DValueError("strip requires canonical 4 phases per side")


def _index_frames(
    label: str, frames: Sequence[VerificationStripFrame]
) -> tuple[tuple[VerificationStripFrame, ...], FidelityTier, EnvelopeStatus]:
    """Validate 4 unique required phases, monotonic timestamps, and single tier."""
    by_phase: dict[str, VerificationStripFrame] = {}
    for frame in frames:
        if frame.phase in by_phase:
            raise BunkerShot3DValueError(f"{label} duplicate phase {frame.phase!r}")
        by_phase[frame.phase] = frame
    missing = [p for p in REQUIRED_STRIP_PHASES if p not in by_phase]
    if missing:
        raise BunkerShot3DValueError(f"{label} missing phase(s): {missing!r}")
    ordered = tuple(by_phase[p] for p in REQUIRED_STRIP_PHASES)
    for prev, curr in zip(ordered, ordered[1:], strict=False):
        if curr.time_s < prev.time_s:
            raise BunkerShot3DValueError(f"{label} timestamps must be monotonic")
    tiers = {_norm_enum(FidelityTier, f.fidelity_tier, "tier") for f in ordered}
    if len(tiers) != 1:
        raise BunkerShot3DValueError(f"{label} frames must share one fidelity_tier")
    verdicts = [_norm_enum(EnvelopeStatus, f.verdict_status, "v") for f in ordered]
    return (
        ordered,
        next(iter(tiers)),
        max(verdicts, key=lambda v: _VERDICT_RANK[v.value]),
    )


def build_four_frame_verification_strip(
    baseline_frames: Sequence[VerificationStripFrame],
    candidate_frames: Sequence[VerificationStripFrame],
    *,
    camera_atol_deg: float = 1e-6,
) -> FourFrameVerificationStrip:
    """Build a 4-frame matched-camera strip and stamp ADR-0032/0033 metadata."""
    atol = _finite("camera_atol_deg", camera_atol_deg, minimum=0.0)
    b_ord, b_tier, b_ver = _index_frames("baseline", baseline_frames)
    c_ord, c_tier, c_ver = _index_frames("candidate", candidate_frames)
    elevs, azims = [], []
    for bf, cf in zip(b_ord, c_ord, strict=True):
        if abs(bf.elevation_deg - cf.elevation_deg) > atol:
            raise BunkerShot3DValueError(f"elevation_deg mismatch at {bf.phase!r}")
        if abs((bf.azimuth_deg - cf.azimuth_deg + 180.0) % 360.0 - 180.0) > atol:
            raise BunkerShot3DValueError(f"azimuth_deg mismatch at {bf.phase!r}")
        elevs.append(bf.elevation_deg)
        azims.append(bf.azimuth_deg)
    overall = max((b_ver, c_ver), key=lambda v: _VERDICT_RANK[v.value])
    meta = MappingProxyType(
        {
            "phases": REQUIRED_STRIP_PHASES,
            "baseline_tier": b_tier.value,
            "candidate_tier": c_tier.value,
            "baseline_verdict_status": b_ver.value,
            "candidate_verdict_status": c_ver.value,
            "overall_verdict_status": overall.value,
            "camera_elevation_deg": tuple(elevs),
            "camera_azimuth_deg": tuple(azims),
        }
    )
    return FourFrameVerificationStrip(
        REQUIRED_STRIP_PHASES, b_ord, c_ord, b_tier, c_tier, b_ver, c_ver, overall, meta
    )
