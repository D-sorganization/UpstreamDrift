"""Reference swing peak segment KE for fixed shading limits (#15396).

Derived from the tour-average anthro driver coordinate swing on the full-body
MuJoCo model via :class:`MujocoSegmentKineticEnergyProvider`. Joint positions are
despiked and low-pass filtered on **pre- and post-ball-contact segments
separately**, with a ±25-frame gate around ball impact (frames 661–664 at
t ≈ 1.322–1.328 s per ``provenance.json``) so filters do not run across impact.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from scipy.signal import butter, filtfilt, medfilt

# Canonical inputs used by ``compute_reference_swing_peak_segment_ke_j()``.
SWING_Q_DRIVER_FIXTURE: Final[str] = "tests/fixtures/club_face/swing_q_driver.npz"
ADDRESS_POSES_FIXTURE: Final[str] = "tests/fixtures/club_face/address_poses.json"
SWING_PROVENANCE_FIXTURE: Final[str] = "tests/fixtures/club_face/provenance.json"
FULL_BODY_SPEC: Final[str] = (
    "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
SWING_CLUB: Final[str] = "driver"

_EDGE_TRIM_FRAMES: Final[int] = 10
_DESPIKE_KERNEL: Final[int] = 5
_LOWPASS_HZ: Final[float] = 20.0
_IMPACT_EXCLUDE_RADIUS_FRAMES: Final[int] = 25
_WHOLE_BODY_KE_SANITY_J: Final[float] = 3000.0

_REFERENCE_SOURCES_DOC: Final[str] = (
    f"{SWING_Q_DRIVER_FIXTURE} + {FULL_BODY_SPEC} (club {SWING_CLUB!r}); "
    f"impact ±{_IMPACT_EXCLUDE_RADIUS_FRAMES} frames from {SWING_PROVENANCE_FIXTURE}; "
    f"q despike (median-{_DESPIKE_KERNEL}), {_LOWPASS_HZ} Hz zero-phase low-pass "
    f"per pre/post segment; trim {_EDGE_TRIM_FRAMES} edge frames before qdot"
)


@dataclass(frozen=True)
class _ProcessedSwingSegment:
    q: np.ndarray
    time_s: np.ndarray


def _repo_root(explicit: Path | None) -> Path:
    return explicit or Path(__file__).resolve().parents[4]


def _impact_frame_index(times: np.ndarray, repo_root: Path) -> int:
    prov_path = repo_root / SWING_PROVENANCE_FIXTURE
    payload = json.loads(prov_path.read_text(encoding="utf-8"))
    impact_t = float(
        payload["clubs"][SWING_CLUB]["impact_split"]["impact_time_s"]
    )
    times = np.asarray(times, dtype=float)
    index = int(np.argmin(np.abs(times - impact_t)))
    if not math.isfinite(times[index]):
        raise ValueError("impact time does not align with swing samples")
    return index


def _filter_position_segment(q: np.ndarray, *, dt: float) -> np.ndarray:
    if q.ndim != 2:
        raise ValueError("swing q must be a 2-D array (time x joint)")
    if q.shape[0] < 2 * _EDGE_TRIM_FRAMES + _DESPIKE_KERNEL:
        raise ValueError("swing segment too short for filtering")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be a finite positive number")
    despiked = np.apply_along_axis(
        lambda column: medfilt(column, kernel_size=_DESPIKE_KERNEL), 0, q
    )
    sample_hz = 1.0 / dt
    normal_cutoff = _LOWPASS_HZ / (0.5 * sample_hz)
    if not 0.0 < normal_cutoff < 1.0:
        raise ValueError("low-pass cutoff is outside the Nyquist range for this swing")
    b, a = butter(4, normal_cutoff, btype="low")
    filtered = filtfilt(b, a, despiked, axis=0)
    return filtered[_EDGE_TRIM_FRAMES:-_EDGE_TRIM_FRAMES]


def _impact_gated_segments(
    q: np.ndarray,
    times: np.ndarray,
    *,
    dt: float,
    repo_root: Path,
) -> tuple[_ProcessedSwingSegment, ...]:
    impact_index = _impact_frame_index(times, repo_root)
    lo = max(0, impact_index - _IMPACT_EXCLUDE_RADIUS_FRAMES)
    hi = min(q.shape[0], impact_index + _IMPACT_EXCLUDE_RADIUS_FRAMES + 1)
    segments: list[_ProcessedSwingSegment] = []
    for start, slice_end in ((0, lo), (hi, q.shape[0])):
        raw = q[start:slice_end]
        if raw.shape[0] < 2 * _EDGE_TRIM_FRAMES + _DESPIKE_KERNEL:
            continue
        proc = _filter_position_segment(raw, dt=dt)
        offset = start + _EDGE_TRIM_FRAMES
        seg_times = np.asarray(times[offset : offset + proc.shape[0]], dtype=float)
        segments.append(_ProcessedSwingSegment(q=proc, time_s=seg_times))
    if not segments:
        raise ValueError("no swing segments remain after impact gating")
    return tuple(segments)


def _mujoco_engine_provider(spec_bytes: bytes):
    import mujoco

    engine_root = _repo_root(None) / "src/engines/physics_engines/mujoco/python"
    if str(engine_root) not in sys.path:
        sys.path.insert(0, str(engine_root))
    from full_body_mjcf import export_full_body_mjcf
    from mujoco_humanoid_golf.segment_kinetic_energy_provider import (
        MujocoSegmentKineticEnergyProvider,
    )

    model = mujoco.MjModel.from_xml_string(export_full_body_mjcf(spec_bytes)[0])
    return MujocoSegmentKineticEnergyProvider(model)


def compute_reference_swing_peak_segment_ke_j(repo_root: Path | None = None) -> float:
    """Peak per-body total segment KE (J) with impact-gated pre/post q filtering."""
    root = _repo_root(repo_root)
    spec_path = root / FULL_BODY_SPEC
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    names = spec["coordinate_order"]
    from src.shared.python.grip_contact.swing_input import load_coordinate_swing

    swing = load_coordinate_swing(
        root / SWING_Q_DRIVER_FIXTURE,
        root / ADDRESS_POSES_FIXTURE,
        SWING_CLUB,
        names,
    )
    q = np.asarray(swing.q, dtype=float)
    times = np.asarray(swing.time_s, dtype=float)
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.001
    segments = _impact_gated_segments(q, times, dt=dt, repo_root=root)

    provider = _mujoco_engine_provider(spec_path.read_bytes())
    peak = 0.0
    max_system_ke = 0.0
    for segment in segments:
        qvel = np.gradient(segment.q, dt, axis=0, edge_order=2)
        for i in range(segment.q.shape[0]):
            frame = provider.frame_at(
                segment.q[i], qvel[i], time_s=float(segment.time_s[i])
            )
            max_system_ke = max(max_system_ke, float(frame.system_kinetic_j))
            for sample in frame.segments:
                peak = max(peak, float(sample.total_j))
    if not math.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("reference swing peak KE must be finite and positive")
    if max_system_ke >= _WHOLE_BODY_KE_SANITY_J:
        raise RuntimeError(
            f"whole-body KE {max_system_ke:.1f} J exceeds sanity "
            f"{_WHOLE_BODY_KE_SANITY_J:.0f} J after preprocessing"
        )
    return peak


def assert_reference_constant_matches_swing() -> None:
    from .shading import REFERENCE_SWING_PEAK_SEGMENT_KE_J

    measured = compute_reference_swing_peak_segment_ke_j()
    if measured > REFERENCE_SWING_PEAK_SEGMENT_KE_J + 1e-3:
        raise AssertionError(
            f"impact-aware measured peak {measured} J exceeds nominal shading "
            f"default {REFERENCE_SWING_PEAK_SEGMENT_KE_J=} J "
            f"from {_REFERENCE_SOURCES_DOC}"
        )
    if measured <= 0.0:
        raise AssertionError("measured reference swing peak must be positive")


__all__ = [
    "ADDRESS_POSES_FIXTURE",
    "FULL_BODY_SPEC",
    "SWING_CLUB",
    "SWING_PROVENANCE_FIXTURE",
    "SWING_Q_DRIVER_FIXTURE",
    "WHOLE_BODY_KE_SANITY_J",
    "assert_reference_constant_matches_swing",
    "compute_reference_swing_peak_segment_ke_j",
]

WHOLE_BODY_KE_SANITY_J = _WHOLE_BODY_KE_SANITY_J
IMPACT_EXCLUDE_RADIUS_FRAMES = _IMPACT_EXCLUDE_RADIUS_FRAMES