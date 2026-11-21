"""Reference swing peak segment KE for fixed shading limits (#15396).

Derived from the tour-average anthro driver coordinate swing on the full-body
MuJoCo model via :class:`MujocoSegmentKineticEnergyProvider`, with despiking
and low-pass filtering before velocity estimation (see module docstring in
``shading.REFERENCE_SWING_PEAK_SEGMENT_KE_J``).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Final

import numpy as np
from scipy.signal import butter, filtfilt, medfilt

# Canonical inputs used by ``compute_reference_swing_peak_segment_ke_j()``.
SWING_Q_DRIVER_FIXTURE: Final[str] = "tests/fixtures/club_face/swing_q_driver.npz"
ADDRESS_POSES_FIXTURE: Final[str] = "tests/fixtures/club_face/address_poses.json"
FULL_BODY_SPEC: Final[str] = (
    "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
SWING_CLUB: Final[str] = "driver"

_EDGE_TRIM_FRAMES: Final[int] = 10
_DESPIKE_KERNEL: Final[int] = 5
_LOWPASS_HZ: Final[float] = 20.0
_WHOLE_BODY_KE_SANITY_J: Final[float] = 3000.0

_REFERENCE_SOURCES_DOC: Final[str] = (
    f"{SWING_Q_DRIVER_FIXTURE} + {FULL_BODY_SPEC} (club {SWING_CLUB!r}); "
    f"q despike (median-{_DESPIKE_KERNEL}), {_LOWPASS_HZ} Hz zero-phase "
    f"low-pass, trim {_EDGE_TRIM_FRAMES} edge frames before qdot"
)


def _repo_root(explicit: Path | None) -> Path:
    return explicit or Path(__file__).resolve().parents[4]


def _preprocess_swing_q(q: np.ndarray, *, dt: float) -> np.ndarray:
    if q.ndim != 2:
        raise ValueError("swing q must be a 2-D array (time x joint)")
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
    if filtered.shape[0] <= 2 * _EDGE_TRIM_FRAMES:
        raise ValueError("swing too short for edge trimming")
    return filtered[_EDGE_TRIM_FRAMES:-_EDGE_TRIM_FRAMES]


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
    """Peak per-body total segment KE (J) after despike/low-pass q preprocessing."""
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
    dt = float(swing.time_s[1] - swing.time_s[0]) if len(swing.time_s) > 1 else 0.001
    q_proc = _preprocess_swing_q(q, dt=dt)
    qvel = np.gradient(q_proc, dt, axis=0, edge_order=2)
    times = np.asarray(swing.time_s, dtype=float)[_EDGE_TRIM_FRAMES:-_EDGE_TRIM_FRAMES]

    provider = _mujoco_engine_provider(spec_path.read_bytes())
    peak = 0.0
    max_system_ke = 0.0
    for i in range(q_proc.shape[0]):
        frame = provider.frame_at(q_proc[i], qvel[i], time_s=float(times[i]))
        max_system_ke = max(max_system_ke, float(frame.system_kinetic_j))
        for seg in frame.segments:
            peak = max(peak, float(seg.total_j))
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

    computed = compute_reference_swing_peak_segment_ke_j()
    if not math.isclose(
        computed, REFERENCE_SWING_PEAK_SEGMENT_KE_J, rel_tol=0, abs_tol=1e-3
    ):
        raise AssertionError(
            f"{REFERENCE_SWING_PEAK_SEGMENT_KE_J=} stale; recomputed {computed} "
            f"from {_REFERENCE_SOURCES_DOC}"
        )


__all__ = [
    "ADDRESS_POSES_FIXTURE",
    "FULL_BODY_SPEC",
    "SWING_CLUB",
    "SWING_Q_DRIVER_FIXTURE",
    "WHOLE_BODY_KE_SANITY_J",
    "assert_reference_constant_matches_swing",
    "compute_reference_swing_peak_segment_ke_j",
]

# Public alias for tests / legend notes
WHOLE_BODY_KE_SANITY_J = _WHOLE_BODY_KE_SANITY_J
