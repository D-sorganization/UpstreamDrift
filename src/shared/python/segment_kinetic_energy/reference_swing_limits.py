"""Reference swing peak segment KE for fixed shading limits (#15396).

Derived from the tour-average anthro driver coordinate swing on the full-body
MuJoCo model via :class:`MujocoSegmentKineticEnergyProvider`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Final

import numpy as np

from .shading import REFERENCE_SWING_PEAK_SEGMENT_KE_J

# Canonical inputs used by ``compute_reference_swing_peak_segment_ke_j()``.
SWING_Q_DRIVER_FIXTURE: Final[str] = "tests/fixtures/club_face/swing_q_driver.npz"
ADDRESS_POSES_FIXTURE: Final[str] = "tests/fixtures/club_face/address_poses.json"
FULL_BODY_SPEC: Final[str] = (
    "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
SWING_CLUB: Final[str] = "driver"

_REFERENCE_SOURCES_DOC: Final[str] = (
    f"{SWING_Q_DRIVER_FIXTURE} + {FULL_BODY_SPEC} (club {SWING_CLUB!r})"
)


def compute_reference_swing_peak_segment_ke_j(repo_root: Path | None = None) -> float:
    """Peak per-body total segment KE (J) across the recorded driver swing."""
    root = repo_root or Path(__file__).resolve().parents[4]
    import mujoco

    spec_path = root / FULL_BODY_SPEC
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    names = spec["coordinate_order"]
    from src.shared.python.grip_contact.swing_input import load_coordinate_swing
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    engine_root = root / "src/engines/physics_engines/mujoco/python"
    if str(engine_root) not in sys.path:
        sys.path.insert(0, str(engine_root))
    from mujoco_humanoid_golf.segment_kinetic_energy_provider import (
        MujocoSegmentKineticEnergyProvider,
    )

    swing = load_coordinate_swing(
        root / SWING_Q_DRIVER_FIXTURE,
        root / ADDRESS_POSES_FIXTURE,
        SWING_CLUB,
        names,
    )
    xml, _ = export_full_body_mjcf(spec_path.read_bytes())
    model = mujoco.MjModel.from_xml_string(xml)
    provider = MujocoSegmentKineticEnergyProvider(model)
    dt = float(swing.time_s[1] - swing.time_s[0]) if len(swing.time_s) > 1 else 0.001
    qvel = np.gradient(swing.q, dt, axis=0, edge_order=2)
    peak = 0.0
    for i in range(len(swing.time_s)):
        frame = provider.frame_at(swing.q[i], qvel[i], time_s=float(swing.time_s[i]))
        for seg in frame.segments:
            peak = max(peak, float(seg.total_j))
    if not math.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("reference swing peak KE must be finite and positive")
    return peak


def assert_reference_constant_matches_swing() -> None:
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
    "assert_reference_constant_matches_swing",
    "compute_reference_swing_peak_segment_ke_j",
]
