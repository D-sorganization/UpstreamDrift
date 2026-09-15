"""Receipt creation and assembly for full-body pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_markers import (
    FullBodyMarkerKinematics,
)
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.pipeline.address import posture_summary
from src.shared.python.motion_matching.pipeline.constants import (
    IK_UNBOUNDED,
    RATE_HZ,
    STANCE_TOLERANCE_M,
    TOE_SPHERES,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.pipeline.lane import Lane


def build_ground_support_receipt(
    *,
    base_spec: dict[str, Any],
    spec_path: Path,
    scaled_path: Path,
    hipcal_path: Path,
    recalibrate_upper: bool,
    anthropometric: tuple[float, float] | None,
    qualification_note: str,
    spec_bytes: bytes,
    hip_report: dict[str, Any],
    candidate_bytes: bytes,
    c3d_path: Path,
    capture_name: str,
    lane: Lane,
    address_report: dict[str, Any],
    ik_report: dict[str, Any],
    dynamics_report: dict[str, Any],
    kin: FullBodyMarkerKinematics,
    q_ref: np.ndarray,
    elapsed_s: float,
) -> dict[str, Any]:
    """Assemble the standardized receipt for ground-supported full-body pipeline.

    Args:
        base_spec: Input base specification dictionary.
        spec_path: Path to input specification file.
        scaled_path: Path to scaled specification output file.
        hipcal_path: Path to hip-calibrated specification output file.
        recalibrate_upper: Whether upper body was recalibrated.
        anthropometric: Optional (stature_m, mass_kg) tuple.
        qualification_note: Description note for qualification milestone.
        spec_bytes: Raw bytes of final scaled spec.
        hip_report: Summary dictionary from hip calibration stage.
        candidate_bytes: Raw bytes of candidate file.
        c3d_path: Path to C3D capture file.
        capture_name: Label of capture ("driver", "iron", etc.).
        lane: Coordination lane providing ground, stance, labels, and bounds.
        address_report: Summary dictionary from address stage.
        ik_report: Summary dictionary from IK stage.
        dynamics_report: Summary dictionary from forward dynamics stage.
        kin: Final marker kinematics model.
        q_ref: (N, nq) reference trajectory array.
        elapsed_s: Total elapsed execution time in seconds.

    Returns:
        Structured receipt dictionary matching GS milestone schema.
    """
    stance_fraction = {
        name: float(np.mean([name in s for s in lane.stance]))
        for name in ("heel_r", "forefoot_r", "toe_r", "heel_l", "forefoot_l", "toe_l")
    }

    # Top of backswing posture at ~0.83 s
    tob_frame = min(int(round(0.83 * RATE_HZ)), len(q_ref) - 1)
    tob_posture = posture_summary(kin, q_ref[tob_frame])

    return {
        "base_spec_sha256": canonical_sha256(base_spec),
        "base_spec_file": spec_path.name,
        "spec_file": scaled_path.name,
        "hipcal_spec_file": hipcal_path.name,
        "recalibrate_upper": bool(recalibrate_upper),
        "anthropometric": list(anthropometric) if anthropometric else None,
        "posture_top_of_backswing": tob_posture,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "hip_calibration": hip_report,
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "capture": capture_name,
        "capture_sha256": hashlib.sha256(c3d_path.read_bytes()).hexdigest(),
        "club": base_spec.get("club"),
        "grip_rotation_deg": base_spec.get("subject", {}).get("grip_rotation_deg"),
        "wrists_bounded": any(name in lane.bounds for name in IK_UNBOUNDED),
        "labels": list(lane.labels),
        "ground": {
            "height_m": lane.ground_cal.height_m,
            "lowest_toe_marker_m": lane.ground_cal.lowest_marker_height_m,
            "standoff_m": lane.ground_cal.standoff_m,
            "policy": lane.ground_cal.policy,
            "stance_tolerance_m": STANCE_TOLERANCE_M,
            "stance_rule": "heights relative to address; both feet flat at frame 0",
            "toe_spheres": {
                k: {"body": b, "position_m": list(p), "radius_m": r}
                for k, (b, p, r) in TOE_SPHERES.items()
            },
            "stance_fraction": stance_fraction,
        },
        "address": address_report,
        "ik": ik_report,
        "dynamics": dynamics_report,
        "elapsed_s": elapsed_s,
        "qualification": (
            "kinematic IK and computed-torque tracking milestone on the MuJoCo "
            f"full-body model ({qualification_note}); not a fit, not acceptance"
        ),
    }
