"""OS-9: Static address-hold marker re-registration and continuous IK verification.

Addresses UpstreamDrift issue #13434:
Re-places the 14 trunk markers on pelvis and torso using the held address quiet
segment from tour_average_tracked.trc under declared static posture constraints:
- lumbar_extension = lumbar_bending = lumbar_rotation = 0 (neutral spine).
- Pelvis orientation from waist-cluster anatomical axes.
- Non-trunk coordinates from IK on non-trunk markers only.
- 14 trunk markers re-placed as mean quiet observed position in parent body.

Produces:
- evidence/os9_static_address_registration/golf_humanoid_scaled_tour_markers_address_registered.osim
- evidence/os9_static_address_registration/receipt.json
- evidence/os9_static_address_registration/ik_receipt.json
- evidence/os9_static_address_registration/ik_states_continuous.sto
- evidence/os9_static_address_registration/marker_rms.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from docs.development.opensim_tour_matching.os8_continuous_ik import (  # noqa: E402
    OsimKinematics,
)
from src.engines.physics_engines.opensim.python.tour_matching.static_address_registration import (  # noqa: E402
    StaticRegistrationResult,
    register_static_trunk_markers,
    write_registered_model,
)
from src.shared.python.tour_baselines.events import (  # noqa: E402
    SWING_EVENTS_DRIVER,
)

EVIDENCE = ROOT / "docs/development/opensim_tour_matching/evidence"
DEFAULT_MODEL = EVIDENCE / "os7_moco_g1/inputs/golf_humanoid_scaled_tour_markers.osim"
DEFAULT_TRC = EVIDENCE / "tour_average_tracked.trc"
DEFAULT_OUTPUT = EVIDENCE / "os9_static_address_registration"
OS8_STATES = EVIDENCE / "os8_continuous_ik/ik_states_continuous.sto"
LOGGER = logging.getLogger("os9_static_address_registration")
ISSUE_13434_ADDRESS_FRAME = 76

DECLARED_ASSUMPTIONS: list[str] = [
    "Spinal posture during the quiet address segment is the model neutral (lumbar_extension = lumbar_bending = lumbar_rotation = 0).",
    "Waist belt marker cluster anatomical axes coincide with the model pelvis body frame (x anterior, y superior, z right = -l).",
    "Non-trunk markers are sufficient to locate pelvis translation and limb joint angles during the quiet address hold.",
    "Trunk marker positions in parent body frames are stationary during the quiet segment.",
    "Torso-in-pelvis rotation magnitude at address is ~0 deg BY CONSTRUCTION and is not independent evidence; the independent checks are mirror symmetry, whole-swing marker RMS, and top/impact rotation values.",
]


def sha256_file(path: Path) -> str:
    """Compute sha256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_torso_pelvis_rotations(sto_path: Path, osim_path: Path) -> dict[str, float]:
    """Compute 3D torso-in-pelvis rotation magnitude (deg) at address, top, and impact."""
    kin = OsimKinematics(osim_path)
    lines = sto_path.read_text().splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith(("time\t", "time ")):
            header_idx = i
            break
    if header_idx == -1:
        raise ValueError("Could not find column header row in STO file")

    headers = lines[header_idx].split("\t")[1:]
    col_names = [h.split("/")[-2] for h in headers]

    # Events from SWING_EVENTS_DRIVER
    event_frames = {
        "address": SWING_EVENTS_DRIVER.address.frame_index,
        "top": SWING_EVENTS_DRIVER.top_of_backswing.frame_index,
        "impact": SWING_EVENTS_DRIVER.impact.frame_index,
        # The #13434 diagnosis table sampled address at 0.211 s (frame 76),
        # after the quiet segment, so it is not fixed by construction.
        "address_0p211s": ISSUE_13434_ADDRESS_FRAME,
    }

    out = {}
    for event_name, f_idx in event_frames.items():
        row_str = lines[header_idx + 1 + f_idx]
        q_row = np.array([float(x) for x in row_str.split("\t")[1:]])
        q_ordered = np.array([q_row[col_names.index(n)] for n in kin.names])
        poses = kin.poses(q_ordered)
        r_pelvis = poses["pelvis"][0]
        r_torso = poses["torso"][0]
        r_rel = r_pelvis.T @ r_torso
        angle_deg = float(np.degrees(Rotation.from_matrix(r_rel).magnitude()))
        out[event_name] = angle_deg

    return out


def _run_continuous_ik(
    osim_path: Path, trc_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Run OS-8 continuous IK on the registered model and return receipt data."""
    cmd = [
        sys.executable,
        str(ROOT / "docs/development/opensim_tour_matching/os8_continuous_ik.py"),
        "--osim",
        str(osim_path),
        "--trc",
        str(trc_path),
        "--output",
        str(output_dir),
        "--last",
        "529",
        "--judge-last",
        "475",
    ]
    LOGGER.info("Running OS-8 continuous IK: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    LOGGER.info("Continuous IK stdout:\n%s", res.stdout)

    # Rename receipt.json to ik_receipt.json so it doesn't collide with OS-9 receipt
    raw_receipt_p = output_dir / "receipt.json"
    ik_receipt_p = output_dir / "ik_receipt.json"
    raw_data = json.loads(raw_receipt_p.read_text())
    ik_receipt_p.write_text(json.dumps(raw_data, indent=2) + "\n")
    raw_receipt_p.unlink()
    return raw_data


def run_registration_pipeline(
    model_path: Path,
    trc_path: Path,
    output_dir: Path,
    speed_threshold_mps: float = 0.05,
    min_duration_s: float = 0.05,
) -> Path:
    """Execute static registration, write registered model and receipt, and re-run IK."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Registering static trunk markers from %s...", trc_path.name)

    result: StaticRegistrationResult = register_static_trunk_markers(
        model_path=model_path,
        trc_path=trc_path,
        speed_threshold_mps=speed_threshold_mps,
        min_duration_s=min_duration_s,
    )

    reg_osim_path = (
        output_dir / "golf_humanoid_scaled_tour_markers_address_registered.osim"
    )
    write_registered_model(model_path, result.new_offsets, reg_osim_path)
    LOGGER.info("Saved registered model to %s", reg_osim_path)

    # Re-run OS-8 continuous IK
    ik_receipt = _run_continuous_ik(reg_osim_path, trc_path, output_dir)
    sto_path = output_dir / "ik_states_continuous.sto"

    # Compute torso-in-pelvis rotations
    torso_rotations = compute_torso_pelvis_rotations(sto_path, reg_osim_path)
    LOGGER.info("Torso-in-pelvis rotations (deg): %s", torso_rotations)
    baseline_rotations = compute_torso_pelvis_rotations(OS8_STATES, model_path)
    LOGGER.info("OS-8 baseline rotations (deg): %s", baseline_rotations)

    # Build marker offset comparison table
    marker_offsets_summary = {}
    for name, (body, new_loc) in result.new_offsets.items():
        old_body, old_loc = result.old_offsets[name]
        marker_offsets_summary[name] = {
            "parent_body": body,
            "old_offset_m": list(old_loc),
            "new_offset_m": list(new_loc),
            "displacement_mm": result.displacements_mm.get(name, 0.0),
            "status": (
                "re-registered"
                if name in result.re_registered_markers
                else "not re-registered"
            ),
        }

    receipt = {
        "schema": "os9-static-address-registration/v1",
        "issue": "#13434",
        "qualification": "kinematics only; static-registration assumption declared, not validated",
        "inputs": {
            "model": model_path.name,
            "model_sha256": sha256_file(model_path),
            "trc": trc_path.name,
            "trc_sha256": sha256_file(trc_path),
        },
        "declared_assumptions": DECLARED_ASSUMPTIONS,
        "quiet_segment": {
            "first_frame": int(result.quiet_frames[0]),
            "last_frame": int(result.quiet_frames[-1]),
            "count": len(result.quiet_frames),
            "duration_s": result.quiet_duration_s,
            "threshold_mps": result.speed_threshold_mps,
            "time_range_s": list(result.quiet_time_range_s),
        },
        "pelvis_orientation_solve": {
            "fit_residual_rad": result.pelvis_fit_residual_rad,
            "coordinates_rad": result.pelvis_rotation_coords,
        },
        "quiet_non_trunk_marker_rms_m": result.non_trunk_marker_rms_m,
        "marker_offsets": marker_offsets_summary,
        "mirror_symmetry": {
            "before": result.symmetry_before,
            "after": result.symmetry_after,
        },
        "rerun_continuous_ik": {
            "whole_swing_mean_marker_rms_m": ik_receipt["marker_rms_m"]["mean"],
            "whole_swing_max_marker_rms_m": ik_receipt["marker_rms_m"]["max"],
            "baseline_os8_mean_marker_rms_m": json.loads(
                (EVIDENCE / "os8_continuous_ik/receipt.json").read_text()
            )["marker_rms_m"]["mean"],
            "torso_in_pelvis_rotation_deg": {
                "address": torso_rotations["address"],
                "top": torso_rotations["top"],
                "impact": torso_rotations["impact"],
                "address_0p211s": torso_rotations["address_0p211s"],
                "baseline_os8_deg": baseline_rotations,
                "address_note": (
                    "Torso-in-pelvis rotation at address is ~0 deg BY CONSTRUCTION and is not "
                    "evidence; the independent checks are mirror symmetry, whole-swing RMS, and top/impact values."
                ),
            },
            "largest_coordinate_step": ik_receipt["continuity"]["largest_steps"][0],
            "judged_largest_step": ik_receipt["continuity"]["judged_largest_steps"][0],
            "continuous": ik_receipt["continuity"]["continuous"],
        },
        "outputs": {
            "model": reg_osim_path.name,
            "model_sha256": sha256_file(reg_osim_path),
            "states": sto_path.name,
            "states_sha256": sha256_file(sto_path),
            "marker_rms": "marker_rms.npz",
            "ik_receipt": "ik_receipt.json",
        },
    }

    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    LOGGER.info("Wrote registration receipt to %s", receipt_path)
    return receipt_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osim", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--trc", type=Path, default=DEFAULT_TRC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-duration", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    run_registration_pipeline(
        model_path=args.osim,
        trc_path=args.trc,
        output_dir=args.output,
        speed_threshold_mps=args.threshold,
        min_duration_s=args.min_duration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
