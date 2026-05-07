"""Inspect Pinocchio FK vs the numpy reference chain (issue #4136).

Pretty-prints the residuals between ``pin.framesForwardKinematics`` on
``golfer.urdf`` and the independently-implemented numpy spine-chain FK
in ``tests/unit/engines/pinocchio/_kinematic_equivalence_data.py``, at
the three reference poses (address, top-of-backswing, impact).

Designed so future agents debugging the kinematic-equivalence audit do
not have to rerun the full pytest suite. Falls back gracefully when
pinocchio is not installed: the numpy column always populates, the
pinocchio column is rendered as ``-`` and a header note explains the
skip.

Usage
-----
    python3 scripts/inspect_pinocchio_vs_simscape_poses.py

Optional flags
--------------
    --pose ADDRESS|TOP_OF_BACKSWING|IMPACT  - print only one pose
    --json                                  - emit machine-readable JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.unit.engines.pinocchio._kinematic_equivalence_data import (  # noqa: E402
    GOLFER_URDF,
    GRIP_ORIENTATION_TOL_RAD,
    GRIP_POSITION_RMSE_TOL_M,
    REFERENCE_POSES,
    SpineConfig,
    geodesic_angle,
    load_simscape_address_row,
    numpy_spine_fk,
    position_rmse,
)


def _try_pinocchio_fk(cfg: SpineConfig) -> dict[str, np.ndarray] | None:
    """Run pinocchio FK on the URDF for ``cfg``; return None on skip."""
    try:
        import pinocchio as pin
    except ImportError:
        return None
    if not GOLFER_URDF.exists():
        return None
    model = pin.buildModelFromUrdf(str(GOLFER_URDF))
    data = model.createData()

    q = pin.neutral(model)
    spine_joint_angles = {
        "pelvis_to_lumbar1_intermediate": cfg.lumbar1_x,
        "lumbar1_intermediate_to_lumbar1": cfg.lumbar1_y,
        "lumbar1_to_lumbar2_intermediate": cfg.lumbar2_x,
        "lumbar2_intermediate_to_lumbar2": cfg.lumbar2_y,
        "lumbar2_to_lumbar3_intermediate": cfg.lumbar3_x,
        "lumbar3_intermediate_to_lumbar3": cfg.lumbar3_y,
        "lumbar3_to_thorax1": cfg.thorax1_z,
        "thorax1_to_thorax2": cfg.thorax2_z,
        "thorax2_to_thorax3": cfg.thorax3_z,
    }
    for joint_name, angle in spine_joint_angles.items():
        if not model.existJointName(joint_name):
            return None
        joint_id = model.getJointId(joint_name)
        q[model.idx_qs[joint_id]] = angle

    pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)

    out: dict[str, np.ndarray] = {}
    for frame_name in ("mid_hands", "club_head"):
        if not model.existFrame(frame_name):
            continue
        T = data.oMf[model.getFrameId(frame_name)]
        H = np.eye(4)
        H[:3, :3] = T.rotation
        H[:3, 3] = T.translation
        out[frame_name] = H
    return out


def _format_residuals(
    pose_name: str,
    frame_name: str,
    pin_T: np.ndarray | None,
    np_T: np.ndarray,
) -> dict[str, object]:
    p_np = np_T[:3, 3]
    if pin_T is None:
        return {
            "pose": pose_name,
            "frame": frame_name,
            "numpy_position_m": p_np.tolist(),
            "pinocchio_position_m": None,
            "position_rmse_mm": None,
            "orientation_geodesic_deg": None,
            "within_tolerance": None,
        }
    p_pin = pin_T[:3, 3]
    rmse_m = position_rmse(p_pin, p_np)
    ori_rad = geodesic_angle(pin_T[:3, :3], np_T[:3, :3])
    return {
        "pose": pose_name,
        "frame": frame_name,
        "numpy_position_m": p_np.tolist(),
        "pinocchio_position_m": p_pin.tolist(),
        "position_rmse_mm": rmse_m * 1e3,
        "orientation_geodesic_deg": float(np.rad2deg(ori_rad)),
        "within_tolerance": bool(
            rmse_m < GRIP_POSITION_RMSE_TOL_M and ori_rad < GRIP_ORIENTATION_TOL_RAD
        ),
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'pose':18s} {'frame':12s} "
        f"{'pos RMSE [mm]':>14s} {'ori err [deg]':>14s} {'within tol?':>12s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        rmse = r["position_rmse_mm"]
        ori = r["orientation_geodesic_deg"]
        within = r["within_tolerance"]
        rmse_s = f"{rmse:14.4f}" if rmse is not None else f"{'-':>14s}"
        ori_s = f"{ori:14.4f}" if ori is not None else f"{'-':>14s}"
        within_s = "PASS" if within is True else ("FAIL" if within is False else "SKIP")
        print(f"{r['pose']:18s} {r['frame']:12s} {rmse_s} {ori_s} {within_s:>12s}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pose",
        choices=("address", "top_of_backswing", "impact"),
        default=None,
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    poses = [c for c in REFERENCE_POSES if args.pose is None or c.name == args.pose]

    rows: list[dict[str, object]] = []
    pin_available = True
    for cfg in poses:
        np_frames = numpy_spine_fk(cfg)
        pin_frames = _try_pinocchio_fk(cfg)
        if pin_frames is None:
            pin_available = False
        for frame_name in ("mid_hands", "club_head"):
            pin_T = pin_frames.get(frame_name) if pin_frames else None
            np_T = np_frames[frame_name]
            rows.append(_format_residuals(cfg.name, frame_name, pin_T, np_T))

    if args.json:
        print(
            json.dumps({"rows": rows, "pinocchio_available": pin_available}, indent=2)
        )
        return 0

    print("Pinocchio kinematic-equivalence audit (issue #4136)")
    print(f"  URDF: {GOLFER_URDF}")
    print(
        f"  spec tolerances: pos < {GRIP_POSITION_RMSE_TOL_M * 1e3:.1f} mm, "
        f"ori < {np.rad2deg(GRIP_ORIENTATION_TOL_RAD):.2f} deg"
    )
    if not pin_available:
        print(
            "  pinocchio not available - showing numpy reference only "
            "(install pinocchio to populate the residuals)"
        )
    print()
    _print_table(rows)

    # Footer: Simscape ground-truth address row, if available.
    sim = load_simscape_address_row()
    if sim:
        ch = (
            sim.get("ClubLogs_CHGlobalPosition_1", float("nan")),
            sim.get("ClubLogs_CHGlobalPosition_2", float("nan")),
            sim.get("ClubLogs_CHGlobalPosition_3", float("nan")),
        )
        mh = (
            sim.get("MidpointCalcsLogs_MPGlobalPosition_1", float("nan")),
            sim.get("MidpointCalcsLogs_MPGlobalPosition_2", float("nan")),
            sim.get("MidpointCalcsLogs_MPGlobalPosition_3", float("nan")),
        )
        print()
        print(
            "Simscape address-row reference (CSV row 0): "
            f"mid_hands={mh}  club_head={ch}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
