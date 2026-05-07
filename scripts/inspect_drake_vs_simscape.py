"""Pretty-print Drake vs Simscape grip-trajectory residuals (issue #4123).

Companion debug tool for ``tests/test_drake_simscape_equivalence.py``.
Runs the same Drake float-pathway forward-sim (theta=0 + default initial
pose) used by the test and prints per-pose, per-axis residuals against
the Simscape reference CSV. Useful when the equivalence test fails and
you need to triage *which* pose / axis is contributing the error.

Usage:

    python3 scripts/inspect_drake_vs_simscape.py
    python3 scripts/inspect_drake_vs_simscape.py --csv path/to/trial_xxx.csv
    python3 scripts/inspect_drake_vs_simscape.py --simulation-time 0.5

The script exits 0 even on residual gate violations — its purpose is to
*report* the divergence, not enforce it. The test in
``tests/test_drake_simscape_equivalence.py`` is the enforcement gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Reuse the loader + math primitives from the test module so the inspect
# script and the gate stay numerically identical.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_drake_simscape_equivalence import (  # noqa: E402
    ORIENT_GATE_RAD,
    RMSE_POSITION_GATE_M,
    SIMSCAPE_CSV,
    _load_simscape_poses,
    _quaternion_geodesic_angle_rad,
)


def _format_row(
    pose_name: str,
    t: float,
    drake_pos: np.ndarray,
    simscape_pos: np.ndarray,
    rmse_mm: float,
    angle_deg: float,
    pos_pass: bool,
    ori_pass: bool,
) -> str:
    pos_marker = "PASS" if pos_pass else "FAIL"
    ori_marker = "PASS" if ori_pass else "FAIL"
    delta = drake_pos - simscape_pos
    return (
        f"  {pose_name:<18s} t={t:6.3f}s  "
        f"RMSE={rmse_mm:7.2f} mm [{pos_marker}]   "
        f"angle={angle_deg:7.3f} deg [{ori_marker}]\n"
        f"    delta_xyz_mm = "
        f"({delta[0] * 1e3:+8.2f}, {delta[1] * 1e3:+8.2f}, "
        f"{delta[2] * 1e3:+8.2f})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=SIMSCAPE_CSV,
        help=f"Simscape ground-truth CSV (default: {SIMSCAPE_CSV})",
    )
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=0.30,
        help="Drake simulation duration in seconds (default 0.30)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1000.0,
        help="Drake output sample rate in Hz (default 1000)",
    )
    args = parser.parse_args(argv)

    print("=" * 72)
    print("Drake vs Simscape grip-trajectory equivalence  (issue #4123)")
    print("=" * 72)
    print(f"Simscape CSV: {args.csv}")
    print(f"Position gate: {RMSE_POSITION_GATE_M * 1e3:.2f} mm")
    print(f"Orientation gate: {np.rad2deg(ORIENT_GATE_RAD):.2f} deg")
    print()

    # Load Simscape poses up-front (cheap; no Drake needed).
    poses = _load_simscape_poses(args.csv)
    print(f"Loaded {len(poses)} Simscape poses: {[p.name for p in poses]}")
    print()

    # Try to import the Drake forward-sim wrapper.
    try:
        from src.engines.physics_engines.drake.python.motion_matching import (
            simulate as drake_sim,
        )
    except ImportError as exc:
        print(f"[skip] Drake simulate module not available: {exc}")
        print(
            "       This module ships in PR #4169; on a fresh main checkout "
            "it is expected to be missing.\n"
        )
        # Still report Simscape-side numbers so the script is useful pre-merge.
        for p in poses:
            print(
                f"  {p.name:<18s} t={p.t:6.3f}s  butt={p.butt.tolist()}  "
                f"quat={p.grip_quat.tolist()}"
            )
        return 0

    options = drake_sim.SimOptions(
        simulation_time_s=args.simulation_time,
        sample_rate_hz=args.sample_rate,
        time_step_s=1.0e-3,
    )
    n_actuators_max = 64
    theta = np.zeros(n_actuators_max * drake_sim.COEFFS_PER_JOINT, dtype=np.float64)
    print(
        f"Running Drake simulate_with_coefficients(theta=zeros, "
        f"sim_time={args.simulation_time}s, rate={args.sample_rate} Hz) ..."
    )
    out = drake_sim.simulate_with_coefficients(theta, options=options)
    print(f"  solver_status={out.solver_status}  duration={out.duration_s:.3f}s")
    print(f"  metadata={out.metadata}")
    print()

    print("Per-pose residuals:")
    any_fail = False
    for pose in poses:
        j = int(np.argmin(np.abs(out.time - pose.t)))
        drake_pos = np.asarray(out.grip[j], dtype=np.float64)
        drake_quat = np.asarray(out.grip_quat[j], dtype=np.float64)

        if not np.all(np.isfinite(drake_pos)):
            print(
                f"  {pose.name:<18s} t={pose.t:6.3f}s  "
                f"Drake grip is non-finite (URDF body 'club_grip' missing?)"
            )
            any_fail = True
            continue

        delta = drake_pos - pose.butt
        rmse_m = float(np.sqrt(np.mean(delta**2)))
        rmse_mm = rmse_m * 1e3
        if np.all(np.isfinite(drake_quat)) and np.linalg.norm(drake_quat) > 0.0:
            angle_rad = _quaternion_geodesic_angle_rad(drake_quat, pose.grip_quat)
        else:
            angle_rad = float("nan")

        pos_pass = rmse_m < RMSE_POSITION_GATE_M
        ori_pass = np.isfinite(angle_rad) and angle_rad < ORIENT_GATE_RAD
        if not (pos_pass and ori_pass):
            any_fail = True

        print(
            _format_row(
                pose.name,
                pose.t,
                drake_pos,
                pose.butt,
                rmse_mm,
                np.rad2deg(angle_rad) if np.isfinite(angle_rad) else float("nan"),
                pos_pass,
                ori_pass,
            )
        )

    print()
    if any_fail:
        print(
            "[divergence] At least one pose failed the equivalence gate. "
            "See tests/test_drake_simscape_equivalence.py for the enforcement test "
            "and the PR body / linked issues for tracked follow-ups."
        )
    else:
        print("[ok] All three poses pass the equivalence gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
