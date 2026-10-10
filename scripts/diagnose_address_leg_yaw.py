"""Leg-chain yaw budget at the calibrated address (#12109).

Runs the pipeline through the calibrated address and writes, for each leg,
the world heading (yaw about +z, degrees) of the knee flexion axis and the
foot long axis, measured twice: on the capture markers and on the model's
attached markers at the solved address. A hip-rotation coordinate pinned at
its range limit is explained by whichever link of the chain (pelvis, thigh,
shank-to-foot) carries the heading difference.

Usage::

    python3 -m scripts.diagnose_address_leg_yaw --capture iron \
        --spec docs/development/full_body_models/full_body_spec_anthro_iron7.json \
        --out runs/legyaw_iron [--engine mujoco] [-- <extra pipeline args>]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.motion_matching.hip_calibration import (
    LATERAL_SEEDS_M,
    knee_flexion_axis,
)
from src.shared.python.motion_matching.pipeline import cli

PELVIS_FRAME = "Hip"  # the spec's pelvis frame (Simscape LowerTorso)
LEG_BODIES = (PELVIS_FRAME, "femur_r", "femur_l", "tibia_r", "tibia_l")
SIDES = (("right", "R", 0), ("left", "L", 1))


def heading_deg(vector: Sequence[float]) -> float:
    """Heading (deg) of ``vector`` projected on the ground plane (+x is 0)."""
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or float(np.hypot(v[0], v[1])) < 1e-9:
        raise ValueError("vector must be 3-D with a horizontal component")
    return float(np.degrees(np.arctan2(v[1], v[0])))


def wrap_deg(angle: float) -> float:
    """Wrap ``angle`` (deg) to (-180, 180]."""
    return float(-((-angle + 180.0) % 360.0 - 180.0))


def leg_headings(
    markers: Mapping[str, Sequence[float]],
    hip_centres: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Knee-axis and foot headings per leg from one marker set.

    ``markers`` maps capture labels to world points; ``hip_centres`` maps
    ``"right"``/``"left"`` to the hip joint centre. The knee axis uses the
    malleolus/epicondyle-corrected flexion plane (:func:`knee_flexion_axis`)
    and points to the body's right. The forefoot heading is the forward normal
    of the ToeIn-ToeOut line (same convention as the knee axis, rotated 90
    degrees), so ``forefoot_minus_knee_deg`` is 0 for a square shank-to-foot.
    """
    up = np.array([0.0, 0.0, 1.0])
    out: dict[str, dict[str, float]] = {}
    for side, pre, index in SIDES:
        axis = knee_flexion_axis(
            hip_centres[side],
            markers[f"{pre}KneeOut"],
            markers[f"{pre}AnkleOut"],
            side=index,
            lateral=LATERAL_SEEDS_M,
        )
        toe_in = np.asarray(markers[f"{pre}ToeIn"], dtype=float)
        toe_out = np.asarray(markers[f"{pre}ToeOut"], dtype=float)
        rightward = toe_out - toe_in if index == 0 else toe_in - toe_out
        if index == 1:
            axis = -axis  # left knee axis points laterally (left); flip to right
        forefoot = np.cross(up, rightward)
        knee_forward = np.cross(up, axis)
        knee_yaw = heading_deg(knee_forward)
        foot_yaw = heading_deg(forefoot)
        out[side] = {
            "knee_forward_heading_deg": knee_yaw,
            "forefoot_heading_deg": foot_yaw,
            "forefoot_minus_knee_deg": wrap_deg(foot_yaw - knee_yaw),
        }
    return out


def _pipeline_args(ns: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--address-only",
        "--spec",
        ns.spec,
        "--static-seeds",
        "--capture",
        ns.capture,
        "--engine",
        ns.engine,
        "--foot-progression",
        "capture",
        "--out",
        ns.out,
        *ns.extra,
    ]
    return cli.build_parser().parse_args(argv)


def diagnose(ns: argparse.Namespace) -> dict[str, Any]:
    """Run the calibrated address and return the yaw budget report."""
    Path(ns.out).mkdir(parents=True, exist_ok=True)
    run = cli.calibrate_run(_pipeline_args(ns))
    res = run.cal_res
    kin, q = res.kin, np.asarray(res.address2.q, dtype=float)
    lane = run.lane
    frame = int(getattr(res.address2, "frame", 0) or 0)

    poses = kin.body_poses(q, LEG_BODIES)
    hips = {
        "right": np.asarray(poses["femur_r"][1], dtype=float),
        "left": np.asarray(poses["femur_l"][1], dtype=float),
    }
    capture = {
        label: lane.points[frame, i]
        for i, label in enumerate(lane.labels)
        if lane.valid[frame, i]
    }
    model_points = kin.marker_positions(q)
    model = {label: model_points[i] for i, label in enumerate(kin.labels)}

    cap = leg_headings(capture, hips)
    mod = leg_headings(model, hips)
    names = list(kin.coordinate_order)
    pelvis_r = np.asarray(poses[PELVIS_FRAME][0], dtype=float)
    report: dict[str, Any] = {
        "capture": ns.capture,
        "engine": ns.engine,
        "address_frame": frame,
        "hip_zero_twist_deg": res.hip_report.get("hip_zero_twist_deg"),
        "pelvis_axes_heading_deg": {
            axis: heading_deg(pelvis_r[:, k])
            for k, axis in enumerate("xyz")
            if float(np.hypot(*pelvis_r[:2, k])) > 1e-6
        },
        "hip_line_heading_deg": heading_deg(hips["right"] - hips["left"]),
        "coordinates_deg": {
            n: float(np.degrees(q[i]))
            for i, n in enumerate(names)
            if n.startswith(("hip_", "knee_", "ankle_", "subtalar_", "pelvis_"))
        },
        "capture_markers": cap,
        "model_markers": mod,
        "model_minus_capture_deg": {
            side: {
                key: wrap_deg(mod[side][key] - cap[side][key])
                for key in ("knee_forward_heading_deg", "forefoot_heading_deg")
            }
            for side in cap
        },
        "foot_progression": res.address_report.get("foot_progression"),
        "marker_rms_m": res.address_report.get("calibrated", {}).get("marker_rms_m"),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine", default="mujoco")
    parser.add_argument("extra", nargs="*", help="extra pipeline arguments")
    ns = parser.parse_args(argv)
    report = diagnose(ns)
    path = Path(ns.out) / "leg_yaw_budget.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
