"""Emit the HRG-C3 pelvis revision of a fitted spec and compare it with its parent.

Reads a private capture and a fitted spec by path (nothing is copied into the
repository), estimates the hip centres with the HRG-C2 MAP estimator, writes a
NEW spec revision plus a receipt to ``--out-dir`` (refusing to overwrite) and
reports hip lever arms, hip reaction wrench and L5/S1 torques before and after
along the fitted bundle trajectory. No refit is performed.

    PYTHONPATH=.:src python3 -m scripts.research.hrg_c3_pelvis_revision \\
        --spec RUN/full_body_spec_hipcal_scaled.json --receipt RUN/receipt.json \\
        --capture data/C3D_TA_Driver.c3d --bundle BUNDLES/driver.npz \\
        --out-dir OUT --stride 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python import pelvis_revision_report as report
from src.shared.python.motion_matching import pelvis_frame
from src.shared.python.motion_matching.hip_centre_map import (
    HipCentrePrior,
    estimate_hip_centres_from_capture,
)
from src.shared.python.motion_matching.pipeline.address import pelvis_marker_offsets
from src.shared.python.motion_matching.pipeline.lane import Lane

logger = logging.getLogger(__name__)


def _peak(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    def pick(row: dict[str, Any]) -> float:
        node: Any = row
        for key in path:
            node = node[key]
        return float(np.linalg.norm(np.atleast_1d(node)))

    return max(pick(r) for r in rows)


def summarise(comparison: dict[str, Any]) -> dict[str, Any]:
    """Peak magnitudes and the first-frame (address) lever arms, before/after."""
    out: dict[str, Any] = {"frames": comparison["frames"]}
    for key in ("before", "after"):
        rows = comparison[key]
        first = rows[0]["lever_arms"]
        out[key] = {
            "address_lever_arms": {
                side: {
                    "pelvis_com_arm_m": first[side]["pelvis_com_arm_m"],
                    "superior_com_arm_m": first[side]["superior_com_arm_m"],
                    "superior_com_height_m": first[side]["superior_com_height_m"],
                }
                for side in ("r", "l")
            },
            "address_pelvis_com_to_hjc_midpoint_m": first[
                "pelvis_com_to_hjc_midpoint_m"
            ],
            "peak_hip_reaction_force_n": _peak(rows, ("hip_reaction", "force_n")),
            "peak_hip_reaction_moment_nm": _peak(rows, ("hip_reaction", "moment_nm")),
            "peak_l5s1_torque_nm": {
                n: max(abs(r["l5s1_torque_nm"][n]) for r in rows)
                for n in report.L5S1_COORDINATES
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stature", type=float, default=None)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--name", default="full_body_spec_pelvis_rev1")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    labels = tuple(json.loads(args.receipt.read_text(encoding="utf-8"))["labels"])
    stature = args.stature or float(spec["subject"]["stature_m"])
    lane = Lane(labels, str(args.capture))
    estimate = estimate_hip_centres_from_capture(
        lane.points,
        lane.valid,
        lane.labels,
        pelvis_marker_offsets(spec),
        HipCentrePrior.from_stature(stature),
    )
    revision = pelvis_frame.relocate_pelvis_from_estimate(spec, estimate)
    paths = pelvis_frame.write_revision(revision, args.out_dir / f"{args.name}.json")
    logger.info("wrote %s and %s", paths.spec, paths.receipt)

    bundle = np.load(args.bundle, allow_pickle=True)
    manifest = json.loads(str(bundle["manifest"]))
    coordinates = manifest["coordinate_order"]
    q, v = bundle["reference_q"], bundle["reference_v"]
    acceleration = np.gradient(v, manifest["dt_s"], axis=0)
    frames = list(range(0, q.shape[0], max(1, args.stride)))
    comparison = report.compare(
        spec, revision.spec, coordinates, q, v, acceleration, frames
    )
    result = {
        "spec_revision": json.loads(paths.receipt.read_text(encoding="utf-8")),
        "pelvic_plane": pelvis_frame.pelvic_plane_report(estimate),
        "summary": summarise(comparison),
        "frames_run": len(frames),
        "stride": args.stride,
        "trajectory": "bundle reference_q/reference_v, acceleration by central "
        "difference; not refitted",
    }
    out = args.out_dir / f"{args.name}.comparison.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    full = args.out_dir / f"{args.name}.comparison_frames.json"
    full.write_text(json.dumps(comparison) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))  # noqa: T201 - CLI
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
