"""Subject-specific lengths for the anthropometric geometry from the capture.

Offsets come from the static trial, lengths from the dynamic trial: for each
(trunk, arm, shoulder) scale the full-body candidate is built, the neutral
address is fitted (scapulae undepressed, spine within 10 deg), every marker
is placed from the first address frames, and the decimated swing is solved
with those offsets and no further calibration. A rigid offset cannot hide a
wrong joint centre once the segment turns, so the whole-swing residual of
the body markers (head excluded: the chain has no neck) ranks the lengths.
Writes ``scan_geometry_receipt.json``; nothing here is qualified.

    python scan_geometry.py 1.15,1.1,1.0 1.25,1.1,1.0 ...   (trunk,arm,shoulder)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import src.shared.python.motion_matching.pipeline as drv  # noqa: E402

from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    MARKER_SEGMENTS,
)

HERE = Path(__file__).resolve().parent
BUILD = ROOT / "docs/development/full_body_models/build_anthropometric_spec.py"
NATIVE = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
OSIM = ROOT / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
STATURE, MASS = 1.71, 78.0
DEFAULT_VARIANTS = ((1.15, 1.1, 1.0),)


def body_rms(labels: tuple[str, ...], errors: np.ndarray, valid: np.ndarray) -> float:
    """RMS over every valid marker outside the head segment."""
    keep = np.array([m not in MARKER_SEGMENTS["head"] for m in labels])
    e, v = errors[:, keep], valid[:, keep]
    return float(np.sqrt(np.mean(e[v] ** 2)))


def evaluate(
    lane: drv.Lane | None, trunk: float, arm: float, shoulder: float, out: Path
):
    name = f"anthro_t{trunk:.2f}_a{arm:.2f}_s{shoulder:.2f}"
    subprocess.run(
        [
            sys.executable, str(BUILD), "--native", str(NATIVE), "--osim", str(OSIM),
            "--native-candidate", str(drv.CANDIDATE), "--stature", str(STATURE),
            "--mass", str(MASS), "--trunk-scale", str(trunk), "--arm-scale", str(arm),
            "--shoulder-scale", str(shoulder), "--output", str(out), "--name", name,
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    spec = json.loads((out / f"{name}.json").read_text())
    upper = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    seeds = {**upper, **drv.LEG_SEEDS}
    if lane is None:
        lane = drv.Lane(tuple(seeds))
    drv.configure_lane(lane, spec)
    spec_bytes = json.dumps(drv.add_toe_spheres(spec), sort_keys=True).encode()
    adapter, kin = lane.kinematics(spec_bytes, seeds)
    static, neutral, kin = lane.static_trial(
        spec_bytes, seeds, drv.document_seed(spec, kin)
    )
    address = lane.best_address(kin, neutral.q)
    frames = lane.calibration_frames
    q, _ = lane.trajectory(kin, address.q, frames=frames)
    errors = drv.marker_errors(kin, q, lane.points[frames])
    valid = lane.valid[frames]
    row = {
        "trunk_scale": trunk,
        "arm_scale": arm,
        "shoulder_scale": shoulder,
        "spec_sha256": spec["upper_body_sha256"],
        "neutral_fit_rms_m": neutral.marker_rms_m,
        "static_address_rms_m": address.marker_rms_m,
        "address_posture": drv.posture_summary(kin, address.q),
        "swing_rms_all_m": float(np.sqrt(np.mean(errors[valid] ** 2))),
        "swing_rms_body_m": body_rms(lane.labels, errors, valid),
        "swing_segment_rms_m": drv.segment_rms(lane.labels, errors, valid),
        "address_elbows_deg": {
            name: float(np.degrees(address.q[kin.coordinate_order.index(name)]))
            for name in ("LEInput", "REInput")
        },
        "frames": len(frames),
    }
    return lane, row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("scan")
    variants = [tuple(map(float, v.split(","))) for v in sys.argv[1:]] or list(
        DEFAULT_VARIANTS
    )
    out = HERE / "scan"
    out.mkdir(exist_ok=True)
    receipt_path = HERE / "scan_geometry_receipt.json"
    rows = json.loads(receipt_path.read_text())["rows"] if receipt_path.exists() else []
    lane = None
    for trunk, arm, shoulder in variants:
        lane, row = evaluate(lane, trunk, arm, shoulder, out)
        rows = [
            r
            for r in rows
            if (r["trunk_scale"], r["arm_scale"], r["shoulder_scale"])
            != (trunk, arm, shoulder)
        ]
        rows.append(row)
        log.info(
            "t %.2f a %.2f s %.2f: neutral %.1f address %.1f swing all %.1f body %.1f mm %s links %s bend fwd %.1f lat %.1f",
            trunk, arm, shoulder, row["neutral_fit_rms_m"] * 1e3,
            row["static_address_rms_m"] * 1e3, row["swing_rms_all_m"] * 1e3,
            row["swing_rms_body_m"] * 1e3,
            {k: round(v * 1e3) for k, v in row["swing_segment_rms_m"].items()},
            {k: round(v) for k, v in row["address_posture"]["clavicle_link_below_horizontal_deg"].items()},
            row["address_posture"]["spine_bend_deg"]["forward_deg"],
            row["address_posture"]["spine_bend_deg"]["lateral_deg"],
        )  # fmt: skip
        log.info(
            "   elbows %s", {k: round(v) for k, v in row["address_elbows_deg"].items()}
        )
        receipt_path.write_text(
            json.dumps(
                {
                    "stature_m": STATURE,
                    "mass_kg": MASS,
                    "metric": "swing_rms_body_m: decimated-swing IK RMS of non-head markers with static-trial offsets, no calibration",
                    "rows": sorted(rows, key=lambda r: r["swing_rms_body_m"]),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
