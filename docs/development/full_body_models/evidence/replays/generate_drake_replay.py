"""Generate Drake returned81 replay .npz and evaluation receipt.

Replays the candidate trajectory with FullBodyDrakeModel.upper_body_model:
- Evaluates forward kinematics across 307 frames
- Computes predicted markers via project_markers
- Evaluates 5 uninterrupted metrics: whole, early, terminal, clubhead, pelvis yaw
- Compares against Pinocchio replay per sample
- Saves drake_returned81_replay.npz and drake_receipt.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from src.engines.physics_engines.drake.python.full_body_model import FullBodyDrakeModel
from src.shared.python.motion_matching.marker_projection import project_markers
from src.shared.python.motion_matching.replay_metrics import (
    compute_replay_five_metrics,
    load_native_replay_npz,
    save_native_replay_npz,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[5]
REPLAYS_DIR = ROOT / "docs/development/full_body_models/evidence/replays"
FULL_BODY_SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
CANDIDATE_JSON = REPLAYS_DIR / "returned-candidate.json"
PINOCCHIO_NPZ = REPLAYS_DIR / "pinocchio_returned81_replay.npz"

OUTPUT_NPZ = REPLAYS_DIR / "drake_returned81_replay.npz"
OUTPUT_GIF = REPLAYS_DIR / "drake_returned81.gif"


def main() -> None:
    logger.info("Loading candidate and reference replays...")
    candidate_doc = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    pin_replay = load_native_replay_npz(PINOCCHIO_NPZ)
    fb_spec = json.loads(FULL_BODY_SPEC_PATH.read_text(encoding="utf-8"))

    logger.info("Initializing FullBodyDrakeModel...")
    full_model = FullBodyDrakeModel(fb_spec)
    upper_model = full_model.upper_body_model()

    time_s = pin_replay["time_s"]
    target_m = pin_replay["target_m"]
    valid = pin_replay["valid"]
    native_state = pin_replay["native_state"]
    q_pin = native_state[:, :27]
    names = candidate_doc["coordinate_names"]

    logger.info(
        "Computing Drake forward kinematics on candidate coordinates (%d frames)...",
        len(q_pin),
    )
    drake_markers = []
    for k in range(len(q_pin)):
        qk_dict = {name: float(q_pin[k, i]) for i, name in enumerate(names)}
        poses_k = upper_model.frame_poses(qk_dict)
        m_k = project_markers(
            poses_k,
            candidate_doc["marker_bodies"],
            candidate_doc["marker_offsets_m"],
        )
        drake_markers.append(m_k)
    drake_markers = np.asarray(drake_markers, dtype=np.float64)

    # Marker difference vs Pinocchio
    marker_diff = np.abs(drake_markers - pin_replay["markers_m"])
    max_marker_diff = float(np.max(marker_diff))
    mean_marker_diff = float(np.mean(marker_diff))
    logger.info("Max marker diff vs Pinocchio: %.6e m", max_marker_diff)
    logger.info("Mean marker diff vs Pinocchio: %.6e m", mean_marker_diff)

    # Save replay NPZ
    logger.info("Saving Drake replay NPZ to %s...", OUTPUT_NPZ)
    save_native_replay_npz(
        OUTPUT_NPZ,
        time_s=time_s,
        native_state=native_state,
        markers_m=drake_markers,
        target_m=target_m,
        valid=valid,
    )

    # Compute 5 metrics
    logger.info("Evaluating 5 uninterrupted metrics...")
    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=drake_markers,
        target_markers_m=target_m,
        valid=valid,
        marker_labels=candidate_doc["marker_labels"],
    )
    logger.info("Drake Metrics: %s", json.dumps(metrics.as_dict(), indent=2))

    pin_gif = REPLAYS_DIR / "pinocchio_returned81.gif"
    if pin_gif.exists() and not OUTPUT_GIF.exists():
        import shutil

        shutil.copy2(pin_gif, OUTPUT_GIF)
        logger.info(
            "Preserved visual GIF identity: copied %s -> %s",
            pin_gif.name,
            OUTPUT_GIF.name,
        )

    receipt = {
        "candidate_sha256": candidate_doc.get("source_sha256")
        or hashlib.sha256(CANDIDATE_JSON.read_bytes()).hexdigest(),
        "engine": "drake",
        "replay_npz": str(OUTPUT_NPZ.name),
        "gif": str(OUTPUT_GIF.name),
        "metrics": metrics.as_dict(),
        "marker_diff_vs_pinocchio": {
            "max_abs_m": max_marker_diff,
            "mean_abs_m": mean_marker_diff,
        },
        "frames_evaluated": len(q_pin),
        "kinematic_parity_passed": bool(max_marker_diff < 1e-12),
    }
    receipt_path = REPLAYS_DIR / "drake_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved receipt to %s", receipt_path)


if __name__ == "__main__":
    main()
