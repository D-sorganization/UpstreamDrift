"""Generate MuJoCo returned81 replay .npz and animated GIF.

Replays the candidate trajectory with NativeMujocoModel:
- Evaluates forward kinematics across 307 frames
- Computes predicted markers
- Evaluates 5 metrics: whole, early, terminal, clubhead, pelvis yaw
- Compares against Pinocchio replay per sample
- Renders animated GIF via mujoco.Renderer at (240, 320), stride 3
- Saves mujoco_returned81_replay.npz and mujoco_returned81.gif
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.engines.physics_engines.mujoco.python.native_model import NativeMujocoModel
from src.shared.python.motion_matching.full_body_spec import upper_body_slice
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

OUTPUT_NPZ = REPLAYS_DIR / "mujoco_returned81_replay.npz"
OUTPUT_GIF = REPLAYS_DIR / "mujoco_returned81.gif"

SIZE = (240, 320)
STRIDE = 3


def camera(lookat: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -12.0
    return cam


def evaluate_forward_kinematics(
    model: NativeMujocoModel,
    q_pin: np.ndarray,
    names: list[str],
    candidate_doc: dict[str, Any],
) -> np.ndarray:
    """Evaluate forward kinematics across all candidate frames."""
    mj_markers = []
    for k in range(len(q_pin)):
        qk_dict = {name: float(q_pin[k, i]) for i, name in enumerate(names)}
        poses_k = model.frame_poses(qk_dict)
        m_k = project_markers(
            poses_k,
            candidate_doc["marker_bodies"],
            candidate_doc["marker_offsets_m"],
        )
        mj_markers.append(m_k)
    return np.asarray(mj_markers, dtype=np.float64)


def render_replay_gif(
    q_pin: np.ndarray,
    names: list[str],
    lookat: np.ndarray,
) -> int:
    """Render animated GIF using MuJoCo visual layer."""
    spec_bytes = FULL_BODY_SPEC_PATH.read_bytes()
    xml, _ = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    visual_model = mujoco.MjModel.from_xml_string(xml)
    visual_data = mujoco.MjData(visual_model)

    addresses = [visual_model.joint(n).qposadr[0] for n in names]
    renderer = mujoco.Renderer(visual_model, *SIZE)

    pil_frames: list[Image.Image] = []
    for k in range(0, q_pin.shape[0], STRIDE):
        visual_data.qpos[addresses] = q_pin[k]
        mujoco.mj_forward(visual_model, visual_data)
        renderer.update_scene(visual_data, camera=camera(lookat))
        arr = renderer.render().copy()
        pil_frames.append(Image.fromarray(arr))

    frame_duration_ms = int(round(1000.0 * STRIDE / 360.0))
    pil_frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    return len(pil_frames)


def main() -> None:
    logger.info("Loading candidate and reference replays...")
    candidate_doc = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    pin_replay = load_native_replay_npz(PINOCCHIO_NPZ)
    fb_spec = json.loads(FULL_BODY_SPEC_PATH.read_text(encoding="utf-8"))

    upper_spec = upper_body_slice(fb_spec)
    upper_bytes = json.dumps(upper_spec).encode("utf-8")
    model = NativeMujocoModel(upper_bytes)

    time_s = pin_replay["time_s"]
    target_m = pin_replay["target_m"]
    valid = pin_replay["valid"]
    native_state = pin_replay["native_state"]
    q_pin = native_state[:, :27]
    names = candidate_doc["coordinate_names"]

    logger.info("Computing MuJoCo forward kinematics on candidate coordinates...")
    mj_markers = evaluate_forward_kinematics(model, q_pin, names, candidate_doc)

    marker_diff = np.abs(mj_markers - pin_replay["markers_m"])
    max_marker_diff = float(np.max(marker_diff))
    mean_marker_diff = float(np.mean(marker_diff))
    logger.info("Max marker diff vs Pinocchio: %.6e m", max_marker_diff)

    save_native_replay_npz(
        OUTPUT_NPZ,
        time_s=time_s,
        native_state=native_state,
        markers_m=mj_markers,
        target_m=target_m,
        valid=valid,
    )

    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=mj_markers,
        target_markers_m=target_m,
        valid=valid,
        marker_labels=candidate_doc["marker_labels"],
    )

    lookat = np.nanmean(target_m[0], axis=0)
    n_frames = render_replay_gif(q_pin, names, lookat)

    receipt = {
        "candidate_sha256": candidate_doc.get("source_sha256")
        or hashlib.sha256(CANDIDATE_JSON.read_bytes()).hexdigest(),
        "engine": "mujoco",
        "replay_npz": str(OUTPUT_NPZ.name),
        "gif": str(OUTPUT_GIF.name),
        "metrics": metrics.as_dict(),
        "marker_diff_vs_pinocchio": {
            "max_abs_m": max_marker_diff,
            "mean_abs_m": mean_marker_diff,
        },
        "render_config": {
            "resolution": list(SIZE),
            "stride": STRIDE,
            "frames": n_frames,
            "camera": {
                "distance": 3.2,
                "azimuth": 135.0,
                "elevation": -12.0,
                "lookat": lookat.tolist(),
            },
        },
    }
    receipt_path = REPLAYS_DIR / "mujoco_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved receipt to %s", receipt_path)


if __name__ == "__main__":
    main()
