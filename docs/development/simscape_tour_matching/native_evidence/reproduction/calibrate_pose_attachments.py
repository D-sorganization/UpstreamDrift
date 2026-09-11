"""Fit fixed marker attachments using the shared offset estimator and native poses."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
for name in ("repo", "frames", "seed", "capture", "output"):
    parser.add_argument(f"--{name}", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))
sys.path.insert(0, str(args.repo / "src"))
from src.shared.python.pose_estimation import estimate_keypoint_offset

seed = json.loads(args.seed.read_text(encoding="utf-8"))
capture = json.loads(args.capture.read_text(encoding="utf-8"))
if seed["source_sha256"] != capture["source_sha256"]:
    raise ValueError("capture identity differs")
if args.output.exists():
    raise FileExistsError("preserve existing calibration output")
paths = sorted(args.frames.glob("frame-*.json"))
if len(paths) < 3:
    raise ValueError("at least three native frames are required")
frames = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
indices = np.asarray([frame["frame_index"] - 1 for frame in frames])
clock = np.asarray(capture["time_s"])
if np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= len(clock):
    raise ValueError("native frame indices must increase within the capture")
bodies = frames[0]["body_names"]
for frame, index in zip(frames, indices, strict=True):
    parity = frame["saved_marker_max_difference_m"]
    if (
        frame["source_sha256"] != seed["source_sha256"]
        or frame["geometry_in"] != seed["geometry_in"]
        or frame["body_names"] != bodies
        or abs(frame["time_s"] - clock[index]) > 1e-12
        or not np.isfinite(parity)
        or not 0 <= parity < 1e-8
    ):
        raise ValueError("native frame provenance or marker parity differs")
origins = np.asarray([frame["origins_m"] for frame in frames])
rotations = np.asarray([frame["rotations_world_from_body"] for frame in frames])
points = np.asarray(capture["points_world_m"])[indices]
valid = np.asarray(capture["valid"], dtype=bool)[indices]
old_offsets = np.asarray(seed["offsets_m"])
estimates, offsets, before_squares = [], [], []
after_energy, count = 0.0, 0
for marker, (label, body) in enumerate(
    zip(seed["labels"], seed["body_names"], strict=True)
):
    column, body_index = capture["labels"].index(label), bodies.index(body)
    retained = valid[:, column]
    centers, turns = origins[retained, body_index], rotations[retained, body_index]
    observed = points[retained, column]
    estimate = estimate_keypoint_offset(
        keypoint_name=label,
        canonical_site=label,
        segment_name=body,
        joint_center_name=body,
        joint_centers_world_m=centers,
        keypoints_world_m=observed,
        segment_rotations_world_from_segment=turns,
        min_samples=3,
    )
    old_prediction = centers + np.einsum("nij,j->ni", turns, old_offsets[marker])
    before_squares.extend(np.sum((old_prediction - observed) ** 2, axis=1).tolist())
    offsets.append(list(estimate.offset_m))
    estimates.append(estimate.to_dict())
    after_energy += estimate.rms_residual_m**2 * estimate.sample_count
    count += estimate.sample_count
before, after = (
    float(np.sqrt(np.mean(before_squares))),
    float(np.sqrt(after_energy / count)),
)
if after > before + 1e-12:
    raise ValueError("fixed-pose least-squares calibration must not worsen residuals")
report = {
    "qualification": "fixed-pose attachment calibration candidate; not anatomical identification or forward acceptance",
    "source_sha256": seed["source_sha256"],
    "geometry_in": seed["geometry_in"],
    "frame_count": len(frames),
    "valid_observation_count": count,
    "fixed_pose_before_rms_m": before,
    "fixed_pose_after_rms_m": after,
    "offset_change_m": (np.asarray(offsets) - old_offsets).tolist(),
    "estimates": estimates,
    "input_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.seed, args.capture, *paths]
    },
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "estimator_sha256": hashlib.sha256(
        (
            args.repo / "src/shared/python/pose_estimation/keypoint_offsets.py"
        ).read_bytes()
    ).hexdigest(),
}
candidate = deepcopy(seed)
candidate.update(
    offsets_m=offsets,
    initial_state_verified=False,
    status="attachment-calibration-candidate",
    qualification=report["qualification"],
)
candidate.pop("prediction_m", None)
args.output.mkdir(parents=True, exist_ok=False)
for name, value in [("calibration_report", report), ("candidate_seed", candidate)]:
    (args.output / f"{name}.json").write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )
shutil.copyfile(__file__, args.output / Path(__file__).name)
