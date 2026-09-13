"""Replay the fixed-pose run35 diagnostic using the shared offset estimator."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.shared.python.pose_estimation import estimate_keypoint_offset
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
data = np.load(args.input, allow_pickle=False)
target, valid = data['targets'], data['valid'].astype(bool)
train = np.arange(len(target)) % 2 == 0
errors = np.zeros((2, *valid.shape))
new = []
for j, (label, body, offset) in enumerate(zip(data['labels'], data['bodies'], data['offsets'], strict=True)):
    tf = data['transforms'][:, j]
    rotation, origin = tf[:, :3, :3], tf[:, :3, 3]
    mask = train & valid[:, j]
    estimate = estimate_keypoint_offset(keypoint_name=str(label), canonical_site=str(label), segment_name=str(body), joint_center_name=str(body), joint_centers_world_m=origin[mask], segment_rotations_world_from_segment=rotation[mask], keypoints_world_m=target[mask,j], min_samples=3)
    new.append(list(estimate.offset_m))
    for k, value in enumerate((offset, np.array(estimate.offset_m))):
        errors[k,:,j] = np.linalg.norm(origin + rotation @ value - target[:,j], axis=1)
report = {'scope': 'Fixed-pose offset calibration; held-out offsets are not independent end-to-end validation because poses were fitted to target.', 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'marker_offsets_m': new}
for name, rows in [('training',train), ('heldout',~train), ('initial',np.arange(len(target))==0)]:
    report[name+'_rms_mm'] = [float(np.sqrt(np.mean(e[valid & rows[:,None]]**2))*1000) for e in errors]
report['max_offset_change_mm'] = float(np.max(np.linalg.norm(np.array(new)-data['offsets'],axis=1))*1000)
args.output.write_text(json.dumps(report,indent=2)+'\n')
