"""Report a conditional rigidity floor using the tested shared diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--state", type=Path, required=True)
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--prefix-end", type=float, action="append", default=[])
args = parser.parse_args()
sys.path.insert(0, str(args.repo))
from src.shared.python.motion_matching.rigidity import rigid_attachment_residuals

state_raw = args.state.read_bytes()
capture_raw = args.capture.read_bytes()
seed = json.loads(state_raw)
capture = json.loads(capture_raw)
if seed["source_sha256"] != capture["source_sha256"]:
    raise ValueError("state and capture source identity differ")
indices = [capture["labels"].index(label) for label in seed["labels"]]
observed = np.asarray(capture["points_world_m"], dtype=float)[:, indices]
valid = np.asarray(capture["valid"], dtype=bool)[:, indices]
observed[~valid] = np.nan
clock = np.asarray(capture["time_s"], dtype=float)
if (
    clock.shape != (len(observed),)
    or not np.isfinite(clock).all()
    or np.any(np.diff(clock) <= 0)
):
    raise ValueError("capture clock must be finite, ordered and match observations")
if any(
    not np.isfinite(end) or end < clock[0] or end > clock[-1] for end in args.prefix_end
):
    raise ValueError("prefix ends must lie within the capture clock")
errors = rigid_attachment_residuals(
    np.asarray(seed["offsets_m"]), observed, seed["body_names"]
)
body_names = np.asarray(seed["body_names"])


def rms_mm(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2)) * 1000) if finite.size else None


report = {
    "qualification": "independent-body rigid relaxation, conditional on fixed attachments; not forward dynamics",
    "source_sha256": capture["source_sha256"],
    "state_sha256": hashlib.sha256(state_raw).hexdigest(),
    "payload_sha256": hashlib.sha256(capture_raw).hexdigest(),
    "diagnostic_sha256": hashlib.sha256(
        (args.repo / "src/shared/python/motion_matching/rigidity.py").read_bytes()
    ).hexdigest(),
    "frames": len(clock),
    "markers": len(indices),
    "rms_mm": rms_mm(errors),
    "body_rms_mm": {
        body: rms_mm(errors[:, body_names == body]) for body in np.unique(body_names)
    },
    "prefix_rms_mm": {
        str(end): rms_mm(errors[clock <= end]) for end in args.prefix_end
    },
    "limitations": [
        "Offsets and body assignments are fixed; recalibration can change this bound.",
        "Every body pose is independent per frame; no connectivity, dynamics or continuity constraints.",
        "One-marker bodies have zero residual and no rotation identifiability.",
    ],
}
with args.output.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
