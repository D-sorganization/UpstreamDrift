"""Reproduce fixed-attachment rigidity diagnostics with shared tested providers."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))
from src.shared.python.motion_matching.rigidity import rigid_attachment_residuals

base = args.repo / "docs/development/simscape_tour_matching/native_evidence"
candidate_raw = (base / "two_window_fit_9967_102/returned-candidate.json").read_bytes()
doc = json.loads(candidate_raw)
with tarfile.open(base / "run102_original_inputs.tar.gz") as archive:
    capture_raw = archive.extractfile("driver_marker_payload_9967.json").read()
capture = json.loads(capture_raw)
assert doc["capture_sha256"] == capture["source_sha256"]
indices = [capture["labels"].index(label) for label in doc["marker_labels"]]
points = np.asarray(capture["points_world_m"], float)[:, indices]
valid = np.asarray(capture["valid"], bool)[:, indices]
points[~valid] = np.nan
errors = rigid_attachment_residuals(
    np.array(doc["marker_offsets_m"]), points, doc["marker_bodies"]
)
clock = np.asarray(capture["time_s"])
assert len(clock) == 654 and np.all(np.diff(clock) > 0)
bodies = np.array(doc["marker_bodies"])
# Pairwise distances are invariant to any rigid body pose; preserve their changes.
pairs = []
for i, label in enumerate(doc["marker_labels"]):
    for j in range(i + 1, len(bodies)):
        if bodies[i] != bodies[j]:
            continue
        distances = np.linalg.norm(points[:, i] - points[:, j], axis=1)
        model_distance = float(
            np.linalg.norm(
                np.array(doc["marker_offsets_m"][i]) - doc["marker_offsets_m"][j]
            )
        )
        pairs.append(
            {
                "body": str(bodies[i]),
                "labels": [label, doc["marker_labels"][j]],
                "model_distance_m": model_distance,
                "min_observed_m": float(np.nanmin(distances)),
                "max_observed_m": float(np.nanmax(distances)),
                "terminal_085_m": float(distances[306]),
            }
        )


def frame_rms(values: np.ndarray) -> list[float | None]:
    count = np.sum(np.isfinite(values), axis=1)
    mean = np.divide(
        np.nansum(values**2, axis=1),
        count,
        out=np.full(len(values), np.nan),
        where=count > 0,
    )
    return [
        float(1000 * np.sqrt(value)) if np.isfinite(value) else None for value in mean
    ]


report = {
    "qualification": "independent rigid-body lower bound conditional on fixed attachments; not articulated or dynamic feasibility",
    "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
    "payload_sha256": hashlib.sha256(capture_raw).hexdigest(),
    "time_s": clock.tolist(),
    "rms_mm": frame_rms(errors),
    "body_rms_mm": {
        str(b): frame_rms(errors[:, bodies == b]) for b in np.unique(bodies)
    },
    "pairs": pairs,
}
serialized = json.dumps(report, indent=2, allow_nan=False)
with args.output.open("x", encoding="utf-8") as stream:
    stream.write(serialized)
