"""Audit native Hub topology and observed pair spacing across the entire capture."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.marker_rigidity import marker_pair_statistics
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "model", "capture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()),
        spec["coordinate_order"],
        hashlib.sha256(raw).hexdigest(),
    )
    doc = candidate.document
    payload = json.loads(args.capture.read_text())
    if payload["source_sha256"] != doc["capture_sha256"]:
        raise ValueError("Capture identity mismatch")
    labels = [
        label
        for label, frame in zip(doc["marker_labels"], doc["marker_bodies"], strict=True)
        if frame == "Hub"
    ]
    indices = [payload["labels"].index(label) for label in labels]
    points = np.asarray(payload["points_world_m"])[:, indices]
    mask = np.asarray(payload["valid"], dtype=bool)[:, indices]
    body = next(frame["body"] for frame in spec["frames"] if frame["name"] == "Hub")
    solids = next(value["solids"] for value in spec["bodies"] if value["name"] == body)
    report = {
        "qualification": "observed pair-spacing audit; pair-only bounds allow independent pose each frame, not full-marker acceptance",
        "candidate_sha256": candidate.sha256,
        "native_hub_body": body,
        "native_hub_solids": [solid["name"] for solid in solids],
        "capture_samples": len(payload["time_s"]),
        "capture_end_s": payload["time_s"][-1],
        "pairs": marker_pair_statistics(points, mask, labels),
        "input_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ("candidate", "model", "capture")
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
