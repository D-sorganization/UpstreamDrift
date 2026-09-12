"""Measure a fixed-attachment rigidity lower bound from a portable replay archive."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.marker_rigidity import rigid_marker_lower_bound
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", type=Path)
    source.add_argument("--capture-payload", type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    doc = json.loads(args.candidate.read_text())
    candidate = NativeReplayCandidate.from_document(
        doc, doc["coordinate_names"], doc["model_sha256"]
    )
    rows = []
    if args.capture_payload is not None:
        payload = json.loads(args.capture_payload.read_text())
        if payload["source_sha256"] != doc["capture_sha256"]:
            raise ValueError("Capture identity mismatch")
        indices = [payload["labels"].index(label) for label in doc["marker_labels"]]
        data = {
            "time_s": np.asarray(payload["time_s"]),
            "target_m": np.asarray(payload["points_world_m"])[:, indices],
            "valid": np.asarray(payload["valid"], dtype=bool)[:, indices],
        }
        source_name = "capture_payload"
    else:
        with np.load(args.trajectory, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        if (
            str(data["candidate_sha256"]) != candidate.sha256
            or data["labels"].tolist() != doc["marker_labels"]
        ):
            raise ValueError("Trajectory/candidate identity mismatch")
        source_name = "trajectory"
    for time in args.times:
        matches = np.flatnonzero(np.isclose(data["time_s"], time, atol=1e-12, rtol=0))
        if len(matches) != 1:
            raise ValueError("Exact sample unavailable")
        k = int(matches[0])
        row = rigid_marker_lower_bound(
            np.asarray(doc["marker_offsets_m"]),
            data["target_m"][k],
            doc["marker_bodies"],
            data["valid"][k],
        )
        row["time_s"] = time
        rows.append(row)
    args.output.write_text(
        json.dumps(
            {
                "qualification": "fixed-offset rigid-frame lower bound; joints and closures relaxed; no forward-fit acceptance",
                "candidate_sha256": candidate.sha256,
                "input_sha256": {
                    name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                    for name in ("candidate", source_name)
                },
                "poses": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
