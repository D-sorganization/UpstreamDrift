"""Replay two immutable native candidates and plot measured marker residuals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "baseline", "candidate", "target", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    model_hash = hashlib.sha256(raw).hexdigest()
    candidates = [
        NativeReplayCandidate.from_document(
            json.loads(path.read_text()), spec["coordinate_order"], model_hash
        )
        for path in (args.baseline, args.candidate)
    ]
    document = candidates[0].document
    assert candidates[1].document["marker_labels"] == document["marker_labels"]
    assert candidates[1].document["duration_s"] == document["duration_s"]
    payload = json.loads(args.target.read_text())
    assert all(
        c.document["capture_sha256"] == payload["source_sha256"] for c in candidates
    )
    labels = document["marker_labels"]
    indices = [payload["labels"].index(label) for label in labels]
    all_time = np.asarray(payload["time_s"])
    mask = all_time <= document["duration_s"]
    time = all_time[mask]
    target = np.asarray(payload["points_world_m"])[mask][:, indices].copy()
    valid = np.asarray(payload["valid"], dtype=bool)[mask][:, indices]
    valid &= np.isfinite(target).all(axis=2)
    target[~valid] = np.nan
    predictions = []
    receipts = []
    for candidate in candidates:
        result = replay_candidate(
            raw, candidate, time, rtol=1e-11, atol=1e-13, max_step=0.00025
        )
        predictions.append(result.markers_m)
        errors = np.sum((result.markers_m - target) ** 2, axis=2)
        receipts.append(
            {
                "candidate_sha256": candidate.sha256,
                "whole_rms_m": float(np.sqrt(np.mean(errors[valid]))),
                "terminal_rms_m": float(np.sqrt(np.mean(errors[-1, valid[-1]]))),
                "integration_s": result.integration.elapsed_s,
            }
        )
    np.savez_compressed(
        args.output / "sampled-markers.npz",
        time_s=time,
        target_m=target,
        valid=valid,
        baseline_m=predictions[0],
        returned_m=predictions[1],
        labels=np.asarray(labels),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    colors = ("#ad6a23", "#176b9b")
    names = ("Original Run19", "Returned Run62")
    for prediction, color, name in zip(predictions, colors, names, strict=True):
        squared = np.sum((prediction - target) ** 2, axis=2)
        rms = np.sqrt(np.nansum(squared, axis=1) / np.sum(valid, axis=1))
        axes[0, 0].plot(time, 1000 * rms, color=color, label=name)
        for label in ("LWristTop", "RWristTop", "Marker_2:2:1", "Marker_3:3:1"):
            idx = labels.index(label)
            error = np.sqrt(squared[:, idx]) * 1000
            if name == names[1]:
                axes[0, 1].plot(time, error, label=label)
    axes[0, 0].set_title("Observed Marker Euclidean RMS")
    axes[0, 1].set_title("Run62 Selected Marker Euclidean Errors")
    for axis in axes[0]:
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Error (mm)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    selected = ("LWristTop", "RWristTop", "Marker_2:2:1", "Marker_3:3:1")
    for axis, label in zip(axes[1:].flat, selected, strict=True):
        idx = labels.index(label)
        axis.plot(
            target[:, idx, 0] * 1000, target[:, idx, 2] * 1000, "k--", label="Capture"
        )
        for prediction, color, name in zip(predictions, colors, names, strict=True):
            axis.plot(
                prediction[:, idx, 0] * 1000,
                prediction[:, idx, 2] * 1000,
                color=color,
                label=name,
            )
        axis.set_title(label + " — World X–Z Projection")
        axis.set_xlabel("World X (mm)")
        axis.set_ylabel("World Z (mm)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Exploratory / Rejected Fit — 0–0.85 s Prefix Only", fontsize=16)
    figure.savefig(args.output / "marker-comparison.png", dpi=160)
    plt.close(figure)
    receipt = {
        "qualification": "Measured scalar forward replays; rejected exploratory prefix, not full swing acceptance",
        "replays": receipts,
        "input_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.model,
                args.baseline,
                args.candidate,
                args.target,
                Path(__file__),
            )
        },
        "output_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.output.iterdir()
        },
        "validity": "Capture validity AND finite target XYZ; Euclidean norm per marker, RMS over observed markers",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
