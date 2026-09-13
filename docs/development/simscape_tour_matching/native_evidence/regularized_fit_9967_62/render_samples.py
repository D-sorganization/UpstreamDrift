"""Render preserved measured samples without rerunning physics."""
import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.samples)
    time, target, valid = data["time_s"], data["target_m"], data["valid"]
    labels = data["labels"].tolist()
    predictions = [data["baseline_m"], data["returned_m"]]
    args.output.mkdir(exist_ok=True)
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


if __name__ == "__main__":
    main()
