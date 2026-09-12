"""Render a portable marker-replay archive without requiring the physics engine."""

import argparse
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.marker_replay_report import plot_marker_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.trajectory, allow_pickle=False) as data:
        figure = plot_marker_replay(
            data["time_s"],
            data["target_m"],
            data["prediction_m"],
            data["valid"],
            data["labels"].tolist(),
            candidate_sha256=str(data["candidate_sha256"]),
        )
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
