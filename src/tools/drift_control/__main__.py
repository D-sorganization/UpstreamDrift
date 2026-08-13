"""Command-line interface for drift-control force-ratio analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.tools.drift_control.analyzer import DriftControlAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute DIR(t)=||f(x)||/||G(x)u|| from realized trajectory data."
    )
    parser.add_argument("trajectory", type=Path, help="NPZ trajectory path")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Minimum denominator norm used for numerical stability.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyzer = DriftControlAnalyzer(epsilon=args.epsilon)
    trajectory = analyzer.load_expert_trajectory(args.trajectory)
    ratio = analyzer.compute_ratio(trajectory)
    payload = {
        "trajectory": {
            "path": str(args.trajectory),
            "sample_count": trajectory.sample_count,
            "dimensions": trajectory.dimensions,
        },
        "ratio": analyzer.summarize_ratio(ratio),
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
