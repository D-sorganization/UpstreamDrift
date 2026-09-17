"""Playback GIF for a Crocoddyl fit candidate (MS-31, #10338).

Reads the ``candidate.npz`` written by ``full_body_fit`` and renders the
predicted markers against the capture targets with the shared overlay
renderer, so every lane's animations look the same. No engine import.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.motion_matching.cross_engine_replay import (
    render_marker_overlay_animation,
)


def render_candidate_gif(
    candidate_path: Path, output_path: Path, *, stride: int = 3
) -> Path:
    """Render ``candidate.npz`` (replay markers vs targets) to ``output_path``."""
    require(candidate_path.exists(), "candidate file must exist", str(candidate_path))
    data = np.load(candidate_path)
    for key in ("time_s", "markers_m", "target_m", "valid"):
        require(key in data, f"candidate is missing '{key}'", key)
    return render_marker_overlay_animation(
        np.asarray(data["time_s"], dtype=float),
        np.asarray(data["target_m"], dtype=float),
        np.asarray(data["markers_m"], dtype=float),
        output_path,
        engine_name="pinocchio",
        stride=stride,
        valid_mask=np.asarray(data["valid"], dtype=bool),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args(argv)
    render_candidate_gif(args.candidate, args.out, stride=args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
