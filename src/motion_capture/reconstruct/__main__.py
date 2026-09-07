"""``python3 -m motion_capture.reconstruct synth``: write a synthetic bundle.

The bundle has the shape ``motion_capture.rig ingest`` produces (per-view
``observations/<view>.json``) plus ``truth.json``, so any reconstruction
stage can be exercised without cameras and scored with :mod:`.metrics`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from .cameras import PinholeCamera, intrinsics_from_fov, look_at
from .synthetic import RenderOptions, SyntheticScene, write_synthetic_bundle

logger = get_logger(__name__)

LAB_RIG: dict[str, tuple[float, float, float]] = {
    "face_on": (0.0, 1.2, 4.0),
    "down_line": (-4.0, 1.2, 0.0),
    "overhead": (1.5, 3.2, 2.5),
}


def lab_rig(
    width: int = 1920, height: int = 1200, fov_deg: float = 70.0
) -> list[PinholeCamera]:
    """Three cameras around a golfer at the origin looking at chest height."""
    k = intrinsics_from_fov(width, height, fov_deg)
    target = np.array([0.0, 1.0, 0.0])
    return [
        PinholeCamera(
            name, k, look_at(np.array(p), target), np.array(p), (width, height)
        )
        for name, p in LAB_RIG.items()
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motion_capture.reconstruct")
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("synth", help="render a synthetic three-view bundle")
    synth.add_argument("--out", type=Path, required=True)
    synth.add_argument("--frames", type=int, default=120)
    synth.add_argument("--fps", type=float, default=60.0)
    synth.add_argument("--noise-px", type=float, default=1.0)
    synth.add_argument("--occlusion", type=float, default=0.05)
    synth.add_argument("--outliers", type=float, default=0.02)
    synth.add_argument("--seed", type=int, default=0)
    return parser


def cmd_synth(args: argparse.Namespace) -> int:
    scene = SyntheticScene(lab_rig(), fps=args.fps, n_frames=args.frames)
    options = RenderOptions(
        noise_px=args.noise_px,
        occlusion_rate=args.occlusion,
        outlier_rate=args.outliers,
        seed=args.seed,
    )
    views, truth = scene.render(options)
    out = write_synthetic_bundle(args.out, views, truth)
    n_out = sum(len(v) for v in truth.outliers.values())
    n_occ = sum(len(v) for v in truth.occluded.values())
    logger.info(
        "synthetic bundle %s: %d views x %d frames, %d outliers, %d occluded",
        out,
        len(views),
        args.frames,
        n_out,
        n_occ,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    return {"synth": cmd_synth}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
