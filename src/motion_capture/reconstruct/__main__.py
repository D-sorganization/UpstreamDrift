"""``python3 -m motion_capture.reconstruct``: synthetic bundles and the joint fit.

- ``synth --out DIR``: write a synthetic three-view bundle with truth.
- ``fit --bundle DIR --anchor SEGMENT=METRES [--cameras records.json]``: run
  the joint camera/joint/bone-length fit and write ``reconstruction.json``
  (metrics against ``truth.json`` when present). Exit 0 when the fit ran.

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
from .fit import cameras_from_records, fit_bundle
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
    fit = sub.add_parser("fit", help="joint camera + skeleton fit on a bundle")
    fit.add_argument("--bundle", type=Path, required=True)
    fit.add_argument(
        "--anchor",
        action="append",
        required=True,
        metavar="SEGMENT=METRES",
        help="tape-measured segment, repeatable (shank=0.42 forearm=0.26 ...); "
        "everyday names apply to both sides; the first sets the scale",
    )
    fit.add_argument(
        "--cameras",
        type=Path,
        default=None,
        help="JSON list of CameraCalibration records to start from (default: truth.json)",
    )
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


def _parse_anchor(text: str) -> tuple[str, float]:
    name, sep, value = text.partition("=")
    if not sep or not name.strip():
        raise SystemExit("--anchor must look like SEGMENT=METRES, e.g. neck=0.53")
    try:
        metres = float(value)
    except ValueError as exc:
        raise SystemExit(f"--anchor length is not a number: {value!r}") from exc
    if metres <= 0:
        raise SystemExit("--anchor length must be positive")
    return name.strip(), metres


def cmd_fit(args: argparse.Namespace) -> int:
    import json

    start = None
    if args.cameras is not None:
        records = json.loads(args.cameras.read_text(encoding="utf-8"))
        start = cameras_from_records(records)
    from .measurements import expand_measurements, gauge

    try:
        measured = expand_measurements(args.anchor)
    except ValueError as exc:  # includes contract violations: a usage error
        raise SystemExit(f"--anchor: {exc}") from exc
    record = fit_bundle(
        args.bundle,
        scale_anchor=gauge(measured),
        start_cameras=start,
        measured_lengths_m=measured,
    )
    logger.info(
        "fit %s: rms %.2f px (start %.2f), %d rejected, %d unobservable points",
        args.bundle,
        record.rms_px,
        record.initial_rms_px,
        len(record.rejected),
        record.unobservable_points,
    )
    for view in record.views:
        logger.info(
            "  %s: %d obs, %d rejected, rms %s px",
            view.view,
            view.observations,
            view.rejected,
            view.rms_px and round(view.rms_px, 2),
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    return {"synth": cmd_synth, "fit": cmd_fit}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
