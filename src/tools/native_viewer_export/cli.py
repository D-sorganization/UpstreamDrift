"""Command line for the native viewer export tool (NV-5, #11678)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
import logging
from pathlib import Path
import sys

from src.shared.python.body_part_viz.force_colors import (
    BLEND_MODES,
    DEFAULT_COMPRESSION_COLOR,
    DEFAULT_TENSION_COLOR,
    ForceColorScale,
)
from src.shared.python.golf_view_presets import VIEW_ORDER
from src.tools.native_viewer_export.core import (
    DEFAULT_SPEEDS,
    ENGINES,
    ExportSettings,
)
from src.tools.native_viewer_export.overlay import build_overlay_feed
from src.tools.native_viewer_export.runner import ExportJob, run_export

logger = logging.getLogger(__name__)


def _add_overlay_arguments(p: argparse.ArgumentParser) -> None:
    """Add the overlay and segment-shading options."""
    p.add_argument(
        "--no-overlay",
        action="store_true",
        help="disable every overlay (force and torque vectors, segment shading)",
    )
    p.add_argument(
        "--force-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw straight force arrows (default: on)",
    )
    p.add_argument(
        "--torque-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw curling joint-torque arcs (default: on)",
    )
    shade = p.add_argument_group(
        "segment shading",
        "Translucent tension/compression shading of the golfer's segments "
        "(MuJoCo viewer only; axial loads from inverse dynamics, tension "
        "positive). Loads beyond the limit saturate.",
    )
    shade.add_argument(
        "--segment-shading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="shade segments by axial load (default: off)",
    )
    shade.add_argument(
        "--shading-opacity",
        type=float,
        default=0.6,
        metavar="0..1",
        help="shading strength, 0 (invisible) to 1 (full colour; default 0.6)",
    )
    shade.add_argument(
        "--shading-mode",
        choices=BLEND_MODES,
        default="tint",
        help="tint: unloaded segments keep their colour (default); "
        "replace: every loaded segment blends to the load colour",
    )
    shade.add_argument(
        "--tension-color",
        default=DEFAULT_TENSION_COLOR,
        metavar="#RRGGBB",
        help="tension colour (default %(default)s)",
    )
    shade.add_argument(
        "--compression-color",
        default=DEFAULT_COMPRESSION_COLOR,
        metavar="#RRGGBB",
        help="compression colour (default %(default)s)",
    )
    shade.add_argument(
        "--shading-limit-n",
        type=float,
        default=1000.0,
        metavar="N",
        help="load in newtons that saturates the colour, both signs (default 1000)",
    )
    shade.add_argument(
        "--tension-limit-n",
        type=float,
        default=None,
        metavar="N",
        help="tension saturation load (default: --shading-limit-n)",
    )
    shade.add_argument(
        "--compression-limit-n",
        type=float,
        default=None,
        metavar="N",
        help="compression saturation load (default: --shading-limit-n)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.tools.native_viewer_export",
        description=(
            "Render a same-input swing in each engine's native viewer (Drake MeshCat, "
            "Pinocchio MeshCat, OpenSim simbody under xvfb, MyoSuite arena, MuJoCo Renderer) as mp4 "
            "clips: one per camera view plus a labelled 2x2."
        ),
    )
    p.add_argument("--bundle", type=Path, required=True, help="same-input bundle .npz")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--swing", required=True, help="swing name used in file names")
    p.add_argument("--club", default="Driver", help="club label for the HUD")
    p.add_argument(
        "--engines", default=",".join(ENGINES), help="comma-separated engines"
    )
    for engine in ENGINES:
        p.add_argument(
            f"--receipt-{engine}",
            type=Path,
            help=f"{engine} same-input receipt .json (sibling .npz holds q)",
        )
    p.add_argument(
        "--views", default=",".join(VIEW_ORDER), help="comma-separated views"
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="DEPRECATED alias: render every Nth state (use --speeds and --fps)",
    )
    p.add_argument("--fps", type=int, default=60, help="video frame rate")
    p.add_argument(
        "--speeds",
        default=",".join(f"{v:g}" for v in DEFAULT_SPEEDS),
        help="comma-separated playback speeds, each in (0, 4]; one clip set per "
        "speed with _1x / _0p5x suffixes (default 1,0.5,0.25: full, half, quarter)",
    )
    p.add_argument(
        "--impact-window",
        type=float,
        default=None,
        metavar="SECONDS",
        help="also write a clip of this many swing seconds centred on impact "
        "at 0.1x speed (suffix _impact_0p1x)",
    )
    p.add_argument(
        "--impact-speed",
        type=float,
        default=None,
        help="playback speed of the --impact-window clip (default 0.1)",
    )
    p.add_argument(
        "--no-hud",
        action="store_true",
        help="clean frames: no view label, HUD or legend text",
    )
    p.add_argument(
        "--impact-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="impact time in the swing; default: bundle provenance impact_time_s, "
        "else the last sample",
    )
    p.add_argument(
        "--preset",
        choices=("hq", "preview"),
        default="hq",
        help="hq: 1280x720 tiles, libx264 yuv420p CRF 18 (default); "
        "preview: 640x544, CRF 23",
    )
    p.add_argument(
        "--size", default=None, help="tile size WIDTHxHEIGHT (overrides the preset)"
    )
    _add_overlay_arguments(p)
    p.add_argument(
        "--grip",
        action="store_true",
        help="add the grip overlay (per-hand, net and couple; GCV-10) and write a "
        "<swing>_<engine>_grip_wrench.json plot payload; pair with --views "
        "hands_closeup",
    )
    p.add_argument(
        "--no-ball",
        action="store_true",
        help="do not draw the decorative golf ball (shown until impact)",
    )
    p.add_argument("--no-grid", action="store_true", help="skip the 2x2 clip")
    return p


def parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"--size must look like 640x544, got {text!r}") from None
    return w, h


def parse_speeds(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise ValueError(f"--speeds must look like 1,0.5,0.25, got {text!r}") from None


def build_shading(args: argparse.Namespace) -> ForceColorScale | None:
    """The one ``ForceColorScale`` of the shading options (``ValueError`` if invalid).

    Postcondition: ``None`` when shading is off or ``--no-overlay`` is set.
    """
    if not args.segment_shading or args.no_overlay:
        return None
    tension, compression = (
        args.shading_limit_n if own is None else own
        for own in (args.tension_limit_n, args.compression_limit_n)
    )
    return ForceColorScale(
        enabled=True,
        tension_limit_n=tension,
        compression_limit_n=compression,
        tension_color=args.tension_color,
        compression_color=args.compression_color,
        opacity=args.shading_opacity,
        blend_mode=args.shading_mode,
    )


def build_settings(args: argparse.Namespace) -> ExportSettings:
    """``ExportSettings`` from parsed arguments (raises ``ValueError``)."""
    base = ExportSettings.preview() if args.preset == "preview" else ExportSettings()
    width, height = (
        parse_size(args.size) if args.size is not None else (base.width, base.height)
    )
    return replace(
        base,
        views=tuple(v for v in args.views.split(",") if v),
        width=width,
        height=height,
        fps=args.fps,
        speeds=parse_speeds(args.speeds),
        stride=args.stride,
        impact_time_s=args.impact_time,
        impact_window_s=args.impact_window,
        overlays=not args.no_overlay,
        shading=build_shading(args),
        force_vectors=args.force_vectors,
        torque_vectors=args.torque_vectors,
        multiview=not args.no_grid,
        grip=args.grip,
        hud=not args.no_hud,
        ball=not args.no_ball,
        **(
            {"impact_speed": args.impact_speed} if args.impact_speed is not None else {}
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.stride is not None:
        logger.warning(
            "--stride is deprecated: playback is time-based; use --speeds and --fps"
        )
    try:
        settings = build_settings(args)
        receipts = {
            e: getattr(args, f"receipt_{e}")
            for e in ENGINES
            if getattr(args, f"receipt_{e}") is not None
        }
        job = ExportJob(
            args.bundle,
            args.out,
            args.swing,
            args.club,
            tuple(e for e in args.engines.split(",") if e),
            receipts,
        )
        results = run_export(
            job,
            settings,
            overlay_factory=partial(
                build_overlay_feed,
                grip=settings.grip,
                axial=settings.shading_enabled,
                show_force=settings.force_vectors,
                show_torque=settings.torque_vectors,
            ),
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    for r in results:
        if r.skipped:
            sys.stdout.write(f"{r.engine}: SKIPPED ({r.skipped_reason})\n")
        else:
            if r.ball is not None:
                sys.stdout.write(f"{r.engine}: ball {r.ball}\n")
            sys.stdout.write(
                f"{r.engine}: "
                + ", ".join(
                    f"{n} frames ({s.lstrip('_')})"
                    for s, n in r.frames_by_suffix.items()
                )
                + " -> "
                + ", ".join(p.name for p in r.paths.values())
                + "\n"
            )
    return 0 if any(not r.skipped for r in results) else 1
