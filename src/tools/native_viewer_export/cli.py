"""Command line for the native viewer export tool (NV-5, #11678)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
import json
import logging
from pathlib import Path
import sys
from typing import Any

from src.shared.python.golf_view_presets import VIEW_ORDER
from src.shared.python.model_appearance.appearance_selection import (
    AppearanceOverrides,
    resolve_appearance,
)
from src.shared.python.model_appearance.schema import document_from_dict
from src.tools.native_viewer_export.core import (
    DEFAULT_SPEEDS,
    ENGINES,
    ExportSettings,
)
from src.tools.native_viewer_export.overlay import build_overlay_feed
from src.tools.native_viewer_export.runner import ExportJob, run_export

logger = logging.getLogger(__name__)


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
    p.add_argument("--no-overlay", action="store_true", help="skip force/torque glyphs")
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
    add_appearance_arguments(p)
    return p


def add_appearance_arguments(p: argparse.ArgumentParser) -> None:
    """Appearance flags (HRG-B2, #13167). Precedence: flag, then ``--appearance``
    document, then the contract defaults; nothing is read from GUI settings."""
    group = p.add_argument_group(
        "appearance",
        "explicit flag > --appearance document or sidecar > contract defaults",
    )
    group.add_argument(
        "--appearance",
        type=Path,
        default=None,
        metavar="PATH",
        help="appearance-v1 document or <spec>.appearance.json sidecar",
    )
    group.add_argument(
        "--skin-tone",
        default=None,
        metavar="ID_OR_HEX",
        help="skin palette id (skin_1..skin_8) or a #rrggbb colour",
    )
    group.add_argument(
        "--hair",
        default=None,
        metavar="ID_OR_HEX",
        help="hair palette id (hair_brown, ...) or a #rrggbb colour",
    )
    group.add_argument(
        "--clothing", default=None, metavar="ID", help="clothing preset id"
    )
    group.add_argument(
        "--top-color", default=None, metavar="HEX", help="shirt colour #rrggbb"
    )
    group.add_argument(
        "--bottom-color",
        default=None,
        metavar="HEX",
        help="trousers or shorts colour #rrggbb",
    )


def load_appearance_document(path: Path) -> dict[str, Any]:
    """Parse an appearance-v1 document file (``ValueError`` if unreadable/invalid)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read --appearance {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"--appearance {path} must hold a JSON object")
    document_from_dict(data)  # validates (fail closed)
    return data


def build_appearance(args: argparse.Namespace) -> dict[str, Any] | None:
    """Resolved appearance document, or ``None`` when no appearance flag is given
    (backends then render the contract default, exactly as before)."""
    overrides = AppearanceOverrides(
        skin_tone=args.skin_tone,
        hair=args.hair,
        clothing=args.clothing,
        top_color=args.top_color,
        bottom_color=args.bottom_color,
    )
    if args.appearance is None and overrides.is_empty:
        return None
    document = (
        None if args.appearance is None else load_appearance_document(args.appearance)
    )
    return resolve_appearance(overrides, document)


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
        multiview=not args.no_grid,
        grip=args.grip,
        hud=not args.no_hud,
        ball=not args.no_ball,
        appearance=build_appearance(args),
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
            overlay_factory=partial(build_overlay_feed, grip=settings.grip),
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
