"""Owner showcase clips: golfer realism, OpenSim club, two-hand grip, gaze, feet.

OSV-5 (#11731, epic #11726). Headless-capable showcase runner and manifest planner
for the realistic golfer presentation across five engines (MuJoCo, MyoSuite,
Drake, Pinocchio, OpenSim), two clubs (driver, 7-iron), 60 fps at 1x and 0.5x,
and three views (face_on, dtl, hands_closeup).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Final

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

LOG = logging.getLogger("render_golfer_realism_showcase")

MANIFEST_SCHEMA: Final[str] = "golfer-realism-showcase-manifest/v1"
DEFAULT_OUT: Final[Path] = Path("~/Videos/Parity Audit/golfer_realism/")
CAPTURES: Final[tuple[str, ...]] = ("capture-A", "capture-B", "capture-O")
DEFAULT_ENGINES: Final[tuple[str, ...]] = (
    "mujoco",
    "myosuite",
    "drake",
    "pinocchio",
    "opensim",
)
DEFAULT_VIEWS: Final[tuple[str, ...]] = ("face_on", "dtl", "hands_closeup")
VIEW_CANONICAL: Final[dict[str, str]] = {
    "face_on": "face_on",
    "dtl": "down_the_line",
    "down_the_line": "down_the_line",
    "hands_closeup": "hands_closeup",
}
HQ_SIZE: Final[tuple[int, int]] = (1280, 720)
FPS: Final[int] = 60
CRF: Final[int] = 18
DEFAULT_SPEEDS: Final[str] = "1,0.5"
REALISM_FEATURES: Final[dict[str, bool]] = {
    "club": True,
    "wrapped_grip": True,
    "gaze_stabilized_head": True,
    "neutral_address_toe_out": True,
    "decorative_ball": True,
}


class EngineUnavailableError(RuntimeError):
    """A requested engine cannot render in this environment."""


@dataclass(frozen=True)
class ShowcasePlan:
    """Validated inputs for one golfer realism showcase run."""

    capture: str
    swings: dict[str, Path]
    engines: tuple[str, ...]
    views: tuple[str, ...]
    out_dir: Path
    size: tuple[int, int]
    speeds: tuple[float, ...]
    fps: int = FPS
    overlays: bool = False
    decorative_ball: bool = True
    skip_unavailable: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line parser for the golfer realism showcase."""
    p = argparse.ArgumentParser(
        prog="python -m scripts.render_golfer_realism_showcase",
        description=__doc__,
    )
    p.add_argument("--capture", default="capture-A", help=f"one of {CAPTURES}")
    p.add_argument(
        "--swing",
        action="append",
        metavar="NAME=PATH",
        help="swing label and same-input bundle .npz (repeatable, e.g. driver=...)",
    )
    p.add_argument(
        "--driver",
        type=Path,
        default=None,
        help="convenience shorthand for --swing driver=PATH",
    )
    p.add_argument(
        "--iron7",
        type=Path,
        default=None,
        help="convenience shorthand for --swing 7_iron=PATH",
    )
    p.add_argument(
        "--engines",
        default=",".join(DEFAULT_ENGINES),
        help=f"comma-separated engines, default: {','.join(DEFAULT_ENGINES)}",
    )
    p.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEWS),
        help=f"comma-separated views, default: {','.join(DEFAULT_VIEWS)}",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--size", default=f"{HQ_SIZE[0]}x{HQ_SIZE[1]}", help="WIDTHxHEIGHT")
    p.add_argument(
        "--speeds", default=DEFAULT_SPEEDS, help="playback speeds (e.g. 1,0.5)"
    )
    p.add_argument("--fps", type=int, default=FPS, help="video frame rate (default 60)")
    p.add_argument(
        "--overlays",
        action="store_true",
        help="enable force/torque glyph overlays (default: clean frames)",
    )
    p.add_argument(
        "--no-ball",
        action="store_true",
        help="omit decorative golf ball (shown by default)",
    )
    p.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="record unavailable engines in manifest rather than failing",
    )
    return p


def resolve_out_dir(out: Path) -> Path:
    """Expand user tilde and return absolute path."""
    return Path(os.path.expanduser(str(out))).resolve()


def _split(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def plan_from_args(args: argparse.Namespace) -> ShowcasePlan:
    """Validate arguments and produce a validated ShowcasePlan.

    Raises:
        ValueError: when capture, swing syntax, engine, view, size, or speed is invalid.
        FileNotFoundError: when a referenced bundle .npz does not exist.
    """
    from src.tools.native_viewer_export.cli import parse_size, parse_speeds
    from src.tools.native_viewer_export.core import ENGINES

    if args.capture not in CAPTURES:
        raise ValueError(f"capture must be one of {CAPTURES}, got {args.capture!r}")

    swings: dict[str, Path] = {}
    if args.swing:
        for item in args.swing:
            name, sep, path = item.partition("=")
            if not sep or not name.strip() or not path.strip():
                raise ValueError(f"--swing must look like NAME=PATH, got {item!r}")
            clean_name = name.strip()
            if clean_name in swings:
                raise ValueError(f"duplicate swing name {clean_name!r}")
            bundle = Path(path).expanduser()
            if not bundle.is_file():
                raise FileNotFoundError(
                    f"bundle for swing {clean_name!r} not found: {bundle}"
                )
            swings[clean_name] = bundle

    if args.driver is not None:
        if "driver" in swings:
            raise ValueError("duplicate swing name 'driver'")
        bundle = Path(args.driver).expanduser()
        if not bundle.is_file():
            raise FileNotFoundError(f"bundle for swing 'driver' not found: {bundle}")
        swings["driver"] = bundle

    if args.iron7 is not None:
        if "7_iron" in swings:
            raise ValueError("duplicate swing name '7_iron'")
        bundle = Path(args.iron7).expanduser()
        if not bundle.is_file():
            raise FileNotFoundError(f"bundle for swing '7_iron' not found: {bundle}")
        swings["7_iron"] = bundle

    if not swings:
        raise ValueError(
            "at least one swing must be specified via --swing, --driver, or --iron7"
        )

    engines = _split(args.engines)
    if not engines:
        raise ValueError("engines must not be empty")
    for engine in engines:
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")

    views = _split(args.views)
    if not views:
        raise ValueError("views must not be empty")
    for view in views:
        if view not in VIEW_CANONICAL:
            raise ValueError(
                f"unknown view {view!r}; expected one of {list(VIEW_CANONICAL)}"
            )

    size = parse_size(args.size)
    speeds = parse_speeds(args.speeds)
    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    plan = ShowcasePlan(
        capture=args.capture,
        swings=swings,
        engines=engines,
        views=views,
        out_dir=resolve_out_dir(args.out),
        size=size,
        speeds=speeds,
        fps=args.fps,
        overlays=args.overlays,
        decorative_ball=not args.no_ball,
        skip_unavailable=args.skip_unavailable,
    )
    settings_for(plan)
    return plan


def settings_for(plan: ShowcasePlan) -> Any:
    """Create ExportSettings for the plan (validates speed and dimensions)."""
    from src.tools.native_viewer_export.core import ExportSettings

    canonical_views = tuple(VIEW_CANONICAL[v] for v in plan.views)
    return replace(
        ExportSettings(),
        views=canonical_views,
        width=plan.size[0],
        height=plan.size[1],
        fps=plan.fps,
        speeds=plan.speeds,
        crf=CRF,
        overlays=plan.overlays,
        ball=plan.decorative_ball,
        multiview=False,
        hud=True,
    )


def engine_unavailable_reason(engine: str) -> str | None:
    """Report why an engine cannot render, or None if ready."""
    from src.tools.native_viewer_export.backends.registry import make_backend

    return make_backend(engine).unavailable_reason()


def check_engines(plan: ShowcasePlan) -> dict[str, str]:
    """Check availability of requested engines.

    Raises:
        EngineUnavailableError: when an engine is unavailable and skip_unavailable is False.
    """
    missing: dict[str, str] = {}
    for engine in plan.engines:
        reason = engine_unavailable_reason(engine)
        if reason is not None:
            missing[engine] = reason
    if missing and not plan.skip_unavailable:
        detail = "; ".join(f"{e}: {r}" for e, r in missing.items())
        raise EngineUnavailableError(
            f"engine unavailable ({detail}); install it or pass --skip-unavailable"
        )
    return missing


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _clip_entries_for_result(
    swing_name: str,
    engine: str,
    views: Sequence[str],
    result: Any,
    out_dir: Path,
    speeds: Sequence[float],
    fps: int,
) -> list[dict[str, Any]]:
    """Produce structured clip entries per view for one engine export result."""
    entries: list[dict[str, Any]] = []
    for view in views:
        canonical = VIEW_CANONICAL[view]
        entry: dict[str, Any] = {
            "swing": swing_name,
            "engine": engine,
            "view": view,
            "canonical_view": canonical,
            "speeds": list(speeds),
            "fps": fps,
        }
        if result.skipped:
            entry.update(
                status="unavailable",
                reason=result.skipped_reason,
                files=[],
                frames_by_suffix={},
            )
        else:
            matching_files: list[str] = []
            matching_frames: dict[str, int] = {}
            for path_key, clip_path in result.paths.items():
                if path_key.startswith(f"{canonical}_") or path_key == canonical:
                    matching_files.append(clip_path.relative_to(out_dir).as_posix())
                    suffix = path_key[len(canonical) :]
                    if suffix in result.frames_by_suffix:
                        matching_frames[suffix] = result.frames_by_suffix[suffix]
            matching_files.sort()
            entry.update(
                status="rendered",
                reason=None,
                files=matching_files,
                frames_by_suffix=matching_frames,
            )
        entries.append(entry)
    return entries


def _clip_entries_for_unavailable(
    swing_name: str,
    engine: str,
    reason: str,
    views: Sequence[str],
    speeds: Sequence[float],
    fps: int,
) -> list[dict[str, Any]]:
    """Produce unavailable clip entries for an engine that cannot render."""
    entries: list[dict[str, Any]] = []
    for view in views:
        entries.append(
            {
                "swing": swing_name,
                "engine": engine,
                "view": view,
                "canonical_view": VIEW_CANONICAL[view],
                "status": "unavailable",
                "reason": reason,
                "files": [],
                "speeds": list(speeds),
                "fps": fps,
                "frames_by_suffix": {},
            }
        )
    return entries


def run_showcase(
    plan: ShowcasePlan,
    *,
    runner: Callable[..., Any] | None = None,
    unavailable: dict[str, str] | None = None,
) -> Path:
    """Render golfer realism showcase clips and write manifest.

    Postconditions:
        - Output directory exists.
        - Per-swing subdirectories exist.
        - Manifest file is written and returned.
    """
    from src.tools.native_viewer_export.overlay import build_overlay_feed
    from src.tools.native_viewer_export.runner import ExportJob, run_export

    run = runner or run_export
    unavailable = unavailable or {}
    active_engines = tuple(e for e in plan.engines if e not in unavailable)
    entries: list[dict[str, Any]] = []

    plan.out_dir.mkdir(parents=True, exist_ok=True)
    settings = settings_for(plan)
    factory = partial(build_overlay_feed, grip=True) if plan.overlays else None

    for name, bundle in plan.swings.items():
        swing_dir = plan.out_dir / name
        swing_dir.mkdir(parents=True, exist_ok=True)

        for engine, reason in unavailable.items():
            entries.extend(
                _clip_entries_for_unavailable(
                    name, engine, reason, plan.views, plan.speeds, plan.fps
                )
            )

        if not active_engines:
            continue

        job = ExportJob(
            bundle_path=bundle,
            out_dir=swing_dir,
            swing=f"{plan.capture}_{name}",
            club=name.replace("_", " ").title(),
            engines=active_engines,
        )
        LOG.info(
            "rendering %s (%s) views: %s",
            name,
            ",".join(active_engines),
            ",".join(plan.views),
        )
        results = run(job, settings, overlay_factory=factory)
        for result in results:
            entries.extend(
                _clip_entries_for_result(
                    name,
                    result.engine,
                    plan.views,
                    result,
                    plan.out_dir,
                    plan.speeds,
                    plan.fps,
                )
            )

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture": plan.capture,
        "swings": list(plan.swings),
        "inputs_sha256": {n: _sha256(p) for n, p in plan.swings.items()},
        "engines": list(plan.engines),
        "views": list(plan.views),
        "speeds": list(plan.speeds),
        "fps": plan.fps,
        "encoding": {
            "width": plan.size[0],
            "height": plan.size[1],
            "fps": plan.fps,
            "codec": "libx264",
            "pix_fmt": "yuv420p",
            "crf": CRF,
        },
        "realism_features": dict(REALISM_FEATURES),
        "git_commit": _git_commit(),
        "summary": {
            "total_clips": len(entries),
            "rendered": sum(1 for c in entries if c["status"] == "rendered"),
            "unavailable": sum(1 for c in entries if c["status"] == "unavailable"),
        },
        "clips": entries,
    }
    manifest_path = plan.out_dir / f"manifest_{plan.capture}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for golfer realism showcase."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        plan = plan_from_args(args)
        unavailable = check_engines(plan)
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except EngineUnavailableError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    manifest = run_showcase(plan, unavailable=unavailable)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    rendered = doc["summary"]["rendered"]
    total = doc["summary"]["total_clips"]
    sys.stdout.write(f"{rendered}/{total} clip sets rendered; manifest {manifest}\n")
    return 0 if (rendered > 0 or plan.skip_unavailable) else 1


if __name__ == "__main__":
    sys.exit(main())
