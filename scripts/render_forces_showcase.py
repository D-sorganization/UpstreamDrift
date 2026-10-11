"""Owner showcase clips: ground-reaction forces, club forces, clean slow motion.

GCV-19 (#11725, epic #11706). One reproducible driver over the existing
``native_viewer_export`` renderer and encoder (no rendering or encoding is
reimplemented here). Per run it lays out::

    <out>/clean/         no overlays; 1x and 0.5x plus a 0.1x impact window
    <out>/club_forces/   hands close-up with the grip overlay (per-hand, net, couple)
    <out>/grf/           ground-reaction / joint-torque overlay, face-on and DTL
    <out>/manifest_<capture>.json

    MUJOCO_GL=egl python3 -m scripts.render_forces_showcase \\
        --capture capture-A --swing driver=driver.npz --swing iron=iron.npz \\
        --engines mujoco,drake

``--swing NAME=PATH`` is a same-input bundle ``.npz`` (see
``scripts/same_input_bundle.py``). Unavailable is never zero: a missing bundle
or an unavailable engine stops the run with a message (``--skip-unavailable``
records the engine as unavailable in the manifest and renders the rest).

The grf/ and club_forces/ clips carry the GCV-17 impact parameter card from the
impact frame onward (``native_viewer_export.impact_card``); each manifest clip
entry records the card (``impact_card.rows``) or the reason it is unavailable.

They also carry a synchronised inset plot strip with a moving time cursor
(``native_viewer_export.inset_strip``): ``grf/`` shows the vertical ground
reaction of the left foot, right foot and net in body weights plus the net CoP
(X, Y in metres); ``club_forces/`` shows the net grip force (N) and couple
(N m). The curves come once per swing from the source samples; unavailable
samples are gaps, never zero. Each manifest clip entry records the strip
(``inset_strip``: panels and per-curve sample counts) or the reason it is
unavailable. ``clean/`` carries neither the card nor a strip.

No known gaps remain in the exporter; the manifest still lists ``known_gaps``
(empty). Full clips are generated once the stale committed fixtures are refreshed
(#12223, #12230); this script only renders what its inputs support.
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

LOG = logging.getLogger("render_forces_showcase")

MANIFEST_SCHEMA: Final[str] = "forces-showcase-manifest/v1"
DEFAULT_OUT: Final[Path] = Path("~/Videos/Parity Audit/forces_and_impact/")
CAPTURES: Final[tuple[str, ...]] = ("capture-A", "capture-B", "capture-O")
HQ_SIZE: Final[tuple[int, int]] = (1280, 720)
FPS: Final[int] = 60
CRF: Final[int] = 18
DEFAULT_SPEEDS: Final[str] = "1,0.5"
IMPACT_WINDOW_S: Final[float] = 0.4
IMPACT_SPEED: Final[float] = 0.1
KNOWN_GAPS: Final[tuple[str, ...]] = ()


@dataclass(frozen=True)
class ClipKind:
    """One output sub-directory: its views and which overlays it carries."""

    views: tuple[str, ...]
    overlays: bool
    grip: bool = False

    @property
    def impact_card(self) -> bool:
        """The GCV-17 card rides on the overlay clips; ``clean`` stays clean."""
        return self.overlays

    @property
    def inset_strip(self) -> bool:
        """GRF strip on ``grf``, grip strip on ``club_forces``; none on ``clean``."""
        return self.overlays


CLIP_KINDS: Final[dict[str, ClipKind]] = {
    "clean": ClipKind(("face_on", "down_the_line"), overlays=False),
    "club_forces": ClipKind(("hands_closeup",), overlays=True, grip=True),
    "grf": ClipKind(("face_on", "down_the_line"), overlays=True),
}


class EngineUnavailableError(RuntimeError):
    """A requested engine cannot render here."""


@dataclass(frozen=True)
class ShowcasePlan:
    """Validated inputs for one run."""

    capture: str
    swings: dict[str, Path]
    engines: tuple[str, ...]
    out_dir: Path
    clips: tuple[str, ...]
    size: tuple[int, int]
    speeds: tuple[float, ...]
    impact_clip: bool
    skip_unavailable: bool


def build_parser() -> argparse.ArgumentParser:
    from src.tools.native_viewer_export.core import ENGINES

    p = argparse.ArgumentParser(
        prog="python3 -m scripts.render_forces_showcase", description=__doc__
    )
    p.add_argument("--capture", default="capture-A", help=f"one of {CAPTURES}")
    p.add_argument(
        "--swing",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="swing label and its same-input bundle .npz (repeatable, e.g. driver=...)",
    )
    p.add_argument(
        "--engines", default="mujoco", help=f"comma-separated, from {list(ENGINES)}"
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument(
        "--clips", default=",".join(CLIP_KINDS), help="comma-separated clip kinds"
    )
    p.add_argument("--size", default=f"{HQ_SIZE[0]}x{HQ_SIZE[1]}", help="WIDTHxHEIGHT")
    p.add_argument("--speeds", default=DEFAULT_SPEEDS, help="playback speeds")
    p.add_argument(
        "--no-impact-clip", action="store_true", help="omit the 0.1x impact window"
    )
    p.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="record unavailable engines in the manifest instead of failing",
    )
    return p


def resolve_out_dir(out: Path) -> Path:
    return Path(os.path.expanduser(str(out))).resolve()


def _split(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def plan_from_args(args: argparse.Namespace) -> ShowcasePlan:
    """Validate arguments (``ValueError`` / ``FileNotFoundError`` with a reason)."""
    from src.tools.native_viewer_export.cli import parse_size, parse_speeds
    from src.tools.native_viewer_export.core import ENGINES

    if args.capture not in CAPTURES:
        raise ValueError(f"capture must be one of {CAPTURES}, got {args.capture!r}")
    swings: dict[str, Path] = {}
    for item in args.swing:
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"--swing must look like NAME=PATH, got {item!r}")
        if name in swings:
            raise ValueError(f"duplicate swing name {name!r}")
        bundle = Path(path).expanduser()
        if not bundle.is_file():
            raise FileNotFoundError(f"bundle for swing {name!r} not found: {bundle}")
        swings[name.strip()] = bundle
    engines = _split(args.engines)
    if not engines:
        raise ValueError("engines must not be empty")
    for engine in engines:
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    clips = _split(args.clips)
    if not clips:
        raise ValueError("clips must not be empty")
    for clip in clips:
        if clip not in CLIP_KINDS:
            raise ValueError(f"unknown clip kind {clip!r}; expected {list(CLIP_KINDS)}")
    size = parse_size(args.size)
    speeds = parse_speeds(args.speeds)
    plan = ShowcasePlan(
        args.capture,
        swings,
        engines,
        resolve_out_dir(args.out),
        clips,
        size,
        speeds,
        not args.no_impact_clip,
        args.skip_unavailable,
    )
    settings_for(plan, "clean")  # validates speeds and size via ExportSettings
    return plan


def settings_for(plan: ShowcasePlan, kind: str) -> Any:
    """``ExportSettings`` of one clip kind (1280x720, 60 fps, libx264 CRF 18)."""
    from src.tools.native_viewer_export.core import ExportSettings

    spec = CLIP_KINDS[kind]
    extra: dict[str, Any] = {}
    if plan.impact_clip:
        extra = {"impact_window_s": IMPACT_WINDOW_S, "impact_speed": IMPACT_SPEED}
    return replace(
        ExportSettings(),
        views=spec.views,
        width=plan.size[0],
        height=plan.size[1],
        fps=FPS,
        speeds=plan.speeds,
        crf=CRF,
        overlays=spec.overlays,
        grip=spec.grip,
        impact_card=spec.impact_card,
        inset_strip=spec.inset_strip,
        multiview=False,
        **extra,
    )


def engine_unavailable_reason(engine: str) -> str | None:
    from src.tools.native_viewer_export.backends.registry import make_backend

    return make_backend(engine).unavailable_reason()


def check_engines(plan: ShowcasePlan) -> dict[str, str]:
    """Unavailable engines and why; raises unless ``skip_unavailable``."""
    missing = {}
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


def _clip_entry(
    kind: str, swing: str, engine: str, result: Any, out_dir: Path
) -> dict[str, Any]:
    entry: dict[str, Any] = {"kind": kind, "swing": swing, "engine": engine}
    if result.skipped:
        entry.update(status="unavailable", reason=result.skipped_reason, files=[])
        return entry
    files = sorted(p.relative_to(out_dir).as_posix() for p in result.paths.values())
    entry.update(
        status="rendered",
        files=files,
        frames_by_suffix=dict(result.frames_by_suffix),
    )
    if result.grip_json is not None:
        entry["grip_json"] = result.grip_json.relative_to(out_dir).as_posix()
    if result.impact_card is not None:  # card record, or why it is unavailable
        entry["impact_card"] = result.impact_card
    if result.inset_strip is not None:  # strip record, or why it is unavailable
        entry["inset_strip"] = result.inset_strip
    return entry


def run_showcase(
    plan: ShowcasePlan,
    *,
    runner: Callable[..., Any] | None = None,
    unavailable: dict[str, str] | None = None,
) -> Path:
    """Render every clip kind and swing; return the manifest path.

    Postcondition: ``clean/``, ``club_forces/`` and ``grf/`` (those requested)
    exist under the output directory and the manifest lists every clip as
    ``rendered`` or ``unavailable`` with the reason.
    """
    from src.tools.native_viewer_export.overlay import build_overlay_feed
    from src.tools.native_viewer_export.runner import ExportJob, run_export

    run = runner or run_export
    unavailable = unavailable or {}
    engines = tuple(e for e in plan.engines if e not in unavailable)
    entries: list[dict[str, Any]] = []
    for kind in plan.clips:
        kind_dir = plan.out_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        settings = settings_for(plan, kind)
        factory = partial(build_overlay_feed, grip=settings.grip)
        for name, bundle in plan.swings.items():
            for engine, reason in unavailable.items():
                entries.append(
                    {
                        "kind": kind,
                        "swing": name,
                        "engine": engine,
                        "status": "unavailable",
                        "reason": reason,
                        "files": [],
                    }
                )
            if not engines:
                continue
            job = ExportJob(
                bundle, kind_dir, f"{plan.capture}_{name}", name.title(), engines
            )
            LOG.info("rendering %s / %s (%s)", kind, name, ",".join(engines))
            for result in run(job, settings, overlay_factory=factory):
                entries.append(
                    _clip_entry(kind, name, result.engine, result, plan.out_dir)
                )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture": plan.capture,
        "swings": list(plan.swings),
        "inputs_sha256": {n: _sha256(p) for n, p in plan.swings.items()},
        "engines": list(plan.engines),
        "encoding": {
            "width": plan.size[0],
            "height": plan.size[1],
            "fps": FPS,
            "codec": "libx264",
            "pix_fmt": "yuv420p",
            "crf": CRF,
        },
        "speeds": list(plan.speeds),
        "impact_clip": (
            {"window_s": IMPACT_WINDOW_S, "speed": IMPACT_SPEED}
            if plan.impact_clip
            else None
        ),
        "git_commit": _git_commit(),
        "known_gaps": list(KNOWN_GAPS),
        "clips": entries,
    }
    path = plan.out_dir / f"manifest_{plan.capture}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
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
    rendered = sum(
        c["status"] == "rendered" for c in json.loads(manifest.read_text())["clips"]
    )
    sys.stdout.write(f"{rendered} clip sets rendered; manifest {manifest}\n")
    return 0 if rendered else 1


if __name__ == "__main__":
    sys.exit(main())
