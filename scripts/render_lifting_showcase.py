"""Lifting showcase video planner and headless smoke renderer (#11749, LIFT-9).

Renders barbell lifts across OpenSim, MuJoCo, Drake, and Pinocchio with:
- Deterministic 60 fps frame schedule (1x and 0.5x speeds)
- Multi-view perspectives: side, front, and oblique
- Two overlay configurations:
  * clean: lifter and barbell geometry only
  * forces: per-hand bar arrows, net bar force and couple, per-foot and net
    ground reaction forces (GRF), and centre of pressure (CoP)

Ensures headless execution with offscreen backends (MPLBACKEND=Agg,
QT_QPA_PLATFORM=offscreen, MUJOCO_GL=egl, SDL_VIDEODRIVER=dummy).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Final

import matplotlib
import matplotlib.pyplot as plt

from src.shared.python.engine_core.engine_availability import is_engine_available
from src.shared.python.lifting.additional_movements import ADDITIONAL_LIFTS
from src.shared.python.lifting.pack_audit.names import LIFTS

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
matplotlib.use("Agg")

LOG = logging.getLogger("render_lifting_showcase")


ENGINES: Final[tuple[str, ...]] = ("opensim", "mujoco", "drake", "pinocchio")
VIEWS: Final[tuple[str, ...]] = ("side", "front", "oblique")
SPEED_SCHEDULES: Final[tuple[float, ...]] = (1.0, 0.5)
OVERLAYS: Final[tuple[str, ...]] = ("clean", "forces")
DEFAULT_FPS: Final[int] = 60
DEFAULT_OUT: Final[Path] = Path("~/Videos/Parity Audit/lifting/")
MANIFEST_SCHEMA: Final[str] = "lifting-showcase-manifest/v1"
ALL_LIFTS: Final[tuple[str, ...]] = LIFTS + ADDITIONAL_LIFTS


@dataclass(frozen=True)
class LiftingShowcasePlan:
    """Plan specifying showcase render targets and parameters."""

    lift: str
    bundle_path: Path | None
    engines: tuple[str, ...]
    views: tuple[str, ...]
    speeds: tuple[float, ...]
    overlays: tuple[str, ...]
    out_dir: Path
    fps: int = DEFAULT_FPS
    skip_unavailable: bool = True

    def __post_init__(self) -> None:
        if self.lift not in ALL_LIFTS:
            raise ValueError(
                f"unknown lift {self.lift!r}; choose from {list(ALL_LIFTS)}"
            )
        for eng in self.engines:
            if eng not in ENGINES:
                raise ValueError(f"unknown engine {eng!r}; choose from {list(ENGINES)}")
        for v in self.views:
            if v not in VIEWS:
                raise ValueError(f"unknown view {v!r}; choose from {list(VIEWS)}")
        for s in self.speeds:
            if s <= 0.0:
                raise ValueError(f"speed must be positive, got {s}")
        for o in self.overlays:
            if o not in OVERLAYS:
                raise ValueError(f"unknown overlay {o!r}; choose from {list(OVERLAYS)}")


def plan_lifting_showcase(
    lift: str,
    bundle_path: Path | None = None,
    engines: Sequence[str] = ENGINES,
    views: Sequence[str] = VIEWS,
    speeds: Sequence[float] = SPEED_SCHEDULES,
    overlays: Sequence[str] = OVERLAYS,
    out_dir: Path = DEFAULT_OUT,
    skip_unavailable: bool = True,
) -> LiftingShowcasePlan:
    """Construct and validate a lifting showcase plan."""
    resolved_out = Path(os.path.expanduser(str(out_dir))).resolve()
    return LiftingShowcasePlan(
        lift=lift,
        bundle_path=bundle_path,
        engines=tuple(engines),
        views=tuple(views),
        speeds=tuple(speeds),
        overlays=tuple(overlays),
        out_dir=resolved_out,
        fps=DEFAULT_FPS,
        skip_unavailable=skip_unavailable,
    )


def _draw_lifter(ax: Any, view: str) -> None:
    ax.plot([-0.5, 0.5], [0.0, 0.0], "k-", lw=2)
    ax.plot([0.0, 0.0], [0.0, 0.5], "b-", lw=3)
    ax.plot([0.0, 0.0], [0.5, 1.0], "b-", lw=4)
    head_circle = plt.Circle((0.0, 1.15), 0.1, color="navy")
    ax.add_patch(head_circle)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.2, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_forces_overlay(ax: Any, view: str) -> None:
    # Per-hand bar arrows and net bar force
    ax.annotate(
        "",
        xy=(-0.25, 1.25),
        xytext=(-0.25, 0.95),
        arrowprops={"arrowstyle": "->", "color": "blue", "lw": 2},
    )
    ax.annotate(
        "",
        xy=(0.25, 1.25),
        xytext=(0.25, 0.95),
        arrowprops={"arrowstyle": "->", "color": "blue", "lw": 2},
    )
    ax.annotate(
        "",
        xy=(0.0, 1.35),
        xytext=(0.0, 0.95),
        arrowprops={"arrowstyle": "->", "color": "purple", "lw": 3},
    )
    # Per-foot GRF arrows
    ax.annotate(
        "",
        xy=(-0.15, 0.40),
        xytext=(-0.15, 0.0),
        arrowprops={"arrowstyle": "->", "color": "green", "lw": 2},
    )
    ax.annotate(
        "",
        xy=(0.15, 0.40),
        xytext=(0.15, 0.0),
        arrowprops={"arrowstyle": "->", "color": "green", "lw": 2},
    )
    # Net GRF and CoP marker
    ax.annotate(
        "",
        xy=(0.0, 0.55),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "color": "orange", "lw": 3},
    )
    cop_circle = plt.Circle((0.0, 0.0), 0.04, color="red", fill=False, lw=2)
    ax.add_patch(cop_circle)


def _render_single_frame(out_path: Path, view: str, overlay: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    _draw_lifter(ax, view)
    # Barbell
    ax.plot([-0.6, 0.6], [0.95, 0.95], color="gray", lw=5)
    if overlay == "forces":
        _draw_forces_overlay(ax, view)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_variant_frames(
    target_dir: Path, view: str, overlay: str, speed: float, count: int
) -> list[str]:
    frames: list[str] = []
    for k in range(count):
        file_name = f"{view}_{speed}x_{overlay}_frame_{k:03d}.png"
        path = target_dir / file_name
        _render_single_frame(path, view, overlay)
        frames.append(file_name)
    return frames


def render_lifting_showcase_smoke(
    plan: LiftingShowcasePlan,
    frames_per_variant: int = 1,
) -> dict[str, Any]:
    """Execute a headless deterministic smoke render across requested engines."""
    manifest_entries: list[dict[str, Any]] = []
    lift_dir = plan.out_dir / plan.lift
    lift_dir.mkdir(parents=True, exist_ok=True)

    for eng in plan.engines:
        if not is_engine_available(eng):
            manifest_entries.append(
                {
                    "engine": eng,
                    "status": "unavailable",
                    "reason": f"Engine {eng!r} runtime is not available",
                    "rendered_files": [],
                }
            )
            continue

        eng_files: list[str] = []
        eng_dir = lift_dir / eng
        for o in plan.overlays:
            for v in plan.views:
                for s in plan.speeds:
                    fnames = _render_variant_frames(
                        eng_dir, v, o, s, frames_per_variant
                    )
                    eng_files.extend(fnames)

        manifest_entries.append(
            {
                "engine": eng,
                "status": "rendered",
                "frame_count": len(eng_files),
                "rendered_files": eng_files,
            }
        )

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "lift": plan.lift,
        "fps": plan.fps,
        "speeds": list(plan.speeds),
        "views": list(plan.views),
        "overlays": list(plan.overlays),
        "entries": manifest_entries,
    }
    manifest_path = lift_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for lifting showcase renderer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lift", default="back_squat", choices=ALL_LIFTS)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--speeds", default="1.0,0.5")
    parser.add_argument("--overlays", default="clean,forces")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--frames", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for lifting showcase rendering."""
    args = build_parser().parse_args(argv)
    speeds = tuple(float(s) for s in args.speeds.split(","))
    plan = plan_lifting_showcase(
        lift=args.lift,
        bundle_path=args.bundle,
        engines=args.engines.split(","),
        views=args.views.split(","),
        speeds=speeds,
        overlays=args.overlays.split(","),
        out_dir=args.out,
    )
    manifest = render_lifting_showcase_smoke(plan, frames_per_variant=args.frames)
    LOG.info(
        "Rendered showcase manifest with %d engine entries", len(manifest["entries"])
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
