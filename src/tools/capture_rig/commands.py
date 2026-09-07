"""Argument vectors for the rig CLI, built without Qt so they can be tested.

The Capture Rig tool never re-implements a rig command. It launches
``python -m src.motion_capture.rig`` as a child process with the vectors
built here, so the desktop tool and the terminal run identical code and the
session bundle on disk is the same either way (#9619).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.motion_capture.rig.plan import CameraControls, CaptureMode
from src.shared.python.core.contracts import require

RIG_MODULE = "src.motion_capture.rig"

#: Advertised MJPEG modes of the ELP AR0234 units (measured in #9613).
MODE_PRESETS: tuple[CaptureMode, ...] = (
    CaptureMode(width=1920, height=1200, fps=60),
    CaptureMode(width=1920, height=1200, fps=120),
    CaptureMode(width=1920, height=1080, fps=120),
    CaptureMode(width=1280, height=960, fps=120),
    CaptureMode(width=1280, height=720, fps=120),
    CaptureMode(width=1280, height=720, fps=200),
    CaptureMode(width=640, height=480, fps=200),
)


def repo_root() -> Path:
    """The checkout the rig module is imported from (the child's cwd)."""
    return Path(__file__).resolve().parents[3]


def mode_text(mode: CaptureMode) -> str:
    """``WxH@FPS:FOURCC`` as ``rig --mode`` parses it."""
    return f"{mode.width}x{mode.height}@{mode.fps}:{mode.fourcc}"


@dataclass(frozen=True)
class PlanSelection:
    """A plan file plus the operator overrides every camera command accepts."""

    plan: Path
    mode: CaptureMode | None = None
    views: tuple[str, ...] = ()
    controls: CameraControls = field(default_factory=CameraControls)

    def __post_init__(self) -> None:
        require(str(self.plan).strip() != "", "plan path must not be blank")

    def args(self) -> list[str]:
        out = ["--plan", str(self.plan)]
        if self.mode is not None:
            out += ["--mode", mode_text(self.mode)]
        if self.views:
            out += ["--views", ",".join(self.views)]
        if self.controls.exposure is not None:
            out += ["--exposure", f"{self.controls.exposure:g}"]
        if self.controls.gain is not None:
            out += ["--gain", f"{self.controls.gain:g}"]
        if self.controls.auto_exposure is not None:
            out += ["--auto-exposure", "on" if self.controls.auto_exposure else "off"]
        return out


def python_module_command(args: Sequence[str]) -> list[str]:
    """``[python, -m, src.motion_capture.rig, *args]``; run from :func:`repo_root`."""
    return [sys.executable, "-m", RIG_MODULE, *args]


def plan_check_command(selection: PlanSelection) -> list[str]:
    return python_module_command(["plan-check", *selection.args()])


def record_command(
    selection: PlanSelection,
    out: Path,
    *,
    duration_s: float = 10.0,
    warmup_s: float | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Precondition: a positive duration."""
    require(duration_s > 0, "duration must be positive", duration_s)
    args = ["record", *selection.args(), "--duration", f"{duration_s:g}", "--out"]
    args.append(str(out))
    if warmup_s is not None:
        args += ["--warmup", f"{warmup_s:g}"]
    if dry_run:
        args.append("--dry-run")
    return python_module_command(args)


def proxy_command(session: Path, *, encoder: str | None = None) -> list[str]:
    args = ["proxy", "--session", str(session)]
    if encoder:
        args += ["--encoder", encoder]
    return python_module_command(args)


def ingest_command(
    session: Path, *, estimator: str = "mediapipe", max_frames: int | None = None
) -> list[str]:
    require(estimator.strip() != "", "estimator must be named")
    args = ["ingest", "--session", str(session), "--estimator", estimator]
    if max_frames is not None:
        require(max_frames > 0, "max_frames must be positive", max_frames)
        args += ["--max-frames", str(max_frames)]
    return python_module_command(args)


def reconstruct_command(
    session: Path,
    *,
    anchor_segment: str,
    anchor_m: float,
    cameras: Path | None = None,
    intrinsics: Path | None = None,
) -> list[str]:
    """Exactly one of ``cameras`` (later take) or ``intrinsics`` (first take)."""
    require(anchor_segment.strip() != "", "anchor segment must be named")
    require(anchor_m > 0, "anchor length must be positive metres", anchor_m)
    require(
        (cameras is None) != (intrinsics is None),
        "give a cameras file or an intrinsics file, not both",
    )
    args = ["reconstruct", "--session", str(session)]
    args += ["--anchor", f"{anchor_segment}={anchor_m:g}"]
    if cameras is not None:
        args += ["--cameras", str(cameras)]
    else:
        args += ["--intrinsics", str(intrinsics)]
    return python_module_command(args)


def calibrate_command(
    session: Path, *, board: str = "9x6", square_m: float = 0.025, every: int = 10
) -> list[str]:
    require(square_m > 0, "square size must be positive", square_m)
    args = ["calibrate-intrinsics", "--session", str(session), "--board", board]
    args += ["--square", f"{square_m:g}", "--every", str(every)]
    return python_module_command(args)


@dataclass(frozen=True)
class EstimatorChoice:
    """One registry estimator as the tool offers it."""

    name: str
    display_name: str
    available: bool
    hint: str | None


def estimator_choices() -> tuple[EstimatorChoice, ...]:
    """Every registered estimator with its availability on this host.

    Registry-driven (#8392): nothing is offered that is not implemented.
    """
    from src.shared.python.pose_estimation.registry import (
        estimator_availability,
        list_estimators,
    )

    out = []
    for info in list_estimators():
        ok, reason = estimator_availability(info.name)
        out.append(EstimatorChoice(info.name, info.display_name, ok, reason))
    return tuple(out)
