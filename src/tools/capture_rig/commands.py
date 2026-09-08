"""Argument vectors for the rig CLI, built without Qt so they can be tested.

The Capture Rig tool never re-implements a rig command. It launches
``python -m src.motion_capture.rig`` as a child process with the vectors
built here, so the desktop tool and the terminal run identical code and the
session bundle on disk is the same either way (#9619).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
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


def child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The child's environment: ``<repo>/src`` first on ``PYTHONPATH``.

    The pose stack imports ``bunkershot3d`` and friends by their bare names,
    which resolve only with ``src`` on the path (the test suite adds it the
    same way). Postcondition: every other variable of ``base`` is kept.
    """
    env = dict(os.environ if base is None else base)
    src = str(repo_root() / "src")
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p and p != src]
    env["PYTHONPATH"] = os.pathsep.join([src, *parts])
    return env


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


@dataclass(frozen=True)
class OptionSpec:
    """One estimator setting the tool exposes: name, kind, default, help."""

    name: str
    kind: str  # "float" | "int" | "bool" | "str"
    default: float | int | bool | str
    help: str
    minimum: float | None = None
    maximum: float | None = None


#: Settings per estimator, mirrored from the estimator constructors (#9661).
ESTIMATOR_OPTIONS: dict[str, tuple[OptionSpec, ...]] = {
    "mediapipe": (
        OptionSpec("min_detection_confidence", "float", 0.5, "pose found", 0.0, 1.0),
        OptionSpec("min_tracking_confidence", "float", 0.5, "pose kept", 0.0, 1.0),
        OptionSpec("model_variant", "str", "full", "lite | full | heavy"),
        OptionSpec("enable_temporal_smoothing", "bool", True, "MediaPipe smoothing"),
    ),
    "openpose_dnn": (
        OptionSpec("input_height", "int", 368, "network input rows", 64, 1024),
        OptionSpec("min_peak", "float", 0.1, "heat-map peak threshold", 0.0, 0.99),
    ),
}


def option_args(options: dict[str, float | int | bool | str]) -> list[str]:
    """``--option k=v`` pairs; bools as true/false."""
    out: list[str] = []
    for key, value in options.items():
        text = str(value).lower() if isinstance(value, bool) else f"{value}"
        out += ["--option", f"{key}={text}"]
    return out


def ingest_command(
    session: Path,
    *,
    estimator: str = "mediapipe",
    max_frames: int | None = None,
    options: dict[str, float | int | bool | str] | None = None,
    out: Path | None = None,
) -> list[str]:
    """``out`` defaults to ``observations`` (the set reconstruct reads)."""
    require(estimator.strip() != "", "estimator must be named")
    args = ["ingest", "--session", str(session), "--estimator", estimator]
    if max_frames is not None:
        require(max_frames > 0, "max_frames must be positive", max_frames)
        args += ["--max-frames", str(max_frames)]
    args += option_args(options or {})
    if out is not None:
        args += ["--out", str(out)]
    return python_module_command(args)


def compare_command(
    session: Path,
    *,
    estimators: tuple[str, str] = ("mediapipe", "openpose_dnn"),
    max_frames: int | None = None,
) -> list[str]:
    args = ["compare", "--session", str(session), "--estimators", ",".join(estimators)]
    if max_frames is not None:
        args += ["--max-frames", str(max_frames)]
    return python_module_command(args)


def import_command(
    out: Path, views: Sequence[tuple[str, Path]], *, name: str | None = None
) -> list[str]:
    require(len(views) >= 1, "import needs at least one view")
    args = ["import", "--out", str(out)]
    for view, path in views:
        args += ["--view", f"{view}={path}"]
    if name:
        args += ["--name", name]
    return python_module_command(args)


def reliability_command(session: Path) -> list[str]:
    return python_module_command(["reliability", "--session", str(session)])


def analyze_command(session: Path, *, observations: str = "observations") -> list[str]:
    args = ["analyze", "--session", str(session), "--observations", observations]
    return python_module_command(args)


def clip_command(
    session: Path,
    view: str,
    out: Path,
    *,
    start: str = "address-30",
    end: str = "finish+30",
    speed: float = 0.25,
    observation_set: str | None = None,
) -> list[str]:
    require(view.strip() != "", "view must be named")
    require(0.0 < speed <= 1.0, "speed must be in (0, 1]", speed)
    args = ["clip", "--session", str(session), "--view", view, "--from", start]
    args += ["--to", end, "--speed", f"{speed:g}", "--out", str(out)]
    if observation_set:
        args += ["--set", observation_set]
    return python_module_command(args)


def compare_takes_command(
    session: Path,
    view: str,
    other_session: Path,
    other_view: str,
    out: Path,
    *,
    align: str = "top",
    speed: float = 0.5,
) -> list[str]:
    require(align in ("address", "top", "peak", "finish"), "align event", align)
    args = ["compare-takes", "--session", str(session), "--view", view]
    args += ["--other-session", str(other_session), "--other-view", other_view]
    args += ["--align", align, "--speed", f"{speed:g}", "--out", str(out)]
    return python_module_command(args)


def fit_model_command(
    session: Path, *, sigma_accel: float = 300.0, max_velocity: float = 25.0
) -> list[str]:
    require(sigma_accel > 0 and max_velocity > 0, "positive priors")
    args = ["fit-model", "--session", str(session), "--sigma-accel", f"{sigma_accel:g}"]
    args += ["--max-velocity", f"{max_velocity:g}"]
    return python_module_command(args)


def export_command(session: Path) -> list[str]:
    return python_module_command(["export", "--session", str(session)])


def reconstruct_command(
    session: Path,
    *,
    measurements: Sequence[str],
    cameras: Path | None = None,
    intrinsics: Path | None = None,
    exclude_joints: Sequence[str] = (),
) -> list[str]:
    """Exactly one of ``cameras`` (later take) or ``intrinsics`` (first take)."""
    require(len(measurements) >= 1, "at least one measured segment is needed")
    for item in measurements:
        require("=" in item, "measurement must be NAME=METRES", item)
    require(
        (cameras is None) != (intrinsics is None),
        "give a cameras file or an intrinsics file, not both",
    )
    args = ["reconstruct", "--session", str(session)]
    for item in measurements:
        args += ["--anchor", item]
    if cameras is not None:
        args += ["--cameras", str(cameras)]
    else:
        args += ["--intrinsics", str(intrinsics)]
    if exclude_joints:
        args += ["--exclude-joints", ",".join(exclude_joints)]
    return python_module_command(args)


def calibrate_command(
    session: Path, *, board: str = "9x6", square_m: float = 0.025, every: int = 10
) -> list[str]:
    require(square_m > 0, "square size must be positive", square_m)
    args = ["calibrate-intrinsics", "--session", str(session), "--board", board]
    if not board.lower().startswith("charuco:"):
        args += ["--square", f"{square_m:g}"]
    args += ["--every", str(every)]
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
