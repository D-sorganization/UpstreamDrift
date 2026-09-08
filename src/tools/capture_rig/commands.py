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
    live_preview: Path | None = None,
    stop_file: Path | None = None,
    cameras: Mapping[str, str] | None = None,
) -> list[str]:
    """Precondition: a positive duration. ``live_preview``/``stop_file`` pass
    the recorder's live snapshot directory and early-stop file through.
    """
    require(duration_s > 0, "duration must be positive", duration_s)
    args = ["record", *selection.args(), "--duration", f"{duration_s:g}", "--out"]
    args.append(str(out))
    if live_preview is not None:
        args += ["--live-preview", str(live_preview)]
    if stop_file is not None:
        args += ["--stop-file", str(stop_file)]
    for view, instance in (cameras or {}).items():
        args += ["--camera", f"{view}={instance}"]
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


def variant_args(variant: str = "") -> list[str]:
    """``--variant NAME`` when a named match is requested (#9793)."""
    return ["--variant", variant] if variant else []


@dataclass(frozen=True)
class ImageSpaceArgs:
    """An image-space fit request: the views, whose cameras, which set (#9794).

    ``cameras_from == ""`` is the session's default variant.
    """

    views: tuple[str, ...]
    cameras_from: str = ""
    observations: str = "observations"

    def __post_init__(self) -> None:
        require(len(self.views) >= 1, "an image-space fit needs a view")

    def args(self) -> list[str]:
        return [
            "--from-views",
            ",".join(self.views),
            "--cameras-from",
            self.cameras_from,
            "--observations",
            self.observations,
        ]


def fit_model_command(
    session: Path,
    *,
    model: str = "golfer",
    sigma_accel: float = 300.0,
    max_velocity: float = 25.0,
    fit_lengths: bool = False,
    variant: str = "",
    image_space: ImageSpaceArgs | None = None,
) -> list[str]:
    """``image_space`` selects the image-space fit (1..N views, borrowed cameras)."""
    require(sigma_accel > 0 and max_velocity > 0, "positive priors")
    require(model.strip() != "", "model must be named")
    args = ["fit-model", "--session", str(session), "--model", model]
    args += ["--sigma-accel", f"{sigma_accel:g}", "--max-velocity", f"{max_velocity:g}"]
    if fit_lengths:
        args.append("--fit-lengths")
    args += variant_args(variant)
    if image_space is not None:
        args += image_space.args()
    return python_module_command(args)


def compare_models_command(
    session: Path,
    *,
    models: Sequence[str] = (),
    fit_lengths: bool = False,
    variant: str = "",
) -> list[str]:
    args = ["compare-models", "--session", str(session)]
    if models:
        args += ["--models", ",".join(models)]
    if fit_lengths:
        args.append("--fit-lengths")
    return python_module_command(args + variant_args(variant))


def kinetics_command(
    session: Path, *, model: str, body_mass_kg: float, variant: str = ""
) -> list[str]:
    require(body_mass_kg > 0, "body mass must be positive kg", body_mass_kg)
    args = ["kinetics", "--session", str(session), "--model", model]
    args += ["--body-mass", f"{body_mass_kg:g}"]
    return python_module_command(args + variant_args(variant))


def overlay_command(
    session: Path,
    view: str,
    out: Path,
    *,
    variants: Sequence[str] = ("",),
    speed: float = 1.0,
    observations: str = "observations",
) -> list[str]:
    """``rig overlay`` for one view and one or more variants (#9795)."""
    require(view.strip() != "", "view must be named")
    require(speed > 0, "speed must be positive", speed)
    args = ["overlay", "--session", str(session), "--view", view, "--out", str(out)]
    for name in variants:
        args += ["--variant", name]
    args += ["--speed", f"{speed:g}", "--observations", observations]
    return python_module_command(args)


@dataclass(frozen=True)
class MultipictureArgs:
    """Optional arguments of ``rig multipicture`` (parameter budget)."""

    variants: Sequence[str] = ()
    observation_set: str | None = None
    start: int | None = None
    stop: int | None = None
    speed: float = 1.0
    size: tuple[int, int] | None = None


def multipicture_command(
    session: Path,
    layout: str,
    out: Path,
    args_spec: MultipictureArgs | None = None,
) -> list[str]:
    """``rig multipicture``: composite video through a layout (#9815).

    ``layout`` is a preset name, a saved layout name or a JSON path.
    Preconditions: a named layout, positive speed, positive size when given.
    """
    require(layout.strip() != "", "layout must be named")
    spec = args_spec or MultipictureArgs()
    variants, observation_set = spec.variants, spec.observation_set
    start, stop, speed, size = spec.start, spec.stop, spec.speed, spec.size
    require(speed > 0, "speed must be positive", speed)
    args = ["multipicture", "--session", str(session), "--layout", layout]
    args += ["--out", str(out)]
    if variants:
        args += ["--variants", *variants]
    if observation_set:
        args += ["--set", observation_set]
    if start is not None:
        args += ["--from", str(start)]
    if stop is not None:
        args += ["--to", str(stop)]
    args += ["--speed", f"{speed:g}"]
    if size is not None:
        require(size[0] > 0 and size[1] > 0, "size must be positive", size)
        args += ["--size", f"{size[0]}x{size[1]}"]
    return python_module_command(args)


def compare_variants_command(session: Path, *, reference: str = "") -> list[str]:
    return python_module_command(
        ["compare-variants", "--session", str(session), "--reference", reference]
    )


def annotations_command(
    session: Path, *, views: Sequence[str] = (), merge_with: str = ""
) -> list[str]:
    """``rig annotations-to-observations`` (#9801, #9803)."""
    args = ["annotations-to-observations", "--session", str(session)]
    for view in views:
        args += ["--view", view]
    if merge_with:
        args += ["--merge-with", merge_with]
    return python_module_command(args)


def lineage_command(session: Path, path: Path) -> list[str]:
    """``rig lineage`` for a file inside the session (#9792)."""
    return python_module_command(
        ["lineage", "--session", str(session), "--path", str(path)]
    )


def model_choices() -> tuple[tuple[str, str], ...]:
    """``(name, description)`` for every registered fittable model."""
    from src.motion_capture.reconstruct.model.registry import descriptions

    return tuple(descriptions().items())


def export_command(session: Path, *, variant: str = "") -> list[str]:
    return python_module_command(
        ["export", "--session", str(session)] + variant_args(variant)
    )


def reconstruct_command(
    session: Path,
    *,
    measurements: Sequence[str],
    cameras: Path | None = None,
    intrinsics: Path | None = None,
    exclude_joints: Sequence[str] = (),
    variant: str = "",
    views: Sequence[str] = (),
    observations: str = "observations",
) -> list[str]:
    """Exactly one of ``cameras`` (later take) or ``intrinsics`` (first take).

    ``views`` (>= 2) restricts the cameras used; ``variant`` names the match.
    """
    require(not views or len(views) >= 2, "triangulation needs at least two views")
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
    if views:
        args += ["--views", ",".join(views)]
    if observations != "observations":
        args += ["--observations", observations]
    return python_module_command(args + variant_args(variant))


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
