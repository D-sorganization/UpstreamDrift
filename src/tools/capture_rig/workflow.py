"""The guided workflow as data: steps, requirements, readiness (#9660).

Each :class:`Step` carries what the operator must do and have, and a
readiness rule evaluated against :class:`~.session.SessionMedia` (what is
on disk). The GUI renders this; the user guide is generated from it, so the
two cannot drift. Nothing here touches Qt or files directly.

Single- and multi-camera sessions share the same steps; a single view skips
intrinsics and reconstruction and takes the 2-D analysis route.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .session import SessionMedia


class Status(str, Enum):
    DONE = "done"
    READY = "ready"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Step:
    """One stage of the workflow, in order."""

    key: str
    title: str
    purpose: str
    requirements: tuple[str, ...]
    instructions: tuple[str, ...]
    actions: tuple[str, ...]  # CaptureRigWidget action keys this step enables
    done: Callable[[SessionMedia], bool]
    ready: Callable[[SessionMedia], tuple[bool, str]]
    applies: Callable[[SessionMedia], bool] = lambda m: True


@dataclass(frozen=True)
class StepState:
    step: Step
    status: Status
    reason: str


def _multi(m: SessionMedia) -> bool:
    return len(m.views) >= 2


def _has_recordings(m: SessionMedia) -> bool:
    return any(v.recording is not None for v in m.views)


def _ready_if(cond: bool, why: str) -> tuple[bool, str]:
    return (True, "") if cond else (False, why)


SETUP = Step(
    key="setup",
    title="Set up Cameras and Plan",
    purpose="Bind each camera to a named view so takes are repeatable.",
    requirements=(
        "One camera per USB 2.0 root port (30 ft powered cables are fine: one camera per cable).",
        "A plan file naming each view (cam_a, cam_b, ...) by serial, port path or the one unserialized unit.",
        "All cameras on the same capture mode; 1280x720@120 or 1920x1200@60 for swings.",
    ),
    instructions=(
        "Choose the plan file and a session folder.",
        "Pick a mode preset or leave 'plan default'; optionally restrict the views.",
        "Run *Plan check*: every view must resolve to a camera before recording.",
        "Importing existing video files instead? Use *Import videos* and skip to Detect.",
    ),
    actions=("plan_check", "import"),
    done=lambda m: bool(m.views),
    ready=lambda m: (True, ""),
)

INTRINSICS = Step(
    key="intrinsics",
    title="Calibrate Each Camera Once",
    purpose="Lens focal length and distortion per camera; the only precise setup step.",
    requirements=(
        "A printed chessboard with 9x6 inner corners (10x7 squares), flat, one square measured to the millimetre.",
        "A 20-40 s recording per camera at the capture mode you will use: move the board slowly through the whole frame, corners and edges included, tilting up to ~45 degrees; sharp and well lit; board at least a fifth of the frame.",
        "At least 8 frames with the board found; reprojection RMS under 1 px is the standard.",
    ),
    instructions=(
        "Record the board session (one take, all cameras, 30 s) or import the board files.",
        "Enter the board size and square length, run *Calibrate intrinsics*.",
        "Keep intrinsics.json: it is the --intrinsics input of the first swing reconstruction.",
    ),
    actions=("record", "calibrate"),
    done=lambda m: m.intrinsics is not None,
    ready=lambda m: _ready_if(
        _has_recordings(m), "record or import a board take first"
    ),
    applies=_multi,
)

CAPTURE = Step(
    key="capture",
    title="Record or Import the Swing Take",
    purpose="The footage every later step works from.",
    requirements=(
        "Whole body in every camera for the whole swing; ball in view.",
        "Start at address and hold still about one second (the first frames define up and the origin).",
        "Multi-camera: at least two cameras, three preferred, at least 30 degrees apart, same mode.",
        "Single camera: face-on or down-the-line; gives 2-D events and tempo, no 3-D.",
        "10 s takes at 60-120 fps; a tape-measured segment on the golfer (hip-to-neck by default) for scale.",
    ),
    instructions=(
        "Set the duration, press *Record*, walk to address during the warm-up, swing, hold the finish.",
        "Or *Import videos* to build a session from files (one or many).",
        "Run *Proxies* for smooth playback of large MJPEG recordings.",
    ),
    actions=("record", "import", "proxy"),
    done=_has_recordings,
    ready=lambda m: (True, ""),
)

DETECT = Step(
    key="detect",
    title="Detect the Pose in Every View",
    purpose="2-D joints per frame from MediaPipe or OpenPose; re-run with other settings any time.",
    requirements=(
        "A recorded or imported session.",
        "MediaPipe (fast, primary) or OpenPose BODY_25 via OpenCV DNN (slow, second opinion).",
    ),
    instructions=(
        "Choose the estimator and settings; press *Ingest*. Each estimator writes its own observation set.",
        "Press *Compare* to run both and get coverage / confidence / jitter / agreement per joint.",
        "Scrub the playback with the overlay on; low-confidence joints are red.",
    ),
    actions=("ingest", "compare"),
    done=lambda m: m.ingested,
    ready=lambda m: _ready_if(_has_recordings(m), "record or import first"),
)

REVIEW = Step(
    key="review",
    title="Review Joint Reliability",
    purpose="Know which joints to trust before fitting.",
    requirements=("At least one observation set.",),
    instructions=(
        "Press *Reliability*: joints are graded reliable / usable / weak from coverage, confidence, jitter and (after a fit) outlier rejections.",
        "Weak joints are pre-filled in *Exclude joints*; edit the list as you see fit.",
    ),
    actions=("reliability",),
    done=lambda m: m.reliability is not None,
    ready=lambda m: _ready_if(m.ingested, "ingest first"),
)

RECONSTRUCT = Step(
    key="reconstruct",
    title="Reconstruct in 3-D",
    purpose="Camera placement learned from the golfer, rigid skeleton, outliers rejected.",
    requirements=(
        "Two or more ingested views.",
        "First take of a placement: intrinsics.json from the calibration step; later takes: the previous reconstruction.json.",
        "The anchor segment and its tape-measured length in metres.",
    ),
    instructions=(
        "Pick the start file (intrinsics or previous reconstruction), the anchor and length, optionally joints to exclude.",
        "Press *Reconstruct*. The summary shows RMS, rejections and the swing metrics.",
    ),
    actions=("reconstruct",),
    done=lambda m: m.reconstruction is not None,
    ready=lambda m: _ready_if(
        m.ingested and _multi(m), "two ingested views are needed"
    ),
    applies=_multi,
)

ANALYZE_2D = Step(
    key="analyze_2d",
    title="Analyse the Single View",
    purpose="Events, tempo and normalised hand speed from one camera.",
    requirements=("One ingested view.",),
    instructions=(
        "Press *Analyze 2-D*. Results are in subject box heights, not metres.",
        "*Export clip* writes the swing (address to finish, slow motion, overlay, frame clock) as a video; *Compare takes* puts another session's view beside this one aligned on the top of the backswing, with metric deltas.",
    ),
    actions=("analyze", "clip", "compare_takes"),
    done=lambda m: bool(m.analysis_2d),
    ready=lambda m: _ready_if(m.ingested, "ingest first"),
    applies=lambda m: not _multi(m),
)

EXPORT = Step(
    key="export",
    title="Export to the Motion Pipeline",
    purpose="TRC and canonical JSON for scaling, IK and model matching.",
    requirements=("A reconstruction.",),
    instructions=(
        "Press *Export*. reconstruction.trc loads in the motion pipeline and the model-matching tools as a marker file.",
        "*Export clip* and *Compare takes* produce annotated, slowed videos of this take, alone or beside another session, for coaching.",
    ),
    actions=("export", "clip", "compare_takes"),
    done=lambda m: m.export is not None,
    ready=lambda m: _ready_if(m.reconstruction is not None, "reconstruct first"),
    applies=_multi,
)

STEPS: tuple[Step, ...] = (
    SETUP,
    INTRINSICS,
    CAPTURE,
    DETECT,
    REVIEW,
    RECONSTRUCT,
    ANALYZE_2D,
    EXPORT,
)


def evaluate(media: SessionMedia | None) -> tuple[StepState, ...]:
    """Every step's status for the session (all steps ready-or-blocked when None)."""
    out = []
    for step in STEPS:
        if media is None:
            status = Status.READY if step.key == "setup" else Status.BLOCKED
            out.append(
                StepState(
                    step,
                    status,
                    "" if status is Status.READY else "load or record a session",
                )
            )
            continue
        if not step.applies(media):
            out.append(StepState(step, Status.SKIPPED, "not for this session"))
        elif step.done(media):
            out.append(StepState(step, Status.DONE, ""))
        else:
            ok, why = step.ready(media)
            out.append(StepState(step, Status.READY if ok else Status.BLOCKED, why))
    return tuple(out)


def current(states: tuple[StepState, ...]) -> StepState | None:
    """The first step that is ready, else the first blocked one."""
    for status in (Status.READY, Status.BLOCKED):
        for s in states:
            if s.status is status:
                return s
    return None


def enabled_actions(states: tuple[StepState, ...]) -> frozenset[str]:
    """Actions of every ready or done step that applies."""
    out: set[str] = set()
    for s in states:
        if s.status in (Status.READY, Status.DONE):
            out.update(s.step.actions)
    return frozenset(out)
