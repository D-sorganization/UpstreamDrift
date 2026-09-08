"""Operator CLI: ``python3 -m motion_capture.rig <command>``.

Commands:

- ``plan-check --plan P``: enumerate cameras (Windows), match the plan, flag
  missing cameras and USB 2.0 bandwidth conflicts. Exit 0 when the plan is
  realizable, 1 otherwise, 2 when nothing is enumerated.
- ``capture --plan P --duration S --out DIR [--synthetic]``: open every planned
  camera, capture together, write ``session_manifest.json``. Exit 0 for
  ``supported``, 1 for ``degraded``/``blocked``, 2 for ``unavailable``.
- ``record --plan P --duration S --out DIR [--dry-run]``: stream-copy every
  planned camera's compressed video to disk and write a session bundle
  (``plan.json``, ``recordings.json``, ``session_manifest.json``). Same exit
  codes as ``capture``. ``--dry-run`` records nothing and exercises the bundle.
- ``session-check --session DIR``: validate a bundle on disk. Exit 0 when sound.
- ``proxy --session DIR [--encoder E] [--crf N]``: write browser-playable H.264
  ``.mp4`` proxies beside each recording and ``proxies.json``. Exit 0 when
  every usable recording has a proxy.
- ``reconstruct --session DIR --cameras records.json --anchor SEGMENT=METRES``:
  map, clean and jointly fit the ingested views into 3-D joints, camera
  placement and bone lengths under ``<session>/reconstruct/``. Exit 0 when
  the fit ran.
- ``calibrate-intrinsics --session DIR --board 9x6 --square 0.025 [--every N]``:
  find a printed chessboard in every recording of a bundle and write
  ``intrinsics.json`` (K, distortion, RMS, frames used) keyed by view. Exit 0
  when every view calibrated within the RMS bound.
- ``compare --session DIR --estimators a,b [--max-frames N]``: ingest the bundle
  with each named estimator and write ``comparison_<view>.json`` / ``.md``
  (coverage, confidence, jitter, cross-detector agreement). Exit 0 when every
  estimator produced every view.
- ``ingest --session DIR [--out DIR] [--estimator NAME] [--max-frames N]``: run the
  registered pose estimator over every recording and write per-view 2-D
  observations with provenance and the session's timing block. Exit 0 when
  every view produced observations, 1 when some did, 2 when none did.

``plan-check``, ``capture`` and ``record`` accept ``--mode WxH@FPS[:FOURCC]``
(one capture mode for every selected view) and ``--views a,b`` (a subset of
the plan, in plan order) so a condition can change without editing the plan
file; the derived plan name records the overrides in the bundle.
"""

from __future__ import annotations

import argparse
import json
from typing import Any
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from src.motion_capture.provenance import write_stamped
from src.motion_capture.variants import variant_dir
from src.shared.python.logging_pkg.logging_config import get_logger

from .bundle import build_index, check_bundle, write_bundle
from .plan import CameraControls, RigPlan, check_plan, parse_mode
from .probe import RecordingProbe, probe_recording
from .proxy import DEFAULT_CRF, DEFAULT_ENCODER, ENCODERS, make_proxies
from .recorder import (
    DEFAULT_WARMUP_S,
    FfmpegStreamCopyRecorder,
    NullRecorder,
    Recorder,
    dshow_device_ref,
    record_all,
)
from .session import CaptureOutcome, CaptureSession, CaptureTuning
from .sources import FrameSource, OpenCvMsmfSource, SyntheticFrameSource
from .tools_bridge import probe_tools_schema
from .topology import (
    CameraLocation,
    attach_capture_indices,
    dshow_order,
    query_topology,
)

logger = get_logger(__name__)

_EXIT_BY_OUTCOME = {
    CaptureOutcome.SUPPORTED: 0,
    CaptureOutcome.DEGRADED: 1,
    CaptureOutcome.BLOCKED: 1,
    CaptureOutcome.UNAVAILABLE: 2,
}


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    """``--plan`` plus the two operator overrides shared by the camera commands."""
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=parse_mode,
        default=None,
        metavar="WxH@FPS[:FOURCC]",
        help="capture mode for every selected view, e.g. 1280x720@120",
    )
    parser.add_argument(
        "--views",
        default=None,
        metavar="a,b",
        help="comma-separated subset of plan views to use (plan order)",
    )
    parser.add_argument(
        "--exposure", type=float, default=None, help="UVC exposure for every view"
    )
    parser.add_argument("--gain", type=float, default=None, help="UVC gain")
    parser.add_argument(
        "--auto-exposure",
        choices=("on", "off"),
        default=None,
        help="UVC auto-exposure for every selected view",
    )


def _load_plan(args: argparse.Namespace) -> RigPlan:
    """The plan file with ``--mode``/``--views`` applied."""
    plan = RigPlan.load(args.plan)
    views = None
    if getattr(args, "views", None):
        views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    auto = getattr(args, "auto_exposure", None)
    controls = CameraControls(
        exposure=getattr(args, "exposure", None),
        gain=getattr(args, "gain", None),
        auto_exposure=None if auto is None else auto == "on",
    )
    return plan.with_overrides(
        mode=getattr(args, "mode", None),
        views=views,
        controls=controls if controls.as_overrides() else None,
    )


def _add_coaching_parsers(sub: Any) -> None:
    """Printable boards, annotated clips and take comparison (#9679-#9681)."""
    brd = sub.add_parser("board", help="write a printable ChArUco board image")
    brd.add_argument("--board", default="charuco:7x5:0.04:0.03")
    brd.add_argument("--width", type=int, default=2100, help="pixels (A4 @ 254 dpi)")
    brd.add_argument("--out", type=Path, required=True)
    clp = sub.add_parser("clip", help="trimmed clip with overlay and slow motion")
    clp.add_argument("--session", type=Path, required=True)
    clp.add_argument("--view", required=True)
    clp.add_argument(
        "--from", dest="start", default="address-30", metavar="EVENT|FRAME"
    )
    clp.add_argument("--to", dest="end", default="finish+30", metavar="EVENT|FRAME")
    clp.add_argument("--speed", type=float, default=0.25, help="1 = real time")
    clp.add_argument("--set", default=None, help="observation set for the overlay")
    clp.add_argument("--out", type=Path, required=True)
    cmt = sub.add_parser("compare-takes", help="two takes side by side on an event")
    cmt.add_argument("--session", type=Path, required=True)
    cmt.add_argument("--view", required=True)
    cmt.add_argument("--other-session", type=Path, required=True)
    cmt.add_argument("--other-view", required=True)
    cmt.add_argument(
        "--align", default="top", choices=("address", "top", "peak", "finish")
    )
    cmt.add_argument("--speed", type=float, default=0.5)
    cmt.add_argument("--out", type=Path, required=True)


def _add_model_parsers(sub: Any) -> None:
    """Articulated-model fit, comparison and kinetics (#9709)."""
    fm = sub.add_parser(
        "fit-model", help="articulated golfer (scapula) fit, continuous"
    )
    fm.add_argument("--session", type=Path, required=True)
    fm.add_argument(
        "--sigma-accel",
        type=float,
        default=300.0,
        help="acceleration prior on every joint angle, rad/s^2 (smaller = stiffer)",
    )
    fm.add_argument(
        "--max-velocity",
        type=float,
        default=25.0,
        help="report joint-angle speeds above this, rad/s",
    )
    fm.add_argument("--sigma-landmark", type=float, default=0.01, help="metres")
    fm.add_argument("--model", default="golfer", help="registered model name")
    fm.add_argument(
        "--fit-lengths",
        action="store_true",
        help="learn the model's learnable segment lengths from the data",
    )
    fm.add_argument("--max-iterations", type=int, default=60, help="solver budget")
    _add_variant_args(fm, observations=True)
    fm.add_argument(
        "--from-views",
        default="",
        metavar="a,b",
        help="image-space fit to these views' 2-D keypoints (1..N), no triangulation",
    )
    fm.add_argument(
        "--cameras-from",
        default="",
        metavar="VARIANT",
        help="variant whose reconstruction supplies the cameras for --from-views",
    )
    fm.add_argument("--sigma-px", type=float, default=4.0, help="pixel noise")
    cmm = sub.add_parser("compare-models", help="fit several models, rank them")
    cmm.add_argument("--session", type=Path, required=True)
    cmm.add_argument("--models", default=None, help="comma list; default: all")
    cmm.add_argument("--fit-lengths", action="store_true")
    cmm.add_argument("--sigma-accel", type=float, default=300.0)
    cmm.add_argument("--max-iterations", type=int, default=60, help="solver budget")
    _add_variant_args(cmm)
    kin = sub.add_parser("kinetics", help="inverse dynamics + replay check of a fit")
    kin.add_argument("--session", type=Path, required=True)
    kin.add_argument("--model", default="golfer", help="registered model name")
    kin.add_argument("--body-mass", type=float, required=True, help="kg")
    _add_variant_args(kin)
    _add_variant_tools_parsers(sub)
    lin = sub.add_parser("lineage", help="provenance chain of a pipeline output")
    lin.add_argument("--session", type=Path, required=True)
    lin.add_argument("--path", type=Path, required=True, help="file inside the session")
    lin.add_argument("--json", action="store_true", help="machine-readable")


def _add_variant_tools_parsers(sub: Any) -> None:
    """overlay, compare-variants, annotations-to-observations (#9795/#9796/#9801)."""
    ov = sub.add_parser("overlay", help="draw variants' 3-D results on a view")
    ov.add_argument("--session", type=Path, required=True)
    ov.add_argument("--view", required=True)
    ov.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME",
        help="variant to draw, repeatable (default: the session's default match)",
    )
    ov.add_argument("--out", type=Path, required=True, help="mp4 to write")
    ov.add_argument("--from", dest="start", type=int, default=0, metavar="FRAME")
    ov.add_argument("--to", dest="stop", type=int, default=None, metavar="FRAME")
    ov.add_argument("--speed", type=float, default=1.0)
    ov.add_argument("--observations", default="observations", metavar="SET")
    ov.add_argument("--no-legend", action="store_true")
    cv = sub.add_parser("compare-variants", help="held-out and 3-D error per variant")
    cv.add_argument("--session", type=Path, required=True)
    cv.add_argument("--reference", default="", metavar="NAME")
    cv.add_argument("--observations", default="observations", metavar="SET")
    a2o = sub.add_parser(
        "annotations-to-observations",
        help="manual clicks as an observation set (optionally over a detector set)",
    )
    a2o.add_argument("--session", type=Path, required=True)
    a2o.add_argument("--view", action="append", default=[], help="repeatable")
    a2o.add_argument("--merge-with", default="", metavar="SET")
    a2o.add_argument("--out", default="", metavar="SET")


def _add_variant_args(
    parser: Any, *, observations: bool = False, views: bool = False
) -> None:
    """``--variant`` (and for reconstruct: ``--observations``, ``--views``), #9793."""
    parser.add_argument(
        "--variant",
        default="",
        metavar="NAME",
        help="named match: outputs under variants/NAME/ (default: the session)",
    )
    if observations:
        parser.add_argument(
            "--observations",
            default="observations",
            metavar="SET",
            help="observation-set directory to use (observations, observations_x)",
        )
    if views:
        parser.add_argument(
            "--views",
            default="",
            metavar="a,b",
            help="subset and order of the cameras to use (default: all with cameras)",
        )


def _add_offline_parsers(sub: Any) -> None:
    """Commands that work on a session bundle rather than on cameras."""
    ing = sub.add_parser("ingest", help="pose-estimate every recording in a bundle")
    ing.add_argument("--session", type=Path, required=True)
    ing.add_argument(
        "--out", type=Path, default=None, help="default: <session>/observations"
    )
    ing.add_argument("--estimator", default="mediapipe")
    ing.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="estimator option, repeatable (e.g. min_detection_confidence=0.6)",
    )
    ing.add_argument("--max-frames", type=int, default=None)
    cmp = sub.add_parser("compare", help="run two estimators on one bundle")
    cmp.add_argument("--session", type=Path, required=True)
    cmp.add_argument("--estimators", default="mediapipe,openpose_dnn")
    cmp.add_argument("--max-frames", type=int, default=None)
    cmp.add_argument("--min-confidence", type=float, default=0.5)
    rec3 = sub.add_parser("reconstruct", help="clean and jointly fit ingested views")
    rec3.add_argument("--session", type=Path, required=True)
    rec3.add_argument(
        "--cameras",
        type=Path,
        default=None,
        help="camera records or a reconstruction.json to start from (later takes)",
    )
    rec3.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="intrinsics-only records for a first take: placement from the joints",
    )
    rec3.add_argument(
        "--anchor",
        action="append",
        required=True,
        metavar="SEGMENT=METRES",
        help="tape-measured segment, repeatable; everyday names (shank, forearm, "
        "upper_arm, thigh, shoulder_width, hip_width, torso) constrain both sides; "
        "the first one sets the scale",
    )
    rec3.add_argument(
        "--exclude-joints",
        default="",
        metavar="a,b",
        help="fit joints to treat as unobserved (kept by the segment priors)",
    )
    rec3.add_argument(
        "--accel-sigma-px",
        type=float,
        default=20_000.0,
        help="acceleration prior in px/s^2",
    )
    _add_variant_args(rec3, observations=True, views=True)
    imp = sub.add_parser("import", help="build a bundle from existing video files")
    imp.add_argument("--out", type=Path, required=True)
    imp.add_argument(
        "--view",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="one per view; a single --view is a single-camera session",
    )
    imp.add_argument("--name", default=None, help="plan name (default import:<out>)")
    _add_coaching_parsers(sub)
    _add_model_parsers(sub)
    ana = sub.add_parser("analyze", help="2-D events and tempo per ingested view")
    ana.add_argument("--session", type=Path, required=True)
    ana.add_argument("--observations", default="observations", help="set directory")
    ana.add_argument("--min-confidence", type=float, default=0.5)
    rel = sub.add_parser("reliability", help="grade joints from every observation set")
    rel.add_argument("--session", type=Path, required=True)
    rel.add_argument("--min-confidence", type=float, default=0.5)
    exp = sub.add_parser("export", help="reconstruction -> TRC + canonical JSON")
    exp.add_argument("--session", type=Path, required=True)
    exp.add_argument("--trc", type=Path, default=None)
    exp.add_argument("--json", type=Path, default=None)
    _add_variant_args(exp)
    cal = sub.add_parser("calibrate-intrinsics", help="chessboard intrinsics per view")
    cal.add_argument("--session", type=Path, required=True)
    cal.add_argument(
        "--board",
        default="9x6",
        help="chessboard COLSxROWS (inner corners) or charuco:COLSxROWS:SQ_M:MK_M",
    )
    cal.add_argument("--square", type=float, default=None, help="chessboard square, m")
    cal.add_argument("--every", type=int, default=10, help="sample every Nth frame")
    cal.add_argument(
        "--out", type=Path, default=None, help="default: <session>/intrinsics.json"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motion_capture.rig", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("plan-check", help="match a plan against enumerated cameras")
    _add_plan_args(check)
    cap = sub.add_parser("capture", help="capture every planned camera together")
    _add_plan_args(cap)
    cap.add_argument("--duration", type=float, default=8.0)
    cap.add_argument("--out", type=Path, default=Path.cwd() / "capture")
    cap.add_argument("--settle", type=float, default=2.0, help="seconds between opens")
    cap.add_argument(
        "--synthetic", action="store_true", help="use deterministic synthetic sources"
    )
    cap.add_argument(
        "--timing",
        action="store_true",
        help="record per-frame brightness and align cameras on a shared strobe",
    )
    rec = sub.add_parser("record", help="stream-copy every planned camera to disk")
    _add_plan_args(rec)
    rec.add_argument("--duration", type=float, default=10.0)
    rec.add_argument("--out", type=Path, default=Path.cwd() / "session")
    rec.add_argument(
        "--warmup",
        type=float,
        default=DEFAULT_WARMUP_S,
        help="seconds for the devices to open before the duration clock starts",
    )
    rec.add_argument(
        "--dry-run",
        action="store_true",
        help="record nothing; write the bundle with NullRecorder results",
    )
    chk = sub.add_parser("session-check", help="validate a session bundle on disk")
    chk.add_argument("--session", type=Path, required=True)
    prx = sub.add_parser("proxy", help="write H.264 mp4 proxies beside the recordings")
    prx.add_argument("--session", type=Path, required=True)
    prx.add_argument("--encoder", choices=ENCODERS, default=DEFAULT_ENCODER)
    prx.add_argument("--crf", type=int, default=DEFAULT_CRF, help="libx264 only")
    _add_offline_parsers(sub)
    return parser


def _located_plan(plan: RigPlan) -> dict[str, CameraLocation]:
    """Enumerate cameras and map each plan view to its location; exit when unrealizable."""
    cams = attach_capture_indices(query_topology(), dshow_order())
    check = check_plan(plan, cams)
    if not check.ok:
        raise SystemExit(
            f"plan not realizable: missing={list(check.missing)} "
            f"conflicts={list(check.conflicts)}"
        )
    by_instance = {c.camera: c for c in cams}
    return {view: by_instance[inst] for view, inst in check.matched.items()}


def _real_sources(plan: RigPlan) -> dict[str, FrameSource]:
    sources: dict[str, FrameSource] = {}
    for view, cam in _located_plan(plan).items():
        if cam.index is None:
            raise SystemExit(f"camera {cam.identity} has no capture index")
        sources[view] = OpenCvMsmfSource(cam.identity, cam.index)
    return sources


def _device_refs(plan: RigPlan) -> dict[str, str]:
    return {
        view: dshow_device_ref(cam.camera) for view, cam in _located_plan(plan).items()
    }


def cmd_plan_check(args: argparse.Namespace) -> int:
    plan = _load_plan(args)
    cams = attach_capture_indices(query_topology(), dshow_order())
    if not cams:
        logger.error("no cameras enumerated")
        return 2
    check = check_plan(plan, cams)
    for view, instance in check.matched.items():
        logger.info("view %s -> %s", view, instance)
    for view in check.missing:
        logger.error("view %s: camera not enumerated", view)
    for view in check.conflicts:
        logger.error(
            "view %s: shares a USB 2.0 root port with another planned camera", view
        )
    for identity in check.unplanned:
        logger.warning("camera %s enumerated but not in the plan", identity)
    return 0 if check.ok else 1


def cmd_capture(args: argparse.Namespace) -> int:
    plan = _load_plan(args)
    if args.synthetic:
        sources: dict[str, FrameSource] = {
            c.view: SyntheticFrameSource(c.identity, realtime=True)
            for c in plan.cameras
        }
        settle = 0.0
    else:
        sources = _real_sources(plan)
        settle = args.settle
    tuning = CaptureTuning(settle_s=settle, collect_timing=args.timing)
    session = CaptureSession(plan, sources, duration_s=args.duration, tuning=tuning)
    manifest = session.run()
    manifest = manifest.model_copy(
        update={"tools_schema": probe_tools_schema().to_dict()}
    )
    path = manifest.save(args.out / "session_manifest.json")
    for cam in manifest.cameras:
        logger.info(
            "%s (%s): %.1f fps failed=%d reopens=%d %s%s",
            cam.view,
            cam.identity,
            cam.achieved_fps,
            cam.failed_reads,
            cam.reopens,
            cam.state,
            f" - {cam.reason}" if cam.reason else "",
        )
    logger.info("outcome=%s manifest=%s", manifest.outcome.value, path)
    return _EXIT_BY_OUTCOME[manifest.outcome]


def _dry_run_probe(path: Path) -> RecordingProbe:
    """Dry runs write no video; report a probe that reflects that honestly."""
    return RecordingProbe(
        frames=0, duration_s=0.0, width=None, height=None, nominal_fps=None
    )


def cmd_record(args: argparse.Namespace) -> int:
    plan = _load_plan(args)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    factory: Callable[[], Recorder]
    if args.dry_run:
        refs = {c.view: f"dry-run:{c.identity}" for c in plan.cameras}
        factory = NullRecorder
    else:
        refs = _device_refs(plan)
        factory = FfmpegStreamCopyRecorder
    results = record_all(
        plan, refs, args.duration, args.out, factory, warmup_s=args.warmup
    )
    prober = _dry_run_probe if args.dry_run else probe_recording
    index = build_index(plan, results, args.duration, args.out, prober=prober)
    manifest = write_bundle(
        args.out,
        plan,
        index,
        started_utc=started,
        tools_schema=probe_tools_schema().to_dict(),
    )
    for entry in index.recordings:
        logger.info(
            "%s (%s): %s %d bytes rc=%s frames=%s duration=%s",
            entry.view,
            entry.identity,
            entry.file,
            entry.bytes,
            entry.returncode,
            entry.frames,
            entry.duration_s,
        )
    logger.info("outcome=%s bundle=%s", manifest.outcome.value, args.out)
    return _EXIT_BY_OUTCOME[manifest.outcome]


def cmd_session_check(args: argparse.Namespace) -> int:
    check = check_bundle(args.session)
    for problem in check.problems:
        logger.error("%s", problem)
    logger.info("session %s: %s", args.session, "ok" if check.ok else "problems found")
    return 0 if check.ok else 1


def cmd_proxy(args: argparse.Namespace) -> int:
    index = make_proxies(args.session, encoder=args.encoder, crf=args.crf)
    for entry in index.proxies:
        logger.info(
            "%s: %s%s",
            entry.view,
            entry.file or "no proxy",
            f" - {entry.reason}" if entry.reason else "",
        )
    return 0 if index.ok else 1


def parse_option_value(text: str) -> bool | int | float | str:
    """``true``/``false``, ints, floats, else the string itself."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def parse_options(items: list[str]) -> dict[str, bool | int | float | str]:
    """``KEY=VALUE`` pairs → typed estimator options."""
    out: dict[str, bool | int | float | str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if sep != "=" or not key.strip():
            raise SystemExit(f"--option must be KEY=VALUE, got {item!r}")
        out[key.strip()] = parse_option_value(value)
    return out


def cmd_import(args: argparse.Namespace) -> int:
    from .importer import import_videos, parse_view_spec

    views = [parse_view_spec(v) for v in args.view]
    manifest = import_videos(views, args.out, plan_name=args.name)
    logger.info("import %s: %s", args.out, manifest.outcome.value)
    return 0 if manifest.outcome.value != "blocked" else 1


def cmd_board(args: argparse.Namespace) -> int:
    import cv2

    from src.motion_capture.reconstruct.intrinsics import CharucoBoard, parse_board_spec

    board = parse_board_spec(args.board)
    if not isinstance(board, CharucoBoard):
        raise SystemExit("board images are generated for charuco:... boards only")
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), board.image(args.width))
    logger.info(
        "board %s -> %s (print at 100 %%; measure a square)", args.board, args.out
    )
    return 0


def cmd_clip(args: argparse.Namespace) -> int:
    # The tool package owns playback and overlays; imported lazily (Qt-free modules).
    from src.tools.capture_rig.clips import clip_from_session

    result = clip_from_session(
        args.session,
        args.view,
        start=args.start,
        end=args.end,
        out=args.out,
        speed=args.speed,
        observation_set=args.set,
    )
    logger.info(
        "clip %s: frames %d-%d -> %s",
        args.view,
        result["first"],
        result["last"],
        args.out,
    )
    return 0


def cmd_compare_takes(args: argparse.Namespace) -> int:
    from src.tools.capture_rig.clips import compare_from_sessions

    result = compare_from_sessions(
        args.session,
        args.view,
        args.other_session,
        args.other_view,
        out=args.out,
        align=args.align,
        speed=args.speed,
    )
    logger.info(
        "compare-takes aligned on %s: %d frames -> %s",
        args.align,
        result["frames"],
        args.out,
    )
    return 0


def cmd_fit_model(args: argparse.Namespace) -> int:
    from src.motion_capture.reconstruct.model import FitOptions
    from src.motion_capture.reconstruct.model.registry import get_model
    from src.motion_capture.reconstruct.model.session import fit_session_model

    registered = get_model(args.model)
    options = FitOptions(
        sigma_landmark_m=args.sigma_landmark,
        sigma_accel_rad_s2=args.sigma_accel,
        max_velocity_rad_s=args.max_velocity,
        max_iterations=args.max_iterations,
        fit_lengths=registered.learnable_lengths if args.fit_lengths else (),
    )
    out_subdir = None if args.model == "golfer" else args.model
    from_views = [v.strip() for v in args.from_views.split(",") if v.strip()]
    if from_views:
        # --cameras-from "" is the session's default variant.
        from src.motion_capture.reconstruct.model.fit2d import fit_session_model_2d

        fit, out_dir = fit_session_model_2d(
            args.session,
            registered.spec,
            registered.landmark_map,
            views=from_views,
            cameras_from=args.cameras_from,
            observation_set=args.observations,
            variant=args.variant,
            options=options,
            sigma_px=args.sigma_px,
            out_subdir=out_subdir,
        )
    else:
        fit, out_dir = fit_session_model(
            variant_dir(args.session, args.variant),
            registered.spec,
            registered.landmark_map,
            session_root=args.session,
            options=options,
            out_subdir=out_subdir,
        )
    logger.info(
        "fit-model %s: rms %.1f mm (%s px), %d rejected, %d velocity violations -> %s",
        args.session,
        1000 * fit.rms_m,
        f"{fit.rms_px:.2f}" if fit.rms_px is not None else "n/a",
        len(fit.rejected),
        fit.velocity_violations,
        out_dir,
    )
    return 0 if fit.velocity_violations == 0 else 1


def cmd_compare_models(args: argparse.Namespace) -> int:
    from src.motion_capture.reconstruct.model import FitOptions
    from src.motion_capture.reconstruct.model.compare import compare_models

    names = [n.strip() for n in args.models.split(",")] if args.models else None
    report = compare_models(
        variant_dir(args.session, args.variant),
        names,
        options=FitOptions(
            sigma_accel_rad_s2=args.sigma_accel, max_iterations=args.max_iterations
        ),
        fit_lengths=args.fit_lengths,
    )
    logger.info("compare-models:\n%s", report.markdown())
    return 0


def cmd_kinetics(args: argparse.Namespace) -> int:
    import json

    import numpy as np

    from src.motion_capture.reconstruct.model.dynamics import kinetics_report
    from src.motion_capture.reconstruct.model.golfer import simscape_variable_names
    from src.motion_capture.reconstruct.model.kinematics import ArticulatedModel
    from src.motion_capture.reconstruct.model.registry import get_model

    registered = get_model(args.model)
    model_dir = variant_dir(args.session, args.variant) / "model"
    if args.model != "golfer":
        model_dir = model_dir / args.model
    angles_file = model_dir / "joint_angles.json"
    if not angles_file.is_file():
        raise SystemExit(f"run fit-model --model {args.model} first: {angles_file}")
    payload = json.loads(angles_file.read_text(encoding="utf-8"))
    model = ArticulatedModel(registered.spec)
    report = kinetics_report(
        model,
        np.asarray(payload["q"], dtype=float),
        float(payload["fps"]),
        body_mass_kg=args.body_mass,
        names=simscape_variable_names() if args.model == "golfer" else None,
    )
    out = model_dir / "kinetics.json"
    write_stamped(
        out,
        report,
        schema_version=str(report["schema_version"]),
        module=__name__,
        inputs=[angles_file],
        parameters={"model": args.model, "body_mass_kg": args.body_mass},
        derived_from=[angles_file],
        base=args.session,
    )
    logger.info(
        "kinetics %s: replay max |error| %.4f rad (worst %s) -> %s",
        args.model,
        report["replay_max_abs_error"],
        report["replay_worst_dof"],
        out,
    )
    return 0


def cmd_overlay(args: argparse.Namespace) -> int:
    from src.tools.capture_rig.overlay_render import export_overlay

    sidecar = export_overlay(
        args.session,
        args.view,
        tuple(args.variant) or ("",),
        args.out,
        start=args.start,
        stop=args.stop,
        speed=args.speed,
        observation_set=args.observations,
        legend=not args.no_legend,
    )
    for track in sidecar["tracks"]:
        logger.info(
            "overlay %s: %s rms %s px%s",
            args.view,
            track["label"],
            track["reprojection_rms_px"],
            " (held out)" if track["held_out"] else "",
        )
    return 0


def cmd_compare_variants(args: argparse.Namespace) -> int:
    from src.motion_capture.compare_variants import compare_variants, markdown

    payload = compare_variants(
        args.session, reference=args.reference, observation_set=args.observations
    )
    logger.info("compare-variants:\n%s", markdown(payload))
    return 0


def cmd_annotations_to_observations(args: argparse.Namespace) -> int:
    from src.motion_capture.annotate.store import AnnotationSet, annotation_path
    from src.motion_capture.annotate.to_observations import (
        DEFAULT_OUT,
        merge,
        to_view_observations,
        write_observation_set,
    )

    from .bundle import load_bundle

    plan, _index, _ = load_bundle(args.session)
    views = args.view or [
        p.stem for p in sorted((args.session / "annotations").glob("*.json"))
    ]
    if not views:
        raise SystemExit("no annotations in the session")
    out_set = args.out or (
        f"{args.merge_with}_edited" if args.merge_with else DEFAULT_OUT
    )
    payloads = {}
    inputs = []
    counts: dict[str, int] = {}
    for view in views:
        path = annotation_path(args.session, view)
        store = AnnotationSet.load(path)
        inputs.append(path)
        if args.merge_with:
            base = args.session / args.merge_with / f"{view}.json"
            if not base.is_file():
                raise SystemExit(f"no {args.merge_with} observations for {view}")
            detector = json.loads(base.read_text(encoding="utf-8"))
            payloads[view], done = merge(store, detector)
            inputs.append(base)
            counts = {k: counts.get(k, 0) + v for k, v in done.items()}
        else:
            payloads[view] = to_view_observations(store, annotation_file=path)
    out_dir = write_observation_set(
        args.session,
        out_set,
        payloads,
        plan_name=plan.name,
        inputs=inputs,
        parameters={
            "estimator": "manual",
            "merge_with": args.merge_with or None,
            "corrections": counts,
        },
    )
    logger.info("annotations -> %s (%s)", out_dir, counts or "manual only")
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    from src.motion_capture.provenance import lineage, lineage_markdown

    path = args.path if args.path.is_absolute() else args.session / args.path
    records = lineage(path, base=args.session)
    if args.json:
        sys.stdout.write(json.dumps([r.__dict__ for r in records], indent=2) + "\n")
    else:
        sys.stdout.write(lineage_markdown(records))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from src.motion_capture.reconstruct.analytics2d import analyze_session_2d

    written = analyze_session_2d(
        args.session,
        observations_dir=args.observations,
        min_confidence=args.min_confidence,
    )
    for view, path in written.items():
        logger.info("analysis_2d %s: %s", view, path)
    return 0


def cmd_reliability(args: argparse.Namespace) -> int:
    from .reliability import reliability_report, write_reliability

    report = reliability_report(args.session, min_confidence=args.min_confidence)
    path = write_reliability(report, args.session)
    logger.info("reliability -> %s", path)
    logger.info("%s", report.markdown())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from src.motion_capture.reconstruct.export import export_reconstruction

    root = variant_dir(args.session, args.variant)
    written = export_reconstruction(
        root / "reconstruct", trc_path=args.trc, json_path=args.json
    )
    model_angles = root / "model" / "joint_angles.json"
    if model_angles.is_file():
        from src.motion_capture.reconstruct.model.golfer import write_simscape_csv

        written["simscape_csv"] = str(write_simscape_csv(model_angles))
    for kind, path in written.items():
        logger.info("export %s: %s", kind, path)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    # Imported here: the pose stack (MediaPipe, simulation backends) takes
    # seconds to import and no other command needs it.
    from .ingest import ingest_bundle, registry_estimator_factory

    out_dir = args.out or (args.session / "observations")
    index = ingest_bundle(
        args.session,
        out_dir,
        registry_estimator_factory(args.estimator, **parse_options(args.option)),
        max_frames=args.max_frames,
    )
    for view in index.views:
        logger.info(
            "%s: %s frames_with_pose=%s/%s%s",
            view.view,
            view.status,
            view.frames_with_pose,
            view.frames_total,
            f" - {view.reason}" if view.reason else "",
        )
    produced = sum(1 for v in index.views if v.status == "available")
    logger.info(
        "ingest %s: %d/%d views -> %s",
        args.session,
        produced,
        len(index.views),
        out_dir,
    )
    return 0 if produced == len(index.views) else (1 if produced else 2)


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_view, load_series
    from .ingest import ingest_bundle, registry_estimator_factory

    names = [n.strip() for n in args.estimators.split(",") if n.strip()]
    if len(names) not in (1, 2):
        raise SystemExit("--estimators takes one or two names")
    per_view: dict[str, dict[str, Path]] = {}
    complete = True
    for name in names:
        out_dir = args.session / f"observations_{name}"
        index = ingest_bundle(
            args.session,
            out_dir,
            registry_estimator_factory(name),
            max_frames=args.max_frames,
        )
        for view in index.views:
            if view.status != "available" or not view.file:
                complete = False
                continue
            per_view.setdefault(view.view, {})[name] = out_dir / view.file
    for view_name, files in per_view.items():
        series = {n: load_series(p) for n, p in files.items()}
        report = compare_view(view_name, series, min_confidence=args.min_confidence)
        (args.session / f"comparison_{view_name}.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        (args.session / f"comparison_{view_name}.md").write_text(
            report.markdown(), encoding="utf-8"
        )
        logger.info("comparison for %s:\n%s", view, report.markdown())
    return 0 if complete else 1


def cmd_reconstruct(args: argparse.Namespace) -> int:
    from src.motion_capture.reconstruct.pipeline import (
        intrinsics_from,
        reconstruct_session,
        start_cameras_from,
    )

    if (args.cameras is None) == (args.intrinsics is None):
        raise SystemExit("give exactly one of --cameras or --intrinsics")
    from src.motion_capture.reconstruct.measurements import expand_measurements

    try:
        expand_measurements(args.anchor)  # usage errors before any heavy work
    except ValueError as exc:
        raise SystemExit(f"--anchor: {exc}") from exc
    wanted = [v.strip() for v in args.views.split(",") if v.strip()]
    summary = reconstruct_session(
        args.session,
        start_cameras=start_cameras_from(args.cameras) if args.cameras else None,
        intrinsics=intrinsics_from(args.intrinsics) if args.intrinsics else None,
        measurements=tuple(args.anchor),
        acceleration_sigma_px=args.accel_sigma_px,
        exclude_joints=tuple(
            j.strip() for j in args.exclude_joints.split(",") if j.strip()
        ),
        observation_set=args.observations,
        views=wanted or None,
        variant=args.variant,
        camera_source=(
            f"cameras:{args.cameras}"
            if args.cameras
            else f"intrinsics:{args.intrinsics}"
        ),
    )
    logger.info(
        "reconstructed %s: rms %.2f px, rejections %s, %d unobservable points -> %s",
        args.session,
        summary.rms_px,
        summary.cleaned_rejections,
        summary.unobservable_points,
        summary.reconstruction_file,
    )
    return 0


def cmd_calibrate_intrinsics(args: argparse.Namespace) -> int:
    from src.motion_capture.reconstruct.intrinsics import (
        calibrate_video,
        parse_board_spec,
        write_intrinsics,
    )

    from .bundle import load_bundle

    board = parse_board_spec(args.board, args.square)
    _plan, index, _manifest = load_bundle(args.session)
    records = []
    for entry in index.recordings:
        if not entry.ok:
            logger.warning("%s: recording unusable, skipped", entry.view)
            continue
        record = calibrate_video(
            entry.view, args.session / entry.file, board, every=args.every
        )
        logger.info(
            "%s: rms %.3f px from %d frames (%d without board) %s",
            entry.view,
            record.rms_px,
            record.frames_used,
            record.frames_without_board,
            "ok" if record.ok else "BELOW STANDARD",
        )
        records.append(record)
    out = write_intrinsics(records, args.out or (args.session / "intrinsics.json"))
    logger.info("intrinsics -> %s", out)
    return 0 if records and all(r.ok for r in records) else 1


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "plan-check": cmd_plan_check,
    "capture": cmd_capture,
    "record": cmd_record,
    "session-check": cmd_session_check,
    "proxy": cmd_proxy,
    "compare": cmd_compare,
    "reconstruct": cmd_reconstruct,
    "calibrate-intrinsics": cmd_calibrate_intrinsics,
    "ingest": cmd_ingest,
    "import": cmd_import,
    "export": cmd_export,
    "analyze": cmd_analyze,
    "fit-model": cmd_fit_model,
    "compare-models": cmd_compare_models,
    "lineage": cmd_lineage,
    "overlay": cmd_overlay,
    "compare-variants": cmd_compare_variants,
    "annotations-to-observations": cmd_annotations_to_observations,
    "kinetics": cmd_kinetics,
    "board": cmd_board,
    "clip": cmd_clip,
    "compare-takes": cmd_compare_takes,
    "reliability": cmd_reliability,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
