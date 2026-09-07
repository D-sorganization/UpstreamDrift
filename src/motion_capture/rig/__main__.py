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
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from src.shared.python.logging_pkg.logging_config import get_logger

from .bundle import build_index, check_bundle, write_bundle
from .plan import RigPlan, check_plan, parse_mode
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


def _load_plan(args: argparse.Namespace) -> RigPlan:
    """The plan file with ``--mode``/``--views`` applied."""
    plan = RigPlan.load(args.plan)
    views = None
    if getattr(args, "views", None):
        views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    return plan.with_overrides(mode=getattr(args, "mode", None), views=views)


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
    ing = sub.add_parser("ingest", help="pose-estimate every recording in a bundle")
    ing.add_argument("--session", type=Path, required=True)
    ing.add_argument(
        "--out", type=Path, default=None, help="default: <session>/observations"
    )
    ing.add_argument("--estimator", default="mediapipe")
    ing.add_argument("--max-frames", type=int, default=None)
    cmp = sub.add_parser("compare", help="run two estimators on one bundle")
    cmp.add_argument("--session", type=Path, required=True)
    cmp.add_argument("--estimators", default="mediapipe,openpose_dnn")
    cmp.add_argument("--max-frames", type=int, default=None)
    cmp.add_argument("--min-confidence", type=float, default=0.5)
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


def cmd_ingest(args: argparse.Namespace) -> int:
    # Imported here: the pose stack (MediaPipe, simulation backends) takes
    # seconds to import and no other command needs it.
    from .ingest import ingest_bundle, registry_estimator_factory

    out_dir = args.out or (args.session / "observations")
    index = ingest_bundle(
        args.session,
        out_dir,
        registry_estimator_factory(args.estimator),
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


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "plan-check": cmd_plan_check,
    "capture": cmd_capture,
    "record": cmd_record,
    "session-check": cmd_session_check,
    "proxy": cmd_proxy,
    "compare": cmd_compare,
    "ingest": cmd_ingest,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
