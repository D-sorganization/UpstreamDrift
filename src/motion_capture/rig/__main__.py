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
from .plan import RigPlan, check_plan
from .recorder import (
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motion_capture.rig", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("plan-check", help="match a plan against enumerated cameras")
    check.add_argument("--plan", type=Path, required=True)
    cap = sub.add_parser("capture", help="capture every planned camera together")
    cap.add_argument("--plan", type=Path, required=True)
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
    rec.add_argument("--plan", type=Path, required=True)
    rec.add_argument("--duration", type=float, default=10.0)
    rec.add_argument("--out", type=Path, default=Path.cwd() / "session")
    rec.add_argument(
        "--dry-run",
        action="store_true",
        help="record nothing; write the bundle with NullRecorder results",
    )
    chk = sub.add_parser("session-check", help="validate a session bundle on disk")
    chk.add_argument("--session", type=Path, required=True)
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
    by_identity = {c.identity: c for c in cams}
    return {b.view: by_identity[b.identity] for b in plan.cameras}


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
    plan = RigPlan.load(args.plan)
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
    plan = RigPlan.load(args.plan)
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


def cmd_record(args: argparse.Namespace) -> int:
    plan = RigPlan.load(args.plan)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    factory: Callable[[], Recorder]
    if args.dry_run:
        refs = {c.view: f"dry-run:{c.identity}" for c in plan.cameras}
        factory = NullRecorder
    else:
        refs = _device_refs(plan)
        factory = FfmpegStreamCopyRecorder
    results = record_all(plan, refs, args.duration, args.out, factory)
    index = build_index(plan, results, args.duration, args.out)
    manifest = write_bundle(
        args.out,
        plan,
        index,
        started_utc=started,
        tools_schema=probe_tools_schema().to_dict(),
    )
    for entry in index.recordings:
        logger.info(
            "%s (%s): %s %d bytes rc=%s",
            entry.view,
            entry.identity,
            entry.file,
            entry.bytes,
            entry.returncode,
        )
    logger.info("outcome=%s bundle=%s", manifest.outcome.value, args.out)
    return _EXIT_BY_OUTCOME[manifest.outcome]


def cmd_session_check(args: argparse.Namespace) -> int:
    check = check_bundle(args.session)
    for problem in check.problems:
        logger.error("%s", problem)
    logger.info("session %s: %s", args.session, "ok" if check.ok else "problems found")
    return 0 if check.ok else 1


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "plan-check": cmd_plan_check,
    "capture": cmd_capture,
    "record": cmd_record,
    "session-check": cmd_session_check,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
