#!/usr/bin/env python3
"""Diagnose a multi-camera USB rig for markerless motion capture (Windows).

Thin operator CLI over :mod:`motion_capture.rig`: enumerate the cameras and
their USB root ports, predict how many can stream (one ELP AR0234 per USB 2.0
root port — see ``docs/motion_capture/usb_camera_rig_bringup.md``), then
measure a solo and a concurrent capture and compare. Exit 0 when every camera
streams at 90 % of target or better, 1 on a conflict, 2 when nothing can run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.motion_capture.rig.plan import (  # noqa: E402
    CameraBinding,
    CaptureMode,
    RigPlan,
)
from src.motion_capture.rig.session import CaptureSession, SessionManifest  # noqa: E402
from src.motion_capture.rig.sources import OpenCvMsmfSource  # noqa: E402
from src.motion_capture.rig.topology import (  # noqa: E402
    PERIODIC_BUDGET_BYTES,
    RESERVE_BYTES,
    VENDOR_ID,
    CameraLocation,
    attach_capture_indices,
    dshow_order,
    predict_streaming,
    query_topology,
)

SETTLE_SECONDS = 2.0  # Media Foundation teardown is asynchronous


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vendor", default=VENDOR_ID, help="USB vendor id (hex)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--solo-seconds", type=float, default=2.0)
    parser.add_argument("--concurrent-seconds", type=float, default=8.0)
    parser.add_argument("--out-dir", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def _plan_from_topology(cams: list[CameraLocation], mode: CaptureMode) -> RigPlan:
    """One view per enumerated camera, bound by serial or port path."""
    bindings = tuple(
        CameraBinding(
            view=cam.identity,
            serial=cam.serial,
            port_path=None if cam.serial else cam.identity,
            mode=mode,
        )
        for cam in cams
        if cam.index is not None
    )
    return RigPlan(name="diagnose", cameras=bindings)


def _sources(cams: list[CameraLocation]) -> dict[str, OpenCvMsmfSource]:
    return {
        cam.identity: OpenCvMsmfSource(cam.identity, cam.index)
        for cam in cams
        if cam.index is not None
    }


def _print_topology(cams: list[CameraLocation]) -> int:
    for cam in cams:
        print(
            f"  idx {cam.index}  {cam.identity:<22} "
            f"serial={'yes' if cam.serial else 'NO '} root_port={cam.root_port}  "
            f"hubs={cam.hub_depth}/5  root_hub={cam.root_hub}"
        )
    expected, lines = predict_streaming(cams)
    print(
        f"\n== Prediction ({RESERVE_BYTES} B reserved per camera, "
        f"{PERIODIC_BUDGET_BYTES} B periodic budget per root port) =="
    )
    print("\n".join(lines))
    print(f"  expected streaming: {expected}/{len(cams)}")
    return expected


def _run_solo(plan: RigPlan, cams: list[CameraLocation], seconds: float) -> None:
    print("\n== Solo health check ==")
    for binding in plan.cameras:
        single = RigPlan(name="solo", cameras=(binding,))
        cam = next(c for c in cams if c.identity == binding.identity)
        manifest = CaptureSession(
            single, _sources([cam]), duration_s=seconds, max_reopens=0
        ).run()
        stats = manifest.cameras[0]
        size = stats.effective_mode and (
            stats.effective_mode.width,
            stats.effective_mode.height,
        )
        print(
            f"  {stats.identity:<22} {stats.achieved_fps:5.1f} fps  {size}  "
            f"failed={stats.failed_reads}"
        )
        time.sleep(SETTLE_SECONDS)


def _run_concurrent(
    plan: RigPlan, cams: list[CameraLocation], args: argparse.Namespace
) -> SessionManifest:
    print(
        f"\n== Concurrent capture: {len(plan.cameras)} cameras, "
        f"{args.width}x{args.height}@{args.fps} MJPG, {args.concurrent_seconds:.0f}s =="
    )
    manifest = CaptureSession(
        plan, _sources(cams), duration_s=args.concurrent_seconds, max_reopens=0
    ).run()
    by_identity = {c.identity: c for c in cams}
    for stats in manifest.cameras:
        state = {"ok": "OK", "degraded": "DEGRADED"}.get(stats.state, "NO STREAM")
        print(
            f"  {stats.identity:<22} root_port={by_identity[stats.identity].root_port}  "
            f"{stats.achieved_fps:5.1f} fps  failed={stats.failed_reads:>8}  "
            f"worst_gap={stats.worst_gap_ms:.0f}ms  {state}"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Run topology, prediction, solo, and concurrent checks; return exit code."""
    args = _parse_args(argv)
    try:
        import cv2  # noqa: F401 - probe availability before touching hardware
    except ImportError:
        print("OpenCV (cv2) is not installed; nothing to test.")
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("== PnP topology (slow: one PowerShell property query per hub tier) ==")
    cams = attach_capture_indices(query_topology(args.vendor), dshow_order())
    if not cams:
        print(f"  no cameras with VID_{args.vendor} enumerated by Windows PnP.")
        return 2
    expected = _print_topology(cams)
    mode = CaptureMode(width=args.width, height=args.height, fps=args.fps)
    plan = _plan_from_topology(cams, mode)
    _run_solo(plan, cams, args.solo_seconds)
    manifest = _run_concurrent(plan, cams, args)

    streaming = sum(1 for c in manifest.cameras if c.state == "ok")
    verdict = "PASS" if streaming == len(plan.cameras) else "FAIL"
    note = "" if streaming == expected else "   [prediction mismatch - investigate]"
    print(
        f"\n  streaming {streaming}/{len(plan.cameras)}  "
        f"(predicted {expected}/{len(cams)})  -> {verdict}{note}"
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = args.out_dir / f"rig_report_{stamp}.json"
    report.write_text(
        json.dumps(
            {
                "cameras": [cam.to_dict() for cam in cams],
                "expected_streaming": expected,
                "measured_streaming": streaming,
                "verdict": verdict,
                "manifest": json.loads(manifest.model_dump_json()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {report}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
