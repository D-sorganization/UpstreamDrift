#!/usr/bin/env python3
"""Diagnose a multi-camera USB rig for markerless motion capture (Windows).

The ELP AR0234 "Global Shutter Camera" (``VID_32E4``) requests its largest
isochronous alt-setting (3060 bytes per microframe) for every video format, and
USB 2.0 High-Speed allows 6000 bytes of periodic traffic per microframe. Two of
these cameras therefore never fit on one USB 2.0 bus, and xHCI accounts that
budget per root port. This script predicts how many cameras can stream from the
hub topology, then measures it. See ``docs/motion_capture/usb_camera_rig_bringup.md``.

It is an operator diagnostic. It defines no capture contract; those belong to
Tools ``sidekick.lab.mocap`` under ADR-0041.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

VENDOR_ID = "32E4"
PERIODIC_BUDGET_BYTES = 6000  # 80 % of a 7500-byte USB 2.0 High-Speed microframe
RESERVE_BYTES = 3060  # alt-setting 0x0B: 3 transactions x 1020 bytes per microframe
SETTLE_SECONDS = 2.0  # Media Foundation teardown is asynchronous

PS_TOPOLOGY = r"""
$ErrorActionPreference = 'SilentlyContinue'
$cams = Get-PnpDevice -PresentOnly -Class Camera |
  Where-Object { $_.InstanceId -match 'VID_@VENDOR@' }
$out = @()
foreach ($c in $cams) {
  $chain = @(); $cur = $c.InstanceId
  while ($cur) {
    $d = Get-PnpDevice -InstanceId $cur
    if (-not $d) { break }
    $loc = (Get-PnpDeviceProperty -InstanceId $cur `
      -KeyName 'DEVPKEY_Device_LocationInfo').Data
    $chain += [PSCustomObject]@{ name = $d.FriendlyName; id = $cur; loc = [string]$loc }
    if ($cur -match '^PCI\\') { break }
    $cur = (Get-PnpDeviceProperty -InstanceId $cur -KeyName 'DEVPKEY_Device_Parent').Data
  }
  $out += [PSCustomObject]@{ camera = $c.InstanceId; chain = $chain }
}
ConvertTo-Json -InputObject @($out) -Depth 5 -Compress
"""

DSHOW_NAME = re.compile(r'\]\s+"(.+?)"\s+\(video\)')
DSHOW_ALT = re.compile(r'Alternative name\s+"@device_pnp_\\\\\?\\(.+?)#\{')
ROOT_PORT = re.compile(r"Port_#(\d+)")


@dataclass
class Camera:
    """One enumerated camera and its position in the USB tree."""

    camera: str
    composite: str | None
    serial: str | None
    identity: str
    root_hub: str | None
    root_port: int | None
    host: str | None
    hub_depth: int
    chain: list[dict[str, str]] = field(default_factory=list)
    index: int | None = None


def derive_camera(entry: dict[str, Any], vendor: str = VENDOR_ID) -> Camera:
    """Build a :class:`Camera` from one PnP chain (camera first, host last).

    The composite device's instance tail is the USB serial when the unit
    exposes one; a tail containing ``&`` is a port path, so identity falls back
    to the camera's own instance tail. The element directly below the root hub
    carries the root-port number in its location string.
    """
    chain = entry["chain"] if isinstance(entry["chain"], list) else [entry["chain"]]
    composite_re = re.compile(
        rf"USB\\VID_{vendor}&PID_[0-9A-F]{{4}}\\[^\\]+$", re.IGNORECASE
    )
    composite = next((x for x in chain if composite_re.match(x["id"])), None)
    tail = composite["id"].rsplit("\\", 1)[1] if composite else ""
    serial = tail if tail and "&" not in tail else None
    root_idx = next(
        (i for i, x in enumerate(chain) if x["id"].upper().startswith("USB\\ROOT_HUB")),
        None,
    )
    below = chain[root_idx - 1] if root_idx else None
    port = ROOT_PORT.search(below.get("loc", "")) if below else None
    return Camera(
        camera=entry["camera"],
        composite=composite["id"] if composite else None,
        serial=serial,
        identity=serial or "path_" + entry["camera"].split("\\")[-1].replace("&", "-"),
        root_hub=chain[root_idx]["id"] if root_idx is not None else None,
        root_port=int(port.group(1)) if port else None,
        host=next(
            (x["id"] for x in chain if x["id"].upper().startswith("PCI\\")), None
        ),
        hub_depth=sum(
            1 for x in chain if "Hub" in x["name"] and "Root" not in x["name"]
        ),
        chain=chain,
    )


def parse_dshow_listing(text: str) -> list[tuple[str, str]]:
    """Return ``(friendly name, instance path)`` for each DirectShow video device.

    Order matters: OpenCV's MSMF and DSHOW backends index devices in this order.
    The alternative name ``usb#vid_...#6&fadbf3b&0&0000`` becomes the PnP
    instance id ``USB\\VID_...\\6&FADBF3B&0&0000``.
    """
    order: list[tuple[str, str]] = []
    name: str | None = None
    for line in text.splitlines():
        if match := DSHOW_NAME.search(line):
            name = match.group(1)
            continue
        if name is not None and (match := DSHOW_ALT.search(line)):
            order.append((name, match.group(1).replace("#", "\\").upper()))
            name = None
    return order


def predict(cams: list[Camera]) -> tuple[int, list[str]]:
    """Return the expected streaming count and one report line per root port."""
    groups: dict[tuple[str | None, str | None, int | None], list[Camera]] = {}
    for cam in cams:
        groups.setdefault((cam.host, cam.root_hub, cam.root_port), []).append(cam)
    fit = PERIODIC_BUDGET_BYTES // RESERVE_BYTES
    expected = 0
    lines = []
    for (_, hub, port), group in groups.items():
        expected += min(len(group), fit)
        flag = "   <-- CONFLICT" if len(group) > fit else ""
        lines.append(
            f"  root_port {port} of {hub}: {len(group)} camera(s) -> "
            f"{min(len(group), fit)} can stream{flag}"
        )
    return expected, lines


def query_topology(vendor: str = VENDOR_ID) -> list[Camera]:
    """Walk every camera's hub chain through Windows PnP (slow, seconds per tier)."""
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            PS_TOPOLOGY.replace("@VENDOR@", vendor),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    data = json.loads(result.stdout.strip() or "[]")
    return [
        derive_camera(e, vendor) for e in ([data] if isinstance(data, dict) else data)
    ]


def dshow_order() -> list[tuple[str, str]]:
    """List DirectShow video devices with the bundled imageio-ffmpeg binary."""
    import imageio_ffmpeg

    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return parse_dshow_listing(result.stderr + result.stdout)


def capture(
    index: int,
    spec: tuple[int, int, int],
    duration: float,
    barrier: threading.Barrier | None,
    out: dict[int, dict[str, Any]],
    frame_path: Path,
) -> None:
    """Stream one camera and record fps, failed reads, and the worst frame gap."""
    import cv2

    width, height, fps = spec
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    opened = cap.isOpened()
    if opened:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(5):
            cap.read()
    if barrier is not None:
        barrier.wait()
    reads = good = 0
    worst = 0.0
    last = None
    start = time.time()
    while time.time() - start < duration:
        tick = time.time()
        ok, frame = cap.read() if opened else (False, None)
        reads += 1
        if ok and frame is not None:
            good += 1
            last = frame
            worst = max(worst, time.time() - tick)
    elapsed = time.time() - start
    out[index] = {
        "opened": opened,
        "fps": good / elapsed,
        "good": good,
        "failed": reads - good,
        "worst_gap_ms": worst * 1000,
        "size": [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        if opened
        else None,
    }
    if last is not None:
        cv2.imwrite(str(frame_path), last)
    cap.release()


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


Results = dict[int, dict[str, Any]]


def _enumerate(vendor: str) -> list[Camera]:
    """Query PnP, attach capture indices, and print one line per camera."""
    print("== PnP topology (slow: one PowerShell property query per hub tier) ==")
    cams = query_topology(vendor)
    if not cams:
        print(f"  no cameras with VID_{vendor} enumerated by Windows PnP.")
        return cams
    index_by_path = {path: i for i, (_, path) in enumerate(dshow_order())}
    for cam in cams:
        cam.index = index_by_path.get(cam.camera.upper())
    cams.sort(key=lambda c: (c.index is None, c.index or 0))
    for cam in cams:
        print(
            f"  idx {cam.index}  {cam.identity:<22} "
            f"serial={'yes' if cam.serial else 'NO '} root_port={cam.root_port}  "
            f"hubs={cam.hub_depth}/5  root_hub={cam.root_hub}"
        )
    return cams


def _print_prediction(cams: list[Camera]) -> int:
    expected, lines = predict(cams)
    print(
        f"\n== Prediction ({RESERVE_BYTES} B reserved per camera, "
        f"{PERIODIC_BUDGET_BYTES} B periodic budget per root port) =="
    )
    print("\n".join(lines))
    print(f"  expected streaming: {expected}/{len(cams)}")
    return expected


def _run_solo(live: list[Camera], args: argparse.Namespace) -> Results:
    """Open each camera on its own, with a settle pause between opens."""
    print("\n== Solo health check ==")
    solo: Results = {}
    spec = (args.width, args.height, args.fps)
    for cam in live:
        assert cam.index is not None
        frame = args.out_dir / f"frame_solo_{cam.identity}.png"
        capture(cam.index, spec, args.solo_seconds, None, solo, frame)
        result = solo[cam.index]
        print(
            f"  {cam.identity:<22} {result['fps']:5.1f} fps  {result['size']}  "
            f"failed={result['failed']}"
        )
        time.sleep(SETTLE_SECONDS)
    return solo


def _run_concurrent(live: list[Camera], args: argparse.Namespace) -> Results:
    """Start every camera behind one barrier so reservations compete for real."""
    print(
        f"\n== Concurrent capture: {len(live)} cameras, "
        f"{args.width}x{args.height}@{args.fps} MJPG, {args.concurrent_seconds:.0f}s =="
    )
    concurrent: Results = {}
    barrier = threading.Barrier(len(live))
    spec = (args.width, args.height, args.fps)
    threads = [
        threading.Thread(
            target=capture,
            args=(
                cam.index,
                spec,
                args.concurrent_seconds,
                barrier,
                concurrent,
                args.out_dir / f"frame_{cam.identity}.png",
            ),
        )
        for cam in live
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return concurrent


def _count_streaming(live: list[Camera], concurrent: Results, fps: int) -> int:
    """Print one verdict line per camera; a camera streams at >= 90 % of target."""
    streaming = 0
    for cam in live:
        assert cam.index is not None
        result = concurrent[cam.index]
        ok = result["fps"] >= 0.9 * fps
        streaming += int(ok)
        state = "OK" if ok else ("NO STREAM" if result["fps"] < 1 else "DEGRADED")
        print(
            f"  {cam.identity:<22} root_port={cam.root_port}  {result['fps']:5.1f} fps  "
            f"failed={result['failed']:>8}  worst_gap={result['worst_gap_ms']:.0f}ms  "
            f"{state}"
        )
    return streaming


def _write_report(
    args: argparse.Namespace,
    cams: list[Camera],
    solo: Results,
    concurrent: Results,
    expected: int,
    streaming: int,
    verdict: str,
) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = args.out_dir / f"rig_report_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "spec": {"w": args.width, "h": args.height, "fps": args.fps},
        "reserve_bytes": RESERVE_BYTES,
        "budget_bytes": PERIODIC_BUDGET_BYTES,
        "cameras": [asdict(cam) for cam in cams],
        "solo": {str(k): v for k, v in solo.items()},
        "concurrent": {str(k): v for k, v in concurrent.items()},
        "expected_streaming": expected,
        "measured_streaming": streaming,
        "verdict": verdict,
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {report}")


def main(argv: list[str] | None = None) -> int:
    """Run topology, prediction, solo, and concurrent checks; return exit code."""
    args = _parse_args(argv)
    try:
        import cv2  # noqa: F401 - probe availability before touching hardware
    except ImportError:
        print("OpenCV (cv2) is not installed; nothing to test.")
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cams = _enumerate(args.vendor)
    if not cams:
        return 2
    expected = _print_prediction(cams)
    live = [cam for cam in cams if cam.index is not None]
    solo = _run_solo(live, args)
    concurrent = _run_concurrent(live, args)
    streaming = _count_streaming(live, concurrent, args.fps)

    verdict = "PASS" if streaming == len(live) else "FAIL"
    note = "" if streaming == expected else "   [prediction mismatch - investigate]"
    print(
        f"\n  streaming {streaming}/{len(live)}  "
        f"(predicted {expected}/{len(cams)})  -> {verdict}{note}"
    )
    _write_report(args, cams, solo, concurrent, expected, streaming, verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
