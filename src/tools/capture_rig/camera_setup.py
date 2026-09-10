"""Player-authored camera plans using the existing rig contracts and probes."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path
from threading import Event
from uuid import uuid4

from src.motion_capture.rig.binding import locate_plan
from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.motion_capture.rig.topology import (
    CameraLocation,
    attach_capture_indices,
    dshow_order,
    query_topology,
)

MAX_CAMERAS = 12
MAX_PLAN_BYTES = 1024 * 1024
PROBE_TIMEOUT_SECONDS = 12
_VIEW_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")


def bind_camera(view: str, camera: CameraLocation, mode: CaptureMode) -> CameraBinding:
    """Use a stable serial or port identity, never a transient capture index."""
    return CameraBinding(
        view=view,
        serial=camera.serial,
        port_path=None if camera.serial else camera.identity,
        mode=mode,
    )


def create_plan(
    name: str, cameras: Sequence[CameraBinding], notes: str = ""
) -> RigPlan:
    """Validate a player-editable plan without declaring live readiness."""
    name = name.strip()
    if not name or len(name) > 120 or any(ord(char) < 32 for char in name):
        raise ValueError("Give this camera setup a name of 1–120 visible characters")
    if not 1 <= len(cameras) <= MAX_CAMERAS:
        raise ValueError(f"Choose between 1 and {MAX_CAMERAS} camera views")
    if len(notes) > 10_000:
        raise ValueError("Camera setup notes must be at most 10,000 characters")
    for camera in cameras:
        if not _VIEW_NAME.fullmatch(camera.view):
            raise ValueError(
                f"Invalid view {camera.view!r}: start with a letter and use only "
                "letters, numbers, underscores or hyphens (up to 48 characters)"
            )
    return RigPlan(name=name, cameras=tuple(cameras), notes=notes)


def load_plan(path: Path) -> RigPlan:
    """Read a bounded plan before replacing any unsaved editor choices."""
    if path.stat().st_size > MAX_PLAN_BYTES:
        raise ValueError("Camera plan exceeds the 1 MiB document limit")
    plan = RigPlan.load(path)
    return create_plan(plan.name, plan.cameras, plan.notes)


def save_revision(plan: RigPlan, library: Path) -> Path:
    """Write a new atomic revision; existing plans and captures remain intact."""
    checked = create_plan(plan.name, plan.cameras, plan.notes)
    folder = library / "camera-plans"
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", plan.name).strip("-") or "camera-setup"
    target = folder / f"{slug}-{uuid4().hex}.json"
    write_document(target, checked.model_dump(mode="json"))
    return target


def connection_status(
    plan: RigPlan, cameras: Sequence[CameraLocation]
) -> tuple[bool, str]:
    """Use the same identity/index/bus checks as live preview and recording."""
    try:
        locate_plan(plan, cameras)
    except ValueError as exc:
        return (
            False,
            f"Connections need attention: {exc}. Rescan or edit the selected devices.",
        )
    return True, (
        "All views match detected cameras. Preview and Plan Check must still verify "
        "the recording setup; stream modes and lens calibration are not verified by this scan."
    )


def discover_cameras(cancelled: Event) -> list[CameraLocation]:
    """Run bounded existing Windows rig probes; cancellation discards the scan."""
    if cancelled.is_set():
        return []
    if sys.platform != "win32":
        raise OSError(
            "Live rig discovery currently requires Windows. Load a saved plan "
            "for offline editing, or import video through Capture Library."
        )
    cameras = query_topology(timeout_s=PROBE_TIMEOUT_SECONDS)
    if cancelled.is_set():
        return []
    order = dshow_order(timeout_s=PROBE_TIMEOUT_SECONDS)
    return [] if cancelled.is_set() else list(attach_capture_indices(cameras, order))
