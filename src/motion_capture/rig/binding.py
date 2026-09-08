"""Bind a rig plan to the cameras that are actually plugged in.

The CLI (``plan-check``, ``capture``, ``record``) and the Capture Rig tile's
live preview must open the *same* devices for the same views, so the
enumeration -> plan check -> capture-index step lives here once. Failures
are ``ValueError`` (the CLI turns them into exit codes, the tile into a
status line).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .plan import RigPlan, check_plan
from .sources import FrameSource, OpenCvMsmfSource
from .topology import (
    CameraLocation,
    attach_capture_indices,
    dshow_order,
    query_topology,
)

Enumerator = Callable[[], Sequence[CameraLocation]]


def enumerate_cameras() -> list[CameraLocation]:
    """Every camera on the machine with its DirectShow capture index."""
    return list(attach_capture_indices(query_topology(), dshow_order()))


def locate_plan(
    plan: RigPlan, cams: Sequence[CameraLocation] | None = None
) -> dict[str, CameraLocation]:
    """``{view: camera}`` for a realizable plan; ``ValueError`` otherwise.

    ``cams`` defaults to a fresh enumeration. Postcondition: every plan view
    maps to exactly one camera with a capture index.
    """
    located = list(cams) if cams is not None else enumerate_cameras()
    check = check_plan(plan, located)
    if not check.ok:
        raise ValueError(
            f"plan not realizable: missing={list(check.missing)} "
            f"conflicts={list(check.conflicts)}"
        )
    by_instance = {c.camera: c for c in located}
    out = {view: by_instance[inst] for view, inst in check.matched.items()}
    without_index = [v for v, c in out.items() if c.index is None]
    if without_index:
        raise ValueError(f"cameras without a capture index: {without_index}")
    return out


def real_sources(
    plan: RigPlan, cams: Sequence[CameraLocation] | None = None
) -> dict[str, FrameSource]:
    """An OpenCV frame source per plan view, bound by identity."""
    sources: dict[str, FrameSource] = {}
    for view, cam in locate_plan(plan, cams).items():
        assert cam.index is not None  # locate_plan guarantees it
        sources[view] = OpenCvMsmfSource(cam.identity, cam.index)
    return sources
