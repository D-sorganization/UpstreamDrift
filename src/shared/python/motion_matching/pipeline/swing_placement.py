"""Whole-swing marker placement as a stage of the matching pipeline (#15274).

The pipeline places a marker on its segment from one posture (the address, or
the static trial with ``--static-seeds``). ``--placement-passes N`` adds N
rounds of: solve the swing, re-place every marker at its mean position in its
segment's frame over that solve
(:func:`swing_marker_placement.place_markers_over_swing`), rebuild the marker
kinematics and the address, solve again. Zero passes, the default, leaves the
pipeline as it was.

Kinematics only: no body, joint, mass or mesh changes. Markers on the bodies
named by ``--placement-keep-bodies`` (default the club, whose marker geometry
is measured, not fitted) keep their placement.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.swing_marker_placement import (
    PlacementSettings,
    place_markers_over_swing,
)

Array = NDArray[np.float64]
Offset = tuple[float, float, float]
Attachments = Mapping[str, tuple[str, Sequence[float]]]
#: Builds marker kinematics (``marker_positions(q)``) from attachments.
MakeKinematics = Callable[[Mapping[str, tuple[str, Offset]]], Any]

DEFAULT_PLACEMENT_PASSES = 0
DEFAULT_KEEP_BODIES: tuple[str, ...] = ("Clubhead",)
#: Probe points that read a segment's frame: its origin and its three axes.
_PROBES: tuple[tuple[str, Offset], ...] = (
    ("o", (0.0, 0.0, 0.0)),
    ("x", (1.0, 0.0, 0.0)),
    ("y", (0.0, 1.0, 0.0)),
    ("z", (0.0, 0.0, 1.0)),
)


def _passes(value: str) -> int:
    """Parse a placement-pass count that is zero or more."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or more")
    return number


def _move(value: str) -> float:
    """Parse a non-negative maximum marker move in metres."""
    number = float(value)
    if not number >= 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def add_placement_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the whole-swing placement options to a pipeline parser."""
    defaults = PlacementSettings()
    parser.add_argument(
        "--placement-passes",
        type=_passes,
        default=DEFAULT_PLACEMENT_PASSES,
        help="rounds of re-placing markers on their segments from the solved "
        "swing, each followed by a new solve (0: off; #15274)",
    )
    parser.add_argument(
        "--placement-max-move-m",
        type=_move,
        default=defaults.max_move_m,
        help="furthest a marker may move on its segment in one pass (m)",
    )
    parser.add_argument(
        "--placement-keep-bodies",
        nargs="*",
        default=list(DEFAULT_KEEP_BODIES),
        help="segments whose markers keep their placement",
    )


def placement_settings(args: argparse.Namespace) -> PlacementSettings:
    """The placement settings a parsed command line asks for."""
    return PlacementSettings(max_move_m=args.placement_max_move_m)


def segment_poses(
    make_kinematics: MakeKinematics, bodies: Sequence[str], q: Array
) -> tuple[dict[str, Array], dict[str, Array]]:
    """World rotation (frames, 3, 3) and origin (frames, 3) of each segment.

    Read from the marker kinematics itself, with four probe markers per
    segment, so the frames are the ones marker offsets are expressed in.

    Precondition: ``q`` is (frames, coordinates).
    """
    poses = np.asarray(q, dtype=float)
    if poses.ndim != 2:
        raise ValueError("q must have shape (frames, coordinates)")
    probe = {
        f"{body}|{name}": (body, offset) for body in bodies for name, offset in _PROBES
    }
    kinematics = make_kinematics(probe)
    seen = np.stack([kinematics.marker_positions(row) for row in poses])
    seen = seen.reshape(len(poses), len(bodies), len(_PROBES), 3)
    rotations: dict[str, Array] = {}
    origins: dict[str, Array] = {}
    for column, body in enumerate(bodies):
        origin = seen[:, column, 0]
        origins[body] = origin
        # Columns of the rotation are the segment's axes in the world.
        rotations[body] = np.stack(
            [seen[:, column, axis] - origin for axis in (1, 2, 3)], axis=2
        )
    return rotations, origins


def replaced_attachments(
    make_kinematics: MakeKinematics,
    points_m: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    attachments: Attachments,
    q: Array,
    settings: PlacementSettings,
    keep_bodies: Sequence[str] = DEFAULT_KEEP_BODIES,
) -> tuple[dict[str, tuple[str, Offset]], dict[str, Any]]:
    """Attachments re-placed over the solved swing ``q``, and a report.

    Markers without an attachment and markers on ``keep_bodies`` are passed
    through. The report is plain data for the receipt: the RMS distance of the
    moved markers from their measured positions before and after, at the same
    segment poses, and each marker's move.

    Preconditions: ``points_m`` is (frames, markers, 3) with one marker per
    label and ``q`` has one row per frame (``ValueError`` otherwise).
    Postconditions: ``attachments`` is not changed; every label that had an
    attachment still has one on the same segment.
    """
    poses = np.asarray(q, dtype=float)
    points = np.asarray(points_m, dtype=float)
    if poses.ndim != 2 or len(poses) != len(points):
        raise ValueError(
            f"q must have one row per {len(points)} frames of marker positions"
        )
    kept = set(keep_bodies)
    columns = [
        column
        for column, label in enumerate(labels)
        if label in attachments and attachments[label][0] not in kept
    ]
    moving = [labels[column] for column in columns]
    out: dict[str, tuple[str, Offset]] = {
        label: (body, (float(o[0]), float(o[1]), float(o[2])))
        for label, (body, o) in attachments.items()
    }
    bodies = sorted({attachments[label][0] for label in moving})
    rotations, origins = segment_poses(make_kinematics, bodies, poses)
    placed = place_markers_over_swing(
        points[:, columns],
        np.asarray(valid, dtype=bool)[:, columns],
        moving,
        attachments,
        rotations,
        origins,
        settings,
    )
    frames = np.array([entry.frames_used for entry in placed.values()], dtype=float)
    total = max(float(frames.sum()), 1.0)

    def pooled(values: list[float]) -> float:
        """Frame-weighted root-mean-square of the per-marker values."""
        return float(np.sqrt(np.nansum(frames * np.square(values)) / total))

    for label, entry in placed.items():
        out[label] = (entry.body, entry.offset_m)
    report = {
        "max_move_m": settings.max_move_m,
        "min_frames": settings.min_frames,
        "kept_bodies": sorted(kept),
        "moved": sum(entry.replaced for entry in placed.values()),
        "limited": sorted(label for label, entry in placed.items() if entry.limited),
        "rms_before_m": pooled([entry.rms_before_m for entry in placed.values()]),
        "rms_after_m": pooled([entry.rms_after_m for entry in placed.values()]),
        "markers": {
            label: {
                "body": entry.body,
                "moved_m": entry.moved_m,
                "rms_before_m": entry.rms_before_m,
                "rms_after_m": entry.rms_after_m,
                "frames_used": entry.frames_used,
            }
            for label, entry in placed.items()
        },
    }
    return out, report


def replace_run_markers(
    lane: Any, calibrated: Any, q: Array, args: argparse.Namespace
) -> tuple[Any, dict[str, Any]]:
    """One placement pass on a calibrated pipeline run.

    ``calibrated`` is the pipeline's calibration result; the copy returned has
    the new attachments and leg offsets, marker kinematics and simulator built
    on them, and the address solved again from the old one.
    """
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    attachments, report = replaced_attachments(
        lane.plant.create_ik,
        lane.points,
        lane.valid,
        lane.labels,
        calibrated.attachments,
        q,
        placement_settings(args),
        tuple(args.placement_keep_bodies),
    )
    adapter, kin = lane.kinematics(
        calibrated.spec_bytes,
        attachments,
        ik_backend=getattr(args, "ik_backend", "lm"),
    )
    offsets = {label: attachments[label] for label in calibrated.offsets}
    address = lane.best_address(kin, calibrated.address2.q)
    report["address_marker_rms_m"] = float(address.marker_rms_m)
    updated = replace(
        calibrated,
        attachments=attachments,
        offsets=offsets,
        adapter=adapter,
        kin=kin,
        sim=fs.FullBodySimulator(adapter),
        address2=address,
    )
    return updated, report
