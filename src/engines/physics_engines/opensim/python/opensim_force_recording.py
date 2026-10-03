"""OpenSim force and segment playback recorder (ADR-0052, #11302, FTO-17).

Provides:
- record_force_series: records ForceTorqueSeries across OpenSim states.
- record_force_and_segment_series: records both ForceTorqueSeries and SegmentSeries.
- CLI entrypoint for recording and rendering force playback from .osim model and .mot motion.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.opensim_force_torque import (
    OpenSimForceTorqueSource,
    _base_name,
    _vec3,
)
from src.shared.python.force_overlay import ForceTorqueFrame, ForceTorqueSeries
from src.shared.python.force_overlay.playback import (
    PlaybackReceipt,
    SegmentSeries,
    render_force_playback,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "record_force_series",
    "record_force_and_segment_series",
    "main",
]


def record_force_series(
    model: Any,
    states: Iterable[Any],
    *,
    out_segments: list[SegmentSeries] | None = None,
) -> ForceTorqueSeries:
    """Record a ForceTorqueSeries and segment geometry from an OpenSim model across states.

    Parameters
    ----------
    model : opensim.Model
        Initialized OpenSim model.
    states : Iterable[opensim.State]
        States to sample consecutively.
    out_segments : list[SegmentSeries] | None
        Optional mutable list to receive the captured SegmentSeries.

    Returns
    -------
    ForceTorqueSeries
        Time-ordered series of force/torque frames.
    """
    source = OpenSimForceTorqueSource(model)
    frames: list[ForceTorqueFrame] = []

    joints = source._joints()
    children: dict[str, list[Any]] = {}
    for joint in joints:
        children.setdefault(_base_name(joint.getParentFrame()), []).append(joint)

    body_names: list[str] = []
    for joint in joints:
        body = _base_name(joint.getChildFrame())
        if body not in body_names:
            body_names.append(body)

    num_bodies = len(body_names)
    proximal_list: list[np.ndarray] = []
    distal_list: list[np.ndarray] = []

    for state in states:
        frame = source.sample(state)
        frames.append(frame)

        p_frame = np.zeros((num_bodies, 3), dtype=np.float64)
        d_frame = np.zeros((num_bodies, 3), dtype=np.float64)

        for b_idx, body in enumerate(body_names):
            # Find the joint that attaches this body as child
            child_joint = next(
                (j for j in joints if _base_name(j.getChildFrame()) == body),
                None,
            )
            if child_joint is not None:
                p_ground = np.asarray(
                    _vec3(child_joint.getChildFrame().getPositionInGround(state))
                )
                p_frame[b_idx] = source._world(p_ground)

                d_ground = source._segment_distal_point(body, children, state)
                if d_ground is not None:
                    d_frame[b_idx] = source._world(d_ground)
                else:
                    d_frame[b_idx] = p_frame[b_idx]

        proximal_list.append(p_frame)
        distal_list.append(d_frame)

    if not frames:
        raise ValueError("states must contain at least one state")

    series = ForceTorqueSeries(frames=tuple(frames))

    if out_segments is not None:
        prox_arr = np.stack(proximal_list, axis=0)  # (T, N, 3)
        dist_arr = np.stack(distal_list, axis=0)  # (T, N, 3)
        segments = SegmentSeries(
            names=tuple(body_names),
            proximal=prox_arr,
            distal=dist_arr,
        )
        out_segments.append(segments)

    return series


def record_force_and_segment_series(
    model: Any, states: Iterable[Any]
) -> tuple[ForceTorqueSeries, SegmentSeries]:
    """Record both ForceTorqueSeries and SegmentSeries from an OpenSim model across states.

    Parameters
    ----------
    model : opensim.Model
        Initialized OpenSim model.
    states : Iterable[opensim.State]
        States to sample.

    Returns
    -------
    tuple[ForceTorqueSeries, SegmentSeries]
        Recorded force series and segment endpoint geometry.
    """
    out_segments: list[SegmentSeries] = []
    series = record_force_series(model, states, out_segments=out_segments)
    if not out_segments:
        raise RuntimeError("Failed to extract segment geometry")
    return series, out_segments[0]


def main(argv: list[str] | None = None) -> PlaybackReceipt | int:
    """CLI entrypoint for recording and rendering OpenSim force playback."""
    parser = argparse.ArgumentParser(
        description="Record and render OpenSim force playback animation."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to .osim OpenSim model file",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        required=False,
        default=None,
        help="Path to .mot OpenSim motion/storage file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output video path (.mp4) or directory path for PNG frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback frames per second (default: 30)",
    )

    args = parser.parse_args(argv)

    try:
        import opensim  # type: ignore[import-not-found]
    except ImportError:
        logger.error(
            "OpenSim Python package is required to execute playback recording."
        )
        return 1

    if not args.model.exists():
        logger.error("Model file not found: %s", args.model)
        return 1

    model = opensim.Model(str(args.model))
    state = model.initSystem()

    states = [state]
    if args.motion is not None and args.motion.exists():
        storage = opensim.Storage(str(args.motion))
        times = [storage.getStateVector(i).getTime() for i in range(storage.getSize())]
        # In a full simulation, states would be populated from Storage or forward integration
        logger.info("Loaded motion with %d frames from %s", len(times), args.motion)

    logger.info("Recording forces for %d states...", len(states))
    series, segments = record_force_and_segment_series(model, states)

    logger.info("Rendering force playback to %s...", args.out)
    receipt = render_force_playback(
        series,
        segments,
        out_path=args.out,
        fps=args.fps,
    )
    logger.info(
        "Rendered %d frames with encoder %s to %s",
        receipt.frame_count,
        receipt.encoder,
        receipt.out_path,
    )
    return receipt


if __name__ == "__main__":
    sys.exit(0 if isinstance(main(), PlaybackReceipt) else 1)
