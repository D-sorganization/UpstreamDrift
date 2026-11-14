"""Segment-scale grid search for ground-support address (#14771).

This module lives next to ``pipeline/`` so tests can import it without running
``pipeline.__init__`` (which pulls the workspace stack).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.segment_scaling import scale_segments

#: Must stay aligned with ``pipeline.constants.SCALE_GRID``.
SCALE_GRID: tuple[float, ...] = (0.94, 0.97, 1.00)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.pipeline.lane import Lane


def scaled_offsets(
    offsets: Mapping[str, tuple[str, Sequence[float]]], femur: float, tibia: float
) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Leg marker offsets carried onto scaled femur and tibia bodies.

    Precondition: ``femur`` and ``tibia`` must be strictly positive.
    Postcondition: returned offsets preserve body names with scaled coordinate offsets.
    """
    if femur <= 0 or tibia <= 0:
        raise ValueError("femur and tibia must have positive scale factors")
    out: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for label, (body, offset) in offsets.items():
        if body.startswith("femur"):
            factor = femur
        elif body.startswith("tibia"):
            factor = tibia
        else:
            factor = 1.0
        scaled = factor * np.asarray(offset, dtype=float)
        out[label] = (body, (float(scaled[0]), float(scaled[1]), float(scaled[2])))
    return out


def search_segment_scales(
    lane: Lane,
    hip_spec: Mapping[str, Any],
    fixed: Mapping[str, tuple[str, Sequence[float]]],
    offsets: Mapping[str, tuple[str, Sequence[float]]],
    address_q: np.ndarray,
    grid: Sequence[float] = SCALE_GRID,
    log: logging.Logger | None = None,
) -> tuple[dict[str, Any], float, float, list[dict[str, Any]]]:
    """Grid search over femur and tibia scales minimizing pinned stance RMS.

    Preconditions:
    - ``grid`` must not be empty and must contain strictly positive scale factors.
    - ``address_q`` must be a 1D array of generalized coordinates.

    Postconditions:
    - Returns best scaled specification, femur scale, tibia scale, and table.
    """
    if not grid:
        raise ValueError("grid must not be empty")
    for s in grid:
        if s <= 0:
            raise ValueError(f"grid scale values must be positive, got {s}")
    if address_q.ndim != 1:
        raise ValueError(f"address_q must be a 1D array, got shape {address_q.shape}")

    scale_table: list[dict[str, Any]] = []
    # Deterministic tie-break: same RMS → prefer smaller femur scale, then smaller
    # tibia scale. Pure `<` on RMS alone lets platform-dependent argmin pick among
    # near-ties and breaks receipt reproducibility (#14771).
    best_key: tuple[float, float, float] | None = None
    best_spec = dict(hip_spec)
    femur_scale = 1.0
    tibia_scale = 1.0
    for femur in grid:
        for tibia in grid:
            scales = {f"femur_{s}": femur for s in "rl"} | {
                f"tibia_{s}": tibia for s in "rl"
            }
            doc = scale_segments(hip_spec, scales)
            doc_bytes = json.dumps(doc, sort_keys=True).encode()
            rms = lane.pinned_rms(
                doc_bytes, {**fixed, **scaled_offsets(offsets, femur, tibia)}, address_q
            )
            scale_table.append({"femur": femur, "tibia": tibia, "pinned_rms_m": rms})
            if log:
                log.info(
                    "scale femur %.2f tibia %.2f: pinned RMS %.1f mm",
                    femur,
                    tibia,
                    rms * 1e3,
                )
            candidate = (rms, femur, tibia)
            if best_key is None or candidate < best_key:
                best_key = candidate
                best_spec = doc
                femur_scale = femur
                tibia_scale = tibia
    scaled_spec = best_spec
    return scaled_spec, femur_scale, tibia_scale, scale_table
