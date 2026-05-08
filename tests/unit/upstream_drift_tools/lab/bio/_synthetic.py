"""Synthetic ezc3d-shaped fixtures for unit tests.

Returns dicts shaped like ``ezc3d.c3d(path)`` so tests can exercise the
loader/parser code paths without committing real C3D files.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _synthetic_c3d_dict(
    n_frames: int = 10,
    n_markers: int = 3,
    marker_names: list[str] | None = None,
    n_analog: int = 0,
    analog_labels: list[str] | None = None,
    analog_units: list[str] | None = None,
    analog_rate: float = 1000.0,
    frame_rate: float = 100.0,
    units: str = "m",
    with_events: bool = False,
    event_labels: list[str] | None = None,
    event_times: list[float] | None = None,
    event_times_2d: bool = False,
    event_times_missing: bool = False,
    omit_analog_group: bool = False,
    analog_subframes: int = 10,
    point_data: np.ndarray | None = None,
    analog_data: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a dict shaped like the result of ``ezc3d.c3d(path)``.

    Args:
        n_frames: Number of point frames.
        n_markers: Number of markers.
        marker_names: Optional explicit marker label list.
        n_analog: Number of analog channels.
        analog_labels: Optional explicit analog label list.
        analog_units: Optional analog units list.
        analog_rate: Analog sample rate (Hz).
        frame_rate: Point frame rate (Hz).
        units: Point units string.
        with_events: Include EVENT group.
        event_labels: Event labels.
        event_times: Event times (seconds).
        event_times_2d: If True, store TIMES as a 2-row array.
        event_times_missing: If True, omit TIMES key entirely.
        omit_analog_group: If True, omit the ANALOG parameter group.
        analog_subframes: Analog subframes per point frame.
        point_data: Override point data array (4, n_markers, n_frames).
        analog_data: Override analog data array
            (analog_subframes, n_analog, n_frames).
    """
    marker_names = (
        marker_names
        if marker_names is not None
        else [f"M{i}" for i in range(n_markers)]
    )
    analog_labels = (
        analog_labels
        if analog_labels is not None
        else [f"A{i}" for i in range(n_analog)]
    )
    analog_units = (
        analog_units if analog_units is not None else [""] * len(analog_labels)
    )

    if point_data is None:
        points = np.zeros((4, n_markers, n_frames), dtype=float)
        # provide non-degenerate coordinates inside biomechanical range
        for m in range(n_markers):
            points[0, m, :] = 0.5 + 0.01 * m
            points[1, m, :] = 0.4 + 0.01 * m
            points[2, m, :] = 0.3 + 0.01 * m
            points[3, m, :] = 0.0
    else:
        points = point_data

    if analog_data is None:
        analogs = np.zeros((analog_subframes, n_analog, n_frames), dtype=float)
    else:
        analogs = analog_data

    parameters: dict[str, Any] = {
        "POINT": {
            "USED": {"value": np.array([n_markers])},
            "RATE": {"value": np.array([frame_rate])},
            "FRAMES": {"value": np.array([n_frames])},
            "LABELS": {"value": list(marker_names)},
            "UNITS": {"value": [units]},
        },
    }

    if not omit_analog_group:
        parameters["ANALOG"] = {
            "USED": {"value": np.array([n_analog])},
            "RATE": {"value": np.array([analog_rate])},
            "LABELS": {"value": list(analog_labels)},
            "UNITS": {"value": list(analog_units)},
        }

    if with_events:
        labels = event_labels if event_labels is not None else ["FootStrike", "FootOff"]
        times = event_times if event_times is not None else [0.1, 0.5]
        event_group: dict[str, Any] = {"LABELS": {"value": labels}}
        if not event_times_missing:
            if event_times_2d:
                arr = np.zeros((2, len(times)))
                arr[1, :] = times
                event_group["TIMES"] = {"value": arr}
            else:
                event_group["TIMES"] = {"value": np.asarray(times)}
        parameters["EVENT"] = event_group

    return {
        "header": {},
        "parameters": parameters,
        "data": {"points": points, "analogs": analogs},
    }
