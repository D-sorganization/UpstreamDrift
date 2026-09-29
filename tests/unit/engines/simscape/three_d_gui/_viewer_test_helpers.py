"""Shared synthetic-model factory for C3D viewer unit tests.

The actual ``sys.path`` pivot for the engine package lives in the
sibling ``conftest.py`` (it must run at collection time before any
``import src.apps`` happens).
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


def make_synthetic_model(
    marker_names: list[str],
    n_frames: int = 100,
    point_rate: float = 100.0,
    include_events: bool = False,
    include_analog: bool = False,
):
    """Construct a deterministic ``C3DDataModel`` for tests.

    Each marker's trajectory is a unique 3-D Lissajous-ish curve so the
    artists have something interesting to render.
    """
    from src.apps.core.models import (  # type: ignore
        AnalogData,
        C3DDataModel,
        C3DEvent,
        MarkerData,
    )

    point_time = np.arange(n_frames, dtype=float) / point_rate
    markers: dict[str, MarkerData] = {}
    rng = np.random.default_rng(0)
    for i, name in enumerate(marker_names):
        phase = 2 * np.pi * (i + 1) / max(1, len(marker_names))
        pos = np.column_stack(
            [
                np.cos(point_time * 2.0 + phase)
                + rng.normal(scale=0.01, size=n_frames),
                np.sin(point_time * 2.0 + phase)
                + rng.normal(scale=0.01, size=n_frames),
                point_time * 0.1 + i * 0.05,
            ]
        )
        markers[name] = MarkerData(name=name, position=pos)

    analog: dict[str, AnalogData] = {}
    if include_analog:
        analog["Fz1"] = AnalogData(
            name="Fz1", values=np.linspace(0.0, 1.0, n_frames), unit="N"
        )

    events: list[C3DEvent] = []
    if include_events:
        events = [
            C3DEvent(label="Address", time=0.0),
            C3DEvent(label="Top", time=point_time[n_frames // 3]),
            C3DEvent(label="Impact", time=point_time[2 * n_frames // 3]),
            C3DEvent(label="Finish", time=point_time[-1]),
        ]

    return C3DDataModel(
        filepath="synthetic.c3d",
        markers=markers,
        analog=analog,
        point_rate=point_rate,
        analog_rate=point_rate if include_analog else 0.0,
        point_time=point_time,
        analog_time=point_time if include_analog else None,
        metadata={},
        events=events,
    )


# Canonical 28-marker anatomical subset used by ``default_body_segments``.
ANATOMICAL_28: tuple[str, ...] = (
    "WaistLeft",
    "WaistRight",
    "WaistLBack",
    "WaistRBack",
    "BackTop",
    "BackLeft",
    "BackRight",
    "HeadTop",
    "HeadFront",
    "HeadSide",
    "LShoulderTop",
    "LShoulderBack",
    "LUArmHigh",
    "LElbowOut",
    "LWristTop",
    "RShoulderTop",
    "RShoulderBack",
    "RUArmHigh",
    "RElbowOut",
    "RWristTop",
    "LKneeOut",
    "LAnkleOut",
    "LToeIn",
    "LToeOut",
    "RKneeOut",
    "RAnkleOut",
    "RToeIn",
    "RToeOut",
)
