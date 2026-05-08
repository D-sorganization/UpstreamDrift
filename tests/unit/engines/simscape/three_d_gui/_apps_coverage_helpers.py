"""Shared helpers for the issue #4675 apps/ coverage tests."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


def make_model(
    n_frames: int = 60,
    point_rate: float = 100.0,
    include_analog: bool = True,
    include_force_plate: bool = True,
    include_events: bool = True,
    include_raw_params: bool = True,
):
    """Build a synthetic C3DDataModel with optional force-plate channels."""
    from src.apps.core.models import (  # type: ignore
        AnalogData,
        C3DDataModel,
        C3DEvent,
        MarkerData,
    )

    rng = np.random.default_rng(7)
    names = ["WaistLeft", "WaistRight", "LKneeOut", "RKneeOut"]
    markers = {}
    for i, name in enumerate(names):
        pos = rng.normal(size=(n_frames, 3)) + i
        residuals = rng.uniform(0.0, 0.01, size=n_frames)
        markers[name] = MarkerData(name=name, position=pos, residuals=residuals)

    analog = {}
    if include_analog:
        analog["EMG1"] = AnalogData(
            name="EMG1", values=np.linspace(-1.0, 1.0, n_frames), unit="V"
        )
    if include_force_plate:
        for plate in (1, 2):
            for axis, base in zip("xyz", (10.0, 20.0, 100.0), strict=True):
                analog[f"F{axis}{plate}"] = AnalogData(
                    name=f"F{axis}{plate}",
                    values=base + np.sin(np.arange(n_frames) * 0.1),
                    unit="N",
                )
                analog[f"M{axis}{plate}"] = AnalogData(
                    name=f"M{axis}{plate}",
                    values=np.cos(np.arange(n_frames) * 0.1),
                    unit="N.m",
                )

    events = []
    if include_events:
        events = [
            C3DEvent(label="Address", time=0.0),
            C3DEvent(label="Impact", time=0.3),
        ]

    raw_params = None
    if include_raw_params:
        raw_params = {
            "POINT": {
                "UNITS": {"value": ["m"]},
                "X_SCREEN": {"value": ["+X"]},
                "Y_SCREEN": {"value": ["+Z"]},
                "RATE": {"value": [point_rate]},
                "LABELS": {"value": names},
                "__internal__": {"value": "skip"},
            },
            "ANALOG": {"USED": {"value": [len(analog)]}},
            "FORCE_PLATFORM": {"USED": {"value": [2 if include_force_plate else 0]}},
            "MANUFACTURER": {
                "SOFTWARE": {"value": ["Vicon Nexus"]},
                "VERSION": {"value": ["2.0"]},
            },
            "TRIAL": {
                "CAPTURE_ID": {"value": ["abc-123"]},
                "EXPORTED_AT": {"value": ["2026-01-01"]},
                "PLAYER_ID": {"value": ["P0001"]},
            },
            "EXTRA": {"NUMERIC": {"value": np.array([[1.0, 2.0], [3.0, 4.0]])}},
        }

    return C3DDataModel(
        filepath="/tmp/synthetic.c3d",
        markers=markers,
        analog=analog,
        point_rate=point_rate,
        analog_rate=point_rate if analog else 0.0,
        point_time=np.arange(n_frames, dtype=float) / point_rate,
        analog_time=(np.arange(n_frames, dtype=float) / point_rate if analog else None),
        metadata={"File": "synthetic.c3d", "Units (POINT)": "m"},
        events=events,
        raw_parameters=raw_params,
    )
