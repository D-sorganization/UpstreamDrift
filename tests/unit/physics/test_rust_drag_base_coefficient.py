"""The Rust ball-flight kernel must receive the Reynolds-curve base Cd.

``upstream_physics`` scales its dimpled-sphere drag-crisis curve by
``AeroBallProperties.drag_coefficient``, the same way the enhanced Python
engine scales ``cd_dimpled_sphere`` by ``GOLF_BALL_DRAG_COEFFICIENT`` (0.25).
``BallProperties.cd0`` (0.21) is the constant term of a *different* model,
the spin polynomial ``cd0 + cd1*S + cd2*S**2``. Passing it across the
boundary under-predicted drag, so the production simulator carried a
TrackMan 7-iron 194 yd instead of ~179 yd.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import numpy as np
import pytest
from src.shared.python.core.physics_constants import GOLF_BALL_DRAG_COEFFICIENT
from src.shared.python.physics import rust_kernel
from src.shared.python.physics.ball_flight_physics import (
    BallFlightSimulator,
    LaunchConditions,
)

pytestmark = pytest.mark.unit


class _Recorder:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _fake_kernel(captured: dict[str, Any]) -> types.ModuleType:
    module = types.ModuleType("upstream_physics")

    def ball_props(**kwargs: Any) -> _Recorder:
        captured["ball"] = kwargs
        return _Recorder(**kwargs)

    def simulate(*_args: Any) -> Any:
        return types.SimpleNamespace(get_points=list)

    module.IntegratorConfig = _Recorder  # type: ignore[attr-defined]
    module.AeroBallProperties = ball_props  # type: ignore[attr-defined]
    module.AirProperties = _Recorder  # type: ignore[attr-defined]
    module.simulate_ball_trajectory_py = simulate  # type: ignore[attr-defined]
    return module


def test_rust_kernel_receives_the_reynolds_curve_base_drag_coefficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "upstream_physics", _fake_kernel(captured))
    monkeypatch.setattr(rust_kernel, "is_rust_available", lambda: True)

    BallFlightSimulator().simulate_trajectory(
        LaunchConditions(
            velocity=53.6,
            launch_angle=math.radians(16.3),
            spin_rate=7097.0,
            spin_axis=np.array([0.0, -1.0, 0.0]),
        ),
        max_time=1.0,
        dt=0.01,
    )

    assert captured["ball"]["drag_coefficient"] == pytest.approx(
        float(GOLF_BALL_DRAG_COEFFICIENT)
    )
