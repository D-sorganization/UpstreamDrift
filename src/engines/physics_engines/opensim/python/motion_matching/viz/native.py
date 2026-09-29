"""Optional OpenSim native Visualizer wrapper.

The functions in this module are intentionally import-safe: importing the
module never imports the ``opensim`` wheel or opens a native viewer. OpenSim is
resolved only when ``render_with_opensim_visualizer`` is called.
"""

from __future__ import annotations

import importlib
import time as _time
from typing import Any

import numpy as np

from . import _adapters


class OpenSimVisualizerUnavailableError(RuntimeError):
    """Raised when the optional OpenSim native visualizer cannot be used."""


def _load_opensim_module() -> Any:
    try:
        osim = importlib.import_module("opensim")
    except ImportError as exc:
        raise OpenSimVisualizerUnavailableError(
            "opensim Python bindings are not available; install the OpenSim "
            "Python package and run with a local display to use the native "
            "Visualizer."
        ) from exc
    if not hasattr(osim, "Visualizer"):
        raise OpenSimVisualizerUnavailableError(
            "opensim Python bindings are importable but do not expose "
            "Visualizer; install a build with native visualization support."
        )
    return osim


def render_with_opensim_visualizer(
    *,
    model: Any,
    sim_out: Any,
    realtime_factor: float = 1.0,
) -> None:
    """Drive ``opensim.Visualizer`` over a sim-output state trajectory.

    Args:
        model: An ``opensim.Model`` instance.
        sim_out: Object exposing ``time`` and ``states``.
        realtime_factor: Playback speed multiplier; ``1.0`` is real-time.

    Raises:
        OpenSimVisualizerUnavailableError: If OpenSim or its native Visualizer
            is unavailable.
        ValueError: If model/sim-output inputs cannot drive playback.
    """
    _load_opensim_module()

    if model is None:
        raise ValueError(
            "render_with_opensim_visualizer requires an opensim.Model instance"
        )

    time = _adapters._as_float_array(_adapters._attr(sim_out, "time"))
    states = _adapters._attr(sim_out, "states")
    if time is None or states is None:
        raise ValueError(
            "sim_out must expose 'time' and 'states' to drive the Visualizer"
        )

    states_arr = np.asarray(states, dtype=float)
    if states_arr.ndim != 2 or states_arr.shape[0] != time.size:
        raise ValueError(
            "sim_out.states must be a 2-D array with one row per time sample"
        )

    model.setUseVisualizer(True)
    state = model.initSystem()
    visualizer = model.updVisualizer().updSimbodyVisualizer()
    visualizer.setShowSimTime(True)

    prev = float(time[0])
    for i, t_now in enumerate(time):
        state.setTime(float(t_now))
        y = state.getY()
        for j in range(states_arr.shape[1]):
            y.set(j, float(states_arr[i, j]))
        model.realizePosition(state)
        model.getVisualizer().show(state)
        dt = max(0.0, (float(t_now) - prev) / max(realtime_factor, 1e-6))
        if dt > 0:
            _time.sleep(dt)
        prev = float(t_now)
