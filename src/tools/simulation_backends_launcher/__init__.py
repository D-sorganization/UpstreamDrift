"""Simulation Backends launcher tile.

A polished PyQt6 tool for exploring the GPU-ready simulation layer
(:mod:`src.shared.python.simulation_backends`). It lets the user:

* pick a physics backend (``ode``, ``mujoco``, or ``mjwarp``) and inspect
  its capabilities;
* edit the golf double-pendulum model parameters (segment masses, wrist
  damping, swing-plane inclination, gravity on/off);
* run a passive rollout and plot the joint-angle trajectories;
* sweep the clubhead mass and plot a clubhead-speed proxy;
* cross-validate the analytical ODE backend against MuJoCo;
* export the last rollout to a versioned HDF5 trace.

Public entry point::

    python -m src.tools.simulation_backends_launcher

The PyQt6 GUI lives in :mod:`.gui`; the PyQt6-free registration shim that
the launcher bootstrap imports lives in :mod:`._embed_adapter`.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
