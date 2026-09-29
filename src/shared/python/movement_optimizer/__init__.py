# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Movement Optimizer.

A biomechanics tool for optimising exercise movement trajectories
using Lagrangian inverse dynamics in the sagittal plane.

Top-level exports:
    balance_pose: Helper that adjusts a raw pose so the COM lies inside
        the base of support.
    __version__: Installed package version (``"0.0.0.dev0"`` when running
        from a source checkout that is not pip-installed).

Most users will import the major API surfaces from sub-packages:
    ``movement_optimizer.models`` -- ``BodyModel``, ``LagrangianDynamics``.
    ``movement_optimizer.trajectory`` -- ``TrajectoryOptimizer``,
        ``OptimizationResult``.
    ``movement_optimizer.exercises`` -- per-exercise configuration
        factories (``make_clean_config``, ``make_snatch_config``, ...).
"""

from __future__ import annotations

import importlib.metadata

from .models import balance_pose as balance_pose

try:
    __version__ = importlib.metadata.version("movement-optimizer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"
