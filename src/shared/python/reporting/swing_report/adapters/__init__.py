"""Concrete engine adapters and the cross-engine parity harness (#12328)."""

from .base import EngineResultAdapter, NativeAdapter
from .drake import DrakeAdapter
from .mujoco import MujocoAdapter
from .native import NativeSwingRun
from .parity import (
    VELOCITY_TOLERANCE,
    ParityOutcome,
    build_parity_report,
    compare_clubhead_speed,
)

__all__ = [
    "VELOCITY_TOLERANCE",
    "DrakeAdapter",
    "EngineResultAdapter",
    "MujocoAdapter",
    "NativeAdapter",
    "NativeSwingRun",
    "ParityOutcome",
    "build_parity_report",
    "compare_clubhead_speed",
]
