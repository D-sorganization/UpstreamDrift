"""Backward compatibility shim - module moved to engine_core.engine_probes."""

import sys as _sys

from .engine_core import engine_probes as _real_module  # noqa: E402
from .engine_core.engine_probes import (  # noqa: F401
    DrakeProbe,
    EngineProbe,
    EngineProbeResult,
    MatlabProbe,
    MuJoCoProbe,
    MyoSimProbe,
    OpenPoseProbe,
    OpenSimProbe,
    PendulumProbe,
    PinocchioProbe,
    ProbeStatus,
)

_sys.modules[__name__] = _real_module
