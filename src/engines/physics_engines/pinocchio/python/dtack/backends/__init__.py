"""Backend wrappers for physics engines."""

from __future__ import annotations

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Lazy imports — backends depend on optional heavy packages.
# Import them individually to avoid cascading ImportError.

try:
    from src.engines.physics_engines.pinocchio.python.dtack.backends.pinocchio_backend import (  # noqa: E501
        PinocchioBackend,
    )
except ImportError:
    PinocchioBackend = None  # type: ignore[assignment,misc]

try:
    from src.engines.physics_engines.pinocchio.python.dtack.backends.mujoco_backend import (  # noqa: E501
        MuJoCoBackend,
    )
except ImportError:
    MuJoCoBackend = None  # type: ignore[assignment,misc]

try:
    from src.engines.physics_engines.pinocchio.python.dtack.backends.pink_backend import (  # noqa: E501
        PINKBackend,
    )
except ImportError:
    PINKBackend = None  # type: ignore[assignment,misc]

try:
    from src.engines.physics_engines.pinocchio.python.dtack.backends.backend_factory import (  # noqa: E501
        BackendFactory,
        BackendType,
    )
except ImportError:
    BackendFactory = None  # type: ignore[assignment,misc]
    BackendType = None  # type: ignore[assignment,misc]

__all__ = [
    "BackendFactory",
    "BackendType",
    "MuJoCoBackend",
    "PINKBackend",
    "PinocchioBackend",
]
