"""Physical and mathematical constants with citations.
DEPRECATED: Use shared.python.physics_constants instead.
This file is maintained for backward compatibility.
"""

from src.shared.python.core import physics_constants as _physics_constants

__all__ = getattr(
    _physics_constants,
    "__all__",
    [name for name in dir(_physics_constants) if not name.startswith("_")],
)

globals().update({name: getattr(_physics_constants, name) for name in __all__})
