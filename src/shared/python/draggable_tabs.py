"""Backward compatibility shim - module moved to gui_pkg.draggable_tabs."""

import sys as _sys

from .gui_pkg import draggable_tabs as _real_module  # noqa: E402
from .gui_pkg.draggable_tabs import (  # noqa: F401
    DetachedTabWindow,
    DraggableTabWidget,
    logger,
)

_sys.modules[__name__] = _real_module
