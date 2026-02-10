"""Backward compatibility shim - module moved to gui_pkg.gui_utils."""

import sys as _sys

from .gui_pkg import gui_utils as _real_module  # noqa: E402
from .gui_pkg.gui_utils import (  # noqa: F401
    BaseApplicationWindow,
    LayoutBuilder,
    apply_stylesheet,
    create_button,
    create_dialog,
    create_label,
    get_default_icon,
    get_qapp,
    logger,
    setup_window_geometry,
)

_sys.modules[__name__] = _real_module
