"""Backward compatibility shim - module moved to gui_pkg.image_utils."""

import sys as _sys

from .gui_pkg import image_utils as _real_module  # noqa: E402
from .gui_pkg.image_utils import (  # noqa: F401
    analyze_image_quality,
    auto_crop_to_content,
    create_optimized_icon,
    enhance_icon_source,
    ensure_pillow,
    logger,
    save_ico,
    save_png_icons,
)

_sys.modules[__name__] = _real_module
