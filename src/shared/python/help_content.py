"""Backward compatibility shim - module moved to gui_pkg.help_content."""

import sys as _sys

from .gui_pkg import help_content as _real_module  # noqa: E402
from .gui_pkg.help_content import (  # noqa: F401
    FEATURE_HELP,
    HELP_TOPICS,
    QUICK_TIPS,
    UI_HELP_TOPICS,
    HelpTopic,
    get_all_topics,
    get_component_help,
    get_feature_help,
    get_help_topic,
    get_quick_tip,
    get_related_topics,
    search_help,
)

_sys.modules[__name__] = _real_module
