"""Backward compatibility shim - module moved to gui_pkg.help_system."""

import sys as _sys

from .gui_pkg import help_system as _real_module  # noqa: E402
from .gui_pkg.help_system import (  # noqa: F401
    DOCS_DIR,
    HELP_DIR,
    REPOS_ROOT,
    USER_MANUAL_PATH,
    HelpButton,
    HelpDialog,
    TooltipManager,
    add_help_button_to_widget,
    create_help_menu_actions,
    get_help_topic_content,
    get_user_manual_content,
    list_help_topics,
)

_sys.modules[__name__] = _real_module
