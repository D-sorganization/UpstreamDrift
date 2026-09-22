"""Best-effort application of the shared app theme to a tool window.

Tool GUIs must still open when the sidekick theme is unavailable (headless
CI, trimmed installs), so theming is optional and never fatal. This is the
single home for that policy, shared by the tool windows that previously
carried identical private copies (DRY gate fingerprint ``bec8b6584288``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Failures the theme may raise on a partially initialised or foreign window.
# Anything else is a programming error and is allowed to propagate.
_TOLERATED_THEME_ERRORS = (RuntimeError, AttributeError, TypeError, ValueError)


def apply_theme_best_effort(window: object) -> None:
    """Apply the app theme to ``window`` if the theme module is available.

    Args:
        window: The top-level Qt widget to theme.

    Postcondition:
        Returns normally when the theme module is missing or the theme
        rejects the window with one of ``_TOLERATED_THEME_ERRORS``.
    """
    try:
        from src.shared.python.sidekick.theme import apply_theme_to_window
    except ImportError:
        return
    try:
        apply_theme_to_window(window)
    except _TOLERATED_THEME_ERRORS as exc:
        logger.debug("Theme application skipped: %s", exc)
