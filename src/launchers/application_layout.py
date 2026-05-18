"""Application-layout defaults — Sidekick sidebar tab ordering.

This module is the *consumer-side* manifest of which Sidekick tabs the
UpstreamDrift launcher requests on first run. It is intentionally
data-only: the actual sidebar / embed wiring is owned by other PRs
(#5613 sidebar wiring, #5630 visual hierarchy). This module simply
declares the *intent* — "these features should be available as tabs"
— and exposes :func:`default_sidebar_tab_ids` which the sidebar
factory reads when no user override is present.

When Tools introduces new Sidekick-surfaceable features, the only
launcher-side change required is appending to
:data:`DEFAULT_SIDEBAR_TAB_IDS` here.

Cross-references:

* Tools #2882 — ``os_terminal``
* Tools #2883 — ``python_repl`` and ``workspace`` (MATLAB-style)
* Tools #2884 — ``mcp_servers`` (no tab; menu-only)
* Tools #2888 — ``skills`` (skills browser)
* Tools #2889 — ``jupyter``
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_SIDEBAR_TAB_IDS",
    "MENU_ONLY_FEATURES",
    "default_sidebar_tab_ids",
    "is_menu_only",
]


# The canonical default sidebar tab order. The order here is the order
# users see on first run. Any tab whose id appears in
# :data:`MENU_ONLY_FEATURES` is deliberately *not* listed here — those
# features have a menu entry / shortcut but never claim sidebar real
# estate by default (e.g. the MCP servers settings panel which lives
# in Preferences, not as a tab).
DEFAULT_SIDEBAR_TAB_IDS: tuple[str, ...] = (
    "sidekick",  # existing AI assistant (already wired by #5613)
    "os_terminal",  # Tools #2882
    "python_repl",  # Tools #2883
    "workspace",  # Tools #2883
    "skills",  # Tools #2888
    "jupyter",  # Tools #2889
)

# Features that have a menu entry + shortcut, but are deliberately
# *not* surfaced as a default sidebar tab. (Keyed by feature_id used
# in :mod:`feature_menu`.)
MENU_ONLY_FEATURES: frozenset[str] = frozenset({"mcp_servers"})


def default_sidebar_tab_ids() -> list[str]:
    """Return a fresh list of default sidebar tab ids.

    Returns a copy each call so the caller can safely mutate the
    result without affecting subsequent calls.
    """
    return list(DEFAULT_SIDEBAR_TAB_IDS)


def is_menu_only(feature_id: str) -> bool:
    """Return ``True`` if *feature_id* is a menu-only feature."""
    if not feature_id:
        raise ValueError("feature_id must be non-empty")
    return feature_id in MENU_ONLY_FEATURES
