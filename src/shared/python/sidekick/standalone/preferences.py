"""Standalone Sidekick preferences — typed getters and Design-by-Contract setters.

Preferences are stored in a ``SessionStore`` (injectable, defaults to
``FileSessionStore`` backed by ``platformdirs.user_config_dir``).  The class
exposes **typed getters** only — consumers call ``prefs.profile()``,
**not** ``prefs._store._raw["profile"]``.

Invariants (DbC)
----------------
- Profile must be one of ``VALID_PROFILES`` (validated at set time).
- Theme must be a non-empty string (format/existence validated at set time).
- Data-dir must be a string path (type validated at set time; existence is
  *not* required — the user may choose a path that doesn't exist yet).
- LLM-provider must be a non-empty string (validated at set time).

Invalid values raise ``ValueError`` or ``TypeError`` immediately, before
the value touches the store.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sidekick.standalone.session_store import FileSessionStore

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_PROFILE",
    "DEFAULT_THEME",
    "VALID_PROFILES",
    "StandalonePreferences",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PROFILES: frozenset[str] = frozenset({"chat-first", "calc-first"})
DEFAULT_PROFILE: str = "chat-first"
DEFAULT_THEME: str = "Catppuccin Mocha"
DEFAULT_LLM_PROVIDER: str = "claude"

try:
    import platformdirs

    DEFAULT_DATA_DIR: str = platformdirs.user_data_dir("sidekick")
except ImportError:  # pragma: no cover
    DEFAULT_DATA_DIR = str(Path.home() / ".local" / "share" / "sidekick")

# Store keys
_KEY_PROFILE = "profile"
_KEY_THEME = "theme"
_KEY_DATA_DIR = "data_dir"
_KEY_LLM_PROVIDER = "llm_provider"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class StandalonePreferences:
    """Typed preference surface backed by an injectable ``SessionStore``.

    Args:
        store: A ``SessionStore`` instance.  Defaults to a
               ``FileSessionStore`` in ``platformdirs.user_config_dir("sidekick")``.

    Precondition:
        ``store`` must implement ``get(key, default)`` and ``set(key, value)``.
    """

    def __init__(self, store: Any = None) -> None:
        if store is None:
            store = _default_store()
        assert hasattr(store, "get") and hasattr(store, "set"), (
            "store must implement get() and set()"
        )
        self._store = store

    # ------------------------------------------------------------------
    # Typed getters
    # ------------------------------------------------------------------

    def profile(self) -> str:
        """Return the active layout profile (``"chat-first"`` or ``"calc-first"``)."""
        return str(self._store.get(_KEY_PROFILE, DEFAULT_PROFILE))

    def theme(self) -> str:
        """Return the active theme name."""
        return str(self._store.get(_KEY_THEME, DEFAULT_THEME))

    def data_dir(self) -> str:
        """Return the Sidekick data directory path."""
        return str(self._store.get(_KEY_DATA_DIR, DEFAULT_DATA_DIR))

    def llm_provider(self) -> str:
        """Return the active LLM provider ID."""
        return str(self._store.get(_KEY_LLM_PROVIDER, DEFAULT_LLM_PROVIDER))

    # ------------------------------------------------------------------
    # Typed setters (DbC preconditions enforced here)
    # ------------------------------------------------------------------

    def set_profile(self, profile: str) -> None:
        """Set the layout profile.

        Args:
            profile: Must be one of ``VALID_PROFILES``.

        Raises:
            ValueError: if *profile* is not a valid profile name.
        """
        if profile not in VALID_PROFILES:
            raise ValueError(
                f"Invalid profile {profile!r}; valid values: {sorted(VALID_PROFILES)}"
            )
        self._store.set(_KEY_PROFILE, profile)

    def set_theme(self, theme: str) -> None:
        """Set the active theme name.

        Args:
            theme: Non-empty string theme name.

        Raises:
            ValueError: if *theme* is empty.
        """
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError(f"theme must be a non-empty string, got {theme!r}")
        self._store.set(_KEY_THEME, theme)

    def set_data_dir(self, path: str) -> None:
        """Set the Sidekick data directory.

        Args:
            path: A string path (need not exist yet).

        Raises:
            TypeError: if *path* is not a string.
        """
        if not isinstance(path, str):
            raise TypeError(f"data_dir must be a str, got {type(path).__name__}")
        self._store.set(_KEY_DATA_DIR, path)

    def set_llm_provider(self, provider_id: str) -> None:
        """Set the active LLM provider ID.

        Args:
            provider_id: Non-empty provider identifier (e.g. ``"claude"``).

        Raises:
            ValueError: if *provider_id* is empty.
        """
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError(
                f"provider_id must be a non-empty string, got {provider_id!r}"
            )
        self._store.set(_KEY_LLM_PROVIDER, provider_id)

    # ------------------------------------------------------------------
    # Theme token application
    # ------------------------------------------------------------------

    def apply_tokens(self, theme_colors: dict[str, str]) -> dict[str, str]:
        """Map raw theme colors to Sidekick design tokens.

        Uses ``theme.sidekick_tokens.COLOR_TOKEN_MAP`` — no QSS strings
        generated here, only the token dict returned to the caller.

        Args:
            theme_colors: Dict of theme-key → hex color.

        Returns:
            Dict of sidekick token name → hex color.

        Precondition:  ``theme_colors`` is a non-empty dict.
        Postcondition: every key in ``COLOR_TOKEN_MAP`` that maps to a key
                       present in ``theme_colors`` appears in the result.
        """
        assert isinstance(theme_colors, dict) and theme_colors, (
            "theme_colors must be a non-empty dict"
        )
        from theme.sidekick_tokens import COLOR_TOKEN_MAP, DEFAULT_SIDEKICK_TOKENS

        tokens: dict[str, str] = dict(DEFAULT_SIDEKICK_TOKENS)
        for token_name, theme_key in COLOR_TOKEN_MAP.items():
            if theme_key in theme_colors:
                tokens[token_name] = theme_colors[theme_key]

        assert all(isinstance(v, str) for v in tokens.values()), (
            "postcondition: all token values must be strings"
        )
        return tokens


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _default_store() -> FileSessionStore:
    """Return a FileSessionStore in the platform config directory."""
    from sidekick.standalone.session_store import FileSessionStore

    try:
        import platformdirs

        config_dir = Path(platformdirs.user_config_dir("sidekick"))
    except ImportError:  # pragma: no cover
        config_dir = Path.home() / ".config" / "sidekick"

    return FileSessionStore(config_dir / "preferences.json")
