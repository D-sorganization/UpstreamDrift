"""Headless chat-settings contracts for the Sidekick chat tab."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

_logger = logging.getLogger(__name__)

CHAT_TAB_ID = "chat"
CHAT_SETTINGS_SCHEMA_VERSION = 1
CHAT_PROVIDERS: tuple[str, ...] = (
    "ollama",
    "openai",
    "anthropic",
    "gemini",
    "cline",
    "codex",
)
CHAT_THINKING_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high")
CHAT_AGENT_MODES: tuple[str, ...] = ("agent", "plan", "ask")
_MIN_CONDENSE_THRESHOLD = 500
_MAX_CONDENSE_THRESHOLD = 1_000_000
_DEFAULT_CONDENSE_THRESHOLD = 8000
CHAT_TAB_SETTINGS_DEFAULTS: dict[str, Any] = {
    "provider": "ollama",
    "model": "llama3",
    "thinking_level": "none",
    "agent_mode": "agent",
    "auto_condense_threshold": _DEFAULT_CONDENSE_THRESHOLD,
}
_ALLOWED_KEYS = frozenset(CHAT_TAB_SETTINGS_DEFAULTS)


def chat_settings_defaults() -> dict[str, Any]:
    """Return a deep copy of the chat settings defaults."""
    return copy.deepcopy(CHAT_TAB_SETTINGS_DEFAULTS)


def _clamp_threshold(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_CONDENSE_THRESHOLD
    return max(_MIN_CONDENSE_THRESHOLD, min(_MAX_CONDENSE_THRESHOLD, number))


def _choose(value: Any, allowed: tuple[str, ...], default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


def coerce_chat_settings(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return validated chat settings merged over the defaults.

    Tolerant by design: unknown keys are dropped and invalid values fall
    back to their default, so stale or hand-edited persisted state never
    raises. This is the single normalization point used on both load and
    save (DRY).

    Args:
        values: Raw settings mapping (or ``None``).

    Returns:
        A complete, JSON-safe settings dict with every known key present.

    Raises:
        TypeError: If ``values`` is provided but is not a mapping.
    """
    if values is None:
        return chat_settings_defaults()
    if not isinstance(values, Mapping):
        raise TypeError("chat settings values must be a mapping or None")

    result = chat_settings_defaults()
    result["provider"] = _choose(
        values.get("provider"), CHAT_PROVIDERS, result["provider"]
    )
    model = values.get("model")
    if isinstance(model, str) and model.strip():
        result["model"] = model.strip()
    result["thinking_level"] = _choose(
        values.get("thinking_level"), CHAT_THINKING_LEVELS, result["thinking_level"]
    )
    result["agent_mode"] = _choose(
        values.get("agent_mode"), CHAT_AGENT_MODES, result["agent_mode"]
    )
    result["auto_condense_threshold"] = _clamp_threshold(
        values.get("auto_condense_threshold")
    )
    return result


def apply_chat_settings_to_dock(dock: Any, values: Mapping[str, Any]) -> bool:
    """Apply provider/model/reasoning selections to a live chat dock.

    Uses only the dock's public :meth:`switch_provider` — no reach into
    dock internals (LOD). Non-provider preferences (agent mode,
    auto-condense threshold) are persisted by the caller and honored by
    the dock on its next build, so they are intentionally not forced here.

    Args:
        dock: The live chat dock widget, or ``None``.
        values: Settings mapping to apply.

    Returns:
        ``True`` if the provider switch was applied, ``False`` otherwise.
    """
    if dock is None:
        return False
    switch = getattr(dock, "switch_provider", None)
    if not callable(switch):
        return False
    settings = coerce_chat_settings(values)
    try:
        switch(
            settings["provider"],
            settings["model"],
            settings["thinking_level"],
        )
    except Exception:  # noqa: BLE001 - live apply is best-effort
        _logger.debug("Live chat provider switch failed", exc_info=True)
        return False
    return True


def credential_status(
    manager: Any,
    providers: tuple[str, ...] = CHAT_PROVIDERS,
) -> dict[str, bool]:
    """Return ``{provider: has_key}`` for ``providers`` via ``manager``.

    Args:
        manager: A credential manager exposing ``has_credentials`` (or
            ``None`` — every provider then reports ``False``).
        providers: Providers to query.

    Returns:
        Mapping of provider name to whether a credential is configured.
    """
    if manager is None:
        return dict.fromkeys(providers, False)
    status: dict[str, bool] = {}
    for provider in providers:
        try:
            status[provider] = bool(manager.has_credentials(provider))
        except Exception:  # noqa: BLE001 - a flaky backend must not break the UI
            _logger.debug("Credential probe failed for %s", provider, exc_info=True)
            status[provider] = False
    return status


def _default_credential_manager() -> Any | None:
    """Return a :class:`CredentialManager`, or ``None`` if chat is absent."""
    try:
        from shared.python.chat.credentials import CredentialManager
    except Exception as exc:  # noqa: BLE001 - chat extras are optional
        _logger.debug("Chat credentials unavailable: %s", exc)
        return None
    return CredentialManager(service_name="sidekick_chat_ai")
