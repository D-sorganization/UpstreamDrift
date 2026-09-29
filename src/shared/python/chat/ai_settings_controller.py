# ruff: noqa: E501
"""Headless AI provider / model / thinking-level controller (ADR-0022, #6119).

Extracted from ``ChatDockWidget`` (Tools issue #2871 logic) as a Qt-free
controller. It encapsulates the change-routing + validation rules for the AI
provider/model/thinking-level dropdowns. The *state* (``provider``/``model``/
``thinking_level``) and the ``QComboBox`` widgets stay on the view; the
controller drives the view through a small typed protocol so the rules can be
unit-tested with a fake view — no ``QApplication`` — which sidesteps the
Sidekick multi-widget Qt segfault.

The widget owns one instance (composition) and delegates to it. Behaviour is
byte-for-byte identical to the previous free helpers in
``chat._qt.ai_dropdowns``; those helpers now delegate here so the routing rules
live in exactly one place (DRY). Keeping state on the view preserves the
``ChatDockWidget._current_provider`` / ``_current_model`` /
``_current_thinking_level`` attributes that existing tests read and write, and
the ``switch_provider`` history-immutability invariant.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "AiSettingsController",
    "AiSettingsView",
    "VALID_FIELDS",
    "VALID_THINKING_NAMES",
]

VALID_THINKING_NAMES: frozenset[str] = frozenset({"none", "low", "medium", "high"})
VALID_FIELDS: frozenset[str] = frozenset({"provider", "model", "thinking"})


class AiSettingsView(Protocol):
    """View collaborator the controller reads state from and drives.

    The real view is ``ChatDockWidget`` (whose ``_current_*`` attributes back
    these properties); tests pass a fake recording object. The controller never
    reaches past these members (LOD).
    """

    current_provider: str
    current_model: str
    current_thinking_level: str

    def refresh_model_combo(self) -> None:
        """Repopulate the model combo for the current provider."""

    def refresh_thinking_combo(self) -> None:
        """Repopulate the thinking combo for the current adapter."""

    def sync_ai_view(self) -> None:
        """Push current state into the combos (signals blocked)."""

    def persist_ai_settings(self) -> None:
        """Persist the current selections (QSettings or host override)."""


class AiSettingsController:
    """Change-routing + validation for the AI header dropdowns.

    Stateless with respect to the selections themselves — those live on the
    ``view``. The controller only enforces the contract and orchestrates the
    view's refresh/persist hooks.
    """

    def __init__(self, view: AiSettingsView) -> None:
        if view is None:
            raise TypeError("AiSettingsController: view must be provided")
        self._view = view

    def apply_settings_change(self, field: str, value: str) -> None:
        """Single change router for the three AI header dropdowns.

        DbC (issue #2871):
            Pre: ``field`` is exactly one of ``"provider"``, ``"model"``, or
                 ``"thinking"``.
            Pre: ``value`` is a non-empty / non-whitespace string.
            Post: dependent combos are refreshed and settings are persisted.
        """
        if field not in VALID_FIELDS:
            raise ValueError(
                f"apply_settings_change: unknown field {field!r}; expected "
                f"one of {sorted(VALID_FIELDS)!r}"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"apply_settings_change: value for {field!r} must be non-empty"
            )
        value = value.strip()
        if field == "provider":
            self._view.current_provider = value
            self._view.refresh_model_combo()
            self._view.refresh_thinking_combo()
        elif field == "model":
            self._view.current_model = value
            self._view.refresh_thinking_combo()
        else:  # field == "thinking"
            self._view.current_thinking_level = value
        self._view.persist_ai_settings()

    def switch_provider(self, name: str, model: str, thinking_level: str) -> None:
        """Switch provider / model / thinking-level mid-thread.

        DbC (Tools issue #2871):
            Pre: ``name``/``model`` are non-empty strings after ``.strip()``.
            Pre: ``thinking_level`` ∈ :data:`VALID_THINKING_NAMES` after strip.
            Post: view state reflects the request and the view is synced.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("switch_provider: name must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("switch_provider: model must be non-empty")
        if not isinstance(thinking_level, str):
            raise ValueError("switch_provider: thinking_level must be a string")
        normalized_level = thinking_level.strip()
        if normalized_level not in VALID_THINKING_NAMES:
            raise ValueError(
                f"switch_provider: thinking_level {thinking_level!r} not in "
                f"{sorted(VALID_THINKING_NAMES)!r}"
            )
        self._view.current_provider = name.strip()
        self._view.current_model = model.strip()
        self._view.current_thinking_level = normalized_level
        self._view.sync_ai_view()
