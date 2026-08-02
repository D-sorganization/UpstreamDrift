# ruff: noqa: E501
"""AI provider / model / thinking dropdown helpers (Tools issue #2871).

Extracted from ``_chat_dock_widget_qt`` so the parent module fits the
1500-line budget. The :class:`ChatDockWidget` retains the public methods
(``switch_provider``, ``_build_ai_dropdowns``, etc.) but their bodies
delegate to the free functions defined here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QSizePolicy

from ..ai_settings_controller import (
    VALID_FIELDS,
    VALID_THINKING_NAMES,
    AiSettingsController,
)
from ..cli_provider_availability import list_available_cli_providers

logger = logging.getLogger(__name__)

# Re-exported from the controller so the validation vocabulary lives in one
# place (DRY); ``ChatDockWidget`` still introspects these (Tools issue #2871).
__all__ = ["DEFAULT_PROVIDERS", "VALID_FIELDS", "VALID_THINKING_NAMES"]
DEFAULT_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("Ollama", "ollama"),
    ("OpenAI", "openai"),
    ("Anthropic", "anthropic"),
    ("Gemini", "gemini"),
    ("Cline", "cline"),
)


def build_header_combobox(
    *,
    label: str,
    items: list[tuple[str, str]],
) -> QComboBox:
    """Build a header combo box used by the AI Provider/Model/Thinking row.

    DRY helper used for all three header dropdowns (issue #2871).

    Args:
        label: Short label (e.g. ``"provider"``) used for tool-tip; must
            be non-empty.
        items: Sequence of ``(display_text, user_data)`` pairs;
            must be non-empty.

    Raises:
        ValueError: If ``label`` is empty/whitespace or ``items`` is empty.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError("build_header_combobox: label must be non-empty")
    if not items:
        raise ValueError("build_header_combobox: items must be non-empty")
    combo = QComboBox()
    combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    combo.setMinimumWidth(0)
    for display, data in items:
        combo.addItem(display, data)
    combo.setToolTip(f"Select AI {label}")
    return combo


def build_available_cli_provider_items() -> list[tuple[str, str]]:
    """Return ``(display_name, provider_id)`` pairs for installed CLI agents.

    Probes the local ``PATH`` so the dropdown never shows unavailable
    entries.
    """
    return [
        (entry.display_name, entry.provider_id)
        for entry in list_available_cli_providers()
    ]


def _required_combo(view: Any, field: str) -> QComboBox:
    combo = getattr(view, field, None)
    if not isinstance(combo, QComboBox):
        raise RuntimeError(f"ChatDockView.{field} must be initialized")
    return combo


def build_ai_dropdowns(
    view: Any,
    mode_row: QHBoxLayout,
    *,
    on_combo_changed: Callable[[str], None],
    refresh_model_combo: Callable[[], None],
    refresh_thinking_combo: Callable[[], None],
    sync_dropdowns: Callable[[], None],
) -> None:
    """Construct + wire the three AI header dropdowns on ``view``.

    Side-effect only: populates ``ChatDockView.ai_*_combo`` fields and inserts
    them into ``mode_row``. Legacy ``dock._ai_*_combo`` aliases are generated
    by the view mirror in ``ui_builder.py``.
    """
    api_items = list(DEFAULT_PROVIDERS)
    cli_items = build_available_cli_provider_items()
    all_provider_items = api_items + cli_items
    view.ai_provider_combo = build_header_combobox(
        label="provider", items=all_provider_items
    )
    mode_row.addWidget(view.ai_provider_combo)

    view.ai_model_combo = build_header_combobox(
        label="model", items=[("(default)", "default")]
    )
    mode_row.addWidget(view.ai_model_combo)

    view.ai_thinking_combo = build_header_combobox(
        label="thinking", items=[("Off", "none")]
    )
    mode_row.addWidget(view.ai_thinking_combo)

    view.ai_provider_combo.currentIndexChanged.connect(
        lambda _: on_combo_changed("provider")
    )
    view.ai_model_combo.currentIndexChanged.connect(lambda _: on_combo_changed("model"))
    view.ai_thinking_combo.currentIndexChanged.connect(
        lambda _: on_combo_changed("thinking")
    )
    refresh_model_combo()
    refresh_thinking_combo()
    sync_dropdowns()


def refresh_ai_model_combo(view: Any, adapter: Any | None) -> None:
    """Repopulate the model combo for the currently selected provider."""
    try:
        models = adapter.list_models() if adapter is not None else []
    except Exception:  # noqa: BLE001 - any adapter failure → empty list
        logger.debug("refresh_ai_model_combo: adapter probe failed", exc_info=True)
        models = []
    items = []
    for m in models:
        display = str(getattr(m, "display_name", None) or getattr(m, "name", str(m)))
        data = str(
            getattr(m, "id", None)
            or getattr(m, "model_id", None)
            or getattr(m, "name", str(m))
        )
        items.append((display, data))
    if not items:
        items = [("(default)", "default")]
    combo = _required_combo(view, "ai_model_combo")
    combo.blockSignals(True)
    try:
        combo.clear()
        for display, data in items:
            combo.addItem(display, data)
    finally:
        combo.blockSignals(False)


def refresh_ai_thinking_combo(view: Any, adapter: Any | None) -> None:
    """Repopulate the thinking combo for the currently selected adapter."""
    try:
        caps = adapter.thinking_capabilities() if adapter is not None else None
    except Exception:  # noqa: BLE001
        logger.debug("refresh_ai_thinking_combo: adapter probe failed", exc_info=True)
        caps = None
    if caps is None:
        items = [("Off", "none")]
    else:
        items = [
            (
                getattr(level, "label", str(level)),
                getattr(level, "name", str(level)),
            )
            for level in getattr(caps, "available_levels", getattr(caps, "levels", []))
        ]
    combo = _required_combo(view, "ai_thinking_combo")
    combo.blockSignals(True)
    try:
        combo.clear()
        for display, data in items:
            combo.addItem(display, data)
    finally:
        combo.blockSignals(False)


def sync_ai_dropdowns(
    view: Any,
    *,
    current_provider: str,
    current_model: str,
    current_thinking_level: str,
) -> None:
    """Push current state into the three combos with signals blocked."""
    for combo, value in (
        (_required_combo(view, "ai_provider_combo"), current_provider),
        (_required_combo(view, "ai_model_combo"), current_model),
        (_required_combo(view, "ai_thinking_combo"), current_thinking_level),
    ):
        combo.blockSignals(True)
        try:
            idx = combo.findData(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)


def get_active_ai_adapter(provider_name: str) -> Any | None:
    """Return the adapter for ``provider_name`` or ``None``.

    Adapter construction failures are non-fatal (offline mode, missing
    API key, etc.) — callers fall back to a static catalogue.
    """
    import importlib

    try:
        module = importlib.import_module("src.shared.python.ai.adapters.factory")
        return module.AdapterFactory.create(provider_name)
    except Exception:  # noqa: BLE001 - missing credentials are normal
        return None


def _controller_for(dock: Any) -> AiSettingsController:
    """Return the dock's controller, or a transient one bound to the dock view."""
    existing = getattr(dock, "_ai_settings", None)
    if isinstance(existing, AiSettingsController):
        return existing
    return AiSettingsController(dock)


def apply_settings_change(dock: Any, field: str, value: str) -> None:
    """Single change router for the three AI header dropdowns.

    Backwards-compatible shim — delegates to
    :class:`chat.ai_settings_controller.AiSettingsController` (issue #6119).
    """
    _controller_for(dock).apply_settings_change(field, value)


def switch_provider(
    dock: Any,
    name: str,
    model: str,
    thinking_level: str,
) -> None:
    """Switch AI provider / model / thinking-level mid-thread.

    Backwards-compatible shim — delegates to the headless controller, then
    re-asserts the ``_message_history`` immutability invariant (Tools #2871),
    which the controller cannot violate since it never touches history.
    """
    history_before = dock._message_history
    snapshot_before = list(history_before)
    _controller_for(dock).switch_provider(name, model, thinking_level)
    assert dock._message_history is history_before, (
        "switch_provider invariant: _message_history must remain the same list"
    )
    assert dock._message_history == snapshot_before, (
        "switch_provider invariant: _message_history contents must not change"
    )
