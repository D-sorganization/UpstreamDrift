"""Embedded chat runtime and fallback widgets for Sidekick."""

from __future__ import annotations

import contextlib
import importlib
import logging
import traceback
from collections.abc import Callable
from functools import partial
from typing import Any, cast

from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QtWidgets
from .registry import WorkspaceRegistry

_logger = logging.getLogger(__name__)

SIDEKICK_CHAT_RUNTIME_OBJECT_NAME = "SidekickChatRuntimeTab"
SIDEKICK_CHAT_STATUS_OBJECT_NAME = "SidekickChatStatusTab"

_DEFAULT_CHAT_ACCENT_COLOR = "#FF8800"
_CHAT_INSTALL_HINT = (
    "Install the chat extras to enable the embedded dock: "
    "pip install 'upstream-drift-tools[chat]' "
    "(or the minimum: pip install PyQt6)."
)


def _resolve_accent_color(theme_provider: Any) -> str:
    """Return the accent color for the chat dock from ``theme_provider``.

    Falls back to :data:`_DEFAULT_CHAT_ACCENT_COLOR` when ``theme_provider``
    is ``None`` or exposes none of the supported color APIs. Never raises;
    a misshaped provider must not crash chat-tab construction.
    """
    if theme_provider is None:
        return _DEFAULT_CHAT_ACCENT_COLOR

    # Preferred path: dict-style color map via get_current_colors() (the
    # ThemeProviderProtocol used by ChatDockWidget and theme.theme_manager).
    try:
        getter = getattr(theme_provider, "get_current_colors", None)
        if callable(getter):
            colors = getter()
            if isinstance(colors, dict):
                accent = colors.get("accent")
                if isinstance(accent, str) and accent:
                    return accent
    except Exception as exc:  # noqa: BLE001 - optional theme surface
        _logger.debug("theme_provider.get_current_colors() failed: %s", exc)

    # Token-style providers occasionally expose tokens().accent or accent_color().
    try:
        tokens = getattr(theme_provider, "tokens", None)
        if callable(tokens):
            token_obj = tokens()
            accent = getattr(token_obj, "accent", None)
            if isinstance(accent, str) and accent:
                return accent
    except Exception as exc:  # noqa: BLE001 - optional theme surface
        _logger.debug("theme_provider styling getter failed: %s", exc)

    try:
        accent_attr = getattr(theme_provider, "accent_color", None)
        if callable(accent_attr):
            accent = accent_attr()
            if isinstance(accent, str) and accent:
                return accent
        elif isinstance(accent_attr, str) and accent_attr:
            return accent_attr
    except Exception as exc:  # noqa: BLE001 - optional theme surface
        _logger.debug("theme_provider.accent_color failed: %s", exc)

    return _DEFAULT_CHAT_ACCENT_COLOR


class _SidebarWorkspaceAdapter:
    """Adapt a sidebar :class:`WorkspaceRegistry` to the chat workspace bridge.

    Tools issue #2849. The chat module depends only on the
    ``WorkspaceContextProtocol`` Protocol; this adapter implements that
    contract on top of the existing sidebar registry without leaking the
    Sidekick package back into the chat module.
    """

    def __init__(self, registry: WorkspaceRegistry) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        self._registry = registry

    def describe(self) -> list[Any]:
        """Return :class:`WorkspaceVariableInfo` snapshots for all variables.

        The return type is annotated as ``list[Any]`` so this module does
        not need to import from the chat package at module-import time;
        the chat dock duck-types against the actual values returned.
        """
        _WorkspaceVariableInfo = importlib.import_module(
            "chat._workspace_protocol"
        ).WorkspaceVariableInfo

        items: list[Any] = []
        for variable in self._registry.variables():
            items.append(
                _WorkspaceVariableInfo(
                    name=variable.name,
                    dtype=variable.dtype or variable.type_name,
                    shape=tuple(variable.shape) if variable.shape else None,
                    preview=variable.preview or "",
                )
            )
        return items

    def read(self, name: str) -> Any:
        """Return the registry value for ``name``.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._registry.list_names():
            raise KeyError(name)
        return self._registry.get(name)

    def write(self, name: str, value: Any) -> None:
        """Write ``value`` into the registry under ``name``.

        Raises:
            TypeError: If ``name`` is not a ``str``.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        self._registry.set(name, value)


def _build_sidebar_plot_request_sink(sidebar: Any) -> Callable[[Any], None] | None:
    """Return a sink that routes plot requests to the Calculator Plot tab.

    The sink accepts either a dict in the
    :class:`CalculatorPlotRequest`-shaped JSON form or an already-built
    request object, and submits the resulting :class:`PlotSpec` to the
    sidebar's Calculator Plot tab widget. Returns ``None`` when any
    required sidebar attribute is missing (host without calculator
    plotting); the caller logs at DEBUG and degrades silently.
    """
    try:
        from .calculator_plotting import (
            CALCULATOR_PLOT_TAB_ID,
            CalculatorPlotRequest,
            CalculatorPlotSource,
            CalculatorPlotTabConfig,
            build_calculator_plot_spec,
        )
    except Exception as exc:  # noqa: BLE001 - optional plot dependency
        _logger.debug("Calculator plot module unavailable for chat: %s", exc)
        return None

    registry = getattr(sidebar, "registry", None)
    if registry is None:
        _logger.debug("Sidebar has no registry; chat plot sink disabled.")
        return None

    set_tab_visible = getattr(sidebar, "set_tab_visible", None)
    tab_widgets = getattr(sidebar, "_tab_widgets", None)
    if not callable(set_tab_visible) or tab_widgets is None:
        _logger.debug("Sidebar lacks tab APIs; chat plot sink disabled.")
        return None

    def _coerce_request(spec: Any) -> Any:
        if isinstance(spec, CalculatorPlotRequest):
            return spec
        if not isinstance(spec, dict):
            raise TypeError("plot spec must be a dict or CalculatorPlotRequest")
        source_val = spec.get("source")
        source: CalculatorPlotSource
        if isinstance(source_val, CalculatorPlotSource):
            source = source_val
        elif isinstance(source_val, str):
            try:
                source = CalculatorPlotSource(source_val)
            except ValueError:
                source = cast(
                    CalculatorPlotSource,
                    CalculatorPlotSource.WORKSPACE_RESULT,
                )
        else:
            source = cast(
                CalculatorPlotSource,
                CalculatorPlotSource.WORKSPACE_RESULT,
            )
        config_data = spec.get("config")
        config = (
            CalculatorPlotTabConfig(**config_data)
            if isinstance(config_data, dict)
            else CalculatorPlotTabConfig()
        )
        return CalculatorPlotRequest(
            source=source,
            x_ref=spec.get("x_ref"),
            y_ref=spec.get("y_ref"),
            expression=spec.get("expression"),
            x_min=spec.get("x_min"),
            x_max=spec.get("x_max"),
            points=spec.get("points"),
            title=spec.get("title"),
            config=config,
        )

    def _sink(spec: Any) -> None:
        request = _coerce_request(spec)
        plot_spec = build_calculator_plot_spec(request, registry)
        # Tools issue #2849: prefer a hidden-tab activation over dropping
        # the request silently. ``set_tab_visible`` is the canonical
        # sidebar API for this.
        if CALCULATOR_PLOT_TAB_ID not in tab_widgets:
            set_tab_visible(CALCULATOR_PLOT_TAB_ID, True)
        widget = tab_widgets.get(CALCULATOR_PLOT_TAB_ID)
        if widget is None:
            _logger.warning("Calculator Plot tab not available; dropping plot request.")
            return
        set_spec = getattr(widget, "set_spec", None)
        if not callable(set_spec):
            _logger.warning(
                "Calculator Plot tab does not implement set_spec; "
                "dropping plot request."
            )
            return
        set_spec(plot_spec)

    return _sink


def _build_pyqt_chat_dock(sidebar: Any) -> QtWidgets.QWidget | None:
    try:
        chat_module = importlib.import_module("chat.chat_dock_widget")
    except Exception as exc:  # noqa: BLE001 - optional chat dependency
        _logger.debug("PyQt chat dock unavailable for Sidekick: %s", exc)
        # Tools issue #2851: stash the import error so the fallback tab can
        # render a useful diagnostic and retry the import on demand.
        with contextlib.suppress(Exception):
            sidebar._chat_dock_import_error = exc
        return None

    # Tools issue #2766: chat dock no longer hard-imports theme.theme_manager.
    # Inject the manager explicitly so existing visuals are preserved when
    # the theme package is available; otherwise the dock falls back to its
    # built-in dark theme.
    theme_provider: Any = None
    try:
        theme_module = importlib.import_module("theme.theme_manager")
        theme_provider = theme_module.get_theme_manager()
    except Exception as exc:  # noqa: BLE001 - theme is optional at this layer
        _logger.debug("Theme manager unavailable for chat dock: %s", exc)

    # Tools issue #2849: wire optional workspace + plot bridges. Any
    # failure here degrades gracefully — the chat dock continues to work
    # with workspace_provider=None / plot_request_sink=None.
    workspace_provider: Any = None
    plot_request_sink: Callable[[Any], None] | None = None
    try:
        registry = getattr(sidebar, "registry", None)
        if registry is not None:
            workspace_provider = _SidebarWorkspaceAdapter(registry)
        plot_request_sink = _build_sidebar_plot_request_sink(sidebar)
    except Exception as exc:  # noqa: BLE001 - bridge is best-effort
        _logger.debug("Sidekick workspace bridge unavailable for chat: %s", exc)
        workspace_provider = None
        plot_request_sink = None

    # Tools issue #2850 / #4896: forward sidebar-level overrides for the chat
    # dock's constructor params, grouped into the cohesive config dataclasses.
    # Each value has a safe default so a bare sidebar still builds the dock
    # identically to today.
    dock = chat_module.ChatDockWidget(
        connection=chat_module.ChatConnectionConfig(
            app_context="sidekick",
            app_name="sidekick",
            session_id=getattr(sidebar, "chat_session_id", None),
            project_root=sidebar.project_root,
        ),
        presentation=chat_module.ChatPresentationConfig(
            accent_color=_resolve_accent_color(theme_provider),
            auto_index_on_open=bool(getattr(sidebar, "auto_index_on_open", False)),
            theme_provider=theme_provider,
        ),
        integrations=chat_module.ChatIntegrationHooks(
            terminal_registry=getattr(sidebar, "terminal_registry", None),
            workspace_provider=workspace_provider,
            plot_request_sink=plot_request_sink,
        ),
        parent=sidebar,
    )
    dock.setObjectName(SIDEKICK_CHAT_RUNTIME_OBJECT_NAME)
    dock.setTitleBarWidget(QtWidgets.QWidget(dock))
    _disable_dock_chrome(dock)
    # Clear any previously stashed import error: a successful build means the
    # chat module is reachable again.
    with contextlib.suppress(Exception):
        if hasattr(sidebar, "_chat_dock_import_error"):
            sidebar._chat_dock_import_error = None
    return dock


def _format_chat_import_error(exc: BaseException | None) -> str:
    """Return a human-readable explanation for a chat-dock import failure."""
    if exc is None:
        return "Chat dock module could not be loaded. Reason unknown."
    summary = traceback.format_exception_only(type(exc), exc)
    text = "".join(summary).strip()
    return text or repr(exc)


def _build_chat_status_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build a diagnostic fallback widget for the chat tab.

    Replaces the legacy single-label placeholder with a heading, the captured
    import-error traceback, an install hint, and a Retry button that re-runs
    the chat-dock import and swaps this widget out on success.
    """
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(SIDEKICK_CHAT_STATUS_OBJECT_NAME)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)

    heading = QtWidgets.QLabel("Chat unavailable", widget)
    heading.setObjectName("SidekickChatStatusHeading")
    heading_font = heading.font()
    heading_font.setBold(True)
    point_size = heading_font.pointSize()
    if point_size > 0:
        heading_font.setPointSize(point_size + 2)
    heading.setFont(heading_font)
    heading.setToolTip(
        "The embedded chat dock could not be loaded into this Sidekick session."
    )
    layout.addWidget(heading)

    error_view = QtWidgets.QPlainTextEdit(widget)
    error_view.setObjectName("SidekickChatStatusError")
    error_view.setReadOnly(True)
    monospace = _monospace_font()
    if monospace is not None:
        error_view.setFont(monospace)
    error_view.setToolTip("Captured chat-dock import error.")
    error_view.setPlainText(
        _format_chat_import_error(getattr(sidebar, "_chat_dock_import_error", None))
    )
    layout.addWidget(error_view, stretch=1)

    install_hint = QtWidgets.QLabel(_CHAT_INSTALL_HINT, widget)
    install_hint.setObjectName("SidekickChatStatusInstallHint")
    install_hint.setWordWrap(True)
    install_hint.setToolTip(
        "Suggested install command to enable the embedded chat dock."
    )
    layout.addWidget(install_hint)

    retry = QtWidgets.QPushButton("Retry", widget)
    retry.setObjectName("SidekickChatStatusRetry")
    retry.setToolTip("Re-attempt loading the embedded chat dock.")
    retry.clicked.connect(partial(_retry_chat_dock, sidebar, widget, error_view))
    layout.addWidget(retry)

    return widget


def _monospace_font() -> Any | None:
    """Return a monospace ``QFont`` when QtGui is reachable; else ``None``."""
    try:
        from .qt_compat import QtGui

        font = QtGui.QFont()
        style_hint = getattr(QtGui.QFont, "StyleHint", None)
        if style_hint is not None and hasattr(style_hint, "Monospace"):
            font.setStyleHint(style_hint.Monospace)
        font.setFamily("monospace")
        return font
    except Exception as exc:  # noqa: BLE001 - font tweak is cosmetic only
        _logger.debug("Monospace font unavailable for chat status tab: %s", exc)
        return None


def _retry_chat_dock(
    sidebar: Any,
    fallback_widget: QtWidgets.QWidget,
    error_view: QtWidgets.QPlainTextEdit,
) -> None:
    """Retry the chat-dock import; swap in the real dock on success."""
    dock = _build_pyqt_chat_dock(sidebar)
    if dock is None:
        error_view.setPlainText(
            _format_chat_import_error(getattr(sidebar, "_chat_dock_import_error", None))
        )
        return

    dock.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
    replaced = _replace_sidebar_tab_widget(sidebar, fallback_widget, dock)
    if not replaced:
        # If we cannot swap (e.g. sidebar lacks the helper), leave the
        # fallback in place but record success so users can re-open the tab.
        _logger.debug(
            "Chat dock rebuilt but sidebar tab swap failed; leaving fallback."
        )
        dock.deleteLater()


def _replace_sidebar_tab_widget(
    sidebar: Any,
    old_widget: QtWidgets.QWidget,
    new_widget: QtWidgets.QWidget,
) -> bool:
    """Swap ``old_widget`` for ``new_widget`` inside ``sidebar.tabs``."""
    replace = getattr(sidebar, "replace_tab_widget", None)
    if callable(replace):
        try:
            return bool(replace(old_widget, new_widget))
        except Exception as exc:  # noqa: BLE001 - sidebar-defined helper
            _logger.debug("sidebar.replace_tab_widget failed: %s", exc)
            return False

    tabs = getattr(sidebar, "tabs", None)
    if tabs is None:
        return False
    index = tabs.indexOf(old_widget)
    if index < 0:
        return False
    title = tabs.tabText(index)
    tooltip = tabs.tabToolTip(index)
    tabs.removeTab(index)
    tabs.insertTab(index, new_widget, title)
    if tooltip:
        tabs.setTabToolTip(index, tooltip)
    tabs.setCurrentIndex(index)
    # Keep the sidebar's stable-id -> widget map in sync when present so that
    # future remove/popout/duplicate operations target the new widget.
    tab_widgets = getattr(sidebar, "_tab_widgets", None)
    if isinstance(tab_widgets, dict):
        for tab_id, widget in list(tab_widgets.items()):
            if widget is old_widget:
                tab_widgets[tab_id] = new_widget
                break
    old_widget.setParent(None)
    old_widget.deleteLater()
    return True


def _disable_dock_chrome(dock: Any) -> None:
    feature_type = getattr(QtWidgets.QDockWidget, "DockWidgetFeature", None)
    if feature_type is not None:
        dock.setFeatures(feature_type.NoDockWidgetFeatures)
        return
    dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
