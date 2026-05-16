"""Embedded Sidekick runtime widgets for shared utility tabs."""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import types
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from . import design_tokens as theme
from .calculator_assist import (
    calculator_predictive_text_enabled,
    calculator_startup_config,
    set_calculator_predictive_text_enabled,
)
from .calculator_runtime import (
    SidekickCalculatorWidget,
)
from .calculator_startup import (
    apply_calculator_startup_imports,
    default_calculator_startup_config,
)
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QT_API, QtCore, QtWidgets
from .registry import WorkspaceRegistry

logger = logging.getLogger(__name__)

SIDEKICK_CHAT_RUNTIME_OBJECT_NAME = "SidekickChatRuntimeTab"
SIDEKICK_TERMINAL_OBJECT_NAME = "SidekickTerminalTab"
SIDEKICK_NOTES_OBJECT_NAME = "SidekickNotesTab"

_RESERVED_NAMESPACE_NAMES = {
    "__builtins__",
    "np",
    "numpy",
    "pd",
    "pandas",
    "scipy",
}

SetVariable = Callable[[str, Any], None]


def build_chat_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the embedded chat tab for a Sidekick sidebar."""
    if QT_API == "PyQt6":
        widget = _build_pyqt_chat_dock(sidebar)
        if widget is not None:
            widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
            return widget
    return _build_chat_status_tab(sidebar)


def build_terminal_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build an embedded Python terminal tab bound to the workspace registry."""
    widget = SidekickTerminalWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        terminal_theme=theme.SidekickTerminalTheme.inherited(
            getattr(sidebar, "_design_tokens", None),
        ),
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["terminal"]["summary"])
    return widget


def build_calculator_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build an embedded symbolic calculator tab bound to workspace state."""
    widget = SidekickCalculatorWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        predictive_text_enabled=calculator_predictive_text_enabled(sidebar),
        startup_import_config=calculator_startup_config(sidebar),
        set_predictive_text_enabled=partial(
            set_calculator_predictive_text_enabled,
            sidebar,
        ),
        refresh_workspace=sidebar.refresh_workspace,
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["calculator"]["summary"])
    return widget


def build_notes_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build a project-persistent notes tab."""
    widget = SidekickNotesWidget(project_root=sidebar.project_root, parent=sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["notes"]["summary"])
    return widget


class SidekickTerminalWidget(QtWidgets.QWidget):
    """Small Python execution surface sharing values with Workspace."""

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        set_variable: SetVariable,
        terminal_theme: theme.SidekickTerminalTheme | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        if set_variable is None:
            raise ValueError("set_variable must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_TERMINAL_OBJECT_NAME)
        self._registry = registry
        self._set_variable = set_variable
        self._terminal_theme = terminal_theme or theme.SidekickTerminalTheme.inherited()
        self._namespace: dict[str, Any] = {}
        self._load_workspace_namespace()
        _preload_scientific_namespace(self._namespace)
        self._build_ui()
        self.apply_terminal_theme(self._terminal_theme)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._input = QtWidgets.QPlainTextEdit(self)
        self._input.setObjectName("SidekickTerminalInput")
        self._input.setPlaceholderText("result = np.array([1, 2, 3]).sum()")
        self._input.setToolTip(
            "Enter Python code that can read and write shared workspace variables."
        )
        layout.addWidget(self._input, stretch=2)

        self._run_button = QtWidgets.QPushButton("Run", self)
        self._run_button.setObjectName("SidekickTerminalRun")
        self._run_button.setToolTip(
            "Execute the current terminal script and export assigned variables."
        )
        self._run_button.clicked.connect(self.execute_script)
        layout.addWidget(self._run_button)

        self._output = QtWidgets.QPlainTextEdit(self)
        self._output.setObjectName("SidekickTerminalOutput")
        self._output.setReadOnly(True)
        self._output.setToolTip("Shows terminal stdout, stderr, and execution errors.")
        layout.addWidget(self._output, stretch=3)

    def execute_script(self) -> None:
        """Execute the current script and export user variables."""
        script = self._input.toPlainText()
        if not script.strip():
            self._append_output("No code to run.")
            return
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            compiled = compile(script, "<sidekick-terminal>", "exec")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compiled, self._namespace, self._namespace)  # noqa: S102  # nosec B102
        except Exception as exc:  # noqa: BLE001 - terminal reports user code errors
            logger.debug("Sidekick terminal execution failed: %s", exc)
            self._append_output(_format_terminal_output(stdout, stderr, exc))
            return

        self._sync_namespace_to_registry()
        self._append_output(_format_terminal_output(stdout, stderr, None))

    def _load_workspace_namespace(self) -> None:
        for name in self._registry.list_names():
            self._namespace[name] = self._registry.get(name)

    def _sync_namespace_to_registry(self) -> None:
        for name, value in _exportable_values(self._namespace).items():
            self._set_variable(name, value)

    def _append_output(self, text: str) -> None:
        existing = self._output.toPlainText().strip()
        combined = f"{existing}\n{text}" if existing else text
        self._output.setPlainText(combined.strip())

    def apply_terminal_theme(self, terminal_theme: theme.SidekickTerminalTheme) -> None:
        """Apply terminal-scoped colors without changing global Sidekick QSS."""
        if terminal_theme is None:
            raise ValueError("terminal_theme must be provided")
        self._terminal_theme = terminal_theme
        self.setStyleSheet(terminal_theme.qss(SIDEKICK_TERMINAL_OBJECT_NAME))


class SidekickNotesWidget(QtWidgets.QWidget):
    """Project note-card editor with explicit save and debounced persistence."""

    def __init__(
        self,
        *,
        project_root: Path,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if project_root is None:
            raise ValueError("project_root must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_NOTES_OBJECT_NAME)
        self._store = _note_card_store(project_root)
        self._active_card_id: str | None = None
        self._autosave = QtCore.QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(500)
        self._autosave.timeout.connect(self.save_notes)
        self._build_ui()
        self._load_first_card()
        self._apply_board_style()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status = QtWidgets.QLabel("Ready", self)
        self._status.setObjectName("SidekickNotesStatus")
        self._status.setToolTip("Reports the latest notes persistence status.")
        layout.addWidget(self._status)

        self._card_frame = QtWidgets.QFrame(self)
        self._card_frame.setObjectName("SidekickNotesCard")
        card_layout = QtWidgets.QVBoxLayout(self._card_frame)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(8)

        self._color = QtWidgets.QLineEdit(self._card_frame)
        self._color.setObjectName("SidekickNotesCardColor")
        self._color.setPlaceholderText("#fff7cc")
        self._color.setToolTip("Sets the active note card color as a #RRGGBB value.")
        card_layout.addWidget(self._color)

        self._editor = QtWidgets.QPlainTextEdit(self)
        self._editor.setObjectName("SidekickNotesEditor")
        self._editor.setPlaceholderText("Project notes")
        self._editor.setToolTip("Edit the active project-scoped markdown note card.")
        self._editor.textChanged.connect(self._schedule_autosave)
        card_layout.addWidget(self._editor, stretch=1)
        layout.addWidget(self._card_frame, stretch=1)

        self._board_color = QtWidgets.QLineEdit(self)
        self._board_color.setObjectName("SidekickNotesBoardColor")
        self._board_color.setPlaceholderText("#f7f7f7")
        self._board_color.setToolTip("Sets the notes screen background color.")
        layout.addWidget(self._board_color)

        row = QtWidgets.QHBoxLayout()
        self._save = QtWidgets.QPushButton("Save", self)
        self._save.setObjectName("SidekickNotesSave")
        self._save.setToolTip("Persist the current notes text immediately.")
        self._save.clicked.connect(self.save_notes)
        row.addWidget(self._save)

        clear = QtWidgets.QPushButton("Clear", self)
        clear.setObjectName("SidekickNotesClear")
        clear.setToolTip(
            "Clear the current note text while keeping the notes file available."
        )
        clear.clicked.connect(self.clear_notes)
        row.addWidget(clear)

        restore = QtWidgets.QPushButton("Restore", self)
        restore.setObjectName("SidekickNotesRestore")
        restore.setToolTip(
            "Restore the latest recycled notes snapshot when one exists."
        )
        restore.clicked.connect(self.restore_latest)
        row.addWidget(restore)

        apply_colors = QtWidgets.QPushButton("Apply Colors", self)
        apply_colors.setObjectName("SidekickNotesApplyColors")
        apply_colors.setToolTip("Validate and persist note and screen colors.")
        apply_colors.clicked.connect(self.apply_colors)
        row.addWidget(apply_colors)
        layout.addLayout(row)

    def save_notes(self) -> None:
        """Persist the current notes text to the active markdown card."""
        color = self._color.text().strip() or "#fff7cc"
        if self._active_card_id is None:
            card = self._store.create_note(
                "Project Notes",
                self._editor.toPlainText(),
                color=color,
            )
            self._active_card_id = card.note_id
        else:
            self._store.update_note(
                self._active_card_id,
                title="Project Notes",
                markdown_body=self._editor.toPlainText(),
                color=color,
            )
        self.apply_colors(save_note=False)
        self._status.setText("Saved")

    def clear_notes(self) -> None:
        """Clear notes while preserving the active markdown card."""
        self._editor.setPlainText("")
        self.save_notes()
        self._status.setText("Cleared")

    def restore_latest(self) -> None:
        """Restore the latest recycled note file when available."""
        item_id = self._store.latest_recycled_id()
        restored = None if item_id is None else self._store.restore_note(item_id)
        if restored is None:
            self._status.setText("Nothing to restore")
            return
        self._active_card_id = restored.note_id
        self._editor.setPlainText(restored.markdown_body)
        self._color.setText(restored.color)
        self._apply_card_style(restored.color)
        self._status.setText("Restored")

    def apply_colors(self, *, save_note: bool = True) -> None:
        """Validate and persist note-card and board colors."""
        from notes.models import NotesBoardSettings, normalize_color

        note_color = normalize_color(self._color.text().strip() or "#fff7cc")
        board = NotesBoardSettings(
            background_color=self._board_color.text().strip() or "#f7f7f7"
        )
        self._color.setText(note_color)
        self._board_color.setText(board.background_color)
        self._store.save_settings(board)
        self._apply_card_style(note_color)
        self._apply_board_style()
        if save_note:
            self.save_notes()

    def _load_first_card(self) -> None:
        card = self._store.migrate_legacy_text_note()
        if card is None:
            notes = self._store.list_notes()
            card = notes[0] if notes else None
        if card is not None:
            self._active_card_id = card.note_id
            self._editor.setPlainText(card.markdown_body)
            self._color.setText(card.color)
            self._apply_card_style(card.color)
        else:
            self._color.setText("#fff7cc")
        self._board_color.setText(self._store.load_settings().background_color)

    def _apply_board_style(self) -> None:
        color = self._store.load_settings().background_color
        self.setStyleSheet(f"#{SIDEKICK_NOTES_OBJECT_NAME} {{ background: {color}; }}")

    def _apply_card_style(self, color: str) -> None:
        self._card_frame.setStyleSheet(
            "#SidekickNotesCard { "
            f"background: {color}; border: 1px solid #d0d0d0; border-radius: 6px;"
            " }"
        )

    def _schedule_autosave(self) -> None:
        self._autosave.start()


def _build_pyqt_chat_dock(sidebar: Any) -> QtWidgets.QWidget | None:
    try:
        chat_module = importlib.import_module("chat.chat_dock_widget")
    except Exception as exc:  # noqa: BLE001 - optional chat dependency
        logger.debug("PyQt chat dock unavailable for Sidekick: %s", exc)
        return None

    # Wire application state and diagnostic history into the chat agent context
    try:
        from src.shared.python.ai.chat_context import record_event

        # Record diagnostic history
        try:
            from src.launchers.launcher_diagnostics import LauncherDiagnostics

            diag = LauncherDiagnostics()
            record_event("diagnostic", {"diagnostic_history": diag.run_all_checks()})
        except ImportError:
            pass

        # Record application state via the registry
        if hasattr(sidebar, "registry") and hasattr(sidebar.registry, "export_all"):
            record_event(
                "app_state", {"registry_export": sidebar.registry.export_all()}
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to wire Sidekick chat context: %s", exc)

    # Sidekick: unify React and PyQt chat surfaces and verify cross-shell parity
    # Inject an adapter that implements ThemeProviderProtocol over the Sidekick tokens
    tokens = getattr(sidebar, "_design_tokens", None)
    theme_provider: Any = None
    if tokens is not None:

        class _SidekickThemeAdapter:
            def __init__(self, t: Any) -> None:
                self.t = t

            def get_current_colors(self) -> dict[str, str]:
                return {
                    "bg": self.t.color.canvas,
                    "group_bg": self.t.color.surface,
                    "input_bg": self.t.color.input,
                    "text": self.t.color.text,
                    "text_secondary": self.t.color.text_subtle,
                    "border": self.t.color.border,
                    "button_hover": self.t.color.accent_hover,
                    "accent": self.t.color.accent,
                }

        theme_provider = _SidekickThemeAdapter(tokens)
    else:
        try:
            theme_module = importlib.import_module("theme.theme_manager")
            theme_provider = theme_module.get_theme_manager()
        except Exception as exc:  # noqa: BLE001 - theme is optional at this layer
            logger.debug("Theme manager unavailable for chat dock: %s", exc)

    dock = chat_module.ChatDockWidget(
        app_context="sidekick",
        app_name="sidekick",
        project_root=sidebar.project_root,
        theme_provider=theme_provider,
        parent=sidebar,
    )
    dock.setObjectName(SIDEKICK_CHAT_RUNTIME_OBJECT_NAME)
    dock.setTitleBarWidget(QtWidgets.QWidget(dock))
    _disable_dock_chrome(dock)
    return dock


def _build_chat_status_tab(sidebar: Any) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(SIDEKICK_CHAT_RUNTIME_OBJECT_NAME)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    label = QtWidgets.QLabel(
        "Shared chat is available when the PyQt chat dock is loaded.",
        widget,
    )
    label.setWordWrap(True)
    layout.addWidget(label)
    layout.addStretch(1)
    return widget


def _disable_dock_chrome(dock: Any) -> None:
    feature_type = getattr(QtWidgets.QDockWidget, "DockWidgetFeature", None)
    if feature_type is not None:
        dock.setFeatures(feature_type.NoDockWidgetFeatures)
        return
    dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)


def _note_card_store(project_root: Path) -> Any:
    from notes.card_store import NoteCardStore

    return NoteCardStore(project_dir=project_root)


def _preload_scientific_namespace(namespace: dict[str, Any]) -> None:
    apply_calculator_startup_imports(
        namespace,
        default_calculator_startup_config(),
    )


def _exportable_values(namespace: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in namespace.items()
        if _is_exportable_name(name) and _is_exportable_value(value)
    }


def _is_exportable_name(name: str) -> bool:
    return (
        bool(name)
        and not name.startswith("_")
        and name not in _RESERVED_NAMESPACE_NAMES
    )


def _is_exportable_value(value: Any) -> bool:
    return not isinstance(value, types.ModuleType) and not callable(value)


def _format_terminal_output(
    stdout: io.StringIO,
    stderr: io.StringIO,
    exc: Exception | None,
) -> str:
    parts = [text for text in (stdout.getvalue(), stderr.getvalue()) if text]
    if exc is not None:
        parts.append(f"{type(exc).__name__}: {exc}")
    if not parts:
        return "Executed."
    return "".join(parts).strip()
