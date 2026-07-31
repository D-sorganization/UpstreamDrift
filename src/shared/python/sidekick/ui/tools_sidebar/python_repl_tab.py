"""Python REPL widgets for Sidekick runtime tabs."""

from __future__ import annotations

import contextlib
import io
import logging
import time
import types
from collections.abc import Callable
from typing import Any

from . import design_tokens as theme
from .appearance import (
    DEFAULT_DARK_PANEL_APPEARANCE,
    PanelAppearance,
    is_panel_appearance,
    panel_qss,
)
from .calculator_startup import (
    CalculatorStartupConfig,
    CalculatorStartupResult,
    apply_calculator_startup_imports,
    default_calculator_startup_config,
    default_repl_startup_config,
)
from .qt_compat import QtCore, QtWidgets, Signal
from .registry import WorkspaceRegistry, is_workspace_registry

_logger = logging.getLogger(__name__)

_IMMEDIATE_COMPLETION_DRAIN_SECONDS = 0.05

SIDEKICK_TERMINAL_OBJECT_NAME = "SidekickTerminalTab"
SIDEKICK_PYTHON_REPL_OBJECT_NAME = SIDEKICK_TERMINAL_OBJECT_NAME

_RESERVED_NAMESPACE_NAMES = {
    "__builtins__",
    "np",
    "numpy",
    "pd",
    "pandas",
    "scipy",
}

SetVariable = Callable[[str, Any], None]


class _ReplWorker(QtCore.QThread):
    """Execute a Python script off the GUI thread (F6)."""

    result_ready = Signal(str)

    def __init__(
        self,
        script: str,
        namespace: dict[str, Any],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._script = script
        self._namespace = namespace

    @property
    def namespace(self) -> dict[str, Any]:
        """Return the worker-owned namespace copy after clean execution."""
        return self._namespace

    def run(self) -> None:  # noqa: ANN201 - Qt override
        """Run the user script and emit ``result_ready`` with formatted output."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        exception: Exception | None = None
        last_result: Any = _SENTINEL
        compiled_exec = None
        compiled_eval = None
        try:
            with contextlib.suppress(SyntaxError):
                compiled_eval = compile(self._script, "<sidekick-repl>", "eval")
            if compiled_eval is None:
                compiled_exec = compile(self._script, "<sidekick-repl>", "exec")
        except Exception as exc:  # noqa: BLE001 - report compile errors
            exception = exc

        if exception is None:
            try:
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    if compiled_eval is not None:
                        last_result = eval(  # noqa: S307  # nosec B307
                            compiled_eval, self._namespace, self._namespace
                        )
                    else:
                        assert compiled_exec is not None
                        exec(  # noqa: S102  # nosec B102
                            compiled_exec, self._namespace, self._namespace
                        )
            except Exception as exc:  # noqa: BLE001 - REPL reports user errors
                _logger.debug("Sidekick REPL execution failed: %s", exc)
                exception = exc

        output = _format_terminal_output(stdout, stderr, exception)
        if (
            exception is None
            and last_result is not _SENTINEL
            and last_result is not None
        ):
            output = f"{repr(last_result)}\n{output}" if output else repr(last_result)
        self.result_ready.emit(output)


class PythonReplWidget(QtWidgets.QWidget):
    """Reusable Python REPL bound to a :class:`WorkspaceRegistry`.

    DRY: this is the single Python execution surface — both the Terminal tab
    and the MATLAB-home command window embed an instance of this widget.

    Args:
        registry: Workspace registry holding shared variables. Required.
        set_variable: Callback ``(name, value)`` used to export user
            assignments back to the host registry. Required.
        object_name: Qt object name; defaults to a stable widget id.
        parent: Optional Qt parent.

    Raises:
        TypeError: If ``registry`` or ``set_variable`` is missing or wrong type.
    """

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        set_variable: SetVariable,
        object_name: str = SIDEKICK_PYTHON_REPL_OBJECT_NAME,
        startup_config: CalculatorStartupConfig | None = None,
        appearance: PanelAppearance | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if registry is None:
            raise TypeError("registry must be provided")
        if not is_workspace_registry(registry):
            raise TypeError("registry must be a WorkspaceRegistry")
        if set_variable is None:
            raise TypeError("set_variable must be provided")
        if not callable(set_variable):
            raise TypeError("set_variable must be callable")
        super().__init__(parent)
        self.setObjectName(object_name)
        self._registry = registry
        self._set_variable = set_variable
        self._namespace: dict[str, Any] = {}
        self._history: list[str] = []
        self._startup_config = startup_config or default_repl_startup_config()
        self._appearance = appearance or DEFAULT_DARK_PANEL_APPEARANCE
        self._load_workspace_namespace()
        _preload_scientific_namespace(self._namespace, self._startup_config)
        self._exported_names: set[str] = set(_exportable_values(self._namespace))
        self._build_ui()
        self.apply_appearance(self._appearance)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._input_label = QtWidgets.QLabel("Python input", self)
        self._input_label.setObjectName("SidekickPythonReplInputLabel")
        self._input_label.setToolTip(
            "Type Python here. Assigned names are exported to the Workspace tab."
        )
        layout.addWidget(self._input_label)

        self._input = QtWidgets.QPlainTextEdit(self)
        self._input.setObjectName("SidekickPythonReplInput")
        self._input.setPlaceholderText(
            "Type Python here, then press Run …  e.g.  result = np.array([1, 2]).sum()"
        )
        self._input.setToolTip(
            "Enter Python code that can read and write shared workspace variables."
        )
        layout.addWidget(self._input, stretch=2)

        run_row = QtWidgets.QHBoxLayout()
        run_row.setSpacing(6)

        self._run_button = QtWidgets.QPushButton("Run", self)
        self._run_button.setObjectName("SidekickPythonReplRun")
        self._run_button.setToolTip(
            "Execute the current script and export assigned variables."
        )
        self._run_button.clicked.connect(self._on_run_clicked)
        run_row.addWidget(self._run_button)

        self._cancel_button = QtWidgets.QPushButton("Cancel", self)
        self._cancel_button.setObjectName("SidekickPythonReplCancel")
        self._cancel_button.setToolTip(
            "Attempt to interrupt a long-running script (best-effort)."
        )
        self._cancel_button.setEnabled(False)
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        run_row.addWidget(self._cancel_button)

        layout.addLayout(run_row)

        self._status_label = QtWidgets.QLabel("", self)
        self._status_label.setObjectName("SidekickPythonReplStatus")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        self._output_label = QtWidgets.QLabel("Output", self)
        self._output_label.setObjectName("SidekickPythonReplOutputLabel")
        layout.addWidget(self._output_label)

        self._output = QtWidgets.QPlainTextEdit(self)
        self._output.setObjectName("SidekickPythonReplOutput")
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("stdout, stderr, and results appear here.")
        self._output.setToolTip("Shows stdout, stderr, and execution errors.")
        layout.addWidget(self._output, stretch=3)

        self._worker: _ReplWorker | None = None

    def _on_run_clicked(self) -> None:
        self.execute(self._input.toPlainText())

    def execute(self, script: str) -> None:
        """Execute ``script`` against the shared namespace."""
        if not isinstance(script, str):
            raise TypeError("script must be a str")
        if not script.strip():
            self._append_output("No code to run.")
            return

        if self._worker is not None and self._worker.isRunning():
            _logger.debug(
                "REPL execution already in progress; ignoring re-entrant call"
            )
            return

        self._history.append(script.strip())
        self._set_running(True)

        # Use a copy so cancelled executions cannot half-mutate the live namespace.
        self._worker = _ReplWorker(script, dict(self._namespace))
        self._worker.result_ready.connect(self._on_execution_finished)
        self._worker.start()
        self._drain_immediate_completion(self._worker)

    def _drain_immediate_completion(self, worker: _ReplWorker) -> None:
        """Publish fast completions without making long-running code synchronous."""
        deadline = time.monotonic() + _IMMEDIATE_COMPLETION_DRAIN_SECONDS
        while self._worker is worker and worker.isRunning():
            if time.monotonic() >= deadline:
                return
            QtWidgets.QApplication.processEvents()
            worker.wait(1)
        if self._worker is worker:
            worker.wait()
            QtWidgets.QApplication.processEvents()

    def _retire_worker(self, worker: _ReplWorker | None) -> None:
        """Drop a stopped worker without leaving it parented to widget teardown."""
        if worker is None:
            return
        if worker.isRunning():
            worker.wait()
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()

    def _set_running(self, running: bool) -> None:
        """Toggle Run/Cancel controls and the 'Running…' status label (F6)."""
        self._run_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._cancel_button.setVisible(running)
        self._status_label.setText("Running\u2026" if running else "")
        self._status_label.setVisible(running)

    def _on_cancel_clicked(self) -> None:
        """Best-effort cancel: terminate the worker thread (F6)."""
        if self._worker is not None and self._worker.isRunning():
            worker = self._worker
            # Prevent a race-completed worker from merging after cancellation.
            with contextlib.suppress(TypeError, RuntimeError):
                worker.result_ready.disconnect(self._on_execution_finished)
            worker.terminate()
            worker.wait()
            self._append_output("[Cancelled]")
            self._retire_worker(worker)
        self._set_running(False)

    def _on_execution_finished(self, output: str) -> None:
        """Slot called on the GUI thread when the worker emits ``result_ready`` (F6)."""
        # Merge the worker copy only after clean completion.
        if self._worker is not None:
            worker = self._worker
            self._namespace.clear()
            self._namespace.update(worker.namespace)
        else:
            worker = None
        self._sync_namespace_to_registry()
        self._set_running(False)
        self._append_output(output)
        self._retire_worker(worker)

    def output_text(self) -> str:
        """Return the current output pane text."""
        return str(self._output.toPlainText())

    def history(self) -> tuple[str, ...]:
        """Return submitted scripts in oldest-to-newest order."""
        return tuple(self._history)

    def apply_theme(self, terminal_theme: theme.SidekickTerminalTheme) -> None:
        """Apply REPL-scoped colors. Single-token theme handoff (LOD)."""
        if terminal_theme is None:
            raise TypeError("terminal_theme must be provided")
        self.setStyleSheet(terminal_theme.qss(self.objectName()))

    def apply_appearance(self, appearance: PanelAppearance) -> None:
        """Apply user-adjustable colours/border to the REPL surfaces.

        Single-value handoff (LOD): the panel passes a validated
        :class:`PanelAppearance`; the widget renders it via shared QSS.
        """
        if not is_panel_appearance(appearance):
            raise TypeError("appearance must be a PanelAppearance")
        self._appearance = appearance
        self.setStyleSheet(panel_qss(self.objectName(), appearance))

    def appearance(self) -> PanelAppearance:
        """Return the currently applied appearance."""
        return self._appearance

    def startup_config(self) -> CalculatorStartupConfig:
        """Return the startup-import config currently backing the REPL."""
        return self._startup_config

    @property
    def namespace(self) -> dict[str, Any]:
        """Return the live REPL namespace."""
        return self._namespace

    def apply_startup_config(
        self, config: CalculatorStartupConfig
    ) -> CalculatorStartupResult:
        """Re-import the configured packages into the live namespace.

        Lets the user change preloaded scientific packages without
        rebuilding the tab. Missing optional packages degrade to warnings.
        """
        if not isinstance(config, CalculatorStartupConfig):
            raise TypeError("config must be a CalculatorStartupConfig")
        self._startup_config = config
        result = apply_calculator_startup_imports(self._namespace, config)
        self._sync_namespace_to_registry()
        return result

    def _load_workspace_namespace(self) -> None:
        for name in self._registry.list_names():
            self._namespace[name] = self._registry.get(name)

    def _sync_namespace_to_registry(self) -> None:
        exportable = _exportable_values(self._namespace)
        for name in sorted(self._exported_names - set(exportable)):
            self._registry.remove(name)
        for name, value in exportable.items():
            self._set_variable(name, value)
        self._exported_names = set(exportable)

    def _append_output(self, text: str) -> None:
        existing = self._output.toPlainText().strip()
        combined = f"{existing}\n{text}" if existing else text
        self._output.setPlainText(combined.strip())


_SENTINEL = object()


class SidekickPythonReplWidget(QtWidgets.QWidget):
    """Legacy terminal-tab wrapper around :class:`PythonReplWidget`."""

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        set_variable: SetVariable,
        terminal_theme: theme.SidekickTerminalTheme | None = None,
        startup_config: CalculatorStartupConfig | None = None,
        appearance: PanelAppearance | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(SIDEKICK_TERMINAL_OBJECT_NAME)
        self._terminal_theme = terminal_theme or theme.SidekickTerminalTheme.inherited()
        self._repl = PythonReplWidget(
            registry=registry,
            set_variable=set_variable,
            startup_config=startup_config,
            appearance=appearance,
            parent=self,
        )
        self._registry = registry
        self._set_variable = set_variable
        self._namespace = self._repl.namespace
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._repl)
        # Preserve historic child object names for findChild() callers.
        self._repl._input.setObjectName("SidekickTerminalInput")  # noqa: SLF001
        self._repl._output.setObjectName("SidekickTerminalOutput")  # noqa: SLF001
        self._repl._run_button.setObjectName("SidekickTerminalRun")  # noqa: SLF001
        self.apply_terminal_theme(self._terminal_theme)
        self.apply_appearance(self._repl.appearance())

    def execute_script(self) -> None:
        """Execute the current script and export user variables (legacy API)."""
        self._repl.execute(self._repl._input.toPlainText())  # noqa: SLF001

    def apply_terminal_theme(self, terminal_theme: theme.SidekickTerminalTheme) -> None:
        """Apply terminal-scoped colors without changing global Sidekick QSS."""
        if terminal_theme is None:
            raise ValueError("terminal_theme must be provided")
        self._terminal_theme = terminal_theme
        self.setStyleSheet(terminal_theme.qss(SIDEKICK_TERMINAL_OBJECT_NAME))

    def apply_appearance(self, appearance: PanelAppearance) -> None:
        """Apply user-adjustable colours/border to the REPL (delegates inward)."""
        if not is_panel_appearance(appearance):
            raise TypeError("appearance must be a PanelAppearance")
        self._repl.apply_appearance(appearance)
        self.setStyleSheet(panel_qss(SIDEKICK_TERMINAL_OBJECT_NAME, appearance))

    def appearance(self) -> PanelAppearance:
        """Return the currently applied appearance."""
        return self._repl.appearance()

    def startup_config(self) -> CalculatorStartupConfig:
        """Return the startup-import config backing the REPL."""
        return self._repl.startup_config()

    def apply_startup_config(
        self, config: CalculatorStartupConfig
    ) -> CalculatorStartupResult:
        """Re-import configured packages into the live namespace."""
        return self._repl.apply_startup_config(config)


def _preload_scientific_namespace(
    namespace: dict[str, Any],
    config: CalculatorStartupConfig | None = None,
) -> None:
    apply_calculator_startup_imports(
        namespace,
        config or default_calculator_startup_config(),
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
