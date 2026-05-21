"""Runtime widget for the Sidekick calculator tab."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .calculator_assist import (
    CalculatorPredictionProvider,
    CalculatorPredictiveText,
    StaticCalculatorPredictionProvider,
)
from .calculator_startup import (
    CalculatorStartupConfig,
    apply_calculator_startup_imports,
    default_calculator_startup_config,
)
from .calculator_workspace import (
    CalculatorWorkspaceActions,
    CalculatorWorkspaceFacade,
    build_calculator_workspace_controls,
    default_calculator_workspace_controller,
    evaluate_calculator_expression,
    get_default_sidekick_dir,
)
from .command_history import CommandHistoryController
from .qt_compat import QtCore, QtWidgets
from .registry import WorkspaceRegistry
from .workspace_commands import WorkspaceCommandExecutor

logger = logging.getLogger(__name__)

SIDEKICK_CALCULATOR_OBJECT_NAME = "SidekickCalculatorTab"
_CALCULATOR_RESULT_NAME = "calculator_result"

SetVariable = Callable[[str, Any], None]
SetPredictiveTextEnabled = Callable[[bool], None]
RefreshWorkspace = Callable[[], None]


class SidekickCalculatorWidget(QtWidgets.QWidget):
    """Compact symbolic calculator suitable for narrow sidebars."""

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        set_variable: SetVariable,
        local_registry: WorkspaceRegistry | None = None,
        predictive_text_enabled: bool = False,
        prediction_provider: CalculatorPredictionProvider | None = None,
        startup_import_config: CalculatorStartupConfig | None = None,
        set_predictive_text_enabled: SetPredictiveTextEnabled | None = None,
        refresh_workspace: RefreshWorkspace | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        if set_variable is None:
            raise ValueError("set_variable must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_CALCULATOR_OBJECT_NAME)
        self._workspace_registry = registry
        self._registry = local_registry or WorkspaceRegistry()
        self._workspace = CalculatorWorkspaceFacade(
            local_registry=self._registry,
            global_registry=self._workspace_registry,
        )
        self._set_variable = set_variable
        self._prediction_provider = (
            prediction_provider or StaticCalculatorPredictionProvider()
        )
        self._workspace_controller = default_calculator_workspace_controller(
            self._registry,
            storage_dir=get_default_sidekick_dir("upstream_drift_tools"),
        )
        self._startup_namespace: dict[str, Any] = {}
        self._startup_result = apply_calculator_startup_imports(
            self._startup_namespace,
            startup_import_config or default_calculator_startup_config(),
        )
        self._set_predictive_text_enabled = set_predictive_text_enabled
        self._refresh_workspace = refresh_workspace
        self._predictive_text_enabled = bool(predictive_text_enabled)
        self._workspace_command_history = CommandHistoryController()
        self._workspace_command_executor = WorkspaceCommandExecutor(
            workspace=self._workspace,
            local_controller=self._workspace_controller,
            global_registry=self._workspace_registry,
            global_storage_path=(
                self._workspace_controller.settings.default_directory / "workspace.json"
            ),
        )
        self._build_ui()
        self.set_predictive_text_enabled(self._predictive_text_enabled)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._input = QtWidgets.QLineEdit(self)
        self._input.setObjectName("SidekickCalculatorInput")
        self._input.setPlaceholderText("2 + 2, sin(pi/2), diff(x**2, x)")
        self._input.setToolTip("Enter a symbolic or numeric expression.")
        self._input.returnPressed.connect(self.evaluate_expression)
        self._input.textEdited.connect(self._refresh_predictive_suggestions)
        layout.addWidget(self._input)

        self._completer_model = QtCore.QStringListModel(self)
        self._completer = QtWidgets.QCompleter(self._completer_model, self)
        self._completer.setCaseSensitivity(_case_insensitive_flag())
        self._input.setCompleter(self._completer)

        self._predictive_toggle = QtWidgets.QCheckBox("Predictive text", self)
        self._predictive_toggle.setObjectName("SidekickCalculatorPredictiveText")
        self._predictive_toggle.setToolTip(
            "Show suggestions from calculator commands, Workspace, "
            "and loaded dependencies."
        )
        self._predictive_toggle.toggled.connect(self.set_predictive_text_enabled)
        layout.addWidget(self._predictive_toggle)

        self._startup_status = QtWidgets.QLabel("", self)
        self._startup_status.setObjectName("SidekickCalculatorStartupStatus")
        self._startup_status.setWordWrap(True)
        self._startup_status.setToolTip(
            "Reports optional calculator startup dependencies that are unavailable."
        )
        layout.addWidget(self._startup_status)
        self._refresh_startup_status()

        self._run_button = QtWidgets.QPushButton("Evaluate", self)
        self._run_button.setObjectName("SidekickCalculatorRun")
        self._run_button.setToolTip("Evaluate the current calculator expression.")
        self._run_button.clicked.connect(self.evaluate_expression)
        layout.addWidget(self._run_button)

        self._result = QtWidgets.QLabel("", self)
        self._result.setObjectName("SidekickCalculatorResult")
        self._result.setWordWrap(True)
        self._result.setToolTip("Displays the latest calculator result or error.")
        self._workspace_actions = CalculatorWorkspaceActions(
            self._workspace_controller,
            self._result,
        )
        layout.addLayout(
            build_calculator_workspace_controls(self, self._workspace_actions),
        )
        self._workspace_command_input = QtWidgets.QLineEdit(self)
        self._workspace_command_input.setObjectName("SidekickWorkspaceCommandInput")
        self._workspace_command_input.setPlaceholderText(
            "global alpha = 42 | show local calculator_result | clear global confirm"
        )
        self._workspace_command_input.setToolTip(
            "Run bounded workspace commands for local or global Sidekick variables."
        )
        self._workspace_command_input.returnPressed.connect(
            self.execute_workspace_command
        )
        self._workspace_command_input.installEventFilter(self)
        layout.addWidget(self._workspace_command_input)

        self._workspace_command_run = QtWidgets.QPushButton(
            "Run Workspace Command",
            self,
        )
        self._workspace_command_run.setObjectName("SidekickWorkspaceCommandRun")
        self._workspace_command_run.setToolTip(
            "Execute the current bounded workspace command without opening a terminal."
        )
        self._workspace_command_run.clicked.connect(self.execute_workspace_command)
        layout.addWidget(self._workspace_command_run)
        layout.addWidget(self._result)
        layout.addStretch(1)

    @property
    def predictive_text_enabled(self) -> bool:
        """Return whether predictive text is active for this widget."""
        return self._predictive_text_enabled

    def set_predictive_text_enabled(self, enabled: bool) -> None:
        """Toggle predictive text without evaluating calculator input."""
        self._predictive_text_enabled = bool(enabled)
        if self._predictive_toggle.isChecked() != self._predictive_text_enabled:
            self._predictive_toggle.setChecked(self._predictive_text_enabled)
        if self._set_predictive_text_enabled is not None:
            self._set_predictive_text_enabled(self._predictive_text_enabled)
        self._refresh_predictive_suggestions(self._input.text())

    def suggestions_for(self, prefix: str) -> tuple[str, ...]:
        """Return prediction labels for tests and host integrations."""
        predictive = CalculatorPredictiveText(
            enabled=self._predictive_text_enabled,
            provider=self._prediction_provider,
        )
        suggestions = predictive.suggest(
            prefix,
            workspace_variables=self._workspace.variables(include_global=True),
            loaded_dependencies=self.loaded_startup_dependencies(),
        )
        return tuple(suggestion.label for suggestion in suggestions)

    def loaded_startup_dependencies(self) -> tuple[str, ...]:
        """Return optional dependency modules loaded for this calculator instance."""
        return self._startup_result.loaded_modules

    def startup_warnings(self) -> tuple[str, ...]:
        """Return user-facing optional dependency diagnostics."""
        return tuple(warning.message for warning in self._startup_result.warnings)

    def evaluate_expression(self) -> None:
        """Evaluate the current expression and publish the result."""
        expression = self._input.text().strip()
        if not expression:
            self._result.setText("Enter an expression.")
            return
        try:
            workspace_value, text = evaluate_calculator_expression(expression)
        except Exception as exc:  # noqa: BLE001 - user-facing calculator errors
            logger.debug("Sidekick calculator evaluation failed: %s", exc)
            self._result.setText(f"Error: {exc}")
            return

        self._result.setText(text)
        self._workspace.set_local(_CALCULATOR_RESULT_NAME, workspace_value)
        self._refresh_predictive_suggestions(self._input.text())

    def execute_workspace_command(self) -> None:
        """Execute one bounded workspace command and report the result."""
        command = self._workspace_command_input.text().strip()
        if not command:
            self._result.setText("Enter a workspace command.")
            return
        try:
            normalized = self._workspace_command_history.submit(command)
            result = self._workspace_command_executor.execute(normalized)
        except Exception as exc:  # noqa: BLE001 - user-facing command errors
            logger.debug("Sidekick workspace command failed: %s", exc)
            self._result.setText(f"Workspace command failed: {exc}")
            return

        self._result.setText(result.message)
        self._refresh_predictive_suggestions(self._input.text())
        if result.scope == "global" and self._refresh_workspace is not None:
            self._refresh_workspace()

    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802
        """Handle history previews for the bounded workspace command line."""
        if watched is self._workspace_command_input and _is_key_press(event):
            if _matches_key(event, "Key_Up"):
                preview = self._workspace_command_history.previous_preview(
                    self._workspace_command_input.text()
                )
                if preview is not None:
                    self._workspace_command_input.setText(preview)
                return True
            if _matches_key(event, "Key_Down"):
                preview = self._workspace_command_history.next_preview()
                if preview is not None:
                    self._workspace_command_input.setText(preview)
                return True
        return super().eventFilter(watched, event)

    def _refresh_predictive_suggestions(self, prefix: str) -> None:
        self._completer_model.setStringList(list(self.suggestions_for(prefix)))

    def _refresh_startup_status(self) -> None:
        warnings = self.startup_warnings()
        if not warnings:
            loaded = ", ".join(self.loaded_startup_dependencies())
            self._startup_status.setText(
                f"Startup imports loaded: {loaded}" if loaded else ""
            )
            return
        self._startup_status.setText("Optional dependency unavailable: " + warnings[0])


def _case_insensitive_flag() -> Any:
    case_sensitivity = getattr(QtCore.Qt, "CaseSensitivity", None)
    if case_sensitivity is not None:
        return case_sensitivity.CaseInsensitive
    return QtCore.Qt.CaseInsensitive


def _is_key_press(event: Any) -> bool:
    event_type = getattr(QtCore.QEvent, "Type", None)
    if event_type is not None:
        return event.type() == event_type.KeyPress
    return event.type() == QtCore.QEvent.KeyPress


def _matches_key(event: Any, key_name: str) -> bool:
    key_enum = getattr(QtCore.Qt, "Key", None)
    if key_enum is not None:
        return event.key() == getattr(key_enum, key_name)
    return event.key() == getattr(QtCore.Qt, key_name)
