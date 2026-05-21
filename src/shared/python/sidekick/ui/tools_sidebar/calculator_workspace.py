"""Calculator-local workspace persistence helpers for Sidekick."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import (
    WorkspaceRegistry,
    WorkspaceVariable,
    format_workspace_value_preview,
)
from .workspace_persistence import (
    CALCULATOR_WORKSPACE_FORMAT_VERSION,  # noqa: F401 - re-exported API
    load_workspace_registry,
    save_workspace_registry,
    validate_calculator_workspace_path,  # noqa: F401 - re-exported API
)

CALCULATOR_WORKSPACE_SCOPE = "calculator"
GLOBAL_WORKSPACE_SCOPE = "global"


@dataclass(frozen=True)
class CalculatorWorkspaceSettings:
    """Settings contract for calculator-local workspace files."""

    default_directory: Path
    default_filename: str = "calculator_workspace.json"

    def default_path(self) -> Path:
        """Return the configured default calculator workspace path."""
        return self.default_directory / self.default_filename


@dataclass(frozen=True)
class GlobalWorkspaceSettings:
    """Settings contract for shared global workspace files."""

    default_directory: Path
    default_filename: str = "global_workspace.json"

    def default_path(self) -> Path:
        """Return the configured default global workspace path."""
        return self.default_directory / self.default_filename


@dataclass(frozen=True)
class CalculatorWorkspaceLoadResult:
    """Summary returned after a calculator workspace import."""

    variables: tuple[WorkspaceVariable, ...]
    replaced: bool

    @property
    def summary(self) -> str:
        """Return a compact imported-variable summary for the UI."""
        if not self.variables:
            return "Loaded 0 variables."
        names = ", ".join(variable.name for variable in self.variables)
        return f"Loaded {len(self.variables)} variables: {names}"


class CalculatorWorkspaceFacade:
    """Calculator-local view over a shared global Sidekick workspace."""

    def __init__(
        self,
        *,
        local_registry: WorkspaceRegistry,
        global_registry: WorkspaceRegistry,
        calculator_scope_id: str = CALCULATOR_WORKSPACE_SCOPE,
    ) -> None:
        if local_registry is None:
            raise ValueError("local_registry must be provided")
        if global_registry is None:
            raise ValueError("global_registry must be provided")
        _validate_scope_id(calculator_scope_id)
        self._local_registry = local_registry
        self._global_registry = global_registry
        self._calculator_scope_id = calculator_scope_id

    @property
    def calculator_scope_id(self) -> str:
        """Return the stable calculator-local workspace scope id."""
        return self._calculator_scope_id

    @property
    def local_registry(self) -> WorkspaceRegistry:
        """Return the calculator-local registry for bounded adapters."""
        return self._local_registry

    @property
    def global_registry(self) -> WorkspaceRegistry:
        """Return the shared global registry for bounded adapters."""
        return self._global_registry

    def set_local(self, name: str, value: Any) -> WorkspaceVariable:
        """Set a calculator-local value without mutating global Sidekick state."""
        return self._local_registry.set(name, value)

    def get(
        self,
        name: str,
        default: Any = None,
        *,
        include_global: bool = False,
    ) -> Any:
        """Return a local value, optionally falling back to global Sidekick state."""
        local_missing = object()
        value = self._local_registry.get(name, local_missing)
        if value is not local_missing:
            return value
        if include_global:
            return self._global_registry.get(name, default)
        return default

    def remove_local(self, name: str) -> bool:
        """Remove only the calculator-local value."""
        return bool(self._local_registry.remove(name))

    def clear_local(self) -> None:
        """Clear only the calculator-local registry.

        Postcondition: local workspace is empty; global workspace is unaffected.
        Satisfies: WorkspaceContract.clear_local.
        """
        self._local_registry.clear()

    # ------------------------------------------------------------------
    # WorkspaceContract Protocol surface
    # These thin aliases exist so that CalculatorWorkspaceFacade satisfies
    # the WorkspaceContract structural Protocol without coupling callers to
    # the older set_local/get/list_names naming convention.
    # ------------------------------------------------------------------

    def set_variable(self, name: str, value: Any) -> WorkspaceVariable:
        """Store *value* under *name* in the calculator-local workspace.

        Precondition: ``name`` is a non-empty string.
        Postcondition: variable is retrievable via :meth:`get_variable`.
        Satisfies: WorkspaceContract.set_variable.
        """
        return self.set_local(name, value)

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Return the locally stored *name*, or *default* if absent.

        Precondition: ``name`` is a non-empty string.
        Postcondition: returns stored value or *default*; does not raise.
        Satisfies: WorkspaceContract.get_variable.
        """
        return self.get(name, default)

    def list_variable_names(self) -> list[str]:
        """Return calculator-local variable names in sorted order.

        Postcondition: returns a sorted list; empty list when workspace is empty.
        Satisfies: WorkspaceContract.list_variable_names.
        """
        return self._local_registry.list_names()

    def describe(
        self,
        name: str,
        *,
        include_global: bool = False,
    ) -> WorkspaceVariable:
        """Describe a visible variable without mutating either registry."""
        if name in self._local_registry.list_names():
            return self._local_registry.describe(name)
        if include_global and name in self._global_registry.list_names():
            return self._global_registry.describe(name)
        raise KeyError(name)

    def variables(
        self,
        *,
        include_global: bool = False,
    ) -> tuple[WorkspaceVariable, ...]:
        """Return visible variables with local values shadowing global names."""
        variables = {
            variable.name: variable for variable in self._local_registry.variables()
        }
        if include_global:
            for variable in self._global_registry.variables():
                variables.setdefault(variable.name, variable)
        return tuple(variables[name] for name in sorted(variables))

    def export_variables(self, *, include_global: bool = True) -> dict[str, Any]:
        """Return execution variables with local names shadowing global names."""
        exported: dict[str, Any] = {}
        if include_global:
            exported.update(
                {
                    name: self._global_registry.get(name)
                    for name in self._global_registry.list_names()
                }
            )
        exported.update(
            {
                name: self._local_registry.get(name)
                for name in self._local_registry.list_names()
            }
        )
        return exported

    def promote_to_global(
        self,
        name: str,
        *,
        overwrite: bool = False,
    ) -> WorkspaceVariable:
        """Copy a local value into the global workspace with explicit overwrite."""
        if name not in self._local_registry.list_names():
            raise KeyError(name)
        if not overwrite and name in self._global_registry.list_names():
            raise FileExistsError(f"global workspace variable already exists: {name}")
        return self._global_registry.set(name, self._local_registry.get(name))


class CalculatorWorkspaceController:
    """Persist a calculator-local workspace without touching global state."""

    def __init__(
        self,
        registry: WorkspaceRegistry,
        *,
        settings: CalculatorWorkspaceSettings,
        scope: str = CALCULATOR_WORKSPACE_SCOPE,
    ) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        if settings is None:
            raise ValueError("settings must be provided")
        if scope != CALCULATOR_WORKSPACE_SCOPE:
            raise ValueError("calculator workspace scope must be explicit")
        self._registry = registry
        self._settings = settings
        self._scope = scope

    def save(self, path: str | Path | None = None) -> Path:
        """Save the calculator-local registry to ``path``."""
        return save_workspace_registry(
            self._registry,
            path or self._settings.default_path(),
            scope=self._scope,
        )

    @property
    def settings(self) -> CalculatorWorkspaceSettings:
        """Return the configured calculator workspace persistence settings."""
        return self._settings

    def load(
        self,
        path: str | Path | None = None,
        *,
        replace: bool = False,
        confirm_replace: bool = False,
    ) -> CalculatorWorkspaceLoadResult:
        """Load a calculator-local workspace, merging by default."""
        imported = load_workspace_registry(
            self._registry,
            path or self._settings.default_path(),
            expected_scope=CALCULATOR_WORKSPACE_SCOPE,
            replace=replace,
            confirm_replace=confirm_replace,
        )
        return CalculatorWorkspaceLoadResult(imported, replaced=replace)

    def clear(self, *, confirm_clear: bool = False) -> None:
        """Clear the calculator-local registry after explicit confirmation."""
        if not confirm_clear:
            raise PermissionError("clear requires explicit confirmation")
        self._registry.clear()


class GlobalWorkspaceController:
    """Persist and manage the shared Sidekick global workspace."""

    def __init__(
        self,
        registry: WorkspaceRegistry,
        *,
        settings: GlobalWorkspaceSettings,
        scope: str = GLOBAL_WORKSPACE_SCOPE,
    ) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        if settings is None:
            raise ValueError("settings must be provided")
        if scope != GLOBAL_WORKSPACE_SCOPE:
            raise ValueError("global workspace scope must be explicit")
        self._registry = registry
        self._settings = settings
        self._scope = scope

    @property
    def settings(self) -> GlobalWorkspaceSettings:
        """Return the configured global workspace persistence settings."""
        return self._settings

    def set(self, name: str, value: Any) -> WorkspaceVariable:
        """Set a global workspace variable."""
        return self._registry.set(name, value)

    def describe(self, name: str) -> WorkspaceVariable:
        """Return metadata for a global workspace variable."""
        return self._registry.describe(name)

    def variables(self) -> tuple[WorkspaceVariable, ...]:
        """Return global workspace variables in stable display order."""
        return tuple(self._registry.variables())

    def remove(self, name: str, *, confirm_delete: bool = False) -> bool:
        """Delete a global variable after explicit confirmation."""
        if not confirm_delete:
            raise PermissionError("delete requires explicit confirmation")
        return self._registry.remove(name)

    def clear(self, *, confirm_clear: bool = False) -> None:
        """Clear the global workspace after explicit confirmation."""
        if not confirm_clear:
            raise PermissionError("clear requires explicit confirmation")
        self._registry.clear()

    def save(self, path: str | Path | None = None) -> Path:
        """Save the global registry to ``path``."""
        return save_workspace_registry(
            self._registry,
            path or self._settings.default_path(),
            scope=self._scope,
        )

    def load(
        self,
        path: str | Path | None = None,
        *,
        replace: bool = False,
        confirm_replace: bool = False,
    ) -> CalculatorWorkspaceLoadResult:
        """Load the global workspace, merging by default."""
        imported = load_workspace_registry(
            self._registry,
            path or self._settings.default_path(),
            expected_scope=GLOBAL_WORKSPACE_SCOPE,
            replace=replace,
            confirm_replace=confirm_replace,
        )
        return CalculatorWorkspaceLoadResult(imported, replaced=replace)


class CalculatorWorkspaceActions:
    """UI action adapter around calculator-local workspace persistence."""

    def __init__(
        self,
        controller: CalculatorWorkspaceController,
        status_label: Any,
    ) -> None:
        if controller is None:
            raise ValueError("controller must be provided")
        if status_label is None:
            raise ValueError("status_label must be provided")
        self._controller = controller
        self._status_label = status_label

    def save_workspace(self, path: str | Path | None = None) -> None:
        """Persist calculator-local variables and update the status label."""
        try:
            saved = self._controller.save(path)
        except Exception as exc:  # noqa: BLE001 - user-facing persistence errors
            self._status_label.setText(f"Workspace save failed: {exc}")
            return
        self._status_label.setText(f"Workspace saved: {saved}")

    def load_workspace(
        self,
        path: str | Path | None = None,
        *,
        replace: bool = False,
        confirm_replace: bool = False,
    ) -> None:
        """Load calculator-local variables and update the status label."""
        try:
            result = self._controller.load(
                path,
                replace=replace,
                confirm_replace=confirm_replace,
            )
        except Exception as exc:  # noqa: BLE001 - user-facing persistence errors
            self._status_label.setText(f"Workspace load failed: {exc}")
            return
        self._status_label.setText(result.summary)

    def clear_workspace(self, *, confirm_clear: bool = False) -> None:
        """Clear calculator-local variables and update the status label."""
        try:
            self._controller.clear(confirm_clear=confirm_clear)
        except Exception as exc:  # noqa: BLE001 - user-facing persistence errors
            self._status_label.setText(f"Workspace clear failed: {exc}")
            return
        self._status_label.setText("Cleared calculator workspace.")


def get_default_sidekick_dir(app_name: str | None = None) -> Path:
    """Return the default Sidekick storage directory for an application.

    Args:
        app_name: Optional application name. When given, returns
            ``~/.{app_name}/sidekick``; otherwise returns ``~/.sidekick``.

    Returns:
        Resolved path to the Sidekick storage directory.
    """
    if app_name is not None:
        return Path.home() / f".{app_name}" / "sidekick"
    return Path.home() / ".sidekick"


def default_calculator_workspace_controller(
    registry: WorkspaceRegistry,
    *,
    storage_dir: Path | None = None,
) -> CalculatorWorkspaceController:
    """Build the default Sidekick calculator-local workspace controller.

    Args:
        registry: The calculator-local workspace registry.
        storage_dir: Directory for workspace files. Defaults to
            ``~/.sidekick`` when ``None``. Pass
            ``get_default_sidekick_dir("upstream_drift_tools")`` to preserve
            the legacy ``~/.upstream_drift_tools/sidekick`` path.
    """
    directory = storage_dir if storage_dir is not None else get_default_sidekick_dir()
    return CalculatorWorkspaceController(
        registry,
        settings=CalculatorWorkspaceSettings(default_directory=directory),
    )


def default_global_workspace_controller(
    registry: WorkspaceRegistry,
    *,
    storage_dir: Path | None = None,
) -> GlobalWorkspaceController:
    """Build the default Sidekick global workspace controller.

    Args:
        registry: The shared global workspace registry.
        storage_dir: Directory for workspace files. Defaults to
            ``~/.sidekick`` when ``None``. Pass
            ``get_default_sidekick_dir("upstream_drift_tools")`` to preserve
            the legacy ``~/.upstream_drift_tools/sidekick`` path.
    """
    directory = storage_dir if storage_dir is not None else get_default_sidekick_dir()
    return GlobalWorkspaceController(
        registry,
        settings=GlobalWorkspaceSettings(default_directory=directory),
    )


def build_calculator_workspace_controls(
    parent: Any,
    actions: CalculatorWorkspaceActions,
) -> Any:
    """Build Save/Load Workspace controls for the calculator tab."""
    from .qt_compat import QtWidgets

    row = QtWidgets.QHBoxLayout()
    save_workspace = QtWidgets.QPushButton("Save Workspace", parent)
    save_workspace.setObjectName("SidekickCalculatorSaveWorkspace")
    save_workspace.setToolTip("Save the calculator-local workspace.")
    save_workspace.clicked.connect(lambda: actions.save_workspace())
    row.addWidget(save_workspace)

    load_workspace = QtWidgets.QPushButton("Load Workspace", parent)
    load_workspace.setObjectName("SidekickCalculatorLoadWorkspace")
    load_workspace.setToolTip("Load variables into the calculator-local workspace.")
    load_workspace.clicked.connect(lambda: actions.load_workspace())
    row.addWidget(load_workspace)
    return row


def evaluate_calculator_expression(expression: str) -> tuple[Any, str]:
    """Evaluate a calculator expression and return workspace value plus preview."""
    result = _evaluate_shared_calculator_expression(expression)
    workspace_value = workspace_value_for_calculator_result(result)
    return workspace_value, format_workspace_value_preview(workspace_value)


def _evaluate_shared_calculator_expression(expression: str) -> Any:
    stripped = expression.strip()
    if stripped.startswith("Matrix(") and stripped.endswith(")"):
        return ast.literal_eval(stripped.removeprefix("Matrix(")[:-1])
    from shared.python.safe_eval import safe_eval_math

    return safe_eval_math(stripped)


def workspace_value_for_calculator_result(value: Any) -> Any:
    """Normalize array-like calculator outputs for shared workspace metadata."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, list | tuple):
        return _listify(value)
    return str(value)


def _listify(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return [_listify(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _listify(tolist())
    return value


def _validate_scope_id(scope_id: str) -> None:
    if not isinstance(scope_id, str) or not scope_id.strip():
        raise ValueError("workspace scope id must be a non-empty string")
