"""Calculator-local workspace facade for Sidekick."""

from __future__ import annotations

from typing import Any

from .registry import WorkspaceRegistry, WorkspaceVariable

CALCULATOR_WORKSPACE_SCOPE = "calculator"
GLOBAL_WORKSPACE_SCOPE = "global"


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
        return [str(name) for name in self._local_registry.list_names()]

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


def _validate_scope_id(scope_id: str) -> None:
    if not isinstance(scope_id, str) or not scope_id.strip():
        raise ValueError("calculator_scope_id must be a non-empty string")
