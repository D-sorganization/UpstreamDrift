"""Formal Protocol contract for calculator-local workspace facades.

This module defines the ``WorkspaceContract`` structural Protocol that any
calculator-local workspace facade must satisfy.  The canonical implementation
is :class:`~sidekick.ui.tools_sidebar.calculator_workspace.CalculatorWorkspaceFacade`.

DbC:
    Preconditions are documented per method.
    Postconditions are documented per method.

LOD:
    Protocol methods operate on primitive types and ``WorkspaceVariable``
    metadata objects only — they do not expose the internal ``WorkspaceRegistry``
    objects.  Callers never need to reach through the facade.

DRY:
    The ``WorkspaceContract`` is the single authoritative declaration of the
    workspace facade interface.  All type-checked call sites should annotate
    with ``WorkspaceContract`` rather than the concrete class to reduce coupling.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkspaceContract(Protocol):
    """Structural Protocol for a calculator-local variable workspace.

    Any class that provides ``set_variable``, ``get_variable``,
    ``list_variable_names``, and ``clear_local`` with matching signatures
    satisfies this Protocol via structural subtyping, even without
    explicitly inheriting from it.

    Design principles:
        - LOD: methods expose only name/value/metadata; no internal registries.
        - DbC: callers are responsible for non-empty variable names (str).
        - DRY: this is the single source of truth for the workspace facade API.
    """

    def set_variable(self, name: str, value: Any) -> Any:
        """Store *value* under *name* in the calculator-local workspace.

        Precondition:
            ``name`` is a non-empty string.
        Postcondition:
            The variable is retrievable via :meth:`get_variable`.
            Returns a metadata snapshot (WorkspaceVariable) for the stored value.

        Args:
            name: Non-empty variable name.
            value: JSON-serialisable or representable value.

        Returns:
            A metadata snapshot for the stored variable.

        Raises:
            ValueError: If *name* is empty or contains only whitespace.
        """
        ...

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Return the value stored under *name*, or *default* if absent.

        Precondition:
            ``name`` is a non-empty string.
        Postcondition:
            Returns the stored value when *name* exists, or *default* otherwise.
            Does not raise for missing names.

        Args:
            name: Variable name to look up.
            default: Value returned when *name* is not in the workspace.

        Returns:
            The stored value, or *default*.
        """
        ...

    def list_variable_names(self) -> list[str]:
        """Return calculator-local variable names in stable sorted order.

        Precondition:
            (none)
        Postcondition:
            Returns a list of strings, sorted lexicographically.
            Returns an empty list when no variables are stored.

        Returns:
            Sorted list of variable names.
        """
        ...

    def clear_local(self) -> None:
        """Remove all calculator-local variables without touching global state.

        Precondition:
            (none)
        Postcondition:
            :meth:`list_variable_names` returns an empty list.
            Global workspace variables (if any) are unaffected.
        """
        ...
