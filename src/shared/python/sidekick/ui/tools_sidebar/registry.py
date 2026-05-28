"""WorkspaceRegistry — workspace variable registry with pub-sub notifications.

This module provides a complete WorkspaceRegistry implementation that
shadows the vendor version (vendor/ud-tools/.../registry.py) and adds
a subscribe/unsubscribe pub-sub layer so UI components (e.g.
WorkspaceTableModel) can react to variable changes without polling.

The core WorkspaceRegistry data model (set/get/remove/list/describe/
save_json/load_json) is reproduced from the vendor implementation
verbatim to ensure API compatibility. UpstreamDrift issue #5616 extends
it with the notification surface.

Design-by-Contract:
- subscribe(callback): precondition callable(callback); postcondition returns
  a Subscription whose .dispose() removes the callback.
- set_variable / delete_variable: fire all current subscribers.
- Invariant: re-entrancy is safe — a callback that calls set_variable will
  not cause infinite recursion.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

__all__ = ["Subscription", "WorkspaceRegistry", "WorkspaceVariable"]


@dataclass(frozen=True)
class WorkspaceVariable:
    """Metadata snapshot for one workspace variable."""

    name: str
    value: Any
    type_name: str
    summary: str
    json_safe: bool
    repr_value: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe metadata for UI lists and persisted state."""
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type_name,
            "summary": self.summary,
            "json_safe": self.json_safe,
        }
        if self.json_safe:
            data["value"] = self.value
        else:
            data["repr"] = self.repr_value or repr(self.value)
        return data

    @property
    def preview(self) -> str:
        return format_workspace_value_preview(self.value)

    @property
    def dtype(self) -> str | None:
        dtype_attr = getattr(self.value, "dtype", None)
        return str(dtype_attr) if dtype_attr is not None else None

    @property
    def shape(self) -> tuple[int, ...] | None:
        shape_attr = getattr(self.value, "shape", None)
        if isinstance(shape_attr, tuple):
            return shape_attr
        return None

    @property
    def size(self) -> int | None:
        size_attr = getattr(self.value, "size", None)
        if isinstance(size_attr, int):
            return size_attr
        return None


class Subscription:
    """Disposable handle returned by WorkspaceRegistry.subscribe().

    Calling :meth:`dispose` removes the callback from the registry.
    Subsequent dispose() calls are no-ops.
    """

    def __init__(self, registry: WorkspaceRegistry, callback: Callable) -> None:
        self._registry = registry
        self._callback: Callable | None = callback

    def dispose(self) -> None:
        """Remove this subscription from the registry; idempotent."""
        if self._callback is not None:
            self._registry._unsubscribe(self._callback)
            self._callback = None


class WorkspaceRegistry:
    """In-memory registry for workspace variables, with pub-sub notifications.

    Implements the same interface as the vendor WorkspaceRegistry and adds:
    - subscribe(callback) -> Subscription
    - set_variable(name, value) — fire subscribers after storing
    - delete_variable(name) — fire subscribers with value=None after removing
    - get_variable(name, default=None) — alias for get()

    Invariant: subscriber dispatch is re-entrancy-safe via a snapshot of the
    subscriber list; inner set_variable calls dispatch independently.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._values: dict[str, Any] = {}
        self._repr_values: dict[str, str] = {}
        self._subscribers: list[Callable] = []
        if initial:
            for name, value in initial.items():
                self.set(name, value)

    # ------------------------------------------------------------------
    # Core data store (vendor-compatible API)
    # ------------------------------------------------------------------

    def set(self, name: str, value: Any) -> WorkspaceVariable:
        """Set a workspace variable and return its metadata snapshot."""
        self._validate_name(name)
        self._values[name] = value
        if _is_json_safe(value):
            self._repr_values.pop(name, None)
        else:
            self._repr_values[name] = repr(value)
        return self.describe(name)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a variable value or ``default`` when absent."""
        return self._values.get(name, default)

    def remove(self, name: str) -> bool:
        """Remove a variable. Returns ``True`` when it existed."""
        existed = name in self._values
        self._values.pop(name, None)
        self._repr_values.pop(name, None)
        return existed

    def clear(self) -> None:
        """Remove all variables."""
        self._values.clear()
        self._repr_values.clear()

    def list(self) -> builtins.list[str]:
        """Return registered variable names in stable sorted order."""
        return sorted(self._values)

    def list_names(self) -> builtins.list[str]:
        """Alias for callers that avoid shadowing the built-in ``list``."""
        return self.list()

    def describe(self, name: str) -> WorkspaceVariable:
        """Return a metadata snapshot for one variable."""
        if name not in self._values:
            raise KeyError(name)
        value = self._values[name]
        json_safe = name not in self._repr_values and _is_json_safe(value)
        return WorkspaceVariable(
            name=name,
            value=value,
            type_name=type(value).__name__,
            summary=_summarize_dimensions(value),
            json_safe=json_safe,
            repr_value=(
                None if json_safe else self._repr_values.get(name, repr(value))
            ),
        )

    def variables(self) -> builtins.list[WorkspaceVariable]:
        """Return metadata snapshots for all variables."""
        return [self.describe(name) for name in self.list()]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe payload suitable for persistence."""
        return {
            "version": 1,
            "variables": [variable.to_metadata() for variable in self.variables()],
        }

    def save_json(self, path: str | Path) -> None:
        """Persist registry metadata and JSON-safe values to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> WorkspaceRegistry:
        """Load a registry saved by :meth:`save_json`."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        registry = cls()
        for entry in payload.get("variables", []):
            name = str(entry["name"])
            if entry.get("json_safe", False):
                registry.set(name, entry.get("value"))
            else:
                repr_value = str(entry.get("repr", ""))
                registry._values[name] = repr_value
                registry._repr_values[name] = repr_value
        return registry

    def export_environment(self, prefix: str = "UD_VAR_") -> dict[str, str]:
        """Return stringified variables for terminal/process environments."""
        env: dict[str, str] = {}
        for name, variable in ((name, self.describe(name)) for name in self.list()):
            key = f"{prefix}{_env_key(name)}"
            if variable.json_safe:
                env[key] = json.dumps(variable.value)
            else:
                env[key] = variable.repr_value or repr(variable.value)
        return env

    def export_all(self) -> dict[str, Any]:
        """Return a dict of all variable names → values (for context export)."""
        return dict(self._values)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or not name.strip():
            raise ValueError("Workspace variable name must be non-empty")

    # ------------------------------------------------------------------
    # Pub-sub extension (Issue #5616)
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable) -> Subscription:
        """Register *callback* to fire on every variable change.

        Precondition: callable(callback).
        Postcondition: returns a Subscription whose .dispose() removes callback.

        Args:
            callback: Callable receiving (name: str, value: Any).
                      For delete_variable, value is None.

        Returns:
            Subscription handle.

        Raises:
            TypeError: If callback is not callable.
        """
        if not callable(callback):
            raise TypeError(
                f"subscribe() requires a callable, got {type(callback).__name__!r}"
            )
        self._subscribers.append(callback)
        return Subscription(self, callback)

    def _unsubscribe(self, callback: Callable) -> None:
        """Remove *callback* from the subscriber list (called by Subscription)."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(callback)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a workspace variable and notify all subscribers.

        Postcondition: value is stored and subscribers have been invoked.

        Args:
            name: Variable name. Must be non-empty.
            value: Any Python value.
        """
        self.set(name, value)
        self._notify(name, value)

    def delete_variable(self, name: str) -> None:
        """Remove a variable and notify subscribers with value=None.

        Args:
            name: Variable name to remove.
        """
        self.remove(name)
        self._notify(name, None)

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Return the value for *name*, or *default* when absent.

        Args:
            name: Variable name.
            default: Fallback value (default None).

        Returns:
            Stored value or *default*.
        """
        return self.get(name, default)

    def update_from(self, incoming: WorkspaceRegistry, replace: bool = False) -> None:
        """Update variables in this registry using variables from *incoming*.

        If *replace* is True, existing variables are cleared first.
        """
        if replace:
            for name in list(self._values):
                self.delete_variable(name)
        for name in incoming.list():
            self.set_variable(name, incoming.get(name))

    def _notify(self, name: str, value: Any) -> None:
        """Invoke all subscribers with (name, value).

        Takes a snapshot of the subscriber list to be re-entrancy-safe:
        if a callback calls set_variable/delete_variable, those nested
        calls will dispatch their own _notify without affecting this one.
        """
        for callback in list(self._subscribers):
            try:
                callback(name, value)
            except Exception:  # noqa: BLE001 — subscribers must not crash the registry
                logger.exception(
                    "WorkspaceRegistry subscriber %r raised an exception", callback
                )


# ---------------------------------------------------------------------------
# Helpers (reproduced from vendor for API parity)
# ---------------------------------------------------------------------------


def _is_json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return isinstance(value, str | int | float | bool | type(None) | list | dict)


def _summarize_dimensions(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return "shape=" + "x".join(str(part) for part in shape)
        except TypeError:
            return "shape=unknown"
    if isinstance(value, dict):
        return f"keys={len(value)}"
    if isinstance(value, str):
        return f"length={len(value)}"
    if isinstance(value, list | tuple):
        if value and all(isinstance(row, list | tuple) for row in value):
            row_lengths = {len(row) for row in value}
            if len(row_lengths) == 1:
                return f"{len(value)}x{row_lengths.pop()}"
        return f"length={len(value)}"
    return "scalar"


def _env_key(name: str) -> str:
    return "".join(char.upper() if char.isalnum() else "_" for char in name)


def format_workspace_value_preview(value: Any, max_length: int = 120) -> str:
    """Return a truncated string preview of a workspace value."""
    if isinstance(value, dict) and not value:
        return "{}"
    if isinstance(value, list | tuple) and not value:
        return "[]"

    try:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            preview = f"<{value.dtype} array {value.shape}>"
        else:
            preview = repr(value)
    except Exception:  # noqa: BLE001
        preview = "<unrepresentable>"

    if len(preview) > max_length:
        return preview[: max_length - 3] + "..."
    return preview
