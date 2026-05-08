"""Registry and factory helpers for :class:`MocapSourceAdapter` subclasses.

A registered adapter is selected purely by its :meth:`supports` method;
the first adapter (in registration order) that returns ``True`` wins.
This keeps adapter selection localised - adapters know how to recognise
their own files and the registry knows nothing about format internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from src.shared.python.motion_pipeline.contracts import Calibration
from src.shared.python.motion_pipeline.sources.base import (
    LoadedPayload,
    MocapSourceAdapter,
    UnsupportedFormatError,
)

_REGISTRY: list[type[MocapSourceAdapter]] = []

A = TypeVar("A", bound=type[MocapSourceAdapter])


def register_adapter(cls: A) -> A:
    """Class decorator that appends *cls* to the registry.

    The same adapter class is never registered twice (idempotent).
    """
    if not isinstance(cls, type) or not issubclass(cls, MocapSourceAdapter):
        raise TypeError(
            f"register_adapter expects a MocapSourceAdapter subclass, got {cls!r}"
        )
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)
    return cls


def unregister_adapter(cls: type[MocapSourceAdapter]) -> None:
    """Remove *cls* from the registry if present (used in tests)."""
    if cls in _REGISTRY:
        _REGISTRY.remove(cls)


def detect_format(path: Path) -> type[MocapSourceAdapter]:
    """Return the first registered adapter class whose ``supports(path)`` is True.

    Raises :class:`UnsupportedFormatError` if no adapter claims the file.
    """
    p = Path(path)
    for cls in _REGISTRY:
        try:
            if cls.supports(p):
                return cls
        except Exception:  # noqa: BLE001 - faulty adapter must not block others
            continue
    raise UnsupportedFormatError(
        f"No registered MocapSourceAdapter supports {p!s}. "
        f"Known formats: {list_formats()}"
    )


def load_any(
    path: Path,
    calibration: Calibration | None = None,
) -> LoadedPayload:
    """Detect the format of *path* and load it with post-condition checks."""
    adapter_cls = detect_format(path)
    adapter = adapter_cls()
    return adapter.load_checked(Path(path), calibration=calibration)


def list_formats() -> list[str]:
    """Return the format identifiers of all currently registered adapters."""
    return [cls.format_name for cls in _REGISTRY]


def registered_adapters() -> tuple[type[MocapSourceAdapter], ...]:
    """Snapshot of the registry (useful for diagnostics and tests)."""
    return tuple(_REGISTRY)
