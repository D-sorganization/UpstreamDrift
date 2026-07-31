"""Bridge to the full Data Processor application (single source of truth).

The full-featured Data Processor application lives at
``src/data_processing/data_processor/python/data_processor`` in the Tools repo.
Unlike every other tool (which sits directly under ``src/<tool>`` and is therefore
importable once ``src`` is on ``sys.path``), the Data Processor package is nested
one level deeper, so ``import data_processor`` does not resolve out of the box.

This module is the *single* place that knows how to put that package on the import
path and construct its embeddable widget. Sidekick, Gasification_Model, and
UpstreamDrift all consume the widget through :func:`create_full_data_processor_widget`
so there is exactly one implementation of the real Data Processor UI.

Related issues: D-sorganization/Tools#3111, #3112, #3113.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

__all__ = [
    "REQUIRED_SUBPATHS",
    "DataProcessorUnavailableError",
    "create_full_data_processor_widget",
    "ensure_full_data_processor_on_path",
    "full_data_processor_available",
]

_logger = logging.getLogger(__name__)

#: Repo-root-relative directories that must be importable for the full app:
#: the Data Processor package root and the shared ``utils`` package root used by
#: ``high_performance_loader``.
REQUIRED_SUBPATHS: tuple[str, ...] = (
    "src/data_processing/data_processor/python",
    "src/python/src",
)

#: Module that exposes the canonical embeddable widget class.
_WIDGET_MODULE = "data_processor.pyqt_widget"
_WIDGET_CLASS = "DataProcessorWidget"


class DataProcessorUnavailableError(RuntimeError):
    """Raised when the full Data Processor application cannot be located/imported."""


def _find_repo_root(start: Path) -> Path:
    """Return the repository root containing the full Data Processor package.

    **Pre-conditions** (DbC): ``start`` must be a :class:`~pathlib.Path`.

    The root is identified by containing both ``pyproject.toml`` and the nested
    Data Processor package directory, so this works from any nesting depth.
    """
    if not isinstance(start, Path):
        raise TypeError(f"start must be a Path, got {type(start)!r}")
    marker = REQUIRED_SUBPATHS[0]
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / marker).is_dir():
            return candidate
    raise DataProcessorUnavailableError(
        "could not locate repository root containing the Data Processor package "
        f"(looked for 'pyproject.toml' and '{marker}' above {start})"
    )


def ensure_full_data_processor_on_path(repo_root: Path | None = None) -> Path:
    """Add the full Data Processor package roots to ``sys.path`` (idempotent).

    **Pre-conditions** (DbC): ``repo_root`` is ``None`` or a :class:`~pathlib.Path`.

    The app package directories are appended without reordering or evicting
    cached modules. The bulk-I/O wrapper lives under the distinct
    ``data_processor_io`` import name, so bare ``import data_processor`` remains
    reserved for the full application package.

    Args:
        repo_root: Optional explicit repository root. When omitted it is located
            relative to this module's file.

    Returns:
        The resolved repository root.
    """
    if repo_root is not None and not isinstance(repo_root, Path):
        raise TypeError(f"repo_root must be a Path or None, got {type(repo_root)!r}")
    root = (
        repo_root
        if repo_root is not None
        else _find_repo_root(Path(__file__).resolve())
    )
    import sys

    for sub in reversed(REQUIRED_SUBPATHS):
        target = root / sub
        if not target.is_dir():
            continue
        path_str = str(target)
        if path_str not in sys.path:
            sys.path.append(path_str)
    return root


def full_data_processor_available() -> bool:
    """Return ``True`` if the full Data Processor widget can be imported."""
    try:
        ensure_full_data_processor_on_path()
        return importlib.util.find_spec(_WIDGET_MODULE) is not None
    except (DataProcessorUnavailableError, ImportError, ValueError) as exc:
        _logger.debug("Full Data Processor not available: %s", exc)
        return False


def create_full_data_processor_widget(parent: QWidget | None = None) -> QWidget:
    """Construct the full Data Processor embeddable widget (single source of truth).

    Args:
        parent: Optional Qt parent widget.

    Returns:
        The real :class:`data_processor.pyqt_widget.DataProcessorWidget` instance.

    Raises:
        DataProcessorUnavailableError: If the package or its dependencies cannot
            be imported.
    """
    ensure_full_data_processor_on_path()
    try:
        module = importlib.import_module(_WIDGET_MODULE)
        widget_class = getattr(module, _WIDGET_CLASS)
    except ImportError as exc:
        raise DataProcessorUnavailableError(
            f"failed to import {_WIDGET_MODULE}.{_WIDGET_CLASS}: {exc}"
        ) from exc
    return cast("QWidget", widget_class(parent))
