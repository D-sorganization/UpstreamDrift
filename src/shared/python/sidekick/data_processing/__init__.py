"""Data Processing Components for Upstream Drift Tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .exceptions import (
    ColumnNotFoundError,
    DataNotLoadedError,
    DataProcessingError,
    FileIOError,
    FilterError,
    FitError,
    TransformationError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from .core import DataProcessorEngine, ProcessingResult
    from .io import DataReader, DataWriter, FileFormatDetector

_LAZY_EXPORTS = {
    "DataProcessorEngine": (".core", "DataProcessorEngine"),
    "ProcessingResult": (".core", "ProcessingResult"),
    "DataReader": (".io", "DataReader"),
    "DataWriter": (".io", "DataWriter"),
    "FileFormatDetector": (".io", "FileFormatDetector"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "ColumnNotFoundError",
    "DataNotLoadedError",
    "DataProcessingError",
    "DataProcessorEngine",
    "DataReader",
    "DataWriter",
    "FileFormatDetector",
    "FileIOError",
    "FilterError",
    "FitError",
    "ProcessingResult",
    "TransformationError",
    "UnsupportedOperationError",
]
