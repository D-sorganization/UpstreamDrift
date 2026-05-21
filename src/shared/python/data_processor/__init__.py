"""Bulk-I/O engine package for the Data Processor (issue #2989).

Phase 2: engine contract definition and Python wrapper with pandas fallback.

Public API::

    from shared.python.data_processor import rust_engine

    info   = rust_engine.inspect("data.csv")
    df     = rust_engine.preview("data.csv", nrows=100)
    report = rust_engine.convert("in.csv", "out.parquet", format="parquet")
"""

from .rust_engine import (
    ConversionReport,
    SchemaInfo,
    convert,
    filter_export,
    inspect,
    preview,
    scan_batch,
)

__all__ = [
    "ConversionReport",
    "SchemaInfo",
    "convert",
    "filter_export",
    "inspect",
    "preview",
    "scan_batch",
]
