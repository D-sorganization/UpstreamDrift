"""Diagnostics — system and simulation health monitoring.

Public surface::

    from src.shared.python.diagnostics import (
        DiagnosticsCollector,
        DiagnosticsSnapshot,
        DiagnosticsHistory,
        SystemMetrics,
        SimulationMetrics,
    )

Implements Epic #5698 (first child): DiagnosticsCollector + history ring-buffer.
"""

from __future__ import annotations

from src.shared.python.diagnostics._collector import DiagnosticsCollector
from src.shared.python.diagnostics._history import DiagnosticsHistory
from src.shared.python.diagnostics._snapshot import (
    DiagnosticsSnapshot,
    SimulationMetrics,
    SystemMetrics,
)

__all__ = [
    "DiagnosticsCollector",
    "DiagnosticsHistory",
    "DiagnosticsSnapshot",
    "SimulationMetrics",
    "SystemMetrics",
]
