"""DiagnosticsSnapshot and constituent metrics dataclasses.

All dataclasses are frozen so snapshots are immutable value objects — safe to
store in ring-buffers and hand across thread boundaries without defensive
copying.

Implements part of Epic #5698.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from src.shared.python.contracts import require


@dataclass(frozen=True)
class SystemMetrics:
    """Point-in-time system resource measurements.

    Attributes:
        cpu_percent: Overall CPU utilisation 0–100.  ``-1.0`` when unavailable.
        memory_used_mb: Resident set size in megabytes.  ``-1.0`` when unavailable.
        memory_total_mb: Total physical RAM in megabytes.  ``-1.0`` when unavailable.
        memory_percent: Percentage of RAM in use.  ``-1.0`` when unavailable.
        gpu_percent: GPU utilisation 0–100.  ``-1.0`` when GPU is absent/unavailable.
        open_file_handles: Number of open file descriptors for this process.  ``-1``
            when unavailable (e.g. Windows with missing ``psutil``).
    """

    cpu_percent: float = -1.0
    memory_used_mb: float = -1.0
    memory_total_mb: float = -1.0
    memory_percent: float = -1.0
    gpu_percent: float = -1.0
    open_file_handles: int = -1

    def __post_init__(self) -> None:
        require(
            isinstance(self.cpu_percent, (int, float)),
            "cpu_percent must be numeric",
            self.cpu_percent,
        )
        require(
            isinstance(self.memory_used_mb, (int, float)),
            "memory_used_mb must be numeric",
            self.memory_used_mb,
        )
        require(
            isinstance(self.open_file_handles, int),
            "open_file_handles must be int",
            self.open_file_handles,
        )


@dataclass(frozen=True)
class SimulationMetrics:
    """Point-in-time simulation subsystem measurements.

    Attributes:
        active_simulations: Number of currently running simulations.  ``-1``
            when the registry is unavailable.
        registered_engines: Total number of registered physics engine instances.
    """

    active_simulations: int = -1
    registered_engines: int = 0

    def __post_init__(self) -> None:
        require(
            isinstance(self.active_simulations, int),
            "active_simulations must be int",
            self.active_simulations,
        )
        require(
            isinstance(self.registered_engines, int) and self.registered_engines >= 0,
            "registered_engines must be a non-negative int",
            self.registered_engines,
        )


@dataclass(frozen=True)
class DiagnosticsSnapshot:
    """Immutable point-in-time diagnostics capture.

    Attributes:
        timestamp: UTC datetime when the snapshot was collected.
        system_metrics: CPU, memory, GPU, and file-handle measurements.
        simulation_metrics: Active simulation and engine counts.
        extra: Optional free-form metadata dict for extension points.
    """

    timestamp: datetime
    system_metrics: SystemMetrics
    simulation_metrics: SimulationMetrics
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            isinstance(self.timestamp, datetime),
            "timestamp must be a datetime",
            self.timestamp,
        )
        require(
            isinstance(self.system_metrics, SystemMetrics),
            "system_metrics must be a SystemMetrics instance",
            self.system_metrics,
        )
        require(
            isinstance(self.simulation_metrics, SimulationMetrics),
            "simulation_metrics must be a SimulationMetrics instance",
            self.simulation_metrics,
        )
        require(
            isinstance(self.extra, dict),
            "extra must be a dict",
            self.extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise snapshot to a plain dictionary (JSON-compatible).

        Returns:
            Dict with all fields converted to primitives.

        Postcondition:
            Result is JSON-serialisable.
        """
        d = asdict(self)
        # datetime is not JSON-serialisable by default — convert to ISO 8601
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self) -> str:
        """Serialise snapshot to a JSON string.

        Returns:
            JSON string representation of the snapshot.
        """
        return json.dumps(self.to_dict())
