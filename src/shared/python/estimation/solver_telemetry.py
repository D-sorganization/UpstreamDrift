"""Strict optional measurements; requested budgets never constitute telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any
import math


def _text(value: object, name: str, optional: bool = True) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class SolverBackend:
    """Actual backend, effective method and installed version, when available."""

    name: str
    method: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "name", False)
        _text(self.method, "method")
        _text(self.version, "version")


@dataclass(frozen=True)
class SolverTelemetry:
    """Measured native-Python integer counters and separately scoped durations."""

    nfev: int | None = None
    njev: int | None = None
    solver_elapsed_s: float | None = None
    worker_elapsed_s: float | None = None
    termination_reason: str | None = None
    backend: SolverBackend | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("nfev", "njev"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative native Python integer")
        for name in ("solver_elapsed_s", "worker_elapsed_s"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite nonnegative seconds")
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.backend is not None and not isinstance(self.backend, SolverBackend):
            raise ValueError("backend must be a typed SolverBackend")
        _text(self.termination_reason, "termination_reason")
        _text(self.unavailable_reason, "unavailable_reason")

    def to_record(self) -> dict[str, Any]:
        """Serialize actual measurements with explicit constant timing scopes."""
        return {
            "schema": "upstreamdrift/solver-telemetry/1",
            **asdict(self),
            "solver_scope": "backend_invocation_only",
            "worker_scope": (
                "child_entry_through_computed_payload_or_caught_failure"
                "_excluding_imports_telemetry_publication_and_transport_serialization"
            ),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any] | None) -> SolverTelemetry:
        """Decode strict new records; absent legacy evidence remains unavailable."""
        if record is None:
            return cls(unavailable_reason="legacy_record_has_no_telemetry")
        if not isinstance(record, Mapping):
            raise ValueError("Telemetry record must be a mapping")
        fields = {
            "nfev",
            "njev",
            "solver_elapsed_s",
            "worker_elapsed_s",
            "termination_reason",
            "backend",
            "unavailable_reason",
        }
        constants = cls().to_record()
        if set(record) - fields - {"schema", "solver_scope", "worker_scope"}:
            raise ValueError("Unknown telemetry record fields")
        if set(record) != fields | {"schema", "solver_scope", "worker_scope"}:
            raise ValueError("Missing telemetry record fields")
        for name in ("schema", "solver_scope", "worker_scope"):
            if record.get(name) != constants[name]:
                raise ValueError(f"Unsupported telemetry {name}")
        values = {name: record.get(name) for name in fields}
        backend = values["backend"]
        if backend is not None:
            if not isinstance(backend, Mapping) or set(backend) != {
                "name",
                "method",
                "version",
            }:
                raise ValueError("Malformed backend record")
            values["backend"] = SolverBackend(**backend)
        return cls(**values)
