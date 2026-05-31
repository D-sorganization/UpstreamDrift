"""Canonical provenance stamp helpers for simulation run artifacts.

The stamp is intentionally caller-clocked: ``created_at`` has no default so
tests and orchestrators can make run metadata deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.shared.python.engine_core.checkpoint import StateCheckpoint

    from .protocol import BatchTrace, Trace

PROVENANCE_META_KEY = "provenance"
PROVENANCE_FLAT_PREFIX = "provenance_"


@dataclass(frozen=True)
class ProvenanceStamp:
    """Immutable metadata describing how a simulation artifact was produced."""

    engine: str
    engine_version: str
    model_hash: str
    param_hash: str
    git_commit: str
    solver_settings: Mapping[str, object]
    seed: int | None
    created_at: str
    convention: str
    frame: str
    units: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze nested mappings and reject missing deterministic fields."""
        for name in (
            "engine",
            "engine_version",
            "model_hash",
            "param_hash",
            "git_commit",
            "created_at",
            "convention",
            "frame",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "solver_settings",
            MappingProxyType(_freeze_mapping(self.solver_settings)),
        )
        object.__setattr__(self, "units", MappingProxyType(_freeze_mapping(self.units)))

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "model_hash": self.model_hash,
            "param_hash": self.param_hash,
            "git_commit": self.git_commit,
            "solver_settings": _thaw_mapping(self.solver_settings),
            "seed": self.seed,
            "created_at": self.created_at,
            "convention": self.convention,
            "frame": self.frame,
            "units": _thaw_mapping(self.units),
        }

    def to_flat_meta(self, prefix: str = PROVENANCE_FLAT_PREFIX) -> dict[str, object]:
        """Return scalar trace metadata keys for existing trace persistence."""
        return {
            f"{prefix}engine": self.engine,
            f"{prefix}engine_version": self.engine_version,
            f"{prefix}model_hash": self.model_hash,
            f"{prefix}param_hash": self.param_hash,
            f"{prefix}git_commit": self.git_commit,
            f"{prefix}solver_settings": _canonical_mapping_text(self.solver_settings),
            f"{prefix}seed": "" if self.seed is None else self.seed,
            f"{prefix}created_at": self.created_at,
            f"{prefix}convention": self.convention,
            f"{prefix}frame": self.frame,
            f"{prefix}units": _canonical_mapping_text(self.units),
        }


def serialize_provenance(stamp: ProvenanceStamp) -> dict[str, object]:
    """Serialize ``stamp`` under the canonical provenance metadata key."""
    return {PROVENANCE_META_KEY: stamp.to_dict()}


def attach_provenance_to_trace(
    trace: Trace | BatchTrace,
    stamp: ProvenanceStamp,
    *,
    flatten: bool = True,
) -> Trace | BatchTrace:
    """Return a trace copy with provenance added to ``meta``.

    ``flatten=True`` stores scalar ``provenance_*`` keys so the current HDF5
    trace writer can persist the stamp without changing the on-disk schema.
    """
    meta = dict(trace.meta)
    meta.update(stamp.to_flat_meta() if flatten else serialize_provenance(stamp))
    return replace(trace, meta=meta)


def attach_provenance_to_checkpoint(
    checkpoint: StateCheckpoint,
    stamp: ProvenanceStamp,
) -> StateCheckpoint:
    """Return a checkpoint copy with nested provenance in ``metadata``."""
    metadata = dict(checkpoint.metadata)
    metadata.update(serialize_provenance(stamp))
    return replace(checkpoint, metadata=metadata)


def _freeze_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    """Copy mappings recursively with deterministic key order."""
    return {
        str(key): _freeze_value(mapping[key])
        for key in sorted(mapping.keys(), key=lambda item: str(item))
    }


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(_freeze_mapping(value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    return {key: _thaw_value(value) for key, value in mapping.items()}


def _thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _thaw_mapping(value)
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _canonical_mapping_text(mapping: Mapping[str, object]) -> str:
    items = ", ".join(
        f"{key}={_canonical_value_text(value)}" for key, value in mapping.items()
    )
    return "{" + items + "}"


def _canonical_value_text(value: object) -> str:
    if isinstance(value, Mapping):
        return _canonical_mapping_text(value)
    if isinstance(value, tuple):
        return "[" + ", ".join(_canonical_value_text(item) for item in value) + "]"
    return repr(value)
