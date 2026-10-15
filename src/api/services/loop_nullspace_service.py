"""Service layer of the loop-nullspace report (NSX-40, #12572).

``run(request) -> report`` resolves the request's registered trajectory source,
calls the one facade entry point ``loop_nullspace.facade.analyze`` and caches
the report.  There is no mathematics here and the chain stops at the facade
(service -> facade).

Trajectory sources
------------------

A request names a **server-side trajectory source** (``source_id``) and the
frame indices to analyse; arrays are never uploaded.  A source implements
:class:`TrajectorySource`: ``describe`` returns the cheap provenance the cache
key needs without building anything, and ``load`` builds the snapshots and the
engine context.  Sources are registered with :func:`register_source`; none is
shipped enabled, so a deployment (or a test) opts each in explicitly.

Result cache
------------

The key is the SHA-256 of the canonical JSON of (trajectory input provenance
and frame indices, engine and model provenance, registry version, preset,
options).  A change of any of them -- including the loop-nullspace registry
version -- is a miss; a failed or cancelled run is never cached.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.shared.python.loop_nullspace import facade, registry
from src.shared.python.loop_nullspace.facade import (
    AnalysisOptions,
    EngineContext,
    TrajectoryInput,
)

__all__ = [
    "MAX_FRAMES",
    "RESULT_CACHE",
    "LoopNullspaceOptions",
    "LoopNullspaceRequest",
    "RequestRefusedError",
    "ResultCache",
    "SourceDescription",
    "TrajectorySource",
    "cache_key",
    "lookup",
    "register_source",
    "registered_sources",
    "run",
    "unregister_source",
    "validate",
]

MAX_FRAMES = 400
"""Upper bound on the frames of one request (the NSX-22 benchmark uses 200)."""
CACHE_ENTRIES = 32


class RequestRefusedError(ValueError):
    """A request the service refuses up front; ``code`` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LoopNullspaceOptions(BaseModel):
    """What to compute (mirrors :class:`AnalysisOptions`)."""

    model_config = ConfigDict(extra="forbid")

    metric: str = "identity"
    include_envelopes: bool = True
    include_allocation: bool = False
    effort_weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    rate_weight: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    def to_analysis_options(self) -> AnalysisOptions:
        """The facade options (``ValueError`` on an invalid combination)."""
        return AnalysisOptions(**self.model_dump())


class LoopNullspaceRequest(BaseModel):
    """A job request: a registered source, its frames, the preset and options."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=128)
    engine: str = Field(default="pinocchio", min_length=1, max_length=64)
    preset: str = Field(min_length=1, max_length=128)
    frame_indices: list[int] = Field(min_length=1)
    options: LoopNullspaceOptions = Field(default_factory=LoopNullspaceOptions)

    @field_validator("frame_indices")
    @classmethod
    def _frames(cls, value: list[int]) -> list[int]:
        if len(value) > MAX_FRAMES:
            raise ValueError(f"at most {MAX_FRAMES} frames per request")
        if min(value) < 0:
            raise ValueError("frame indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("frame indices must be unique")
        return value


class SourceDescription(NamedTuple):
    """The provenance that keys the cache, known before anything is built."""

    input_provenance: Mapping[str, Any]
    model_provenance: Mapping[str, Any]


class TrajectorySource(Protocol):
    """A server-side trajectory (and engine) the API can analyse by id."""

    def describe(self, request: LoopNullspaceRequest) -> SourceDescription:
        """Cheap provenance of the inputs; raises ``ValueError`` if unsupported."""
        ...

    def load(
        self, request: LoopNullspaceRequest
    ) -> tuple[TrajectoryInput, EngineContext]:
        """Build the snapshots of ``request.frame_indices`` and the engine."""
        ...


_SOURCES: dict[str, TrajectorySource] = {}
_SOURCES_LOCK = threading.Lock()


def register_source(source_id: str, source: TrajectorySource) -> None:
    """Make ``source`` available under ``source_id`` (replaces an earlier one)."""
    if not source_id:
        raise ValueError("source_id must be non-empty")
    with _SOURCES_LOCK:
        _SOURCES[source_id] = source


def unregister_source(source_id: str) -> None:
    """Remove a source (no error if absent)."""
    with _SOURCES_LOCK:
        _SOURCES.pop(source_id, None)


def registered_sources() -> tuple[str, ...]:
    """The ids of the registered sources, sorted."""
    with _SOURCES_LOCK:
        return tuple(sorted(_SOURCES))


def _source(source_id: str) -> TrajectorySource:
    with _SOURCES_LOCK:
        source = _SOURCES.get(source_id)
    if source is None:
        raise RequestRefusedError(
            "unknown_source",
            f"no trajectory source {source_id!r}; registered: {list(registered_sources())}",
        )
    return source


class ResultCache:
    """A small thread-safe LRU of finished reports keyed by their digest."""

    def __init__(self, max_entries: int = CACHE_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("a cache needs at least one entry")
        self._max = max_entries
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        """The cached report, refreshing its recency, or ``None``."""
        with self._lock:
            report = self._items.get(key)
            if report is not None:
                self._items.move_to_end(key)
            return report

    def put(self, key: str, report: dict[str, Any]) -> None:
        """Store ``report``, evicting the least recently used entry."""
        with self._lock:
            self._items[key] = report
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._items.clear()


RESULT_CACHE = ResultCache()


def _registry_version() -> str:
    return registry.load_registry().registry_version


def validate(request: LoopNullspaceRequest) -> SourceDescription:
    """Refuse a request that could not run, before any job exists.

    Raises:
        RequestRefusedError: ``unknown_source``, ``unknown_preset``,
            ``unsupported_request`` (the source rejects the engine/frames) or
            ``invalid_options``.
    """
    source = _source(request.source_id)
    try:
        registry.load_registry().preset(request.preset)
    except KeyError as exc:
        raise RequestRefusedError(
            "unknown_preset", f"unknown preset {request.preset!r}"
        ) from exc
    try:
        request.options.to_analysis_options()
    except ValueError as exc:
        raise RequestRefusedError("invalid_options", str(exc)) from exc
    try:
        return source.describe(request)
    except ValueError as exc:
        raise RequestRefusedError("unsupported_request", str(exc)) from exc


def cache_key(request: LoopNullspaceRequest) -> str:
    """Digest of (trajectory provenance, engine/model provenance, registry, preset, options)."""
    description = validate(request)
    payload = {
        "trajectory": {
            "provenance": dict(description.input_provenance),
            "frame_indices": list(request.frame_indices),
            "source_id": request.source_id,
        },
        "engine": {
            "name": request.engine,
            "model_provenance": dict(description.model_provenance),
        },
        "registry_version": _registry_version(),
        "preset": request.preset,
        "options": request.options.model_dump(mode="json"),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lookup(request: LoopNullspaceRequest) -> dict[str, Any] | None:
    """The cached report for ``request``, or ``None``."""
    return RESULT_CACHE.get(cache_key(request))


def run(
    request: LoopNullspaceRequest,
    *,
    on_stage: Callable[[str, float, Mapping[str, Any] | None], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Build the request's inputs, run the facade and return the cached report.

    A cache hit returns the stored report without building or analysing
    anything.  ``AnalysisCancelledError`` and every facade error propagate and
    leave the cache untouched.

    Raises:
        RequestRefusedError: See :func:`validate`.
    """
    key = cache_key(request)
    hit = RESULT_CACHE.get(key)
    if hit is not None:
        return hit
    trajectory, engine = _source(request.source_id).load(request)
    report = facade.analyze(
        trajectory,
        engine,
        request.preset,
        request.options.to_analysis_options(),
        on_stage=on_stage,
        should_cancel=should_cancel,
    )
    report["service"] = {"cache_key": key, "source_id": request.source_id}
    RESULT_CACHE.put(key, report)
    return report
