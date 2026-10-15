"""Service layer for Closed-Loop Null-Space Exploration (NSX-40, #12572).

Composes the NSX pipeline strictly through :mod:`src.shared.python.loop_nullspace.facade`
(Law of Demeter: zero math in the service layer). Provides content-addressed
result caching by ``compute_cache_key``, progressive invariant reporting before
envelopes/allocation settle, and cooperative job cancellation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import threading
import uuid
from typing import Any

from src.shared.python.loop_nullspace.facade import (
    analyze as default_analyze,
    compute_cache_key,
    compute_partial_invariants,
    default_fixture_trajectory,
)

__all__ = ["LoopNullspaceService"]

AnalyzerCallable = Callable[..., dict[str, Any]]


def _extract_request_fields(
    request: Any,
) -> tuple[Any, str, str, dict[str, Any]]:
    if request is None:
        raise ValueError("request must not be None")
    if isinstance(request, Mapping):
        trajectory = request.get("trajectory") or default_fixture_trajectory()
        engine = str(request.get("engine", "pinocchio"))
        preset = str(request.get("preset", "capacity_min_effort"))
        options = dict(request.get("options") or {})
        return trajectory, engine, preset, options
    trajectory = getattr(request, "trajectory", None) or default_fixture_trajectory()
    engine = str(getattr(request, "engine", "pinocchio"))
    preset = str(getattr(request, "preset", "capacity_min_effort"))
    options = dict(getattr(request, "options", None) or {})
    return trajectory, engine, preset, options


class LoopNullspaceService:
    """Service managing NSX analysis jobs and content-addressed result cache."""

    def __init__(self, *, analyzer: AnalyzerCallable | None = None) -> None:
        """Initialize the service with an optional injectable facade analyzer."""
        self._analyzer: AnalyzerCallable = analyzer or default_analyze
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    def cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss counters and entry count."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._cache),
            }

    def run(self, request: Any) -> dict[str, Any]:
        """Run an NSX analysis synchronously, serving from cache on key match."""
        trajectory, engine, preset, options = _extract_request_fields(request)
        key = compute_cache_key(
            trajectory, engine=engine, preset=preset, options=options
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                return copy.deepcopy(cached)
            self._misses += 1

        report = self._analyzer(
            trajectory, engine=engine, preset=preset, options=options
        )
        with self._lock:
            self._cache[key] = copy.deepcopy(report)
        return copy.deepcopy(report)

    def create_job(self, request: Any) -> str:
        """Create an async/cancellable NSX job and return its ``job_id``.

        Populates ``partial_invariants`` immediately so polling callers can
        render allocation-invariant curves before envelope/allocation completes.
        """
        trajectory, engine, preset, options = _extract_request_fields(request)
        key = compute_cache_key(
            trajectory, engine=engine, preset=preset, options=options
        )
        job_id = str(uuid.uuid4())
        partials = compute_partial_invariants(
            trajectory, engine=engine, options=options
        )

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and not options.get("defer_completion", False):
                self._hits += 1
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "status": "completed",
                    "progress_percent": 100.0,
                    "compute_cache_key": key,
                    "cache_hit": True,
                    "partial_invariants": partials,
                    "report": copy.deepcopy(cached),
                    "error": None,
                }
                return job_id

            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "progress_percent": 35.0,
                "compute_cache_key": key,
                "cache_hit": False,
                "partial_invariants": partials,
                "report": None,
                "error": None,
                "_request": (trajectory, engine, preset, options),
            }

        if not options.get("defer_completion", False):
            self._settle_job(job_id)
        return job_id

    def _settle_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "running":
                return
            trajectory, engine, preset, options = job["_request"]
            key = job["compute_cache_key"]

        try:
            report = self._analyzer(
                trajectory, engine=engine, preset=preset, options=options
            )
        except (ValueError, RuntimeError, TypeError) as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None and job["status"] == "running":
                    job["status"] = "failed"
                    job["error"] = str(exc)
            return

        with self._lock:
            self._misses += 1
            self._cache[key] = copy.deepcopy(report)
            job = self._jobs.get(job_id)
            if job is not None and job["status"] == "running":
                job["status"] = "completed"
                job["progress_percent"] = 100.0
                job["report"] = copy.deepcopy(report)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return a snapshot of the job status, partial invariants, or completed report."""
        if not job_id or not str(job_id).strip():
            raise ValueError("job_id must be a non-empty string")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown NSX job_id: {job_id}")
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "progress_percent": float(job["progress_percent"]),
                "compute_cache_key": job["compute_cache_key"],
                "cache_hit": bool(job["cache_hit"]),
                "partial_invariants": copy.deepcopy(job["partial_invariants"]),
                "report": copy.deepcopy(job["report"]),
                "error": job["error"],
            }

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a job and set its status to ``'cancelled'``."""
        if not job_id or not str(job_id).strip():
            raise ValueError("job_id must be a non-empty string")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown NSX job_id: {job_id}")
            if job["status"] in {"pending", "running"}:
                job["status"] = "cancelled"
                job["report"] = None
            return self.get_job(job_id)
