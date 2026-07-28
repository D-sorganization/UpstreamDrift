"""Environment resolution for UpstreamDrift API startup gates."""

from __future__ import annotations

import os

_ENVIRONMENT_VARS: tuple[str, ...] = ("ENVIRONMENT", "UPSTREAM_DRIFT_ENV")


def _normalize_environment(value: str) -> str:
    """Normalize a raw environment name to its canonical form."""
    env = value.strip().lower()
    if env in ("dev", "local"):
        return "development"
    if env in ("stage", "test", "testing"):
        return "staging"
    if env in ("prod", "live"):
        return "production"
    return env


def resolve_environment() -> str:
    """Resolve the deployment environment without caching.

    ``ENVIRONMENT`` is the documented spelling; ``UPSTREAM_DRIFT_ENV`` is the
    legacy spelling. The resolver fails closed when either variable names
    production so split configuration cannot disable production-only gates.
    """
    values = [
        _normalize_environment(raw)
        for var in _ENVIRONMENT_VARS
        if (raw := os.environ.get(var, "")).strip()
    ]
    if any(value == "production" for value in values):
        return "production"
    return values[0] if values else "development"


def is_production_environment() -> bool:
    """Return whether startup is running under production semantics."""
    return resolve_environment() == "production"
