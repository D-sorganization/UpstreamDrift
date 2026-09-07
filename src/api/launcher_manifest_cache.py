"""Process-level cache for the launcher manifest (issue #8937).

``LauncherManifest.load()`` re-parses ``launcher_manifest.json`` and
``models.yaml`` and probes provider availability (which historically spawned
dozens of git subprocesses) on every call.  The API layer previously had two
divergent strategies: ``src/api/routes/launcher.py`` cached the manifest
forever, while ``src/api/local_server.py`` reloaded it uncached inside async
handlers, stalling the event loop.

This module is the single shared caching seam for both call sites:

* The cached path IS the uncached path plus a cache — ``get_cached_manifest``
  always delegates to ``LauncherManifest.load`` on a miss, so cached and
  uncached results cannot diverge.
* The cache key is the (mtime, size) signature of the manifest and registry
  files, so edits on disk invalidate the cache without a process restart.
* ``get_cached_manifest_async`` offloads any residual load work to a worker
  thread via ``anyio.to_thread.run_sync`` so async handlers never block the
  event loop.
"""

from __future__ import annotations

import threading
from pathlib import Path
from collections.abc import Callable

import anyio.to_thread

import src.config.launcher_manifest_loader as _loader_module
from src.config.launcher_manifest_loader import LauncherManifest

_CacheKey = tuple[tuple[float, int], ...]

_lock = threading.Lock()
_state: dict[str, object] = {"key": None, "manifest": None}


def _watched_paths() -> tuple[Path, ...]:
    """Files whose on-disk changes must invalidate the cached manifest.

    Only the single registry (``models.yaml``) is read at runtime (#9412);
    the generated ``launcher_manifest.json`` is not an input any more.
    """
    return (_loader_module.REGISTRY_PATH,)


def _file_signature(path: Path) -> tuple[float, int]:
    """Return an (mtime, size) signature; missing files get a sentinel."""
    try:
        stat = path.stat()
    except OSError:
        return (-1.0, -1)
    return (stat.st_mtime, stat.st_size)


def _cache_key() -> _CacheKey:
    return tuple(_file_signature(path) for path in _watched_paths())


def invalidate_manifest_cache() -> None:
    """Drop the cached manifest so the next access reloads from disk."""
    with _lock:
        _state["key"] = None
        _state["manifest"] = None


def get_cached_manifest(
    *,
    loader: Callable[[], LauncherManifest] = LauncherManifest.load,
) -> LauncherManifest:
    """Return the launcher manifest, reloading only when its inputs change.

    Args:
        loader: Seam for tests; defaults to :meth:`LauncherManifest.load`.
            The cached value is exactly one prior return value of ``loader``.

    Returns:
        The loaded (possibly cached) :class:`LauncherManifest`.

    Raises:
        Whatever ``loader`` raises on a cache miss (e.g. ``FileNotFoundError``
        or ``ValueError``); failures are never cached.
    """
    if not callable(loader):
        raise TypeError("loader must be callable")
    key = _cache_key()
    with _lock:
        cached = _state["manifest"]
        if _state["key"] == key and cached is not None:
            assert isinstance(cached, LauncherManifest)
            return cached
    manifest = loader()
    with _lock:
        _state["key"] = key
        _state["manifest"] = manifest
    return manifest


async def get_cached_manifest_async(
    *,
    loader: Callable[[], LauncherManifest] = LauncherManifest.load,
) -> LauncherManifest:
    """Async variant that runs any cache-miss load off the event loop."""

    def _load() -> LauncherManifest:
        return get_cached_manifest(loader=loader)

    manifest: object = await anyio.to_thread.run_sync(_load)
    if not isinstance(manifest, LauncherManifest):
        raise TypeError("manifest loader returned an invalid result")
    return manifest
