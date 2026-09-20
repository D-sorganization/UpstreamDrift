"""Asynchronous engine runtime probe caching (MS-71 #10351, issue #8938).

Prevents blocking launcher GUI startup by caching runtime discovery results in
~/.upstream_drift/engine_probe.json and refreshing in a background worker thread.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

from src.launchers.launcher_provider_compatibility import is_engine_runtime_available

logger = logging.getLogger(__name__)

PROBE_CACHE_DIR = Path.home() / ".upstream_drift"
PROBE_CACHE_FILE = PROBE_CACHE_DIR / "engine_probe.json"

_MEM_CACHE: dict[str, bool] = {}
_CACHE_LOCK = threading.Lock()


def _read_disk_cache() -> dict[str, Any]:
    if not PROBE_CACHE_FILE.is_file():
        return {}
    try:
        data = json.loads(PROBE_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Invalidate if Python executable changed
            if data.get("__python_executable__") == sys.executable:
                return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed reading engine probe cache: %s", exc)
    return {}


def _write_disk_cache(data: dict[str, Any]) -> None:
    try:
        PROBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(data)
        payload["__python_executable__"] = sys.executable
        PROBE_CACHE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Failed writing engine probe cache: %s", exc)


def is_cached_engine_runtime_available(engine_type: str | None) -> bool:
    """Check engine runtime availability using disk/memory cache (non-blocking)."""
    if engine_type is None:
        return True

    key = engine_type.strip().lower()

    with _CACHE_LOCK:
        if key in _MEM_CACHE:
            return _MEM_CACHE[key]

        disk = _read_disk_cache()
        if key in disk and isinstance(disk[key], bool):
            _MEM_CACHE[key] = disk[key]
            return disk[key]

    # Cold cache: probe synchronously once and save
    avail = is_engine_runtime_available(engine_type)
    with _CACHE_LOCK:
        _MEM_CACHE[key] = avail
        disk = _read_disk_cache()
        disk[key] = avail
        _write_disk_cache(disk)
    return avail


def refresh_engine_probe_cache_async(
    engines: tuple[str, ...] = ("mujoco", "drake", "pinocchio", "opensim", "myosuite"),
) -> threading.Thread:
    """Trigger asynchronous background refresh of engine runtime cache."""

    def _worker() -> None:
        disk = _read_disk_cache()
        for eng in engines:
            avail = is_engine_runtime_available(eng)
            with _CACHE_LOCK:
                _MEM_CACHE[eng] = avail
                disk[eng] = avail
        with _CACHE_LOCK:
            _write_disk_cache(disk)

    thread = threading.Thread(target=_worker, name="EngineProbeWorker", daemon=True)
    thread.start()
    return thread
