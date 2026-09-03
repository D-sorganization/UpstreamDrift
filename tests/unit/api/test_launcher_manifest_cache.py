"""Tests for the shared launcher-manifest process cache (issue #8937).

Covers:
* git-subprocess storm elimination (authority memoization),
* cached == uncached result equality (one code path, DRY),
* mtime-based invalidation,
* async offloading so the event loop is never blocked by a manifest load.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.launcher_manifest_cache as manifest_cache
import src.config.launcher_manifest_loader as loader_module
import src.shared.python.config.tools_vendor_authority as authority_module
from src.api.routes.launcher import router
from src.config.launcher_manifest_loader import LauncherManifest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    """Each test starts with empty manifest and authority caches."""
    manifest_cache.invalidate_manifest_cache()
    authority_module.clear_tools_vendor_authority_cache()


@pytest.fixture
def git_spawn_counter(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Count git subprocess spawns made by the vendor-authority probe."""
    calls: list[list[str]] = []
    real_run = subprocess.run

    def counting_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "git":
            calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(authority_module.subprocess, "run", counting_run)
    return calls


def test_authority_inspection_memoized_per_root(
    git_spawn_counter: list[list[str]],
) -> None:
    """Repeat inspections of one repo root spawn zero additional git procs."""
    first = authority_module.inspect_tools_vendor_authority(REPO_ROOT)
    first_count = len(git_spawn_counter)
    second = authority_module.inspect_tools_vendor_authority(REPO_ROOT)

    assert len(git_spawn_counter) == first_count, (
        "second inspection spawned git subprocesses despite memoization"
    )
    assert second == first


def test_manifest_load_spawns_at_most_one_git_batch(
    git_spawn_counter: list[list[str]],
) -> None:
    """N manifest loads spawn at most one bounded git batch total."""
    LauncherManifest.load()
    first_count = len(git_spawn_counter)
    # The authority probe runs at most 7 fixed git queries once per process.
    assert first_count <= 7

    LauncherManifest.load()
    manifest_cache.get_cached_manifest()
    assert len(git_spawn_counter) == first_count, (
        "subsequent manifest loads re-spawned git subprocesses"
    )


def test_cached_manifest_equals_uncached_load() -> None:
    """The cached path is the uncached path + cache: results are identical."""
    cached = manifest_cache.get_cached_manifest()
    uncached = LauncherManifest.load()

    assert cached.to_dict(include_hidden=True) == uncached.to_dict(include_hidden=True)


def test_cache_hit_skips_loader_until_invalidated() -> None:
    """A warm cache never re-invokes the loader; invalidation forces reload."""
    load_count = 0

    def counting_loader() -> LauncherManifest:
        nonlocal load_count
        load_count += 1
        return LauncherManifest(version="1.0.0", tiles=())

    manifest_cache.get_cached_manifest(loader=counting_loader)
    manifest_cache.get_cached_manifest(loader=counting_loader)
    assert load_count == 1

    manifest_cache.invalidate_manifest_cache()
    manifest_cache.get_cached_manifest(loader=counting_loader)
    assert load_count == 2


def test_cache_invalidated_by_watched_file_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Touching the manifest or registry file on disk invalidates the cache."""
    fake_manifest = tmp_path / "launcher_manifest.json"
    fake_registry = tmp_path / "models.yaml"
    fake_manifest.write_text("{}", encoding="utf-8")
    fake_registry.write_text("models: []", encoding="utf-8")
    monkeypatch.setattr(loader_module, "MANIFEST_PATH", fake_manifest)
    monkeypatch.setattr(loader_module, "REGISTRY_PATH", fake_registry)

    load_count = 0

    def counting_loader() -> LauncherManifest:
        nonlocal load_count
        load_count += 1
        return LauncherManifest(version="1.0.0", tiles=())

    manifest_cache.get_cached_manifest(loader=counting_loader)
    manifest_cache.get_cached_manifest(loader=counting_loader)
    assert load_count == 1

    stamp = time.time() + 10
    # launcher_manifest.json is a generated projection (#9412); touching it
    # must NOT invalidate the cache - only the single registry is an input.
    os.utime(fake_manifest, (stamp, stamp))
    manifest_cache.get_cached_manifest(loader=counting_loader)
    assert load_count == 1

    os.utime(fake_registry, (stamp, stamp))
    manifest_cache.get_cached_manifest(loader=counting_loader)
    assert load_count == 2


def test_loader_must_be_callable() -> None:
    with pytest.raises(TypeError, match="loader must be callable"):
        manifest_cache.get_cached_manifest(loader="nope")  # type: ignore[arg-type]


def test_async_load_runs_off_the_event_loop() -> None:
    """A cache-miss load executes on a worker thread, not the loop thread."""
    loader_thread_ids: list[int] = []

    def recording_loader() -> LauncherManifest:
        loader_thread_ids.append(threading.get_ident())
        return LauncherManifest(version="1.0.0", tiles=())

    async def scenario() -> int:
        await manifest_cache.get_cached_manifest_async(loader=recording_loader)
        return threading.get_ident()

    loop_thread_id = asyncio.run(scenario())

    assert loader_thread_ids, "loader was never invoked"
    assert loader_thread_ids[0] != loop_thread_id


def test_manifest_endpoint_uses_shared_cache(
    git_spawn_counter: list[list[str]],
) -> None:
    """Repeated GET /launcher/manifest requests reuse one loaded manifest."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    first = client.get("/launcher/manifest")
    assert first.status_code == 200
    spawns_after_first = len(git_spawn_counter)

    second = client.get("/launcher/manifest")
    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(git_spawn_counter) == spawns_after_first, (
        "second manifest request spawned additional git subprocesses"
    )
