"""Disk cache of built golfer scenes (#15150): a stale scene is worse than a
slow one, so any changed input, setting or source file misses."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any

import pytest

from src.tools.model_explorer import scene_cache as sc
from src.tools.model_explorer.default_match import DEMO_BUNDLE
from src.tools.model_explorer.results_loader_worker import prepare_results

pytestmark = pytest.mark.unit

SPEC = {"spec": b'{"bodies": []}', "appearance": b"{}"}


@pytest.fixture(autouse=True)
def _no_directory_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sc.SceneCacheSettings().directory_env, raising=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "posing.py").write_text("SOLVE = 1\n")
    (root / "grip_settings.json").write_text('{"gap_mm": 2}')
    return root


def _cache(tmp_path: Path, repo: Path, **overrides: Any) -> sc.SceneCache:
    settings = sc.SceneCacheSettings(directory=str(tmp_path / "cache"), **overrides)
    deps = [repo / "posing.py", repo / "grip_settings.json"]
    return sc.SceneCache(settings, repo_root=repo, dependency_paths=lambda: deps)


class _Builder:
    def __init__(self, xml: str = "<mujoco/>") -> None:
        self.calls, self.xml = 0, xml

    def __call__(self) -> str:
        self.calls += 1
        return self.xml


def test_same_inputs_hit_the_cache(tmp_path: Path, repo: Path) -> None:
    cache, build = _cache(tmp_path, repo), _Builder()
    assert cache.get_or_build(SPEC, build) == ("<mujoco/>", False)
    assert not cache.is_cached({**SPEC, "spec": b"other"})
    assert cache.is_cached(SPEC)
    assert cache.get_or_build(SPEC, build) == ("<mujoco/>", True)
    assert build.calls == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda repo: {**SPEC, "spec": b'{"bodies": [1]}'},  # a spec field
        lambda repo: (repo / "grip_settings.json").write_text('{"gap_mm": 3}'),
        lambda repo: (repo / "posing.py").write_text("SOLVE = 2\n"),  # source
        lambda repo: (repo / "posing.py").unlink(),
    ],
    ids=["spec", "setting", "source", "deleted-source"],
)
def test_any_changed_input_misses(tmp_path: Path, repo: Path, change: Any) -> None:
    cache, build = _cache(tmp_path, repo), _Builder()
    cache.get_or_build(SPEC, build)
    changed = change(repo)
    inputs = changed if isinstance(changed, dict) else SPEC
    assert not cache.is_cached(inputs)
    assert cache.get_or_build(inputs, build)[1] is False
    assert build.calls == 2


@pytest.mark.parametrize(
    "damage",
    [
        lambda p: p.write_bytes(b"not gzip"),
        lambda p: p.write_bytes(gzip.compress(b"{broken header\n<mujoco/>")),
        lambda p: p.write_bytes(gzip.compress(gzip.decompress(p.read_bytes())[:-3])),
        lambda p: p.write_bytes(gzip.compress(gzip.decompress(p.read_bytes()) + b"x")),
    ],
    ids=["not-gzip", "bad-header", "truncated", "xml-hash"],
)
def test_corrupt_entry_is_rebuilt(
    tmp_path: Path, repo: Path, damage: Any, caplog: pytest.LogCaptureFixture
) -> None:
    cache, build = _cache(tmp_path, repo), _Builder()
    cache.get_or_build(SPEC, build)
    damage(cache.entry_path(cache.entry_key(SPEC)))
    assert cache.get_or_build(SPEC, build) == ("<mujoco/>", False)
    assert build.calls == 2 and "corrupt scene cache entry" in caplog.text
    assert cache.get_or_build(SPEC, build)[1] is True


def test_writes_are_atomic(tmp_path: Path, repo: Path) -> None:
    cache = _cache(tmp_path, repo)

    def _fail(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sc.os, "replace", _fail)
        assert cache.get_or_build(SPEC, _Builder()) == ("<mujoco/>", False)
    assert not any(Path(tmp_path / "cache").iterdir())  # no partial, no temp
    cache.get_or_build(SPEC, _Builder())
    assert [p.suffix for p in (tmp_path / "cache").iterdir()] == [".scene"]


def test_least_recently_used_entries_are_evicted(tmp_path: Path, repo: Path) -> None:
    payload = "<m>" + "a" * 4000 + "</m>"
    probe = _cache(tmp_path / "probe", repo, compress_level=0)
    probe.get_or_build(SPEC, _Builder(payload))
    size = probe.entry_path(probe.entry_key(SPEC)).stat().st_size
    cache = _cache(tmp_path, repo, max_bytes=int(size * 2.5), compress_level=0)
    a, b, c = ({**SPEC, "spec": name.encode()} for name in "abc")
    for age, inputs in enumerate((a, b), start=1):
        cache.get_or_build(inputs, _Builder(payload))
        os.utime(cache.entry_path(cache.entry_key(inputs)), (age, age))
    assert cache.get_or_build(a, _Builder(payload))[1]  # a hit makes it recent
    cache.get_or_build(c, _Builder(payload))
    assert cache.is_cached(a) and cache.is_cached(c)
    assert not cache.is_cached(b)  # least recently used went first


def test_settings_key_and_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path
) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        sc.SceneCacheSettings(max_bytes=0)
    with pytest.raises(ValueError, match="compress_level"):
        sc.SceneCacheSettings(compress_level=10)
    other = _cache(tmp_path, repo, key_packages=("numpy", "pytest"))
    assert _cache(tmp_path, repo).entry_key(SPEC) != other.entry_key(SPEC)
    off, build = _cache(tmp_path, repo, enabled=False), _Builder()
    off.get_or_build(SPEC, build)
    assert off.get_or_build(SPEC, build)[1] is False and build.calls == 2
    settings = sc.load_scene_cache_settings()
    assert settings.cache_dir().name == "golfer_scenes"
    monkeypatch.setenv(settings.directory_env, str(tmp_path))
    assert settings.cache_dir() == tmp_path


def test_process_dependencies_cover_loaded_code_and_read_files() -> None:
    defaults = sc.REPO_ROOT / "src/config/appearance_defaults.json"
    defaults.read_bytes()  # any repo file read in-process becomes a dependency
    deps = set(sc.process_dependencies())
    assert Path(sc.__file__).resolve() in deps
    assert sc.SETTINGS_PATH.resolve() in deps  # data next to loaded modules
    assert defaults.resolve() in deps
    assert all(p.is_file() and p.is_relative_to(sc.REPO_ROOT) for p in deps)


def test_results_session_reuses_the_cached_scene(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path
) -> None:
    from src.tools.model_explorer import golfer_playback

    cache, build = _cache(tmp_path, repo), _Builder("<cached/>")
    monkeypatch.setattr(sc, "default_scene_cache", lambda: cache)
    monkeypatch.setattr(golfer_playback, "build_golfer_scene_xml", lambda _p: build())
    reports: list[str] = []
    for _ in range(2):
        session = prepare_results(DEMO_BUNDLE, None, reports.append)
        assert session.scene_xml == "<cached/>"
        session.close()
    assert build.calls == 1
    assert reports == [sc.SCENE_BUILDING_MESSAGE, sc.SCENE_CACHED_MESSAGE]
