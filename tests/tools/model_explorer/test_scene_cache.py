"""Disk cache of built golfer scenes (#15150): a stale scene is worse than a
slow one, so any changed input, setting, source or data file misses."""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from src.tools.model_explorer import scene_cache as sc

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
SPEC = {"spec": b'{"bodies": []}', "appearance": b"{}"}
GLOBS = {"extra_dependency_globs": ("assets/*.npz",)}
FILES = {"pkg/posing.py": "1", "pkg/grip.json": "{}", "assets/head.npz": "m"}


@pytest.fixture(autouse=True)
def _tmp_user_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never touch the real user cache, even through ``default_scene_cache``."""
    monkeypatch.setenv(sc.SceneCacheSettings().directory_env, str(tmp_path / "uc"))
    sc.default_scene_cache.cache_clear()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake repository: a loaded module, its data, a declared file, a venv."""
    root = tmp_path / "repo"
    for rel, text in {**FILES, ".venv/numpy.py": ""}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for name in ("pkg/posing.py", ".venv/numpy.py"):
        module = types.ModuleType(f"fake_{name}")
        module.__file__ = str(root / name)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    return root


def _cache(tmp_path: Path, repo: Path, **overrides: Any) -> sc.SceneCache:
    overrides = {**GLOBS, **overrides}
    where = {"directory": str(tmp_path / "cache"), "directory_env": "UD_UNSET_X"}
    return sc.SceneCache(sc.SceneCacheSettings(**where, **overrides), repo_root=repo)


def _builder(xml: str = "<mujoco/>") -> Mock:
    return Mock(return_value=xml)


def test_same_inputs_hit_the_cache(tmp_path: Path, repo: Path) -> None:
    cache, build = _cache(tmp_path, repo), _builder()
    assert cache.get_or_build(SPEC, build) == ("<mujoco/>", False)
    assert not cache.is_cached({**SPEC, "spec": b"other"})
    assert cache.is_cached(SPEC)
    assert cache.get_or_build(SPEC, build) == ("<mujoco/>", True)
    assert build.call_count == 1


def test_manifest_is_loaded_modules_siblings_and_declared_files(repo: Path) -> None:
    settings = sc.SceneCacheSettings(sibling_data_suffixes=(".json",), **GLOBS)
    deps = sc.process_dependencies(repo, settings)
    assert {p.relative_to(repo).as_posix() for p in deps} == set(FILES)


@pytest.mark.parametrize(
    "change",
    [
        lambda repo: {**SPEC, "spec": b'{"bodies": [1]}'},
        lambda repo: (repo / "pkg/grip.json").write_text('{"gap_mm": 3}'),
        lambda repo: (repo / "pkg/posing.py").write_text("2"),
        lambda repo: (repo / "assets/head.npz").write_text("other mesh"),
        lambda repo: (repo / "pkg/posing.py").unlink(),
        lambda repo: (repo / "pkg/grip.json").unlink(),
    ],
    ids=["spec", "sibling-data", "module", "declared", "deleted-mod", "deleted-data"],
)
def test_any_changed_input_misses(tmp_path: Path, repo: Path, change: Any) -> None:
    cache, build = _cache(tmp_path, repo), _builder()
    cache.get_or_build(SPEC, build)
    changed = change(repo)
    inputs = changed if isinstance(changed, dict) else SPEC
    assert not cache.is_cached(inputs)
    assert cache.get_or_build(inputs, build)[1] is False
    assert build.call_count == 2


def test_corrupt_entry_is_a_logged_miss(
    tmp_path: Path, repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache, build = _cache(tmp_path, repo), _builder()
    path = cache.entry_path(cache.entry_key(SPEC))
    for damage in ("not-gzip", "truncated"):
        cache.get_or_build(SPEC, build)
        body = gzip.decompress(path.read_bytes())[:-3]  # XML hash no longer matches
        path.write_bytes(b"not gzip" if damage == "not-gzip" else gzip.compress(body))
        assert cache.get_or_build(SPEC, build) == ("<mujoco/>", False)
    assert build.call_count == 3 and "corrupt scene cache entry" in caplog.text


def test_writes_are_atomic(
    tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cache(tmp_path, repo)
    with monkeypatch.context() as mp:
        mp.setattr(sc.os, "replace", lambda *_: (_ for _ in ()).throw(OSError()))
        assert cache.get_or_build(SPEC, _builder()) == ("<mujoco/>", False)
    assert not any((tmp_path / "cache").iterdir())  # no partial, no temp
    cache.get_or_build(SPEC, _builder())
    assert [p.suffix for p in (tmp_path / "cache").iterdir()] == [".scene"]


def test_least_recently_used_entries_are_evicted(tmp_path: Path, repo: Path) -> None:
    payload, (a, b, c) = "a" * 4000, ({"spec": name.encode()} for name in "abc")
    probe = _cache(tmp_path / "probe", repo, compress_level=0)
    probe.get_or_build(a, _builder(payload))
    size = probe.entry_path(probe.entry_key(a)).stat().st_size
    cache = _cache(tmp_path, repo, max_bytes=int(size * 2.5), compress_level=0)
    for age, inputs in enumerate((a, b), start=1):
        cache.get_or_build(inputs, _builder(payload))
        os.utime(cache.entry_path(cache.entry_key(inputs)), (age, age))
    assert cache.get_or_build(a, _builder(payload))[1]  # a hit makes it recent
    cache.get_or_build(c, _builder(payload))
    assert cache.is_cached(a) and cache.is_cached(c) and not cache.is_cached(b)


def test_settings_are_validated_and_configurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path
) -> None:
    for bad in ({"max_bytes": 0}, {"compress_level": 10}):
        with pytest.raises(ValueError):
            sc.SceneCacheSettings(**bad)
    for glob in ("/etc/*", "../x/*"):
        with pytest.raises(ValueError, match="extra_dependency_globs"):
            sc.SceneCacheSettings(extra_dependency_globs=(glob,))
    other = _cache(tmp_path, repo, key_packages=("numpy", "pytest"))
    assert _cache(tmp_path, repo).entry_key(SPEC) != other.entry_key(SPEC)
    off, build = _cache(tmp_path, repo, enabled=False), _builder()
    off.get_or_build(SPEC, build)
    assert off.get_or_build(SPEC, build)[1] is False and build.call_count == 2
    settings = sc.load_scene_cache_settings()
    assert settings.cache_dir() == tmp_path / "uc"
    monkeypatch.delenv(settings.directory_env)
    assert settings.cache_dir().name == "golfer_scenes"
    assert all(any(REPO.glob(g)) for g in settings.extra_dependency_globs)


def test_cached_scene_is_byte_identical_to_a_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tools.model_explorer import golfer_playback as golfer
    from src.tools.model_explorer.results_loader_worker import prepare_results

    run = tmp_path / "fit"
    shutil.copytree(REPO / "tests/fixtures/results_playback/mujoco_fit", run)
    spec = REPO / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    (run / "full_body_spec_hipcal_scaled.json").write_bytes(spec.read_bytes())
    real_build, builds, reports = golfer.build_golfer_scene_xml, [], []
    monkeypatch.setattr(
        golfer, "build_golfer_scene_xml", lambda p: builds.append(1) or real_build(p)
    )
    cold, warm = (
        prepare_results(run / "dynamics_record.npz", None, reports.append)
        for _ in range(2)
    )
    assert warm.scene_xml == cold.scene_xml == real_build(cold.pose_source)
    expected = [sc.SCENE_BUILDING_MESSAGE, sc.SCENE_CACHED_MESSAGE]
    assert builds == [1] and reports == expected
    assert warm.behind_anchor == cold.behind_anchor is not None
