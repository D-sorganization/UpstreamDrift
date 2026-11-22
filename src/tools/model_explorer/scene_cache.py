"""Disk cache of built golfer scene XML (#15150).

The grip-hand solve dominates opening a matched swing, so the finished MJCF
is kept in the user cache directory. A stale scene is worse than a slow one,
so a hit needs both an equal key (SHA-256 of every caller input, the format
and the Python and ``key_packages`` versions) and an unchanged dependency
manifest: the SHA-256 of every repository file the building process had
loaded or read (module sources, data files beside them, and every file an
audit hook saw opened since import). Any change, even a deletion, misses.
Writes are atomic (temp file plus ``os.replace``); an unreadable entry is a
logged miss and is rebuilt; least recently used entries go above
``max_bytes``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import platformdirs

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_PATH = Path(__file__).with_name("scene_cache_settings.json")
SCENE_BUILDING_MESSAGE = "Building golfer scene (first time only)…"
SCENE_CACHED_MESSAGE = "Loading cached scene…"
_FORMAT, _SUFFIX = 1, ".scene"
_ROOT_TEXT = str(REPO_ROOT) + os.sep
_opened: set[str] = set()
_digests: dict[tuple[str, int, int, int], str] = {}


def _record_open(event: str, args: tuple[Any, ...]) -> None:
    """Audit hook: remember repository files opened for reading."""
    if event != "open" or not isinstance(args[0], str):
        return
    if isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
        return
    path = os.path.abspath(args[0])
    if path.startswith(_ROOT_TEXT) and not path.endswith((".py", ".pyc")):
        _opened.add(path)


sys.addaudithook(_record_open)


@dataclass(frozen=True)
class SceneCacheSettings:
    """Validated ``scene_cache_settings.json`` (``ValueError`` when invalid)."""

    enabled: bool = True
    directory: str = ""
    directory_env: str = "UD_SCENE_CACHE_DIR"
    max_bytes: int = 512 * 1024 * 1024
    compress_level: int = 1
    key_packages: tuple[str, ...] = ("mujoco", "numpy", "scipy", "trimesh")
    sibling_data_suffixes: tuple[str, ...] = (".json", ".npz")
    digest_memo_min_age_s: float = 2.0

    def __post_init__(self) -> None:
        if self.max_bytes <= 0 or not self.digest_memo_min_age_s >= 0:
            raise ValueError(f"max_bytes and digest_memo_min_age_s: {self}")
        if not 0 <= self.compress_level <= 9:
            raise ValueError(f"compress_level must be 0..9: {self.compress_level}")

    def cache_dir(self) -> Path:
        """``directory_env``, else ``directory``, else the user cache dir."""
        chosen = os.environ.get(self.directory_env) or self.directory
        if chosen:
            return Path(chosen).expanduser()
        return Path(platformdirs.user_cache_dir("upstream-drift")) / "golfer_scenes"


def load_scene_cache_settings(path: Path | None = None) -> SceneCacheSettings:
    """Read the settings JSON (``TypeError`` for an unknown key)."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    values: dict[str, Any] = {
        k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()
    }
    return SceneCacheSettings(**values)


def process_dependencies(suffixes: Iterable[str] = (".json", ".npz")) -> set[Path]:
    """Repository files this process's code may have used to build a scene.

    Postcondition: every path is an existing file under :data:`REPO_ROOT`.
    """
    files = (getattr(m, "__file__", None) for m in list(sys.modules.values()))
    modules = {Path(f).resolve() for f in files if isinstance(f, str)}
    modules = {p for p in modules if p.is_relative_to(REPO_ROOT)}
    wanted = tuple(suffixes)
    dirs = {m.parent for m in modules}
    siblings = {p for d in dirs for p in d.iterdir() if p.suffix in wanted}
    found = modules | siblings | {Path(p).resolve() for p in list(_opened)}
    return {p for p in found if p.is_relative_to(REPO_ROOT) and p.is_file()}


def _file_digest(path: Path, memo_min_age_s: float) -> str | None:
    """SHA-256 of ``path`` (None if gone), memoised by inode, size and mtime
    only once older than ``memo_min_age_s``: a quick rewrite keeps its mtime."""
    try:
        stat = path.stat()
        memo = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        digest = _digests.get(memo) or hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
    if time.time() - stat.st_mtime > memo_min_age_s:
        _digests[memo] = digest
    return digest


class SceneCache:
    """Content-addressed scene XML entries in ``settings.cache_dir()``."""

    def __init__(
        self,
        settings: SceneCacheSettings,
        *,
        repo_root: Path = REPO_ROOT,
        dependency_paths: Callable[[], Iterable[Path]] | None = None,
    ) -> None:
        self.settings = settings
        self.directory = settings.cache_dir()
        self._root = repo_root.resolve()
        suffixes = settings.sibling_data_suffixes
        self._dependency_paths = dependency_paths or (
            lambda: process_dependencies(suffixes)
        )

    def entry_key(self, inputs: Mapping[str, bytes]) -> str:
        """SHA-256 over ``inputs``, the format and the package versions."""
        if not inputs or not all(isinstance(v, bytes) for v in inputs.values()):
            raise TypeError("inputs must be a non-empty mapping of name to bytes")
        digest = hashlib.sha256(f"format={_FORMAT};{sys.version}".encode())
        for name in self.settings.key_packages:
            try:
                digest.update(f";{name}={metadata.version(name)}".encode())
            except metadata.PackageNotFoundError:
                digest.update(f";{name}=absent".encode())
        for name in sorted(inputs):
            digest.update(f";{name}:{len(inputs[name])}:".encode() + inputs[name])
        return digest.hexdigest()

    def entry_path(self, key: str) -> Path:
        return self.directory / f"{key}{_SUFFIX}"

    def is_cached(self, inputs: Mapping[str, bytes]) -> bool:
        """True when ``inputs`` have a valid, up-to-date entry."""
        enabled = self.settings.enabled
        return (
            enabled and self._read(self.entry_path(self.entry_key(inputs))) is not None
        )

    def get_or_build(
        self, inputs: Mapping[str, bytes], build: Callable[[], str]
    ) -> tuple[str, bool]:
        """The scene XML for ``inputs`` and whether it came from the cache."""
        if not self.settings.enabled:
            return build(), False
        path = self.entry_path(self.entry_key(inputs))
        xml = self._read(path)
        if xml is not None:
            os.utime(path)  # most recently used
            return xml, True
        xml = build()
        self._write(path, xml)
        return xml, False

    def _read(self, path: Path) -> str | None:
        """The entry's XML; None when absent, stale or (logged) corrupt."""
        if not path.is_file():
            return None
        try:
            with gzip.open(path, "rb") as stream:
                head, body = stream.read().split(b"\n", 1)
            header = json.loads(head)
            if header["format"] != _FORMAT:
                raise ValueError("unknown entry format")
            if hashlib.sha256(body).hexdigest() != header["xml_sha256"]:
                raise ValueError("XML does not match its recorded hash")
            deps: dict[str, str] = dict(header["deps"])
            xml = body.decode("utf-8")
        except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
            logger.warning("corrupt scene cache entry %s (%s); rebuilding", path, exc)
            path.unlink(missing_ok=True)
            return None
        age = self.settings.digest_memo_min_age_s
        current = all(_file_digest(self._root / r, age) == h for r, h in deps.items())
        return xml if current else None

    def _write(self, path: Path, xml: str) -> None:
        age, deps = self.settings.digest_memo_min_age_s, {}
        for dep in sorted(p.resolve() for p in self._dependency_paths()):
            digest = _file_digest(dep, age)
            if digest is not None and dep.is_relative_to(self._root):
                deps[dep.relative_to(self._root).as_posix()] = digest
        body = xml.encode("utf-8")
        header = {"format": _FORMAT, "xml_sha256": hashlib.sha256(body).hexdigest()}
        data = json.dumps({**header, "deps": deps}).encode() + b"\n" + body
        level = self.settings.compress_level
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(suffix=".tmp", dir=self.directory)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(gzip.compress(data, compresslevel=level, mtime=0))
                os.replace(temp, path)
            finally:
                Path(temp).unlink(missing_ok=True)
            self._evict(keep=path)
        except OSError as exc:
            logger.warning("could not cache the golfer scene in %s: %s", path, exc)

    def _evict(self, keep: Path) -> None:
        """Drop least recently used entries until under ``max_bytes``."""
        stats = {p: p.stat() for p in self.directory.glob(f"*{_SUFFIX}")}
        total = sum(s.st_size for s in stats.values())
        for entry in sorted(stats, key=lambda p: stats[p].st_mtime):
            if total > self.settings.max_bytes and entry != keep:
                total -= stats[entry].st_size
                entry.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def default_scene_cache() -> SceneCache:
    """The process-wide cache configured by ``scene_cache_settings.json``."""
    return SceneCache(load_scene_cache_settings())
