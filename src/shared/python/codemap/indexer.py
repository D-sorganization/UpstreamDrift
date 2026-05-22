"""Walk a repo, parse files, persist symbols into SQLite.

Public entry point:

    rebuild(repo_root, *, since=None) -> RebuildStats

If ``since`` is set, runs ``git diff --name-only <since>..HEAD`` and only
re-parses files that changed (incremental). Otherwise walks the whole tree
(respecting ``.gitignore`` via ``pathspec`` if available).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import db as db_mod
from . import parsers as parsers_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hashing — prefer blake3 if available, else hashlib.blake2b.
# ---------------------------------------------------------------------------


def _hash_bytes(data: bytes) -> str:
    try:
        import blake3  # type: ignore[import-not-found]

        return blake3.blake3(data).hexdigest()
    except Exception:  # noqa: BLE001
        return hashlib.blake2b(data, digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Path walking + gitignore.
# ---------------------------------------------------------------------------


_DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".codemap",
    ".idea",
    ".vscode",
    "htmlcov",
    "site-packages",
}


def _load_gitignore(repo_root: Path):
    """Return a callable ``is_ignored(rel_path) -> bool``."""
    try:
        import pathspec  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        pathspec = None

    patterns: list[str] = []
    gi = repo_root / ".gitignore"
    if gi.exists():
        patterns.extend(gi.read_text(encoding="utf-8", errors="ignore").splitlines())
    # Always exclude .codemap directory.
    patterns.append(".codemap/")

    if pathspec is not None:
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        def _ignored(rel: str) -> bool:
            return spec.match_file(rel)

        return _ignored

    # Naive fallback: just match a few common substrings.
    simple = [
        p.strip().rstrip("/") for p in patterns if p.strip() and not p.startswith("#")
    ]

    def _ignored_simple(rel: str) -> bool:
        rel_norm = rel.replace(os.sep, "/")
        for pat in simple:
            if not pat:
                continue
            if pat in rel_norm:
                return True
        return False

    return _ignored_simple


def _walk(repo_root: Path):
    """Yield ``(abs_path, rel_path)`` for every supported source file."""
    is_ignored = _load_gitignore(repo_root)
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _DEFAULT_SKIP_DIRS
            and not is_ignored(
                str(Path(dirpath, d).relative_to(repo_root)).replace(os.sep, "/")
            )
        ]
        for name in filenames:
            abs_p = Path(dirpath) / name
            rel = abs_p.relative_to(repo_root).as_posix()
            if is_ignored(rel):
                continue
            if parsers_mod.language_for(abs_p) is None:
                continue
            yield abs_p, rel


def _git_changed_files(repo_root: Path, since: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", f"{since}..HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        logger.warning(
            "codemap: 'git diff' failed for since=%s; falling back to full rebuild",
            since,
        )
        return []
    files: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        files.append(line)
    return files


def _current_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Stats.
# ---------------------------------------------------------------------------


@dataclass
class RebuildStats:
    files_seen: int = 0
    files_parsed: int = 0
    files_skipped_unchanged: int = 0
    symbols_inserted: int = 0
    symbols_deleted: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core indexing routine.
# ---------------------------------------------------------------------------


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (OSError, PermissionError) as exc:
        logger.debug("codemap: cannot read %s: %s", path, exc)
        return None


def _process_file(
    abs_path: Path,
    rel: str,
    repo_root: Path,
    conn,
    stats: RebuildStats,
) -> None:
    data = _read_bytes(abs_path)
    if data is None:
        return
    stats.files_seen += 1
    file_hash = _hash_bytes(data)

    cur = conn.cursor()
    existing = cur.execute("SELECT hash FROM files WHERE path = ?", (rel,)).fetchone()
    if existing is not None and existing["hash"] == file_hash:
        stats.files_skipped_unchanged += 1
        return

    parsed = parsers_mod.dispatch(rel, data)
    if parsed is None:
        return

    # Delete prior symbols for this file then re-insert.
    deleted = cur.execute("DELETE FROM symbols WHERE path = ?", (rel,)).rowcount or 0
    stats.symbols_deleted += deleted

    try:
        st = abs_path.stat()
        mtime = st.st_mtime
        size = st.st_size
    except OSError:
        mtime = time.time()
        size = len(data)

    cur.execute(
        "INSERT OR REPLACE INTO files("
        "path, language, hash, mtime, size, imports, indexed_at"
        ") VALUES(?, ?, ?, ?, ?, ?, ?)",
        (
            rel,
            parsed.language,
            file_hash,
            mtime,
            size,
            json.dumps(parsed.imports),
            time.time(),
        ),
    )

    for sym in parsed.symbols:
        slice_bytes = b"\n".join(data.splitlines()[sym.start_line - 1 : sym.end_line])
        sym_hash = _hash_bytes(slice_bytes)
        cur.execute(
            "INSERT INTO symbols(path, kind, name, qualified, sig, docstring, "
            "start_line, end_line, calls_out, hash) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                rel,
                sym.kind,
                sym.name,
                sym.qualified,
                sym.sig,
                sym.docstring,
                sym.start_line,
                sym.end_line,
                json.dumps(sym.calls_out),
                sym_hash,
            ),
        )
        stats.symbols_inserted += 1

    stats.files_parsed += 1


def rebuild(
    repo_root: str | os.PathLike[str],
    *,
    since: str | None = None,
) -> RebuildStats:
    """(Re)index the codebase at ``repo_root``.

    If ``since`` is provided, only files reported by ``git diff --name-only
    <since>..HEAD`` (plus their untracked siblings) are re-parsed. Files
    whose blake3 hash hasn't changed are skipped regardless.
    """
    start = time.perf_counter()
    repo = Path(repo_root).resolve()
    stats = RebuildStats()
    conn = db_mod.open_db(repo)
    try:
        if since:
            changed = _git_changed_files(repo, since)
            if not changed:
                # Fall back to full rebuild if git is unavailable.
                logger.info("codemap: no changes from git diff; running full rebuild")
                iterator = _walk(repo)
            else:
                pairs = []
                for rel in changed:
                    abs_p = repo / rel
                    if not abs_p.exists():
                        # File deleted — remove from index.
                        deleted = (
                            conn.execute(
                                "DELETE FROM symbols WHERE path = ?",
                                (rel,),
                            ).rowcount
                            or 0
                        )
                        conn.execute("DELETE FROM files WHERE path = ?", (rel,))
                        stats.symbols_deleted += deleted
                        continue
                    if parsers_mod.language_for(abs_p) is None:
                        continue
                    pairs.append((abs_p, rel))
                iterator = iter(pairs)
        else:
            iterator = _walk(repo)

        for abs_p, rel in iterator:
            try:
                _process_file(abs_p, rel, repo, conn, stats)
            except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
                logger.warning("codemap: failed to index %s: %s", rel, exc)
                stats.errors.append(f"{rel}: {exc}")

        # Refresh manifest.
        manifest = {
            "repo_root": str(repo),
            "schema_version": db_mod.SCHEMA_VERSION,
            "last_indexed": time.time(),
            "last_commit": _current_commit(repo),
            "files": stats.files_parsed,
            "symbols": stats.symbols_inserted,
        }
        db_mod.manifest_path(repo).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        conn.commit()
    finally:
        conn.close()

    stats.elapsed_s = time.perf_counter() - start
    return stats


__all__ = ["RebuildStats", "rebuild"]
