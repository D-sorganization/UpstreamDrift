"""Deterministic file-level sharding for the CI unit-test gate (#13314).

Loaded only by ``ci-standard.yml``'s ``unit-test-gate`` matrix via
``-p scripts.ci.pytest_unit_gate_shard --unit-gate-shard=K/N``. Each test module is
assigned to one shard by a CRC32 of its rootdir-relative POSIX path, and
modules of other shards are skipped in ``pytest_ignore_collect`` -- so a leg
neither imports nor runs them, which splits collection time as well as run
time. CRC32 is independent of ``PYTHONHASHSEED``, so every xdist worker
computes the same collection, as xdist requires.

Postcondition: for any ``N``, the union of shards ``1..N`` is the full
unsharded collection and the shards are pairwise disjoint.
"""

from __future__ import annotations

import fnmatch
import zlib
from pathlib import Path
from typing import Any

import pytest

OPTION = "--unit-gate-shard"
SHARD_KEY = pytest.StashKey[tuple[int, int]]()
_TEST_MODULE_PATTERN = "test_*.py"


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse ``"K/N"`` into ``(K, N)`` with ``1 <= K <= N``.

    Raises:
        ValueError: if *spec* is not two positive integers ``K/N`` with K <= N.
    """
    parts = spec.split("/")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"shard spec must be K/N, got {spec!r}")
    index, total = int(parts[0]), int(parts[1])
    if not 1 <= index <= total:
        raise ValueError(f"shard spec must be K/N with 1 <= K <= N, got {spec!r}")
    return index, total


def shard_of(relative_path: str, total: int) -> int:
    """Return the 1-based shard that owns *relative_path* out of *total*."""
    if total < 1:
        raise ValueError(f"total shards must be >= 1, got {total}")
    return zlib.crc32(relative_path.encode("utf-8")) % total + 1


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        OPTION,
        default=None,
        metavar="K/N",
        help="Collect only test modules hashed to shard K of N (#13314).",
    )


def pytest_configure(config: pytest.Config) -> None:
    spec = config.getoption(OPTION)
    if spec is None:
        return
    try:
        config.stash[SHARD_KEY] = parse_shard(spec)
    except ValueError as exc:
        raise pytest.UsageError(f"{OPTION}: {exc}") from exc


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path: Path, config: Any) -> bool | None:
    """Skip test modules owned by another shard; defer on everything else.

    ``tryfirst`` is load-bearing: the hook is ``firstresult`` and
    ``tests/conftest.py``'s implementation returns ``False`` (not ``None``)
    for every path, which would otherwise end the call before this one runs.
    """
    shard = config.stash.get(SHARD_KEY, None)
    if shard is None or not fnmatch.fnmatch(collection_path.name, _TEST_MODULE_PATTERN):
        return None
    if not collection_path.is_file():
        return None
    try:
        relative = collection_path.relative_to(config.rootpath).as_posix()
    except ValueError:
        return None
    index, total = shard
    return True if shard_of(relative, total) != index else None
