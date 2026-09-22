"""Derived window cache keyed by source + transform version (NM-03)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .record import EpisodeRecord

__all__ = ["WindowCache"]


class WindowCache:
    """Cache derived windows without duplicating raw episode corpora."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        episode: EpisodeRecord,
        *,
        transform_version: str,
        windows: np.ndarray,
    ) -> str:
        if not transform_version.strip():
            raise ValueError("transform_version must be non-empty")
        key = self._key(episode.content_payload_digest(), transform_version)
        path = self._path(key)
        if path.exists():
            return key
        np.savez_compressed(
            path,
            windows=np.asarray(windows, dtype=np.float64),
        )
        meta = {
            "key": key,
            "source_digest": episode.content_payload_digest(),
            "transform_version": transform_version,
            "shape": list(np.asarray(windows).shape),
        }
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return key

    def get(self, source_digest: str, transform_version: str) -> np.ndarray:
        key = self._key(source_digest, transform_version)
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"window cache miss for key {key}")
        with np.load(path, allow_pickle=False) as handle:
            return np.asarray(handle["windows"], dtype=np.float64)

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.npz"

    @staticmethod
    def _key(source_digest: str, transform_version: str) -> str:
        blob = f"{source_digest}:{transform_version}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
