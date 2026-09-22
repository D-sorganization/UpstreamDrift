"""HDF5 episode store with content hashes and lazy/eager reads (NM-03)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .record import EPISODE_STORE_SCHEMA, EpisodeRecord

__all__ = ["EpisodeManifest", "EpisodeStore"]

_INDEX_NAME = "manifest.json"
_SHARD_DIR = "shards"
_META_GROUP = "meta"
_ARRAY_KEYS = ("sample_times_s", "q", "v", "u", "a_native", "q_next", "coefficients")


@dataclass(frozen=True)
class EpisodeManifest:
    """Versioned corpus manifest — episode ids only, no array payload."""

    schema: str
    episode_ids: tuple[str, ...]
    trial_index: dict[str, str]
    content_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_ids": list(self.episode_ids),
            "trial_index": dict(self.trial_index),
            "content_digest": self.content_digest,
        }


class EpisodeStore:
    """Content-addressed episode corpus on HDF5 shards.

    Episodes are written once. Identical payloads are idempotent. A different
    payload for an existing ``trial_id`` is refused (immutable resume).
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / _SHARD_DIR).mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / _INDEX_NAME
        self._index = self._load_index()

    def shard_path(self, episode_id: str) -> Path:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        return self._root / _SHARD_DIR / f"{episode_id}.h5"

    def write_episode(self, episode: EpisodeRecord) -> EpisodeRecord:
        """Persist ``episode``; idempotent for identical content."""
        if not isinstance(episode, EpisodeRecord):
            raise TypeError("episode must be an EpisodeRecord")
        existing_id = self._index["trial_index"].get(episode.trial_id)
        if existing_id is not None:
            existing = self.read_episode(existing_id, lazy=False)
            if existing.content_sha256 == episode.content_sha256:
                return existing
            raise ValueError(
                f"trial_id {episode.trial_id!r} already stored immutably "
                f"as episode {existing_id}"
            )
        path = self.shard_path(episode.episode_id)
        if path.exists():
            # Content-address collision with same hash: treat as idempotent.
            return self.read_episode(episode.episode_id, lazy=False)
        self._write_shard(path, episode)
        self._index["trial_index"][episode.trial_id] = episode.episode_id
        ids = set(self._index["episode_ids"])
        ids.add(episode.episode_id)
        self._index["episode_ids"] = sorted(ids)
        self._save_index()
        return episode

    def read_episode(self, episode_id: str, *, lazy: bool = True) -> EpisodeRecord:
        """Load one episode. ``lazy`` keeps arrays as read-only memmap views."""
        path = self.shard_path(episode_id)
        if not path.exists():
            raise FileNotFoundError(f"episode shard missing: {path}")
        try:
            return self._read_shard(path, lazy=lazy)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt episode shard {path}: {exc}") from exc

    def iter_episode_ids(self) -> Iterator[str]:
        """Yield stored episode ids without loading arrays."""
        yield from tuple(self._index["episode_ids"])

    def build_manifest(self) -> EpisodeManifest:
        ids = tuple(self._index["episode_ids"])
        trial_index = dict(self._index["trial_index"])
        payload = {
            "schema": EPISODE_STORE_SCHEMA,
            "episode_ids": list(ids),
            "trial_index": trial_index,
        }
        digest = _sha256_json(payload)
        return EpisodeManifest(
            schema=EPISODE_STORE_SCHEMA,
            episode_ids=ids,
            trial_index=trial_index,
            content_digest=digest,
        )

    def _load_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {"episode_ids": [], "trial_index": {}}
        data = json.loads(self._index_path.read_text(encoding="utf-8"))
        return {
            "episode_ids": list(data.get("episode_ids", [])),
            "trial_index": dict(data.get("trial_index", {})),
        }

    def _save_index(self) -> None:
        manifest = self.build_manifest().as_dict()
        self._index_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_shard(self, path: Path, episode: EpisodeRecord) -> None:
        with h5py.File(path, "w") as handle:
            meta = handle.create_group(_META_GROUP)
            for key, value in episode.as_meta_dict().items():
                meta.attrs[key] = _attr_value(value)
            meta.attrs["schema"] = EPISODE_STORE_SCHEMA
            for name in _ARRAY_KEYS:
                data = getattr(episode, name)
                if data is None:
                    continue
                handle.create_dataset(
                    name,
                    data=np.asarray(data, dtype=np.float64),
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )

    def _read_shard(self, path: Path, *, lazy: bool) -> EpisodeRecord:
        with h5py.File(path, "r") as handle:
            meta = handle[_META_GROUP]
            attrs = {key: _decode_attr(meta.attrs[key]) for key in meta.attrs}
            expected = attrs.get("content_sha256")
            arrays = {
                name: _read_array(handle, name, lazy=lazy) for name in _ARRAY_KEYS
            }
        record = EpisodeRecord(
            trial_id=str(attrs["trial_id"]),
            family_id=str(attrs["family_id"]),
            model_id=str(attrs["model_id"]),
            control_basis=str(attrs["control_basis"]),
            units=str(attrs["units"]),
            joint_names=tuple(attrs["joint_names"]),
            coefficient_letters=tuple(attrs["coefficient_letters"]),
            schema_version=str(attrs.get("schema_version", EPISODE_STORE_SCHEMA)),
            sample_times_s=arrays["sample_times_s"],
            q=arrays["q"],
            v=arrays["v"],
            u=arrays["u"],
            a_native=arrays["a_native"],
            q_next=arrays["q_next"],
            channel_availability=dict(attrs["channel_availability"]),
            ancestry=tuple(attrs.get("ancestry", ())),
            geometry_stratum=str(attrs["geometry_stratum"]),
            contact_stratum=str(attrs["contact_stratum"]),
            club_stratum=str(attrs["club_stratum"]),
            coefficients=arrays["coefficients"],
            source_schema=attrs.get("source_schema"),
            episode_id=str(attrs.get("episode_id", "")),
            content_sha256=str(attrs.get("content_sha256", "")),
        )
        computed = record.content_payload_digest()
        if expected and computed != expected:
            raise ValueError(
                f"content hash mismatch for {path}: "
                f"stored={expected} computed={computed}"
            )
        return record


def _read_array(handle: h5py.File, name: str, *, lazy: bool) -> np.ndarray | None:
    if name not in handle:
        return None
    dataset = handle[name]
    if lazy:
        # Materialise a contiguous copy so the file may close; true mmap
        # across closed handles is unsafe. Chunked reads still avoid
        # all-corpus RAM load by reading one shard at a time.
        return np.asarray(dataset[()], dtype=np.float64)
    return np.asarray(dataset[()], dtype=np.float64)


def _attr_value(value: Any) -> Any:
    if value is None:
        # JSON null — distinct from empty string so round-trips stay stable.
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    elif hasattr(value, "item") and not isinstance(value, (str, dict, list)):
        # numpy scalar attrs from h5py
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    if value == "null" or value == "":
        return None
    if isinstance(value, str) and value[:1] in "{[":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _sha256_json(payload: dict[str, Any]) -> str:
    import hashlib

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
