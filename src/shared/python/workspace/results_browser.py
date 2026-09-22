"""Index and filter canonical result artifacts for workspace browsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py

from src.shared.python.simulation_backends.provenance import PROVENANCE_FLAT_PREFIX

_RESULT_EXTENSIONS = frozenset({".h5", ".hdf5"})
_META_PREFIX = "meta_"


@dataclass(frozen=True)
class ResultArtifact:
    """A result-browser row backed by one canonical artifact file."""

    path: str
    relative_path: str
    size_bytes: int
    modified_at: str
    schema_version: str | None
    kind: str | None
    backend: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def has_provenance(self) -> bool:
        """Whether this artifact exposes a CC-6 provenance stamp."""
        return bool(self.provenance)


@dataclass(frozen=True)
class ResultFilter:
    """Filter criteria for :class:`ResultsBrowser` view models."""

    project_id: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    dataset_id: str | None = None
    backend: str | None = None
    text: str | None = None
    has_provenance: bool | None = None
    extensions: tuple[str, ...] = (".h5", ".hdf5")


class ResultsBrowser:
    """Filesystem-backed indexer for CC-4 HDF5 result artifacts.

    Postcondition: :meth:`index` returns a deterministic, path-sorted list of
    artifacts that can be filtered without reopening HDF5 files.
    """

    def __init__(self, root: Path | str) -> None:
        if not isinstance(root, (Path, str)):
            raise TypeError("root must be a pathlib.Path or str")
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        """Directory whose result artifacts are indexed."""
        return self._root

    def index(self, result_filter: ResultFilter | None = None) -> list[ResultArtifact]:
        """Index all readable canonical HDF5 artifacts under ``root``."""
        if result_filter is not None and not isinstance(result_filter, ResultFilter):
            raise TypeError("result_filter must be ResultFilter or None")
        filter_value = result_filter or ResultFilter()
        artifacts = [
            artifact
            for artifact in self._iter_artifacts(filter_value.extensions)
            if _matches_filter(artifact, filter_value)
        ]
        return sorted(artifacts, key=lambda artifact: artifact.relative_path)

    def _iter_artifacts(self, extensions: tuple[str, ...]) -> list[ResultArtifact]:
        allowed = {ext.lower() for ext in extensions}
        if not allowed:
            raise ValueError("extensions must contain at least one extension")
        if not self._root.exists():
            return []
        if not self._root.is_dir():
            raise ValueError(f"root must be a directory: {self._root}")
        artifacts: list[ResultArtifact] = []
        for path in self._root.rglob("*"):
            if path.suffix.lower() not in allowed or not path.is_file():
                continue
            artifact = _read_artifact(self._root, path)
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts


def _read_artifact(root: Path, path: Path) -> ResultArtifact | None:
    try:
        with h5py.File(path, "r") as handle:
            metadata = _read_metadata(handle)
            schema_version = _read_optional_attr(handle, "schema_version")
            kind = _read_optional_attr(handle, "kind")
            backend = _read_optional_attr(handle, "backend")
    except OSError:
        return None

    stat = path.stat()
    provenance = _extract_provenance(metadata)
    return ResultArtifact(
        path=str(path),
        relative_path=path.relative_to(root).as_posix(),
        size_bytes=stat.st_size,
        modified_at=_mtime_to_iso(stat.st_mtime),
        schema_version=schema_version,
        kind=kind,
        backend=backend,
        metadata=metadata,
        provenance=provenance,
    )


def _read_metadata(root: h5py.Group) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in root.attrs:
        value = _coerce_attr(root.attrs[name])
        if name.startswith(_META_PREFIX):
            metadata[name[len(_META_PREFIX) :]] = value
        else:
            metadata[name] = value
    return metadata


def _read_optional_attr(root: h5py.Group, name: str) -> str | None:
    if name not in root.attrs:
        return None
    return str(_coerce_attr(root.attrs[name]))


def _extract_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    prefix_len = len(PROVENANCE_FLAT_PREFIX)
    return {
        key[prefix_len:]: value
        for key, value in sorted(metadata.items())
        if key.startswith(PROVENANCE_FLAT_PREFIX)
    }


def _matches_filter(artifact: ResultArtifact, result_filter: ResultFilter) -> bool:
    metadata = artifact.metadata
    if (
        result_filter.project_id
        and metadata.get("project_id") != result_filter.project_id
    ):
        return False
    if (
        result_filter.subject_id
        and metadata.get("subject_id") != result_filter.subject_id
    ):
        return False
    if (
        result_filter.session_id
        and metadata.get("session_id") != result_filter.session_id
    ):
        return False
    if (
        result_filter.dataset_id
        and metadata.get("dataset_id") != result_filter.dataset_id
    ):
        return False
    if result_filter.backend and artifact.backend != result_filter.backend:
        return False
    if (
        result_filter.has_provenance is not None
        and artifact.has_provenance != result_filter.has_provenance
    ):
        return False
    return not (
        result_filter.text and result_filter.text.lower() not in _search_text(artifact)
    )


def _search_text(artifact: ResultArtifact) -> str:
    metadata_text = " ".join(
        f"{key} {value}" for key, value in sorted(artifact.metadata.items())
    )
    return (
        f"{artifact.relative_path} {artifact.backend or ''} "
        f"{artifact.kind or ''} {metadata_text}"
    ).lower()


def _coerce_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        return value.item()
    return value


def _mtime_to_iso(mtime: float) -> str:
    from datetime import datetime, timezone

    timestamp = datetime.fromtimestamp(mtime, timezone.utc)
    iso_text = timestamp.isoformat()
    return iso_text.replace("+00:00", "Z")


__all__ = ["ResultArtifact", "ResultFilter", "ResultsBrowser"]
