"""Dataset library — declarative registry of training datasets.

A :class:`Dataset` is a *handle* to data sitting on disk; the registry
records what exists and where, and validates references made by
:class:`TrainingConfig.dataset_id`. The training subsystem never tries
to guess paths from job configs — every dataset must be registered
explicitly. This is the "idiot-proof" half of the dataset story; the
loader-per-format adapters land in PR5 alongside the engine wiring.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    DuplicateJobError,
    JobNotFoundError,
    TrainingConfigError,
)

__all__ = ["Dataset", "DatasetRegistry"]


_VALID_FORMATS: frozenset[str] = frozenset(
    {"c3d", "csv", "parquet", "json", "pt", "npz", "hdf5", "tfrecord", "custom"}
)


@dataclass(frozen=True, slots=True)
class Dataset:
    """Immutable handle to a registered training dataset.

    Attributes:
        dataset_id: Stable identifier referenced from
            :attr:`TrainingConfig.dataset_id`. Same charset rules as
            :class:`JobId`.
        name: Human-readable label for the dashboard.
        path: Filesystem path. Existence is **not** validated at
            construction time so the registry stays headless-safe;
            consumers should call :meth:`Dataset.exists` before use.
        format: Lower-case format tag; must be in
            :data:`_VALID_FORMATS`. ``"custom"`` is the escape hatch
            for tools whose format isn't yet enumerated.
        size_bytes: On-disk size, in bytes. ``0`` when unknown /
            self-describing. Must be ``>= 0``.
        schema_version: Wire-format version of the dataset's *own*
            schema (independent of training-config schema).
        description: Free-form notes shown in the dashboard.

    Invariants enforced in :meth:`__post_init__`:
        - ``dataset_id`` is a non-empty string matching the same
          charset as :class:`JobId` (validation delegated to
          :class:`JobId` not used here to avoid coupling).
        - ``name`` is a non-empty string.
        - ``path`` is a :class:`Path`.
        - ``format`` is in :data:`_VALID_FORMATS`.
        - ``size_bytes >= 0``.
        - ``schema_version >= 1``.
    """

    dataset_id: str
    name: str
    path: Path
    format: str
    size_bytes: int = 0
    schema_version: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise TrainingConfigError("dataset_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise TrainingConfigError("name must be a non-empty string")
        if not isinstance(self.path, Path):
            raise TrainingConfigError(
                f"path must be a pathlib.Path (got {type(self.path).__name__})"
            )
        if (
            not isinstance(self.format, str)
            or self.format.lower() not in _VALID_FORMATS
        ):
            raise TrainingConfigError(
                f"format must be one of {sorted(_VALID_FORMATS)} (got {self.format!r})"
            )
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise TrainingConfigError(
                f"size_bytes must be a non-negative int (got {self.size_bytes!r})"
            )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise TrainingConfigError(
                f"schema_version must be a positive int (got {self.schema_version!r})"
            )
        if not isinstance(self.description, str):
            raise TrainingConfigError(
                f"description must be a string (got {type(self.description).__name__})"
            )
        object.__setattr__(self, "format", self.format.lower())

    def exists(self) -> bool:
        """Lightweight existence check; does not load the data."""

        return self.path.exists()


class DatasetRegistry:
    """Thread-safe in-memory registry of :class:`Dataset` handles.

    The registry owns lookup + duplicate detection only; it never
    touches disk. Persistence (writing the registry to JSON, reloading
    on launcher start) is layered on top in PR5 by a tiny serializer
    so the in-memory core stays simple and easy to test.
    """

    __slots__ = ("_datasets", "_lock")

    def __init__(self, initial: Iterable[Dataset] | None = None) -> None:
        self._lock = threading.RLock()
        self._datasets: dict[str, Dataset] = {}
        for dataset in initial or ():
            self.register(dataset)

    def register(self, dataset: Dataset) -> None:
        """Add a dataset to the registry.

        Raises:
            DuplicateJobError: When a dataset with the same id is
                already registered. (Reusing the existing exception
                family rather than introducing a parallel one.)
        """

        if not isinstance(dataset, Dataset):
            raise TypeError(f"expected Dataset (got {type(dataset).__name__})")
        with self._lock:
            if dataset.dataset_id in self._datasets:
                raise DuplicateJobError(
                    f"dataset_id {dataset.dataset_id!r} is already registered"
                )
            self._datasets[dataset.dataset_id] = dataset

    def get(self, dataset_id: str) -> Dataset:
        """Look up a dataset by id.

        Raises:
            JobNotFoundError: When ``dataset_id`` is unknown.
        """

        if not isinstance(dataset_id, str):
            raise TypeError("dataset_id must be a string")
        with self._lock:
            try:
                return self._datasets[dataset_id]
            except KeyError as exc:
                raise JobNotFoundError(
                    f"no dataset registered with id {dataset_id!r}"
                ) from exc

    def has(self, dataset_id: str) -> bool:
        """``True`` when ``dataset_id`` is registered. Does not raise."""

        with self._lock:
            return dataset_id in self._datasets

    def remove(self, dataset_id: str) -> Dataset:
        """Remove and return the dataset for ``dataset_id``.

        Raises:
            JobNotFoundError: When ``dataset_id`` is unknown.
        """

        with self._lock:
            try:
                return self._datasets.pop(dataset_id)
            except KeyError as exc:
                raise JobNotFoundError(
                    f"no dataset registered with id {dataset_id!r}"
                ) from exc

    def replace(self, dataset: Dataset) -> Dataset | None:
        """Insert or overwrite. Returns the previous entry if any."""

        if not isinstance(dataset, Dataset):
            raise TypeError(f"expected Dataset (got {type(dataset).__name__})")
        with self._lock:
            previous = self._datasets.get(dataset.dataset_id)
            self._datasets[dataset.dataset_id] = dataset
            return previous

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)

    def __iter__(self) -> Iterator[Dataset]:
        with self._lock:
            return iter(tuple(self._datasets.values()))

    def list(self, *, format: str | None = None) -> tuple[Dataset, ...]:
        """Return all registered datasets, optionally filtered by format."""

        with self._lock:
            entries = tuple(self._datasets.values())
        if format is None:
            return entries
        normalized = format.lower()
        return tuple(d for d in entries if d.format == normalized)
