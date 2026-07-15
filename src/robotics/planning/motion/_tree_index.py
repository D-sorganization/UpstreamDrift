"""Nearest-neighbor index helpers for sampling-based motion planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # SciPy is a project dependency, but keep the vector path as fallback.
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only in stripped installs
    cKDTree = None  # type: ignore[assignment]


@dataclass
class TreeConfigIndex:
    """Append-friendly configuration index for RRT-style trees."""

    use_kd_tree: bool = False
    kd_rebuild_interval: int = 64

    def __post_init__(self) -> None:
        if self.kd_rebuild_interval <= 0:
            raise ValueError("kd_rebuild_interval must be positive")
        self._configs: np.ndarray | None = None
        self._count = 0
        self._dimension: int | None = None
        self._kd_tree: Any | None = None
        self._kd_tree_size = 0

    def clear(self) -> None:
        """Remove all indexed configurations."""
        self._configs = None
        self._count = 0
        self._dimension = None
        self._kd_tree = None
        self._kd_tree_size = 0

    def append(self, config: np.ndarray) -> int:
        """Append one finite 1-D configuration and return its index."""
        config = self._validate_config(config)
        if self._dimension is None:
            self._dimension = int(config.shape[0])
            capacity = max(16, self.kd_rebuild_interval)
            self._configs = np.empty((capacity, self._dimension), dtype=float)
        elif config.shape != (self._dimension,):
            raise ValueError("config shape must match existing tree dimension")

        if self._configs is None:  # defensive narrowing for type checkers
            raise RuntimeError("tree index storage was not initialized")
        if self._count == self._configs.shape[0]:
            new_configs = np.empty((self._configs.shape[0] * 2, self._dimension))
            new_configs[: self._count] = self._configs[: self._count]
            self._configs = new_configs

        idx = self._count
        self._configs[idx] = config
        self._count += 1
        return idx

    def nearest(self, query: np.ndarray) -> int:
        """Return the index of the nearest indexed configuration."""
        query = self._validate_query(query)
        if self._count == 0:
            raise RuntimeError("tree index is empty")
        if self.use_kd_tree and cKDTree is not None:
            return self._nearest_with_kd_tree(query)
        return self._nearest_in_view(query, 0, self._count)[0]

    def within_radius(self, query: np.ndarray, radius: float) -> list[int]:
        """Return sorted indices whose configurations are within ``radius``."""
        query = self._validate_query(query)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and non-negative")
        if self._count == 0:
            return []
        if self.use_kd_tree and cKDTree is not None:
            return self._within_radius_with_kd_tree(query, float(radius))

        squared = self._squared_distances(query, 0, self._count)
        return np.flatnonzero(squared <= radius * radius).astype(int).tolist()

    @property
    def count(self) -> int:
        """Number of indexed configurations."""
        return self._count

    def _validate_config(self, config: np.ndarray) -> np.ndarray:
        arr = np.asarray(config, dtype=float)
        if arr.ndim != 1:
            raise ValueError("config must be a 1-D array")
        if arr.size == 0:
            raise ValueError("config must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError("config must contain only finite values")
        return arr

    def _validate_query(self, query: np.ndarray) -> np.ndarray:
        arr = self._validate_config(query)
        if self._dimension is not None and arr.shape != (self._dimension,):
            raise ValueError("query shape must match tree dimension")
        return arr

    def _view(self) -> np.ndarray:
        if self._configs is None:
            raise RuntimeError("tree index is empty")
        return self._configs[: self._count]

    def _squared_distances(
        self,
        query: np.ndarray,
        start: int,
        stop: int,
    ) -> np.ndarray:
        configs = self._view()[start:stop]
        diff = configs - query
        return np.einsum("ij,ij->i", diff, diff)

    def _nearest_in_view(
        self,
        query: np.ndarray,
        start: int,
        stop: int,
    ) -> tuple[int, float]:
        squared = self._squared_distances(query, start, stop)
        local_idx = int(np.argmin(squared))
        return start + local_idx, float(squared[local_idx])

    def _ensure_kd_tree(self) -> None:
        should_rebuild = (
            self._kd_tree is None
            or self._count - self._kd_tree_size >= self.kd_rebuild_interval
        )
        if should_rebuild:
            self._kd_tree = cKDTree(self._view())
            self._kd_tree_size = self._count

    def _nearest_with_kd_tree(self, query: np.ndarray) -> int:
        self._ensure_kd_tree()
        if self._kd_tree is None:
            return self._nearest_in_view(query, 0, self._count)[0]

        _, tree_idx = self._kd_tree.query(query, k=1)
        best_idx = int(tree_idx)
        best_squared = float(np.vdot(diff := self._view()[best_idx] - query, diff))  # ⚡ Bolt: np.vdot is ~3x faster than np.sum(diff**2)
        if self._kd_tree_size < self._count:
            tail_idx, tail_squared = self._nearest_in_view(
                query,
                self._kd_tree_size,
                self._count,
            )
            if tail_squared < best_squared:
                best_idx = tail_idx
        return best_idx

    def _within_radius_with_kd_tree(
        self,
        query: np.ndarray,
        radius: float,
    ) -> list[int]:
        self._ensure_kd_tree()
        indices: list[int] = []
        if self._kd_tree is not None:
            indices.extend(
                int(i) for i in self._kd_tree.query_ball_point(query, radius)
            )

        if self._kd_tree_size < self._count:
            squared = self._squared_distances(query, self._kd_tree_size, self._count)
            tail = np.flatnonzero(squared <= radius * radius) + self._kd_tree_size
            indices.extend(int(i) for i in tail)
        return sorted(indices)
