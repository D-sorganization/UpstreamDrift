from __future__ import annotations

import json
from typing import Any

import numpy as np


class MotionPrimitiveLibrary:
    """Library of motion primitives for golf swing composition.

    This stores and retrieves pre-computed motion primitives that can be
    combined to create new swings.
    """

    def __init__(self) -> None:
        """Initialize empty library."""
        self.primitives: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict] = {}

    def add_primitive(
        self,
        name: str,
        trajectory: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        """Add a motion primitive to library.

        Args:
            name: Primitive name
            trajectory: Joint trajectory
            metadata: Additional metadata
        """
        if name is None:
            raise ValueError("name must be provided")
        self.primitives[name] = trajectory
        self.metadata[name] = metadata if metadata is not None else {}

    def get_primitive(self, name: str) -> np.ndarray | None:
        """Get primitive by name.

        Args:
            name: Primitive name

        Returns:
            Trajectory or None if not found
        """
        return self.primitives.get(name)

    def blend_primitives(
        self,
        names: list[str],
        weights: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Blend multiple primitives.

        Args:
            names: List of primitive names
            weights: Blending weights (default: equal)

        Returns:
            Blended trajectory
        """
        if names is None:
            raise ValueError("names must be provided")
        if weights is None:
            weights = np.ones(len(names)) / len(names)

        primitives = [
            self.primitives[name] for name in names if name in self.primitives
        ]

        if not primitives:
            return None

        min_len = min(p.shape[0] for p in primitives)
        primitives = [p[:min_len] for p in primitives]

        blended = np.zeros_like(primitives[0])
        for prim, weight in zip(primitives, weights, strict=False):
            blended += weight * prim

        return blended

    def save_library(self, filename: str) -> None:
        """Save library to file.

        Args:
            filename: Output filename (.npz)
        """
        if filename is None:
            raise ValueError("filename must be provided")
        metadata_str = json.dumps(self.metadata)
        save_dict: dict[str, Any] = dict(self.primitives)
        save_dict["metadata"] = metadata_str
        np.savez(filename, **save_dict)  # type: ignore[arg-type]

    def load_library(self, filename: str) -> None:
        """Load library from file.

        Args:
            filename: Input filename (.npz)
        """
        if filename is None:
            raise ValueError("filename must be provided")
        data = np.load(filename, allow_pickle=False)

        for key in data:
            if key == "metadata":
                metadata_value = data[key]
                if isinstance(metadata_value, str):
                    self.metadata = json.loads(metadata_value)
                else:
                    self.metadata = json.loads(metadata_value.item())
            else:
                self.primitives[key] = data[key]
