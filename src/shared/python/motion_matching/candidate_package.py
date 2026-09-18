"""Candidate Package Contract & Serialization (PF-01 / #10431).

Defines the unified, audit-grade candidate package schema preserving:
1. Complete kinematics (time_s, q, v, a) and controls (u, u_min_trail, u_hard_zero_trail).
2. Unactuated root/reserve force histories (delta_tau_root).
3. External ground contact forces and grip loop-closure reaction wrenches.
4. Contact modes, solver status, and dynamic equilibrium residuals.
5. Handedness, coordinate order, and marker label topologies.
6. Interpolation methods and document SHA256 integrity hashes.
7. Truthful naming of soft trail-arm minimization vs hard zero trail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def check_coordinate_mapping(
    source_coords: Sequence[str],
    target_coords: Sequence[str],
) -> list[int]:
    """Verify coordinate topology and return mapping indices.

    Raises ValueError on dimension mismatch (e.g. 44 vs 41 coordinates)
    or missing coordinate names.
    """
    n_src = len(source_coords)
    n_tgt = len(target_coords)
    if n_src != n_tgt:
        raise ValueError(
            f"Coordinate dimension mismatch: source has {n_src} coords, target has {n_tgt} coords"
        )
    src_map = {name: i for i, name in enumerate(source_coords)}
    indices: list[int] = []
    for name in target_coords:
        if name not in src_map:
            raise ValueError(f"Coordinate '{name}' not found in source coordinate set")
        indices.append(src_map[name])
    return indices


@dataclass(frozen=True)
class CandidatePackage:
    """Full-body matched swing candidate package."""

    time_s: FloatArray
    q: FloatArray
    v: FloatArray
    a: FloatArray
    u: FloatArray
    u_min_trail: FloatArray | None
    u_hard_zero_trail: FloatArray | None
    delta_tau_root: FloatArray
    ground_forces: FloatArray
    grip_wrenches: FloatArray
    contact_modes: BoolArray | NDArray[np.int32]
    solver_status: BoolArray
    equilibrium_residuals: FloatArray
    coordinate_order: tuple[str, ...]
    actuated_indices: IntArray
    handedness: str
    marker_labels: tuple[str, ...]
    predicted_markers_m: FloatArray
    target_markers_m: FloatArray
    marker_valid: BoolArray
    interpolation_method: str
    model_sha256: str
    capture_sha256: str
    schema_version: str = "v2.0"

    def __post_init__(self) -> None:
        n_nodes = len(self.time_s)
        require(n_nodes > 0, "time_s must be non-empty")
        require(self.q.shape[0] == n_nodes, "q row count mismatch")
        require(self.v.shape[0] == n_nodes, "v row count mismatch")
        require(self.a.shape[0] == n_nodes, "a row count mismatch")
        require(self.u.shape[0] == n_nodes, "u row count mismatch")
        require(
            self.delta_tau_root.shape == (n_nodes, 6),
            "delta_tau_root must be shape (N, 6)",
        )
        require(
            len(self.coordinate_order) == self.q.shape[1],
            "coordinate_order length mismatch with q cols",
        )

    def save(self, path: Path | str) -> None:
        """Serialize candidate package to compressed NPZ archive."""
        p = Path(path)
        out_dir = p.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        save_dict: dict[str, Any] = {
            "time_s": self.time_s,
            "coordinate_order": np.array(self.coordinate_order),
            "q": self.q,
            "v": self.v,
            "a": self.a,
            "u": self.u,
            "u_optimum": self.u,
            "actuated": self.actuated_indices,
            "delta_tau_root": self.delta_tau_root,
            "ground_forces": self.ground_forces,
            "grip_wrenches": self.grip_wrenches,
            "contact_modes": self.contact_modes,
            "solver_status": self.solver_status,
            "equilibrium_residuals": self.equilibrium_residuals,
            "handedness": np.array(self.handedness),
            "labels": np.array(self.marker_labels),
            "markers_m": self.predicted_markers_m,
            "target_m": self.target_markers_m,
            "valid": self.marker_valid,
            "interpolation_method": np.array(self.interpolation_method),
            "model_sha256": np.array(self.model_sha256),
            "capture_sha256": np.array(self.capture_sha256),
            "schema_version": np.array(self.schema_version),
        }

        # Save truthful controls and legacy backward compatibility aliases
        if self.u_min_trail is not None:
            save_dict["u_min_trail"] = self.u_min_trail
            save_dict["u_trail_zero"] = self.u_min_trail
        else:
            save_dict["u_trail_zero"] = np.zeros_like(self.u)

        if self.u_hard_zero_trail is not None:
            save_dict["u_hard_zero_trail"] = self.u_hard_zero_trail

        save_dict["grip_wrenches_optimum"] = self.grip_wrenches
        save_dict["grip_wrenches_trail_zero"] = self.grip_wrenches
        save_dict["ground_forces_optimum"] = self.ground_forces
        save_dict["ground_forces_trail_zero"] = self.ground_forces

        np.savez_compressed(p, **save_dict)
        logger.info("Saved candidate package to %s", p)

    @staticmethod
    def _unpack_npz_data(raw: Any) -> dict[str, Any]:
        time_s = np.asarray(raw["time_s"], dtype=np.float64)
        n_nodes = len(time_s)
        q = np.asarray(raw["q"], dtype=np.float64)
        v = np.asarray(raw["v"], dtype=np.float64)
        a = np.asarray(raw["a"], dtype=np.float64)
        u = np.asarray(raw["u"], dtype=np.float64)

        u_min_trail: FloatArray | None = None
        if "u_min_trail" in raw:
            u_min_trail = np.asarray(raw["u_min_trail"], dtype=np.float64)
        elif "u_trail_zero" in raw:
            u_min_trail = np.asarray(raw["u_trail_zero"], dtype=np.float64)

        u_hard_zero_trail: FloatArray | None = None
        if "u_hard_zero_trail" in raw:
            u_hard_zero_trail = np.asarray(raw["u_hard_zero_trail"], dtype=np.float64)

        delta_tau_root = (
            np.asarray(raw["delta_tau_root"], dtype=np.float64)
            if "delta_tau_root" in raw
            else np.zeros((n_nodes, 6), dtype=np.float64)
        )
        coord_order = tuple(str(x) for x in raw["coordinate_order"])
        actuated = (
            np.asarray(raw["actuated"], dtype=np.int64)
            if "actuated" in raw
            else np.arange(6, len(coord_order))
        )
        labels_key = "labels" if "labels" in raw else "marker_labels"
        pred_key = "markers_m" if "markers_m" in raw else "predicted_markers_m"
        targ_key = "target_m" if "target_m" in raw else "target_markers_m"

        return {
            "time_s": time_s,
            "q": q,
            "v": v,
            "a": a,
            "u": u,
            "u_min_trail": u_min_trail,
            "u_hard_zero_trail": u_hard_zero_trail,
            "delta_tau_root": delta_tau_root,
            "ground_forces": np.asarray(raw["ground_forces"], dtype=np.float64),
            "grip_wrenches": np.asarray(raw["grip_wrenches"], dtype=np.float64),
            "contact_modes": np.asarray(raw["contact_modes"])
            if "contact_modes" in raw
            else np.ones((n_nodes, 6), dtype=bool),
            "solver_status": np.asarray(raw["solver_status"], dtype=bool)
            if "solver_status" in raw
            else np.ones(n_nodes, dtype=bool),
            "equilibrium_residuals": np.asarray(
                raw["equilibrium_residuals"], dtype=np.float64
            )
            if "equilibrium_residuals" in raw
            else np.zeros(n_nodes, dtype=np.float64),
            "coordinate_order": coord_order,
            "actuated_indices": actuated,
            "handedness": str(raw["handedness"])
            if "handedness" in raw
            else "right_handed",
            "marker_labels": tuple(str(x) for x in raw[labels_key]),
            "predicted_markers_m": np.asarray(raw[pred_key], dtype=np.float64),
            "target_markers_m": np.asarray(raw[targ_key], dtype=np.float64),
            "marker_valid": np.asarray(raw["valid"], dtype=bool),
            "interpolation_method": str(raw["interpolation_method"])
            if "interpolation_method" in raw
            else "pchip",
            "model_sha256": str(raw["model_sha256"]) if "model_sha256" in raw else "",
            "capture_sha256": str(raw["capture_sha256"])
            if "capture_sha256" in raw
            else "",
            "schema_version": str(raw["schema_version"])
            if "schema_version" in raw
            else "v2.0",
        }

    @classmethod
    def load(cls, path: Path | str) -> CandidatePackage:
        """Load candidate package from NPZ with legacy compatibility."""
        p = Path(path)
        with np.load(p, allow_pickle=False) as raw:
            data = cls._unpack_npz_data(raw)
        return cls(**data)
