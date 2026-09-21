"""Identifiability analysis and null-direction resolution for kinematic models (#10361, #9769).

Provides forward kinematics, numerical Jacobians, SVD singular spectrum analysis,
and detection of unobservable/weakly observable degrees of freedom. Supports resolving
identifiability bottlenecks (such as hip vs. trunk rotation about the spine axis)
via anthropometric prior regularization or off-axis marker attachments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.full_body_spec import order_full_body_joints

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Outcome of linearised marker-identifiability analysis at a given pose."""

    dof_names: tuple[str, ...]
    singular_values: Array
    rank: int
    tolerance: float
    condition_number: float
    unobservable_dofs: tuple[str, ...]
    weakly_observable_dofs: tuple[str, ...]
    reliable_dofs: tuple[str, ...]
    nullspace_directions: Mapping[str, Array]
    right_singular_vectors: Array

    @property
    def is_full_rank(self) -> bool:
        """Return True when the Jacobian attains full column rank."""
        return self.rank == len(self.dof_names)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a JSON-compatible dictionary."""
        return {
            "dof_names": list(self.dof_names),
            "n_dof": len(self.dof_names),
            "rank": self.rank,
            "is_full_rank": self.is_full_rank,
            "tolerance": float(self.tolerance),
            "condition_number": float(self.condition_number),
            "singular_values": [float(x) for x in self.singular_values],
            "unobservable_dofs": list(self.unobservable_dofs),
            "weakly_observable_dofs": list(self.weakly_observable_dofs),
            "reliable_dofs": list(self.reliable_dofs),
            "nullspace_directions": {
                k: [float(x) for x in v] for k, v in self.nullspace_directions.items()
            },
        }


def _finite_difference_jacobian(
    model: Callable[[Array], Array],
    q0: Array,
    step: float = 1e-6,
) -> Array:
    """Compute central finite-difference Jacobian of model around q0."""
    baseline = np.asarray(model(q0), dtype=float)
    m = baseline.size
    n = q0.size
    jac = np.empty((m, n), dtype=np.float64)
    for col in range(n):
        delta = np.zeros(n, dtype=np.float64)
        local_step = step * max(1.0, abs(float(q0[col])))
        delta[col] = local_step
        plus = np.asarray(model(q0 + delta), dtype=float)
        minus = np.asarray(model(q0 - delta), dtype=float)
        jac[:, col] = (plus - minus) / (2.0 * local_step)
    return jac


def _classify_dofs(
    names: Sequence[str],
    singular_values: Array,
    right_vectors: Array,
    tol: float,
    weak_cut: float,
    share: float,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Array]]:
    """Classify DOFs into unobservable, weakly observable, and reliable subsets."""
    null_dirs: dict[str, Array] = {}

    def implicated(low: float, high: float) -> list[str]:
        found: list[str] = []
        for idx, sigma in enumerate(singular_values):
            if not (low <= sigma <= high):
                continue
            direction = np.abs(right_vectors[:, idx])
            dominant = float(direction.max())
            if dominant <= 0.0:
                continue
            for dof_idx in np.flatnonzero(direction >= share * dominant):
                name = names[int(dof_idx)]
                if name not in found:
                    found.append(name)
        return found

    for idx, sigma in enumerate(singular_values):
        if sigma <= tol:
            null_dirs[f"null_{idx}"] = right_vectors[:, idx].copy()

    unobs = implicated(-float("inf"), tol)
    weak = [d for d in implicated(tol, weak_cut) if d not in unobs]
    reliable = [d for d in names if d not in unobs and d not in weak]
    return tuple(unobs), tuple(weak), tuple(reliable), null_dirs


def _decompose_jacobian(
    jacobian: Array,
    names: Sequence[str],
    relative_tolerance: float,
    weak_tolerance: float,
    share: float,
    prior_weight: float,
) -> IdentifiabilityResult:
    """Perform SVD and construct an IdentifiabilityResult."""
    n = len(names)
    if prior_weight > 0.0:
        prior_mat = np.sqrt(prior_weight) * np.eye(n, dtype=np.float64)
        j_eval = np.vstack([jacobian, prior_mat])
    else:
        j_eval = jacobian

    _, s, vt = np.linalg.svd(j_eval, full_matrices=False)
    v = vt.T
    sigma_max = float(s[0]) if s.size > 0 else 0.0
    tol = relative_tolerance * sigma_max
    weak_cut = weak_tolerance * sigma_max
    rank = int(np.count_nonzero(s > tol))
    cond = float(s[0] / s[-1]) if (s.size > 0 and s[-1] > 0.0) else float("inf")

    unobs, weak, rel, null_dirs = _classify_dofs(names, s, v, tol, weak_cut, share)
    return IdentifiabilityResult(
        dof_names=tuple(names),
        singular_values=s,
        rank=rank,
        tolerance=tol,
        condition_number=cond,
        unobservable_dofs=unobs,
        weakly_observable_dofs=weak,
        reliable_dofs=rel,
        nullspace_directions=MappingProxyType(null_dirs),
        right_singular_vectors=v,
    )


@precondition(
    lambda dof_names, observation_fn, q, **kwargs: len(dof_names) == len(q),
    "q must match dof_names length",
)
@postcondition(
    lambda r: isinstance(r, IdentifiabilityResult),
    "must return IdentifiabilityResult",
)
def probe_synthetic_chain_identifiability(
    dof_names: Sequence[str],
    observation_fn: Callable[[Array], Array],
    q: Array,
    *,
    step: float = 1e-6,
    relative_tolerance: float = 1e-6,
    weak_tolerance: float = 1e-1,
    share: float = 0.5,
    prior_weight: float = 0.0,
) -> IdentifiabilityResult:
    """Evaluate identifiability for a synthetic kinematic chain."""
    q0 = np.asarray(q, dtype=np.float64).reshape(-1)
    jac = _finite_difference_jacobian(observation_fn, q0, step=step)
    return _decompose_jacobian(
        jac, dof_names, relative_tolerance, weak_tolerance, share, prior_weight
    )


def body_poses_from_coordinates(
    spec: Mapping[str, Any],
    q: Mapping[str, float] | Sequence[float] | Array | None = None,
    coordinate_names: Sequence[str] | None = None,
) -> dict[str, Array]:
    """Compute rigid body transforms from joint coordinate values."""
    ordered_joints = order_full_body_joints(spec)
    all_coords = tuple(spec["coordinate_order"])
    coord_map: dict[str, float] = {}

    if isinstance(q, Mapping):
        coord_map = {k: float(v) for k, v in q.items()}
    elif q is not None:
        names = coordinate_names or all_coords
        coord_map = {name: float(val) for name, val in zip(names, q, strict=True)}
    else:
        seed = spec.get("address_seed_deg", {})
        for name in all_coords:
            val_deg = seed.get(name, 0.0)
            coord_map[name] = float(np.deg2rad(val_deg))

    offsets: dict[str, Array] = {"world": np.eye(4, dtype=np.float64)}
    poses: dict[str, Array] = {"world": np.eye(4, dtype=np.float64)}

    for joint in ordered_joints:
        parent = joint["parent"]
        child = joint["child"]
        child_to_follower = np.asarray(joint["child_to_follower"], dtype=np.float64)
        offsets[child] = np.asarray(np.linalg.inv(child_to_follower), dtype=np.float64)

        parent_to_base = np.asarray(joint["parent_to_base"], dtype=np.float64)
        t_base = poses[parent] @ offsets[parent] @ parent_to_base
        curr_pos = t_base[:3, 3].copy()
        curr_r = t_base[:3, :3].copy()

        for prim in joint["primitives"]:
            kind = prim["primitive"]
            val = coord_map.get(prim["coordinate"], 0.0)
            axis = np.eye(3)["xyz".index(kind[1])]
            if kind[0] == "P":
                curr_pos = curr_pos + curr_r @ (axis * val)
            elif kind[0] == "R":
                curr_r = curr_r @ Rotation.from_rotvec(axis * val).as_matrix()

        t_child = np.eye(4, dtype=np.float64)
        t_child[:3, :3] = curr_r
        t_child[:3, 3] = curr_pos
        poses[child] = t_child

    return poses


def compute_spec_marker_positions(
    spec: Mapping[str, Any],
    q: Mapping[str, float] | Sequence[float] | Array | None = None,
    *,
    marker_offsets: Mapping[str, tuple[str, Sequence[float]]] | None = None,
    coordinate_names: Sequence[str] | None = None,
) -> dict[str, Array]:
    """Compute 3D world positions of all valid markers for a given coordinate state."""
    poses = body_poses_from_coordinates(spec, q, coordinate_names=coordinate_names)
    frames_map: dict[str, tuple[str, Array]] = {
        f["name"]: (f["body"], np.asarray(f["placement"], dtype=np.float64))
        for f in spec.get("frames", [])
    }

    positions: dict[str, Array] = {}
    for label, att in spec.get("marker_attachments", {}).items():
        if marker_offsets and label in marker_offsets:
            body_name, offset = marker_offsets[label]
        else:
            body_name = att["body"]
            offset = att["offset_m"]

        if offset is None:
            continue

        if body_name in poses:
            t_body = poses[body_name]
        elif body_name in frames_map:
            parent_body, placement = frames_map[body_name]
            t_body = poses[parent_body] @ placement
        else:
            continue

        pos_world = t_body[:3, 3] + t_body[:3, :3] @ np.asarray(
            offset, dtype=np.float64
        )
        positions[label] = pos_world

    return positions


@precondition(
    lambda spec, **kwargs: "coordinate_order" in spec, "spec needs coordinate_order"
)
@postcondition(
    lambda r: isinstance(r, IdentifiabilityResult), "must return IdentifiabilityResult"
)
def probe_spec_identifiability(
    spec: Mapping[str, Any],
    q: Mapping[str, float] | Sequence[float] | Array | None = None,
    *,
    marker_offsets: Mapping[str, tuple[str, Sequence[float]]] | None = None,
    relative_tolerance: float = 1e-6,
    weak_tolerance: float = 1e-1,
    share: float = 0.5,
    prior_weight: float = 0.0,
) -> IdentifiabilityResult:
    """Compute marker identifiability for a full-body model specification."""
    coord_names = tuple(spec["coordinate_order"])
    all_coords = coord_names

    if isinstance(q, Mapping):
        q_vec = np.array([float(q.get(c, 0.0)) for c in all_coords], dtype=np.float64)
    elif q is not None:
        q_vec = np.asarray(q, dtype=np.float64)
    else:
        seed = spec.get("address_seed_deg", {})
        q_vec = np.array(
            [float(np.deg2rad(seed.get(c, 0.0))) for c in all_coords],
            dtype=np.float64,
        )

    # Determine which markers are observable
    sample_markers = compute_spec_marker_positions(
        spec, q_vec, marker_offsets=marker_offsets, coordinate_names=all_coords
    )
    obs_labels = sorted(sample_markers.keys())

    def observation_fn(values: Array) -> Array:
        markers = compute_spec_marker_positions(
            spec, values, marker_offsets=marker_offsets, coordinate_names=all_coords
        )
        if not obs_labels:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate([markers[k] for k in obs_labels])

    return probe_synthetic_chain_identifiability(
        all_coords,
        observation_fn,
        q_vec,
        relative_tolerance=relative_tolerance,
        weak_tolerance=weak_tolerance,
        share=share,
        prior_weight=prior_weight,
    )
