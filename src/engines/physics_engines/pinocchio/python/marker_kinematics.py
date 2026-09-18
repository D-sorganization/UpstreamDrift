"""Marker kinematics on the Pinocchio full-body plant (MS-31, #10338).

Marker world positions and their configuration Jacobians for a document's
``marker_attachments`` (which name either a body or a frame of the plant),
plus a damped Gauss-Newton marker IK used as the Crocoddyl warm start.
This is the only module of the fit that touches ``pinocchio`` kinematics
directly; the plant's dynamics stay behind ``FullBodyPinocchioModel``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CoordinateMap:
    """Permutation between the document coordinate order and Pinocchio's q/v layout."""

    names: tuple[str, ...]
    q_index: NDArray[np.int64]
    v_index: NDArray[np.int64]

    @classmethod
    def from_plant(cls, plant: Any) -> CoordinateMap:
        names = tuple(plant.coordinate_order)
        q_index = np.array([plant._coordinates[name] for name in names], dtype=np.int64)
        v_index = np.array(
            [plant._velocity_indices[name] for name in names], dtype=np.int64
        )
        ensure(
            len(set(q_index.tolist())) == len(names),
            "coordinate q indices must be unique",
        )
        return cls(names, q_index, v_index)

    @property
    def n(self) -> int:
        return len(self.names)

    def to_pin_q(self, q_ordered: Array, nq: int) -> Array:
        q_pin = np.zeros(nq)
        q_pin[self.q_index] = q_ordered
        return q_pin

    def columns_from_pin(self, jacobian_pin: Array) -> Array:
        """Reorder the last axis of a Pinocchio (…, nv) Jacobian into coordinate order."""
        return jacobian_pin[..., self.v_index]

    def as_dict(self, values: Array) -> dict[str, float]:
        return {
            name: float(value) for name, value in zip(self.names, values, strict=True)
        }


@dataclass(frozen=True)
class MarkerTable:
    """Markers resolved to (joint id, placement in joint frame) on the plant model."""

    labels: tuple[str, ...]
    joint_ids: tuple[int, ...]
    local_points: Array  # (markers, 3) in the parent joint frame


def _resolve_attachment(
    plant: Any, body_name: str, offset_m: Sequence[float]
) -> tuple[int, Array]:
    offset = np.asarray(offset_m, dtype=float)
    require(offset.shape == (3,), "marker offset must be a 3-vector", offset.shape)
    if body_name in plant._bodies:
        joint_id, placement = plant._bodies[body_name]
        return int(joint_id), np.asarray(placement.act(offset), dtype=float)
    if body_name in plant._frames:
        frame = plant.model.frames[plant._frames[body_name]]
        parent = getattr(frame, "parentJoint", None)
        if parent is None:
            parent = frame.parent
        return int(parent), np.asarray(frame.placement.act(offset), dtype=float)
    raise KeyError(
        f"marker attachment '{body_name}' is neither a body nor a frame of the plant"
    )


def build_marker_table(
    plant: Any, attachments: Mapping[str, Mapping[str, Any]], labels: Sequence[str]
) -> MarkerTable:
    """Resolve ``labels`` through ``attachments`` (``{label: {body, offset_m}}``)."""
    joint_ids: list[int] = []
    points: list[Array] = []
    for label in labels:
        require(label in attachments, f"no attachment for marker {label}", label)
        joint_id, local = _resolve_attachment(
            plant, str(attachments[label]["body"]), attachments[label]["offset_m"]
        )
        joint_ids.append(joint_id)
        points.append(local)
    return MarkerTable(tuple(labels), tuple(joint_ids), np.vstack(points))


def marker_positions(
    pin: Any, model: Any, data: Any, q_pin: Array, table: MarkerTable
) -> Array:
    """World marker positions (markers, 3) after forward kinematics at ``q_pin``."""
    pin.forwardKinematics(model, data, q_pin)
    out = np.empty((len(table.labels), 3))
    for row, (joint_id, local) in enumerate(
        zip(table.joint_ids, table.local_points, strict=True)
    ):
        out[row] = data.oMi[joint_id].act(local)
    return out


def marker_positions_and_jacobians(
    pin: Any, model: Any, data: Any, q_pin: Array, table: MarkerTable
) -> tuple[Array, Array]:
    """World positions (markers, 3) and Jacobians (markers, 3, nv) in Pinocchio's v layout.

    The point velocity of a marker rigidly attached to a joint is
    ``v_lin - r x omega`` with ``r`` the lever from the joint origin, so the
    position Jacobian is ``J_lin - [r]_x J_ang`` in the local-world-aligned frame.
    """
    pin.computeJointJacobians(model, data, q_pin)
    positions = np.empty((len(table.labels), 3))
    jacobians = np.empty((len(table.labels), 3, model.nv))
    frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    for row, (joint_id, local) in enumerate(
        zip(table.joint_ids, table.local_points, strict=True)
    ):
        placement = data.oMi[joint_id]
        point = placement.act(local)
        positions[row] = point
        jac6 = pin.getJointJacobian(model, data, joint_id, frame)
        lever = point - placement.translation
        skew = np.array(
            [
                [0.0, -lever[2], lever[1]],
                [lever[2], 0.0, -lever[0]],
                [-lever[1], lever[0], 0.0],
            ]
        )
        jacobians[row] = jac6[:3] - skew @ jac6[3:]
    return positions, jacobians


@dataclass(frozen=True)
class MarkerIkOptions:
    iterations: int = 40
    damping: float = 1e-3
    closure_weight: float = 1e4
    regularisation: float = 1e-4
    marker_weight: float = 1.0
    ground_barrier_weight: float = 5e4
    step_limit_rad: float = 0.5
    convergence_tol: float = 1e-5
    velocity_extrapolation: bool = True


class MarkerIkSolver:
    """Damped Gauss-Newton marker IK with weld closure, ground barrier, and box bounds.

    Residuals:
    1. Weighted marker errors (valid markers only) with category weighting.
    2. Six-component weld closure error from the plant.
    3. Unilateral ground barrier: foot contact spheres cannot penetrate below ground.
    4. Velocity-extrapolated regularisation toward prior trajectory motion.
    Bounds are enforced by clipping.
    """

    def __init__(
        self,
        pin: Any,
        plant: Any,
        table: MarkerTable,
        lower: Array,
        upper: Array,
        options: MarkerIkOptions | None = None,
    ) -> None:
        self._pin = pin
        self._plant = plant
        self._model = plant.model
        self._data = plant.model.createData()
        self._table = table
        self._map = CoordinateMap.from_plant(plant)
        self._lower = np.asarray(lower, dtype=float)
        self._upper = np.asarray(upper, dtype=float)
        self._options = options or MarkerIkOptions()
        require(
            self._lower.shape == (self._map.n,),
            "lower bounds must match coordinate count",
        )

        # Cache contact spheres and ground height for non-penetration barrier
        self._ground_height_m = (
            float(plant.ground.height_m)
            if hasattr(plant, "ground") and plant.ground is not None
            else 0.0
        )
        self._contact_fids = (
            tuple(plant._contact_frames[s.name] for s in plant.contact_spheres)
            if hasattr(plant, "contact_spheres") and hasattr(plant, "_contact_frames")
            else ()
        )
        self._contact_radii = (
            np.array([s.radius_m for s in plant.contact_spheres], dtype=float)
            if hasattr(plant, "contact_spheres")
            else np.zeros(0)
        )
        rf_type = pin.ReferenceFrame
        self._ref_frame = rf_type.LOCAL_WORLD_ALIGNED

    @property
    def coordinate_map(self) -> CoordinateMap:
        return self._map

    def markers(self, q_ordered: Array) -> Array:
        return marker_positions(
            self._pin,
            self._model,
            self._data,
            self._map.to_pin_q(q_ordered, self._model.nq),
            self._table,
        )

    def _residual_and_jacobian(
        self,
        q: Array,
        target: Array,
        valid: NDArray[np.bool_],
        weights: Array,
        q_prior: Array,
        closure_weight: float | None = None,
    ) -> tuple[Array, Array]:
        opts = self._options
        positions, jac_pin = marker_positions_and_jacobians(
            self._pin,
            self._model,
            self._data,
            self._map.to_pin_q(q, self._model.nq),
            self._table,
        )
        jac = self._map.columns_from_pin(jac_pin)
        rows = np.flatnonzero(valid)
        scale = np.sqrt(opts.marker_weight * weights[rows])[:, None]
        marker_res = (scale * (positions[rows] - target[rows])).reshape(-1)
        marker_jac = (scale[:, :, None] * jac[rows]).reshape(-1, self._map.n)

        # Closure
        closure = self._plant.closure_position_linearization(self._map.as_dict(q))
        closure_scale = np.sqrt(
            opts.closure_weight if closure_weight is None else closure_weight
        )
        closure_res = closure_scale * np.asarray(closure.position)
        closure_jac = closure_scale * np.asarray(closure.jacobian)

        # Ground barrier: foot contact spheres cannot penetrate underground
        g_res_list: list[float] = []
        g_jac_list: list[Array] = []
        if self._contact_fids and opts.ground_barrier_weight > 0.0:
            self._pin.updateFramePlacements(self._model, self._data)
            ref_frame = self._ref_frame
            g_scale = np.sqrt(opts.ground_barrier_weight)
            for fid, radius in zip(
                self._contact_fids, self._contact_radii, strict=True
            ):
                sphere_pos = self._data.oMf[fid].translation
                bottom_z = float(sphere_pos[2]) - radius
                penetration = self._ground_height_m - bottom_z
                if penetration > 0.0:
                    g_res_list.append(g_scale * penetration)
                    j_raw = self._pin.getFrameJacobian(
                        self._model, self._data, fid, ref_frame
                    )
                    j_lin_z = j_raw[2:3, :]
                    j_coord = self._map.columns_from_pin(j_lin_z)
                    g_jac_list.append(-g_scale * j_coord[0])

        if g_res_list:
            g_res = np.array(g_res_list)
            g_jac = np.vstack(g_jac_list)
        else:
            g_res = np.zeros(0)
            g_jac = np.zeros((0, self._map.n))

        # Regularisation toward velocity prior
        reg_scale = np.sqrt(opts.regularisation)
        reg_res = reg_scale * (q - q_prior)
        reg_jac = reg_scale * np.eye(self._map.n)

        residual = np.concatenate([marker_res, closure_res, g_res, reg_res])
        jacobian = np.vstack([marker_jac, closure_jac, g_jac, reg_jac])
        return residual, jacobian

    def solve_frame(
        self,
        target: Array,
        valid: NDArray[np.bool_],
        weights: Array,
        q_init: Array,
        *,
        q_prior: Array | None = None,
        iterations: int | None = None,
        closure_weight: float | None = None,
    ) -> tuple[Array, float, float]:
        """Return (q, marker RMS over valid markers, closure position error norm)."""
        opts = self._options
        q = np.clip(np.asarray(q_init, dtype=float), self._lower, self._upper)
        prior = q.copy() if q_prior is None else np.asarray(q_prior, dtype=float)
        damping = opts.damping
        residual, jacobian = self._residual_and_jacobian(
            q, target, valid, weights, prior, closure_weight
        )
        cost = float(residual @ residual)
        for _ in range(iterations or opts.iterations):
            normal = jacobian.T @ jacobian
            step = -np.linalg.solve(
                normal + damping * np.eye(self._map.n), jacobian.T @ residual
            )
            norm = float(np.max(np.abs(step)))
            if norm > opts.step_limit_rad:
                step *= opts.step_limit_rad / norm
            q_trial = np.clip(q + step, self._lower, self._upper)
            residual_trial, jacobian_trial = self._residual_and_jacobian(
                q_trial, target, valid, weights, prior, closure_weight
            )
            cost_trial = float(residual_trial @ residual_trial)
            if cost_trial < cost:
                q, residual, jacobian, cost = (
                    q_trial,
                    residual_trial,
                    jacobian_trial,
                    cost_trial,
                )
                damping = max(damping / 3.0, 1e-9)
            else:
                damping *= 10.0
            if norm < opts.convergence_tol:
                break
        positions = self.markers(q)
        rows = np.flatnonzero(valid)
        rms = (
            float(
                np.sqrt(np.mean(np.sum((positions[rows] - target[rows]) ** 2, axis=1)))
            )
            if rows.size
            else float("nan")
        )
        closure = self._plant.closure_position_linearization(self._map.as_dict(q))
        return q, rms, float(np.linalg.norm(closure.position))

    def solve_address(
        self,
        target: Array,
        valid: NDArray[np.bool_],
        weights: Array,
        q_init: Array,
        *,
        iterations: int = 60,
    ) -> Array:
        """Marker-first address solve: ramp the closure weight so the weld cannot trap the pose."""
        q = np.asarray(q_init, dtype=float)
        for closure_weight in (0.0, 1.0, 1e2, self._options.closure_weight):
            q, _, _ = self.solve_frame(
                target,
                valid,
                weights,
                q,
                q_prior=q,
                iterations=iterations,
                closure_weight=closure_weight,
            )
        return q

    def solve_trajectory(
        self,
        targets: Array,
        valid: NDArray[np.bool_],
        weights: Array,
        q_init: Array,
        *,
        first_frame_iterations: int = 60,
    ) -> tuple[Array, Array, Array]:
        """Sequential per-frame IK; returns (q (nodes, n), rms (nodes,), closure (nodes,))."""
        n_nodes = targets.shape[0]
        q_out = np.empty((n_nodes, self._map.n))
        rms = np.empty(n_nodes)
        closure = np.empty(n_nodes)
        q_history: list[Array] = []
        q_curr = np.asarray(q_init, dtype=float)

        for node in range(n_nodes):
            if node == 0:
                q_curr = self.solve_address(
                    targets[0],
                    valid[0],
                    weights,
                    q_curr,
                    iterations=first_frame_iterations,
                )
                q_prior = q_curr.copy()
            elif node == 1:
                q_prior = q_history[-1].copy()
            else:
                if self._options.velocity_extrapolation:
                    q_prior = 2.0 * q_history[-1] - q_history[-2]
                else:
                    q_prior = q_history[-1].copy()

            q_curr, rms[node], closure[node] = self.solve_frame(
                targets[node],
                valid[node],
                weights,
                q_curr,
                q_prior=q_prior,
            )
            q_history.append(q_curr.copy())
            q_out[node] = q_curr

        ensure(bool(np.isfinite(q_out).all()), "IK trajectory must be finite")
        return q_out, rms, closure
