"""MuJoCo-native inverse kinematics adapter using mujoco.minimize (MS-16 #10366).

Provides full-body marker IK via `mujoco.minimize.least_squares` with analytical
site/marker Jacobians and native joint-limit box bounds.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.full_body_ik import (
    FullBodyMarkerKinematics,
)
from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import (
    PoseFit,
    SolvePoseOptions,
    _PosePrep,
)

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]
Attachment: TypeAlias = tuple[str, tuple[float, float, float]]

_FINITE_BOUND_CLAMP: float = 1e4


class MujocoMinimizeFullBodyIK(FullBodyMarkerKinematics):
    """Full-body IK solver leveraging `mujoco.minimize.least_squares`."""

    def __init__(
        self,
        adapter: NativeMujocoFullBodyModel,
        attachments: Mapping[str, Attachment],
    ) -> None:
        super().__init__(adapter, attachments)

    def _build_residual_and_jacobian(
        self,
        targets: Array,
        ground: GroundPlane,
        opts: SolvePoseOptions,
        prep: _PosePrep,
        q_init: Array,
    ) -> tuple[Any, Any]:
        """Construct residual and analytical Jacobian callables for least_squares."""
        rot_w = (
            opts.closure_weight
            if opts.closure_rotation_weight is None
            else opts.closure_rotation_weight
        )

        def eval_all(q_k: Array) -> tuple[Array, Array]:
            self._set(q_k)
            positions = self._positions()
            jac = self._marker_jacobian(positions)
            rows: list[Array] = [
                prep.row_scale * (positions[prep.mask] - targets[prep.mask]).reshape(-1)
            ]
            jacs: list[Array] = [
                prep.row_scale[:, None] * jac[prep.mask].reshape(-1, prep.nv)
            ]
            rows.append(prep.sqrt_prior * (q_k - q_init))
            jacs.append(prep.sqrt_prior * np.eye(prep.nv))
            self._append_closure(rows, jacs, opts.closure_weight, rot_w)
            self._append_ground(rows, jacs, ground, opts.ground_weight, prep.pinned)
            self._append_anchors(rows, jacs, prep.planted, opts.ground_weight)
            self._append_balance(rows, jacs, ground, opts.balance_weight)
            self._append_com_target(rows, jacs, ground, prep.com_goal)
            self._append_axes(rows, jacs, prep.axes)
            r_vec = np.concatenate(rows)
            j_mat = np.concatenate(jacs)[:, prep.free]
            return r_vec, j_mat

        def residual_fn(x: Array) -> Array:
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.ndim == 1:
                x_arr = x_arr[:, None]
            _, n_cols = x_arr.shape
            r_cols: list[Array] = []
            for col in range(n_cols):
                q_k = prep.q.copy()
                q_k[prep.free] = x_arr[:, col]
                r_vec, _ = eval_all(q_k)
                r_cols.append(r_vec)
            if n_cols > 1:
                return np.column_stack(r_cols)
            return np.asarray(r_cols[0], dtype=np.float64).reshape(-1, 1)

        def jacobian_fn(x: Array, r: Array) -> Array:
            x_arr = np.asarray(x, dtype=np.float64).ravel()
            q_k = prep.q.copy()
            q_k[prep.free] = x_arr
            _, j_mat = eval_all(q_k)
            return np.asarray(j_mat, dtype=np.float64)

        return residual_fn, jacobian_fn

    @precondition(
        lambda self, targets, valid, q_init, ground, options=None, **kwargs: (
            isinstance(targets, np.ndarray) and isinstance(valid, np.ndarray)
        ),
        "targets and valid must be numpy arrays",
    )
    @postcondition(
        lambda result: bool(
            isinstance(result, PoseFit) and np.isfinite(result.q).all()
        ),
        "PoseFit must contain finite coordinates",
    )
    def solve_pose(
        self,
        targets: Array,
        valid: NDArray[Any],
        q_init: Array,
        *,
        ground: GroundPlane,
        options: SolvePoseOptions | None = None,
        **kwargs: Any,
    ) -> PoseFit:
        """Least-squares pose using `mujoco.minimize.least_squares`."""
        import mujoco.minimize as mm

        opts = SolvePoseOptions(**kwargs) if options is None else options
        targets_arr = np.asarray(targets, dtype=float)
        prep = self._prepare_pose_fit(targets_arr, valid, q_init, ground, opts)

        res_fn, jac_fn = self._build_residual_and_jacobian(
            targets_arr, ground, opts, prep, q_init
        )

        low_free = np.where(
            np.isneginf(prep.low[prep.free]),
            -_FINITE_BOUND_CLAMP,
            prep.low[prep.free],
        )
        high_free = np.where(
            np.isposinf(prep.high[prep.free]),
            _FINITE_BOUND_CLAMP,
            prep.high[prep.free],
        )

        x0 = prep.q[prep.free].copy()
        x0 = np.clip(x0, low_free + 1e-12, high_free - 1e-12)

        x_sol, trace = mm.least_squares(
            x0=x0,
            residual=res_fn,
            bounds=(low_free, high_free),
            jacobian=jac_fn,
            max_iter=int(opts.iterations),
            verbose=0,
        )

        q = prep.q.copy()
        q[prep.free] = np.clip(x_sol.ravel(), prep.low[prep.free], prep.high[prep.free])

        self._set(q)
        positions = self._positions()
        diff = positions - targets_arr
        errors = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        rms = (
            float(np.sqrt(np.mean(errors[prep.mask] ** 2))) if prep.mask.any() else 0.0
        )
        per_marker = {
            label: float(errors[k])
            for k, label in enumerate(self.labels)
            if prep.mask[k]
        }
        heights = self._sphere_heights(ground)
        pos_err, rot_err = self.closure_error(q)

        return PoseFit(
            q=q,
            marker_rms_m=rms,
            per_marker_m=per_marker,
            closure_error_m=pos_err,
            closure_error_rad=rot_err,
            lowest_sphere_height_m=min(heights.values()),
            iterations=len(trace),
        )
