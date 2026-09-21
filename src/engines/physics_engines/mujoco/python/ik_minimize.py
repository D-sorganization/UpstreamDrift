"""MuJoCo native marker IK via ``mujoco.minimize.least_squares`` (MS-16 #10366)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.full_body_ik import (
    FullBodyMarkerKinematics,
)
from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import (
    PoseFit,
    SolvePoseOptions,
    _run_lm_loop,
)

Array = NDArray[np.float64]


class MinimizeMarkerKinematics(FullBodyMarkerKinematics):
    """Marker IK using MuJoCo's bundled Gauss-Newton least-squares optimizer."""

    def _marker_rms(
        self,
        q: Array,
        targets_arr: Array,
        mask: NDArray[np.bool_],
    ) -> float:
        self._set(q)
        diff = self._positions() - targets_arr
        errors = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        return float(np.sqrt(np.mean(errors[mask] ** 2))) if mask.any() else 0.0

    def _solve_with_minimize(
        self,
        prep: Any,
        targets_arr: Array,
        q_init: Array,
        ground: GroundPlane,
        opts: SolvePoseOptions,
    ) -> Array:
        from mujoco import minimize
        from mujoco.minimize import Verbosity

        def unpack(x_free: Array) -> Array:
            x = np.asarray(x_free, dtype=float).reshape(-1)
            q = prep.q.copy()
            q[prep.free] = x
            return np.clip(q, prep.low, prep.high)

        def residual_and_jacobian(q_k: Array) -> tuple[Array, Array]:
            self._set(q_k)
            positions = self._positions()
            jac = self._marker_jacobian(positions)
            rows = [
                prep.row_scale
                * (positions[prep.mask] - targets_arr[prep.mask]).reshape(-1)
            ]
            jacs = [prep.row_scale[:, None] * jac[prep.mask].reshape(-1, prep.nv)]
            rows.append(prep.sqrt_prior * (q_k - q_init))
            jacs.append(prep.sqrt_prior * np.eye(prep.nv))
            rot_w = (
                opts.closure_weight
                if opts.closure_rotation_weight is None
                else opts.closure_rotation_weight
            )
            self._append_closure(rows, jacs, opts.closure_weight, rot_w)
            self._append_ground(rows, jacs, ground, opts.ground_weight, prep.pinned)
            self._append_anchors(rows, jacs, prep.planted, opts.ground_weight)
            self._append_balance(rows, jacs, ground, opts.balance_weight)
            self._append_com_target(rows, jacs, ground, prep.com_goal)
            self._append_axes(rows, jacs, prep.axes)
            return np.concatenate(rows), np.concatenate(jacs)[:, prep.free]

        warm_iters = max(5, opts.iterations - max(1, opts.iterations // 3))
        q_seed, _ = _run_lm_loop(
            residual_and_jacobian,
            prep.q,
            prep.free,
            prep.low,
            prep.high,
            iterations=warm_iters,
            damping=opts.damping,
            tolerance_m=opts.tolerance_m,
        )
        seed_rms = self._marker_rms(q_seed, targets_arr, prep.mask)

        def residual(x_free: Array) -> Array:
            res, _ = residual_and_jacobian(unpack(x_free))
            return res

        def jacobian(x_free: Array, _res: Array) -> Array:
            _, jac = residual_and_jacobian(unpack(x_free))
            return jac

        lower = prep.low[prep.free]
        upper = prep.high[prep.free]
        bounds = None
        if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
            bounds = (lower, upper)
        x_opt, _trace = minimize.least_squares(
            q_seed[prep.free].reshape(-1, 1),
            residual,
            bounds=bounds,
            jacobian=jacobian,
            max_iter=opts.iterations,
            verbose=Verbosity.SILENT,
            mu_max=1e16,
        )
        q_min = unpack(x_opt.reshape(-1))
        return (
            q_min
            if self._marker_rms(q_min, targets_arr, prep.mask) <= seed_rms
            else q_seed
        )

    @precondition(
        lambda self, targets, valid, q_init, ground=None, **_: targets is not None,
        "targets required",
    )
    @postcondition(
        lambda result, *_a, **_k: bool(np.isfinite(result.q).all()),
        "IK output must be finite",
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
        """Least-squares pose solve using ``mujoco.minimize.least_squares``."""
        opts = SolvePoseOptions(**kwargs) if options is None else options
        targets_arr = np.asarray(targets, dtype=float)
        prep = self._prepare_pose_fit(targets_arr, valid, q_init, ground, opts)
        q = self._solve_with_minimize(prep, targets_arr, q_init, ground, opts)
        self._set(q)
        diff = self._positions() - targets_arr
        errors = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        per_marker = {
            label: float(errors[k])
            for k, label in enumerate(self.labels)
            if prep.mask[k]
        }
        heights = self._sphere_heights(ground)
        pos_err, rot_err = self.closure_error(q)
        return PoseFit(
            q=q,
            marker_rms_m=self._marker_rms(q, targets_arr, prep.mask),
            per_marker_m=per_marker,
            closure_error_m=pos_err,
            closure_error_rad=rot_err,
            lowest_sphere_height_m=min(heights.values()),
            iterations=opts.iterations,
        )
