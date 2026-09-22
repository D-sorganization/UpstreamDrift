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
    ) -> tuple[Array, int]:
        from mujoco import minimize
        from mujoco.minimize import Verbosity

        def unpack(x_free: Array) -> Array:
            x = np.asarray(x_free, dtype=float).reshape(-1)
            q = prep.q.copy()
            q[prep.free] = x
            return np.clip(q, prep.low, prep.high)

        def residual_and_jacobian(q_k: Array) -> tuple[Array, Array]:
            return self._pose_residual_stack(
                q_k, prep, targets_arr, q_init, ground, opts
            )

        warm_iters = max(5, opts.iterations - max(1, opts.iterations // 3))
        q_seed, lm_done = _run_lm_loop(
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
        # mujoco.minimize.least_squares rejects non-finite bounds
        # (ValueError: bounds must be finite); omit when any DOF is open.
        bounds = None
        if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
            bounds = (lower, upper)
        x_opt, trace = minimize.least_squares(
            q_seed[prep.free].reshape(-1, 1),
            residual,
            bounds=bounds,
            jacobian=jacobian,
            max_iter=opts.iterations,
            verbose=Verbosity.SILENT,
            mu_max=1e16,
        )
        min_done = len(trace) if trace is not None else 0
        q_min = unpack(x_opt.reshape(-1))
        if self._marker_rms(q_min, targets_arr, prep.mask) <= seed_rms:
            return q_min, lm_done + min_done
        return q_seed, lm_done

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
        q, done = self._solve_with_minimize(prep, targets_arr, q_init, ground, opts)
        return self._finalize_pose_fit(
            q, targets_arr, prep, ground=ground, iterations=done
        )
