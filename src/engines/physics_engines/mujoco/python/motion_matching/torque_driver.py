"""Polynomial-torque driver for MuJoCo.

Implements the Stateflow analogue from ``MUJOCO_PARITY_SPEC.md`` §2.3.

The torque law is the canonical 6th-order polynomial::

    tau_j(t; theta) = A_j + B_j*t + C_j*t^2 + D_j*t^3
                    + E_j*t^4 + F_j*t^5 + G_j*t^6

Coefficients are laid out ascending in power: ``theta[:, k]`` is the
coefficient of ``t^k`` (column 0 = A = constant term, column 6 = G =
``t^6`` term). This matches the canonical cross-engine convention used by
Drake / Pinocchio / OpenSim, restoring θ parity across engines (#7688).

with parameter bounds (mirrored from the Simscape reference):

    |A_j| (t^0), |B_j| (t^1) <= 1000
    |C_j| (t^2), |D_j| (t^3) <=  500
    |E_j| (t^4), |F_j| (t^5) <=  100
    |G_j| (t^6)              <=   25

`mjcb_control` is a *process-global* callback in MuJoCo. The
:class:`PolynomialTorqueDriver` therefore exposes :meth:`install` /
:meth:`uninstall` so callers can scope ownership and use it as a context
manager. Parallel fits MUST use ``multiprocessing`` rather than threading.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.validate_theta import COEFFS_PER_JOINT

# --- Coefficient bounds (mirrored from Simscape build_coefficient_bounds) ---

# Order: A (t^0), B (t^1), C (t^2), D (t^3), E (t^4), F (t^5), G (t^6)
POLY_BOUNDS: tuple[float, float, float, float, float, float, float] = (
    1000.0,  # |A|
    1000.0,  # |B|
    500.0,  # |C|
    500.0,  # |D|
    100.0,  # |E|
    100.0,  # |F|
    25.0,  # |G|
)


def polynomial_torque_bounds(
    n_joints: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(lb, ub)`` element-wise bounds for the flattened theta vector.

    The flattened layout is row-major over joints, i.e.
    ``theta.reshape(n_joints, COEFFS_PER_JOINT)[j, k]`` is the coefficient of ``t^k``
    for joint ``j`` (so column 0 holds A = ``t^0``, ..., column 6 holds
    G = ``t^6``). This matches the contract in
    :class:`PolynomialTorqueDriver`.
    """
    if n_joints <= 0:
        raise ValueError(f"n_joints must be > 0; got {n_joints}")
    per_joint = np.asarray(POLY_BOUNDS, dtype=np.float64)
    ub = np.tile(per_joint, n_joints)
    return -ub.copy(), ub.copy()


def _evaluate_polynomial(
    theta: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    """Evaluate per-joint polynomial torques at scalar time ``t``.

    Args:
        theta: ``(n_joints, 7)`` coefficient matrix. Column ``k`` is the
            coefficient of ``t^k`` (i.e. ``theta[:, 0]`` is A = the
            constant term, ``theta[:, 6]`` is G = the ``t^6`` term).
        t: time in seconds (relative to ``t0``).

    Returns:
        ``(n_joints,)`` torque vector.
    """
    # Horner's method on shape (7,) ascending-power coefficients: start
    # from the highest power (column 6) and fold in descending columns so
    # column 0 is the constant term (canonical cross-engine convention,
    # #7688).
    out = theta[:, -1].astype(np.float64, copy=True)
    for k in range(theta.shape[1] - 2, -1, -1):
        out = out * t + theta[:, k]
    return out


class PolynomialTorqueDriver(AbstractContextManager["PolynomialTorqueDriver"]):
    """Per-joint 6th-order polynomial torque, applied via ``mjcb_control``.

    Args:
        model: a compiled ``mujoco.MjModel``.
        theta: ``(n_joints, 7)`` coefficient matrix or a flat
            ``(n_joints*7,)`` vector. ``n_joints`` must equal
            ``model.nu``. Layout: column 0 = A (t^0), ..., column 6 = G
            (t^6).
        t0: reference time in seconds; the polynomial is evaluated at
            ``data.time - t0``.
        clip_to_ctrlrange: if ``True`` and the model declares
            ``ctrlrange`` on its actuators, clip the torque to that range
            before writing ``data.ctrl``. Default ``True``.

    Use as a context manager to scope the global callback safely::

        with PolynomialTorqueDriver(m, theta) as drv:
            drv.install()
            for _ in range(n_steps):
                mujoco.mj_step(m, d)
    """

    def __init__(
        self,
        model: Any,
        theta: NDArray[np.float64],
        t0: float = 0.0,
        clip_to_ctrlrange: bool = True,
    ) -> None:
        nu = int(model.nu)
        if nu <= 0:
            raise ValueError("model has no actuators (nu == 0)")
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != nu * COEFFS_PER_JOINT:
                raise ValueError(
                    "flat theta must have length "
                    f"nu*{COEFFS_PER_JOINT} = {nu * COEFFS_PER_JOINT}; "
                    f"got {arr.shape[0]}"
                )
            arr = arr.reshape(nu, COEFFS_PER_JOINT)
        elif arr.shape != (nu, COEFFS_PER_JOINT):
            raise ValueError(
                f"theta must have shape ({nu}, {COEFFS_PER_JOINT}); got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("theta must be finite (no NaN/inf)")
        self._theta: NDArray[np.float64] = arr.copy()
        self._t0 = float(t0)
        self._installed = False
        self._clip = bool(clip_to_ctrlrange)
        # Snapshot ctrlrange so we don't reach into model from the callback.
        if (
            self._clip
            and hasattr(model, "actuator_ctrllimited")
            and hasattr(model, "actuator_ctrlrange")
        ):
            limited = np.asarray(model.actuator_ctrllimited).astype(bool)
            ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
            self._ctrl_lo = np.where(limited, ranges[:, 0], -np.inf)
            self._ctrl_hi = np.where(limited, ranges[:, 1], np.inf)
        else:
            self._ctrl_lo = np.full(nu, -np.inf, dtype=np.float64)
            self._ctrl_hi = np.full(nu, np.inf, dtype=np.float64)

    # ------------------------------------------------------------------ API

    @property
    def theta(self) -> NDArray[np.float64]:
        """Return a copy of the current ``(n_joints, 7)`` coefficient matrix."""
        return self._theta.copy()

    def evaluate(self, t: float) -> NDArray[np.float64]:
        """Pure-Python evaluation of the polynomial at time ``t``.

        Useful for tests / debugging without installing the global callback.
        """
        return _evaluate_polynomial(self._theta, t - self._t0)

    def install(self) -> None:
        """Register ``self`` as the process-global ``mjcb_control`` callback."""
        import mujoco  # local import — keeps top-level import-free for tests

        if self._installed:
            return
        theta = self._theta
        t0 = self._t0
        lo = self._ctrl_lo
        hi = self._ctrl_hi

        def _cb(_m: Any, d: Any) -> None:
            t = d.time - t0
            # Horner's method on a fixed (7,) shape — no allocations beyond
            # the temporary scalar product. This is the inner-loop hot path.
            # Coefficients are ascending in power (column k = t^k), so we
            # fold from the highest power (column 6) down to the constant
            # term (column 0) — matching the canonical cross-engine
            # convention (#7688).
            ctrl = theta[:, 6] * t
            ctrl += theta[:, 5]
            ctrl *= t
            ctrl += theta[:, 4]
            ctrl *= t
            ctrl += theta[:, 3]
            ctrl *= t
            ctrl += theta[:, 2]
            ctrl *= t
            ctrl += theta[:, 1]
            ctrl *= t
            ctrl += theta[:, 0]
            np.clip(ctrl, lo, hi, out=ctrl)
            d.ctrl[:] = ctrl

        mujoco.set_mjcb_control(_cb)
        self._installed = True

    def uninstall(self) -> None:
        """Clear the global ``mjcb_control`` callback if this driver set it."""
        import mujoco

        if self._installed:
            mujoco.set_mjcb_control(None)
            self._installed = False

    # ----------------------------------------------------- context-manager

    def __enter__(self) -> PolynomialTorqueDriver:
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.uninstall()
