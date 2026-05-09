"""Canonical MuJoCo forward-sim wrapper for motion matching.

Implements ``simulate_with_coefficients`` per
``CROSS_ENGINE_PARITY_SPEC.md`` §2.2 and the engine spec at
``src/engines/physics_engines/mujoco/MUJOCO_PARITY_SPEC.md`` §2.

Public API:
    SimOptions  -- frozen dataclass of forward-sim options.
    SimOut      -- frozen dataclass of canonical outputs.
    simulate_with_coefficients(theta, options, initial_pose) -> SimOut.

Threading note
--------------
``mjcb_control`` is a process-global callback (see MuJoCo C API). The driver
installed by this function is uninstalled deterministically before return,
including on exception, so back-to-back calls in the same process are safe.
Parallel fits MUST use ``multiprocessing`` rather than threads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts.decorators import postcondition, precondition
from src.shared.python.motion_matching.validate_theta import validate_theta

from .torque_driver import PolynomialTorqueDriver

ModelVariant = Literal["upper", "full", "advanced"]

__all__ = [
    "SimOptions",
    "SimOut",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
]


# --- Body lookup table for grip / clubhead -----------------------------------
#
# Mid-hands "grip" position is the canonical anchor. The full-body and
# advanced models weld the left hand to the club body; the upper-body model
# does the same. We resolve the grip body name per variant so the contract
# is stable even when the underlying MJCF differs.
_GRIP_BODY_BY_VARIANT: dict[str, str] = {
    "upper": "club",
    "full": "club",
    "advanced": "club",
}
_CLUBHEAD_BODY_BY_VARIANT: dict[str, str] = {
    "upper": "clubhead",
    "full": "clubhead",
    "advanced": "clubhead",
}


@dataclass(frozen=True)
class SimOptions:
    """Forward-sim options for ``simulate_with_coefficients``.

    Attributes:
        variant: Which MJCF variant to instantiate.
        T_s: Total simulation horizon in seconds.
        dt: Simulation timestep. ``None`` means use the model's own
            ``opt.timestep`` (recommended).
        t0: Polynomial reference time in seconds.
        output_rate_hz: Sample rate of the returned trajectory. The
            number of rows in every output array is
            ``round(T_s * output_rate_hz) + 1``.
        clip_torque_to_ctrlrange: If ``True`` and the MJCF declares
            ``ctrlrange`` on its actuators, the polynomial torque is
            clipped to that range before being written to ``data.ctrl``.
        compute_qdd: If ``True``, write joint accelerations into the
            output. ``False`` saves a tiny copy per frame.
        rng_seed: Reserved for future stochastic options; currently
            unused but accepted to match the cross-engine signature.
    """

    variant: ModelVariant = "full"
    T_s: float = 0.3
    dt: float | None = None
    t0: float = 0.0
    output_rate_hz: float = 1000.0
    clip_torque_to_ctrlrange: bool = True
    compute_qdd: bool = True
    rng_seed: int = 0


@dataclass(frozen=True)
class SimOut:
    """Canonical forward-sim output (cross-engine contract).

    All arrays have ``N = round(T_s * output_rate_hz) + 1`` rows. Joint-space
    fields have ``J = model.nv`` columns; control-space ``tau`` has
    ``model.nu`` columns (in general ``nu <= nv`` because some joints may
    be unactuated, e.g. the freejoint on the golf ball).

    Attributes:
        time: ``(N,)`` monotonic time stamps in seconds.
        q:    ``(N, nq)`` generalised coordinates over time.
        qd:   ``(N, nv)`` generalised velocities over time.
        qdd:  ``(N, nv)`` generalised accelerations over time.
        tau:  ``(N, nu)`` applied actuator torques (the polynomial driver
            output, post-clip). For unactuated dofs see ``qfrc_applied``
            in MuJoCo — this field reports motor commands only.
        grip:      ``(N, 3)`` mid-hands position (m), world frame.
        grip_quat: ``(N, 4)`` grip orientation, ``[w, x, y, z]``.
        clubhead:  ``(N, 3)`` clubhead position (m), world frame.
        club_quat: ``(N, 4)`` clubhead orientation, ``[w, x, y, z]``.
        solver_status: ``"success"`` if the rollout completed without divergence,
            ``"failed"`` if any frame produced a non-finite state.
        duration_s: wall-clock seconds spent in the rollout (excluding
            model compile).
    """

    time: NDArray[np.float64]
    q: NDArray[np.float64]
    qd: NDArray[np.float64]
    qdd: NDArray[np.float64]
    tau: NDArray[np.float64]
    grip: NDArray[np.float64]
    grip_quat: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    solver_status: str = "success"
    duration_s: float = 0.0

    # ------------------------------------------------------------------ DRY
    # The cost function in src/shared/python/motion_matching/cost.py expects
    # a SimOutput dataclass with ``butt``, ``clubhead``, ``club_quat`` and
    # optionally ``time/tau/omega``. We provide a thin adapter so callers
    # can hand a SimOut to compute_cost without redefining anything.

    def to_cost_simoutput(self) -> Any:
        """Return a shared ``cost.SimOutput`` view of this rollout.

        The shared ``SimOutput`` uses ``butt`` for the mid-hands anchor; we
        map ``grip -> butt`` per CLUB_IK_SPEC.md "grip-primary".
        """
        # Local import to avoid a hard dep if the cost module is unavailable
        # in some build configurations.
        from src.shared.python.motion_matching.cost import SimOutput

        return SimOutput(
            butt=self.grip,
            clubhead=self.clubhead,
            club_quat=self.club_quat,
            time=self.time,
            tau=self.tau,
            omega=self.qd[:, : self.tau.shape[1]] if self.tau.size else None,
        )


# --- Model loader -----------------------------------------------------------


def _load_model_xml(variant: ModelVariant) -> str:
    """Return the MJCF source for ``variant``."""
    if variant == "full":
        from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
            FULL_BODY_GOLF_SWING_XML,
        )

        return FULL_BODY_GOLF_SWING_XML
    if variant == "upper":
        from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
            UPPER_BODY_GOLF_SWING_XML,
        )

        return UPPER_BODY_GOLF_SWING_XML
    if variant == "advanced":
        from src.engines.physics_engines.mujoco._golf_swing_advanced_xml import (
            ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
        )

        return ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
    raise ValueError(
        f"unknown variant {variant!r}; expected 'upper', 'full', or 'advanced'"
    )


def _resolve_body_id(model: Any, name: str) -> int:
    """Look up a body id by name; raise a helpful error if missing."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"body {name!r} not found in MJCF")
    return int(bid)


# --- Pose application -------------------------------------------------------


def _apply_initial_pose(
    data: Any,
    initial_pose: NDArray[np.float64] | None,
    nq: int,
) -> None:
    """Write ``initial_pose`` into ``data.qpos`` (zero-velocity start).

    ``initial_pose`` may be ``None`` (use the MJCF default), a length-``nq``
    vector (full pose), or shorter — in which case the leading entries are
    written and the remainder is left at the MJCF default. This loose
    contract supports the upper/full/advanced models which differ in
    ``nq`` while sharing a common joint head.
    """
    if initial_pose is None:
        return
    pose = np.asarray(initial_pose, dtype=np.float64).reshape(-1)
    if pose.size > nq:
        raise ValueError(f"initial_pose has {pose.size} entries but model nq = {nq}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("initial_pose must be finite")
    data.qpos[: pose.size] = pose


# --- Sampling helpers -------------------------------------------------------


def _output_grid(T_s: float, output_rate_hz: float) -> NDArray[np.float64]:
    """Return the canonical output time grid (linspace, inclusive endpoints).

    ``N = round(T_s * output_rate_hz) + 1``. This matches the contract in
    ``CROSS_ENGINE_PARITY_SPEC.md`` §2.2 and the test expectations.
    """
    if T_s <= 0:
        raise ValueError(f"T_s must be > 0; got {T_s}")
    if output_rate_hz <= 0:
        raise ValueError(f"output_rate_hz must be > 0; got {output_rate_hz}")
    n = int(round(T_s * output_rate_hz)) + 1
    return np.linspace(0.0, T_s, n, dtype=np.float64)


# --- Entry point ------------------------------------------------------------


@precondition(
    # Coerce list-like inputs so the precondition matches the function's
    # historical contract: ``simulate_with_coefficients`` accepts any
    # array-like (list, tuple, ndarray) and normalises via ``np.asarray``
    # internally. Without coercion here, a Python list would raise
    # ``AttributeError`` on ``.size`` inside the decorator before the
    # function's own validation runs, regressing public behaviour.
    lambda theta, *args, **kwargs: bool(np.asarray(theta).size % 7 == 0),
    "theta length must be a multiple of 7",
)
@precondition(
    lambda theta, *args, **kwargs: bool(np.all(np.isfinite(np.asarray(theta)))),
    "theta must be finite",
)
@precondition(
    lambda theta, options=None, initial_pose=None, *args, **kwargs: (
        initial_pose is None or isinstance(initial_pose, (dict, np.ndarray, list))
    ),
    "initial_pose type must be valid",
)
@postcondition(
    lambda result: bool(
        result.time.shape[0] == result.q.shape[0] == result.qd.shape[0]
    ),
    "time, q, qd shape mismatch",
)
@postcondition(
    lambda result: bool(
        np.all(np.isfinite(result.q)) and np.all(np.isfinite(result.qd))
    ),
    "non-finite q or qd",
)
@postcondition(
    lambda result: bool(
        result.time.size > 0
        and result.time[0] == 0.0
        and np.all(np.diff(result.time) > 0)
    ),
    "time not monotonic or does not start at 0",
)
@postcondition(
    lambda result: bool(result.solver_status in ("success", "warning", "failed")),
    "invalid solver_status",
)
def simulate_with_coefficients(  # noqa: C901
    theta: NDArray[np.float64],
    options: SimOptions | None = None,
    initial_pose: NDArray[np.float64] | None = None,
) -> SimOut:
    """Roll out the MuJoCo model under the polynomial-torque driver.

    Args:
        theta: ``(n_joints * 7,)`` flat vector, or ``(n_joints, 7)`` matrix.
            ``n_joints`` must equal the compiled model's ``nu``. Layout per
            joint: ``[A, B, C, D, E, F, G]`` for
            ``A*t^6 + B*t^5 + C*t^4 + D*t^3 + E*t^2 + F*t + G``.
        options: :class:`SimOptions`.
        initial_pose: optional ``(<= nq,)`` initial generalised
            coordinates. ``None`` uses the MJCF default.

    Returns:
        :class:`SimOut` with shapes documented on the dataclass.

    Raises:
        ValueError: if ``theta`` is malformed or options are invalid.
        RuntimeError: if model compilation fails or required bodies are
            missing.
    """
    if options is None:
        options = SimOptions()
    if not isinstance(options, SimOptions):
        raise TypeError(
            f"options must be a SimOptions instance; got {type(options).__name__}"
        )

    import mujoco

    xml = _load_model_xml(options.variant)
    model = mujoco.MjModel.from_xml_string(xml)

    # Optional override of the model timestep.
    if options.dt is not None:
        if options.dt <= 0:
            raise ValueError(f"dt must be > 0; got {options.dt}")
        model.opt.timestep = float(options.dt)

    # Spec §2.2: enforce length + finiteness against the compiled model's
    # actuator count (``nu``). Bounds are enforced separately by the
    # ``PolynomialTorqueDriver`` clip path; the validator focuses on the
    # two checks that prevent silent failures / numerical divergence.
    theta = validate_theta(theta, n_joints=int(model.nu))

    data = mujoco.MjData(model)
    _apply_initial_pose(data, initial_pose, model.nq)
    mujoco.mj_forward(model, data)

    grip_bid = _resolve_body_id(model, _GRIP_BODY_BY_VARIANT[options.variant])
    head_bid = _resolve_body_id(model, _CLUBHEAD_BODY_BY_VARIANT[options.variant])

    # Output grid + per-frame sample stride.
    t_grid = _output_grid(options.T_s, options.output_rate_hz)
    n_out = t_grid.size
    nq = int(model.nq)
    nv = int(model.nv)
    nu = int(model.nu)

    out_q = np.zeros((n_out, nq), dtype=np.float64)
    out_qd = np.zeros((n_out, nv), dtype=np.float64)
    out_qdd = np.zeros((n_out, nv), dtype=np.float64)
    out_tau = np.zeros((n_out, nu), dtype=np.float64)
    out_grip = np.zeros((n_out, 3), dtype=np.float64)
    out_grip_q = np.zeros((n_out, 4), dtype=np.float64)
    out_head = np.zeros((n_out, 3), dtype=np.float64)
    out_head_q = np.zeros((n_out, 4), dtype=np.float64)

    # Capture frame 0 (post mj_forward, pre any mj_step).
    out_q[0] = data.qpos
    out_qd[0] = data.qvel
    if options.compute_qdd:
        out_qdd[0] = data.qacc
    out_grip[0] = data.xpos[grip_bid]
    out_grip_q[0] = data.xquat[grip_bid]
    out_head[0] = data.xpos[head_bid]
    out_head_q[0] = data.xquat[head_bid]
    # Tau at t=0 isn't computed by mj_forward (no ctrl callback yet); evaluate
    # the polynomial directly.

    driver = PolynomialTorqueDriver(
        model,
        theta,
        t0=options.t0,
        clip_to_ctrlrange=options.clip_torque_to_ctrlrange,
    )
    out_tau[0] = driver.evaluate(0.0)
    if options.clip_torque_to_ctrlrange:
        np.clip(
            out_tau[0],
            np.where(
                model.actuator_ctrllimited.astype(bool),
                model.actuator_ctrlrange[:, 0],
                -np.inf,
            ),
            np.where(
                model.actuator_ctrllimited.astype(bool),
                model.actuator_ctrlrange[:, 1],
                np.inf,
            ),
            out=out_tau[0],
        )

    solver_status = "success"
    t_start = time.perf_counter()
    try:
        with driver:
            for i in range(1, n_out):
                t_target = float(t_grid[i])
                # Step until data.time >= t_target (the model timestep may
                # not divide evenly into 1/output_rate_hz).
                # Guard against infinite loops with a generous safety cap.
                max_substeps = int(np.ceil(options.T_s / model.opt.timestep) + 16)
                substeps = 0
                while data.time + 1e-12 < t_target and substeps < max_substeps:
                    mujoco.mj_step(model, data)
                    substeps += 1
                    if not np.all(np.isfinite(data.qpos)) or not np.all(
                        np.isfinite(data.qvel)
                    ):
                        solver_status = "failed"
                        break
                if solver_status != "success":
                    break
                out_q[i] = data.qpos
                out_qd[i] = data.qvel
                if options.compute_qdd:
                    out_qdd[i] = data.qacc
                out_tau[i] = data.ctrl
                out_grip[i] = data.xpos[grip_bid]
                out_grip_q[i] = data.xquat[grip_bid]
                out_head[i] = data.xpos[head_bid]
                out_head_q[i] = data.xquat[head_bid]
    finally:
        # Belt-and-suspenders: even if the context manager already cleared,
        # this guarantees no leftover global callback escapes the function.
        driver.uninstall()

    duration_s = time.perf_counter() - t_start

    return SimOut(
        time=t_grid,
        q=out_q,
        qd=out_qd,
        qdd=out_qdd,
        tau=out_tau,
        grip=out_grip,
        grip_quat=out_grip_q,
        clubhead=out_head,
        club_quat=out_head_q,
        solver_status=solver_status,
        duration_s=duration_s,
    )


from .synthesize import synthesize_target_from_coefficients  # noqa: E402
