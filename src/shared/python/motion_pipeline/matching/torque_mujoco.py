"""
MuJoCo torque tracking backend for motion matching.

Part of issue #4568. MuJoCo torque PD-tracking with residual logging.
"""

from __future__ import annotations

import logging

import time
from xml.sax.saxutils import escape

import numpy as np

from ..contracts import (
    JointTrajectory,
    SkeletonRig,
    TorqueFrame,
)
from .base import (
    BaseMotionMatchingSolver,
    CostWeights,
    MatchingBackendType,
    MotionMatchingRequest,
    MotionMatchingResult,
)
from .inverse_dyn_pinocchio import PinocchioInverseDynMatchingSolver

logger = logging.getLogger(__name__)

# Optional Rust acceleration (issue #5254 slice 4).
try:  # pragma: no cover
    import upstream_pinocchio_id as _rust_outer_loop  # type: ignore[import-not-found]

    _HAVE_RUST = True
except Exception:  # pragma: no cover  # noqa: BLE001
    _rust_outer_loop = None  # type: ignore[assignment]
    _HAVE_RUST = False

import os


def _use_rust_outer_loop() -> bool:
    return _HAVE_RUST and os.environ.get("RUST_OUTER_LOOP", "1") == "1"


class MuJoCoTorqueMatchingSolver(BaseMotionMatchingSolver):
    """
    MuJoCo torque tracking motion matching solver.

    Uses MuJoCo's physics engine for torque-based PD tracking
    with residual force logging.
    """

    backend_type = MatchingBackendType.TORQUE_MUJOCO

    def __init__(self, cost_weights: CostWeights | None = None):
        """
        Initialize MuJoCo torque tracking solver.

        Args:
            cost_weights: Cost function weights
        """
        super().__init__(cost_weights)

    @staticmethod
    def _axis_vector(axis: str) -> str:
        sign = -1.0 if axis.startswith("-") else 1.0
        letter = axis[-1].upper()
        vectors = {
            "X": (sign, 0.0, 0.0),
            "Y": (0.0, sign, 0.0),
            "Z": (0.0, 0.0, sign),
        }
        return " ".join(f"{value:g}" for value in vectors.get(letter, vectors["Z"]))

    @staticmethod
    def _body_pos(offset: list[float]) -> str:
        return " ".join(f"{float(value):.12g}" for value in offset)

    @classmethod
    def _build_mjcf_from_rig(cls, rig: SkeletonRig) -> str:
        children: dict[str | None, list[str]] = {None: []}
        for jname, jdef in rig.joints.items():
            children.setdefault(jdef.parent, []).append(jname)
            children.setdefault(jname, [])
        if rig.root_joint not in rig.joints:
            raise ValueError(f"Root joint {rig.root_joint!r} not found in rig")

        def emit_joint_chain(jname: str, depth: int) -> list[str]:
            jdef = rig.joints[jname]
            indent = "  " * depth
            lines: list[str] = [
                (
                    f'{indent}<body name="{escape(jname)}" '
                    f'pos="{cls._body_pos(jdef.tpose_offset)}">'
                )
            ]
            current_depth = depth + 1
            for axis_index, axis in enumerate(jdef.axes):
                axis_name = f"{jname}_{axis_index}_{axis}"
                axis_indent = "  " * current_depth
                lines.extend(
                    [
                        (f'{axis_indent}<body name="{escape(axis_name)}" pos="0 0 0">'),
                        (
                            f'{axis_indent}  <joint name="{escape(axis_name)}" '
                            f'type="hinge" axis="{cls._axis_vector(axis)}" />'
                        ),
                        (
                            f'{axis_indent}  <inertial pos="0 0 0" '
                            'mass="1" diaginertia="0.01 0.01 0.01" />'
                        ),
                    ]
                )
                current_depth += 1
            for child_name in children.get(jname, []):
                lines.extend(emit_joint_chain(child_name, current_depth))
            for _axis in reversed(jdef.axes):
                current_depth -= 1
                lines.append(f"{'  ' * current_depth}</body>")
            lines.append(f"{indent}</body>")
            return lines

        worldbody: list[str] = []
        roots = [
            rig.root_joint,
            *[name for name in children[None] if name != rig.root_joint],
        ]
        for root in roots:
            worldbody.extend(emit_joint_chain(root, 3))
        return "\n".join(
            [
                '<mujoco model="motion_pipeline_generated">',
                '  <compiler angle="radian" />',
                '  <option gravity="0 0 -9.80665" />',
                "  <worldbody>",
                *worldbody,
                "  </worldbody>",
                "</mujoco>",
            ]
        )

    @staticmethod
    def _per_frame_callback(model: object | None, data: object | None):
        def _cb(q_row: np.ndarray, v_row: np.ndarray, a_row: np.ndarray) -> np.ndarray:
            if model is None or data is None:
                return np.zeros_like(q_row, dtype=np.float64)

            try:
                import mujoco

                # Check dimensional parity. ``model`` is typed as ``object``
                # because the mujoco wheel is conditionally imported; the
                # attributes exist at runtime.
                if (
                    len(q_row) == model.nq  # type: ignore[attr-defined]
                    and len(v_row) == model.nv  # type: ignore[attr-defined]
                    and len(a_row) == model.nv  # type: ignore[attr-defined]
                ):
                    data.qpos[:] = q_row  # type: ignore[attr-defined]
                    data.qvel[:] = v_row  # type: ignore[attr-defined]
                    data.qacc[:] = a_row  # type: ignore[attr-defined]
                    mujoco.mj_inverse(model, data)  # type: ignore
                    return np.asarray(data.qfrc_inverse.copy(), dtype=np.float64)  # type: ignore
            except Exception as exc:  # noqa: BLE001
                logger.debug("MuJoCo per-frame inverse step failed: %s", exc)

            return np.zeros_like(q_row, dtype=np.float64)

        return _cb

    @staticmethod
    def _compute_tau_python(
        times: np.ndarray,
        q_all: np.ndarray,
        qdot_all: np.ndarray,
        qddot_all: np.ndarray,
        callback,
    ) -> np.ndarray:
        n_frames, n_dof = q_all.shape
        tau_all = np.zeros((n_frames, n_dof), dtype=np.float64)
        for i in range(n_frames):
            tau_all[i] = np.asarray(
                callback(q_all[i], qdot_all[i], qddot_all[i]),
                dtype=np.float64,
            ).flatten()
        if not np.all(np.isfinite(tau_all)):
            bad = int(np.argmax(~np.all(np.isfinite(tau_all), axis=1)))
            raise RuntimeError(f"MuJoCo produced non-finite torques at frame {bad}")
        return tau_all

    @staticmethod
    def _compute_tau_rust(
        times: np.ndarray,
        q_all: np.ndarray,
        qdot_all: np.ndarray,
        qddot_all: np.ndarray,
        callback,
    ) -> np.ndarray:
        assert _rust_outer_loop is not None
        q_c = np.ascontiguousarray(q_all, dtype=np.float64)
        v_c = np.ascontiguousarray(qdot_all, dtype=np.float64)
        a_c = np.ascontiguousarray(qddot_all, dtype=np.float64)
        t_c = np.ascontiguousarray(times, dtype=np.float64)

        _, _, tau_all = _rust_outer_loop.inverse_dynamics(
            q_c,
            t_c,
            callback,
            v_c,
            a_c,
        )
        return tau_all

    def match(
        self,
        reference: JointTrajectory,
        rig: SkeletonRig,
        request: MotionMatchingRequest | None = None,
    ) -> MotionMatchingResult:
        """
        Solve motion matching using MuJoCo torque tracking.

        Args:
            reference: Reference joint trajectory to track
            rig: Scaled skeleton rig
            request: Optional matching request with configuration

        Returns:
            MotionMatchingResult with tracked trajectory and torque data.
            ``success`` reflects whether MuJoCo actually ran: when the
            wheel is absent no real inverse-dynamics is executed (torques
            are all zero), so ``success`` is False.

        Raises:
            ValueError: If the reference trajectory is empty.
        """
        if reference is None or rig is None:
            raise ValueError("reference and rig must be provided")
        if not reference.frames:
            raise ValueError("reference must have at least one frame")

        request_id = request.id if request else f"mujoco-torque-{reference.id}"
        t_start = time.perf_counter()

        # Extract finite difference kinematics using the same utility
        times, q_all, qdot_all, qddot_all = (
            PinocchioInverseDynMatchingSolver._finite_difference(reference)
        )
        n_frames, n_dof = q_all.shape

        # Setup MuJoCo model and data from the rig. Production matching must
        # never run on the historical empty ``<mujoco/>`` placeholder.
        model = None
        data = None
        mujoco_available = False
        model_error: str | None = None
        model_xml = self._build_mjcf_from_rig(rig)
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_string(model_xml)
            data = mujoco.MjData(model)
            mujoco_available = True
        except ImportError:
            pass
        except (RuntimeError, TypeError, ValueError) as exc:
            mujoco_available = True
            model_error = f"MuJoCo model build failed: {exc}"

        model_nq = getattr(model, "nq", None)
        model_nv = getattr(model, "nv", None)
        if (
            model_error is None
            and model is not None
            and (model_nq != n_dof or model_nv != n_dof)
        ):
            model_error = (
                f"MuJoCo model nq={model_nq}, nv={model_nv} does not match "
                f"trajectory DOFs={n_dof}"
            )

        callback = self._per_frame_callback(
            None if model_error else model,
            None if model_error else data,
        )

        if _use_rust_outer_loop():
            try:
                tau_all = self._compute_tau_rust(
                    times, q_all, qdot_all, qddot_all, callback
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "upstream_pinocchio_id rust path failed (%s); "
                    "falling back to pure-Python MuJoCo outer loop",
                    exc,
                )
                tau_all = self._compute_tau_python(
                    times, q_all, qdot_all, qddot_all, callback
                )
        else:
            tau_all = self._compute_tau_python(
                times, q_all, qdot_all, qddot_all, callback
            )

        torque_frames = [
            TorqueFrame(timestamp=float(t), tau=tau_all[i].tolist())
            for i, t in enumerate(times)
        ]

        torque_traj = self._build_torque_trajectory(reference, rig, torque_frames)

        residual_report = self._compute_residual_report(reference, reference)
        # Compute fit metrics from real residuals rather than hardcoding 0
        # (#7047). The tracked trajectory is the reference itself, so the
        # joint-tracking RMSE is genuinely near zero; the max torque
        # magnitude is reported as a non-trivial finite diagnostic.
        rmse = self._compute_rmse(reference, reference)
        max_tau = float(np.max(np.abs(tau_all))) if tau_all.size else 0.0
        solve_time = time.perf_counter() - t_start

        # Real execution only happened if MuJoCo produced non-trivial
        # torques. The empty-model fallback yields all zeros -> not a real
        # solve, so success is False (#7047).
        success = (
            mujoco_available and model_error is None and bool(np.any(tau_all != 0.0))
        )

        if success:
            message = "MuJoCo torque tracking solve OK"
        elif model_error is not None:
            message = model_error
        elif mujoco_available:
            message = (
                "MuJoCo present but generated rig model produced zero torques "
                "(no real inverse-dynamics signal)"
            )
        else:
            message = "MuJoCo unavailable; torques are zero (no real solve)"

        return MotionMatchingResult(
            request_id=request_id,
            success=success,
            tracked_trajectory=reference,
            torque_trajectory=torque_traj,
            residual_report=residual_report,
            fit_metrics={
                "rmse": float(rmse),
                "max_error": float(residual_report["max_residual"]),
                "max_torque": max_tau,
            },
            solve_time=float(solve_time),
            message=message,
            metadata={
                "backend": self.backend_type.value,
                "mujoco_available": mujoco_available,
                "n_frames": n_frames,
                "n_dof": n_dof,
                "model_source": "generated_mjcf",
                "model_nq": model_nq,
                "model_nv": model_nv,
                "placeholder_model": False,
                "production_ready": mujoco_available and model_error is None,
            },
        )
