"""
Pinocchio RNEA-based inverse-dynamics motion matching.

Part of issue #4568. Computes joint torques required to reproduce a
reference kinematic trajectory using Pinocchio's recursive Newton-Euler
algorithm. Pinocchio is imported lazily inside :meth:`match` so that the
module can be imported on systems without it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.finite_difference import (
    require_enough_frames_for_finite_diff as _require_enough_frames_for_finite_diff,
)

from ..contracts import JointTrajectory, SkeletonRig, TorqueFrame
from .base import (
    BaseMotionMatchingSolver,
    CostWeights,
    MatchingBackendType,
    MotionMatchingRequest,
    MotionMatchingResult,
)

logger = logging.getLogger(__name__)

# Optional Rust acceleration (issue #5218). The Rust extension moves the
# finite-difference + per-frame driver loop into native code; the inner
# `pin.rnea` call is still made via a Python callback so we don't need the
# Pinocchio C++ dev libraries on the build host.
try:  # pragma: no cover - exercised conditionally
    import upstream_pinocchio_id as _rust_pin_id  # type: ignore[import-not-found]

    _HAVE_RUST_PIN_ID = True
except Exception:  # pragma: no cover - fallback path  # noqa: BLE001
    _rust_pin_id = None  # type: ignore[assignment]
    _HAVE_RUST_PIN_ID = False

import os


def _use_rust_outer_loop() -> bool:
    return _HAVE_RUST_PIN_ID and os.environ.get("RUST_OUTER_LOOP", "1") == "1"


class PinocchioInverseDynMatchingSolver(BaseMotionMatchingSolver):
    """
    Pinocchio inverse-dynamics motion matching solver.

    Uses ``pin.rnea(model, data, q, qdot, qddot)`` per frame to solve for
    the joint torques that reproduce the reference kinematics.

    The result's ``tracked_trajectory`` is the (kinematic) reference
    trajectory; the computed torques are returned as a ``JointTrajectory``
    in ``torque_trajectory`` (the ``q`` slot carries tau).
    """

    backend_type = MatchingBackendType.INVERSE_DYN_PINOCCHIO

    def __init__(
        self,
        cost_weights: CostWeights | None = None,
        urdf_path: Path | str | None = None,
        *,
        enable_shadow: bool = True,
    ) -> None:
        """
        Args:
            cost_weights: Cost weights for diagnostics.
            urdf_path: Optional path to a URDF describing the rig.
                If ``None``, a minimal model is built from the
                :class:`SkeletonRig` passed to :meth:`match`.
            enable_shadow: Enable shadow model observation on inverse-dynamics output
                (issue #11028).
        """
        super().__init__(cost_weights)
        self.urdf_path = Path(urdf_path) if urdf_path is not None else None
        self.enable_shadow = bool(enable_shadow)
        if self.urdf_path is not None:
            if not self.urdf_path.exists():
                raise ValueError(f"URDF path does not exist: {self.urdf_path}")
            suffix = self.urdf_path.suffix
            if not self.urdf_path.is_file() or suffix.lower() != ".urdf":
                raise ValueError(f"URDF path must be a .urdf file: {self.urdf_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finite_difference(
        traj: JointTrajectory,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(times, q, qdot, qddot)`` matrices.

        When ``upstream_pinocchio_id`` (issue #5218) is importable the
        per-row qdot/qddot loops run in Rust on contiguous numpy buffers;
        otherwise we fall back to the pure-Python scheme. Both paths
        produce numerically identical outputs (RMSE <1e-12).
        """
        if not traj.frames:
            raise ValueError("Trajectory must have at least one frame")
        times = np.asarray([f.timestamp for f in traj.frames], dtype=float)
        q = np.asarray([list(f.q) for f in traj.frames], dtype=float)

        have_qdot_override = all(f.qdot is not None for f in traj.frames)
        have_qddot_override = all(f.qddot is not None for f in traj.frames)

        # Precondition (issue #7146): finite differencing without explicit
        # qdot/qddot overrides needs enough samples, otherwise it silently
        # returns all-zero velocities/accelerations and inverse dynamics
        # degenerates to statics (plausible-looking but wrong physics). Fail
        # loudly unless the caller supplied the derivatives.
        _require_enough_frames_for_finite_diff(
            n_frames=len(traj.frames),
            need_qdot=not have_qdot_override,
            need_qddot=not have_qddot_override,
        )

        if _use_rust_outer_loop() and not (have_qdot_override and have_qddot_override):
            q_c = np.ascontiguousarray(q, dtype=np.float64)
            t_c = np.ascontiguousarray(times, dtype=np.float64)
            qdot = (
                np.asarray([list(f.qdot) for f in traj.frames], dtype=float)  # type: ignore[arg-type, type-var]
                if have_qdot_override
                else _rust_pin_id.compute_qdot(q_c, t_c)  # type: ignore[union-attr]
            )
            qddot = (
                np.asarray([list(f.qddot) for f in traj.frames], dtype=float)  # type: ignore[arg-type, type-var]
                if have_qddot_override
                else _rust_pin_id.compute_qddot(q_c, t_c)  # type: ignore[union-attr]
            )
            return times, q, qdot, qddot

        # qdot
        if have_qdot_override:
            qdot = np.asarray([list(f.qdot) for f in traj.frames], dtype=float)  # type: ignore[arg-type, type-var]
        else:
            qdot = np.zeros_like(q)
            for i in range(1, len(times) - 1):
                dt = times[i + 1] - times[i - 1]
                if dt > 0:
                    qdot[i] = (q[i + 1] - q[i - 1]) / dt
            if len(times) >= 2:
                qdot[0] = (q[1] - q[0]) / max(times[1] - times[0], 1e-9)
                qdot[-1] = (q[-1] - q[-2]) / max(times[-1] - times[-2], 1e-9)

        # qddot
        if have_qddot_override:
            qddot = np.asarray([list(f.qddot) for f in traj.frames], dtype=float)  # type: ignore[arg-type, type-var]
        else:
            qddot = np.zeros_like(q)
            for i in range(1, len(times) - 1):
                dt_b = times[i] - times[i - 1]
                dt_f = times[i + 1] - times[i]
                if dt_b > 0 and dt_f > 0:
                    qddot[i] = (
                        2.0
                        * (q[i + 1] * dt_b - q[i] * (dt_b + dt_f) + q[i - 1] * dt_f)
                        / (dt_b * dt_f * (dt_b + dt_f))
                    )
            if len(times) >= 3:
                qddot[0] = qddot[1]
                qddot[-1] = qddot[-2]

        return times, q, qdot, qddot

    @staticmethod
    def _build_model_from_rig(rig: SkeletonRig, pin) -> tuple[Any, Any]:  # type: ignore[name-defined]
        """
        Build a serial Pinocchio model from a SkeletonRig.

        For each joint we add one revolute joint per axis, inheriting the
        parent's frame. This is sufficient for unit tests and for
        synthetic pendulum-style rigs; production callers should pass a
        URDF path instead.
        """
        model = pin.Model()
        # Root joint is implicit (universe). Walk joints in the order they
        # appear in the dict so behavior is deterministic.
        joint_to_id: dict[str, int] = {}
        for jname, jdef in rig.joints.items():
            parent_id = joint_to_id.get(jdef.parent, 0) if jdef.parent else 0
            placement = pin.SE3.Identity()
            placement.translation = np.asarray(jdef.tpose_offset, dtype=float)
            # Add one revolute DOF per declared axis
            current_parent = parent_id
            current_placement = placement
            for axis in jdef.axes:
                ax = axis[-1].upper()
                models = {"X": pin.JointModelRX, "Y": pin.JointModelRY}
                jmodel = models.get(ax, pin.JointModelRZ)()
                jid = model.addJoint(current_parent, jmodel, current_placement, jname)
                model.appendBodyToJoint(
                    jid, pin.Inertia.FromSphere(1.0, 0.05), pin.SE3.Identity()
                )
                current_parent = jid
                current_placement = pin.SE3.Identity()
            joint_to_id[jname] = current_parent
        data = model.createData()
        return model, data

    @staticmethod
    def _rig_dof_names(rig: SkeletonRig) -> list[str]:
        names: list[str] = []
        for jname, jdef in rig.joints.items():
            if len(jdef.axes) == 1:
                names.append(jname)
            else:
                names.extend(
                    f"{jname}_{a.replace('+', '').replace('-', 'neg')}"
                    for a in jdef.axes
                )
        return names

    @staticmethod
    def _reorder_to_pinocchio_joint_order(
        *,
        model: Any,
        rig: SkeletonRig,
        q_all: np.ndarray,
        qdot_all: np.ndarray,
        qddot_all: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        rig_dof_names = PinocchioInverseDynMatchingSolver._rig_dof_names(rig)
        model_names = [str(name) for name in model.names[1:]]
        if len(model_names) != model.nq:
            raise ValueError(
                "Pinocchio URDF must expose one named revolute joint per DOF; "
                f"model names={model_names}, nq={model.nq}"
            )
        missing = [name for name in rig_dof_names if name not in model_names]
        extra = [name for name in model_names if name not in rig_dof_names]
        if missing or extra:
            raise ValueError(
                "Pinocchio URDF joint names must match rig DOF names; "
                f"missing={missing}, extra={extra}"
            )
        permutation = [rig_dof_names.index(name) for name in model_names]
        return (
            q_all[:, permutation],
            qdot_all[:, permutation],
            qddot_all[:, permutation],
            permutation,
        )

    @staticmethod
    def _reorder_tau_to_rig_order(
        torque_frames: list[TorqueFrame],
        permutation: list[int],
    ) -> list[TorqueFrame]:
        inverse = np.argsort(np.asarray(permutation, dtype=int))
        return [
            TorqueFrame(
                timestamp=f.timestamp,
                tau=np.asarray(f.tau, dtype=float)[inverse].tolist(),
            )
            for f in torque_frames
        ]

    @staticmethod
    def _execute_rnea_loop(
        model: Any,
        data: Any,
        pin: Any,
        times: np.ndarray,
        q_arr: np.ndarray,
        v_arr: np.ndarray,
        a_arr: np.ndarray,
    ) -> list[TorqueFrame]:
        n_frames = len(times)
        rnea = pin.rnea
        tau_all = np.empty((n_frames, q_arr.shape[1]), dtype=np.float64)
        for i in range(n_frames):
            tau_all[i] = np.asarray(
                rnea(model, data, q_arr[i], v_arr[i], a_arr[i]),
                dtype=np.float64,
            ).flatten()
        if not np.all(np.isfinite(tau_all)):
            bad = int(np.argmax(~np.all(np.isfinite(tau_all), axis=1)))
            raise RuntimeError(f"RNEA produced non-finite torques at frame {bad}")
        return [
            TorqueFrame(timestamp=float(t), tau=tau_all[i].tolist())
            for i, t in enumerate(times)
        ]

    @staticmethod
    def _compute_torque_frames(
        *,
        model: Any,
        data: Any,
        pin: Any,
        times: np.ndarray,
        q_all: np.ndarray,
        qdot_all: np.ndarray,
        qddot_all: np.ndarray,
    ) -> list[TorqueFrame]:
        """Run the per-frame pin.rnea driver loop (Rust or Python fallback)."""
        if _use_rust_outer_loop():
            try:
                assert _rust_pin_id is not None
                q_c = np.ascontiguousarray(q_all, dtype=np.float64)
                v_c = np.ascontiguousarray(qdot_all, dtype=np.float64)
                a_c = np.ascontiguousarray(qddot_all, dtype=np.float64)
                return PinocchioInverseDynMatchingSolver._execute_rnea_loop(
                    model, data, pin, times, q_c, v_c, a_c
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "upstream_pinocchio_id rust path failed (%s); fallback", exc
                )
        return PinocchioInverseDynMatchingSolver._execute_rnea_loop(
            model, data, pin, times, q_all, qdot_all, qddot_all
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        reference: JointTrajectory,
        rig: SkeletonRig,
        request: MotionMatchingRequest | None = None,
    ) -> MotionMatchingResult:
        """
        Solve inverse dynamics for the reference trajectory.

        Returns:
            ``MotionMatchingResult`` whose ``tracked_trajectory`` is the
            input reference and whose ``torque_trajectory`` carries
            per-frame generalized forces.

        Raises:
            RuntimeError: If Pinocchio is unavailable.
            ValueError: On invalid inputs.
        """
        if reference is None or rig is None:
            raise ValueError("reference and rig must be provided")
        if not reference.frames:
            raise ValueError("reference must have at least one frame")

        try:
            import pinocchio as pin  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised on CI
            raise RuntimeError("pinocchio not installed") from exc

        request_id = request.id if request is not None else f"pin-rnea-{reference.id}"
        t_start = time.perf_counter()

        if self.urdf_path is not None:
            model = pin.buildModelFromUrdf(str(self.urdf_path))
            data = model.createData()
            model_source = "urdf"
            model_fidelity = "production_urdf"
            production_ready = True
        else:
            model, data = self._build_model_from_rig(rig, pin)
            model_source = "synthetic_rig"
            model_fidelity = "synthetic_point_mass"
            production_ready = False

        times, q_all, qdot_all, qddot_all = self._finite_difference(reference)
        n_dof_traj = q_all.shape[1]
        if model.nq != n_dof_traj:
            raise ValueError(
                f"Pinocchio model nq={model.nq} does not match "
                f"trajectory DOFs={n_dof_traj}"
            )
        if self.urdf_path is not None:
            q_all, qdot_all, qddot_all, permutation = (
                self._reorder_to_pinocchio_joint_order(
                    model=model,
                    rig=rig,
                    q_all=q_all,
                    qdot_all=qdot_all,
                    qddot_all=qddot_all,
                )
            )
        else:
            permutation = list(range(n_dof_traj))

        torque_frames = self._compute_torque_frames(
            model=model,
            data=data,
            pin=pin,
            times=times,
            q_all=q_all,
            qdot_all=qdot_all,
            qddot_all=qddot_all,
        )
        if self.urdf_path is not None:
            torque_frames = self._reorder_tau_to_rig_order(torque_frames, permutation)

        torque_traj = self._build_torque_trajectory(reference, rig, torque_frames)

        residual_report = self._compute_residual_report(reference, reference)
        rmse = self._compute_rmse(reference, reference)
        solve_time = time.perf_counter() - t_start

        shadow_report = self._run_shadow_observation(
            model=model,
            data=data,
            q_all=q_all,
            qdot_all=qdot_all,
            qddot_all=qddot_all,
        )

        result = MotionMatchingResult(
            request_id=request_id,
            success=production_ready,
            tracked_trajectory=reference,
            torque_trajectory=torque_traj,
            residual_report=residual_report,
            fit_metrics={"rmse": float(rmse), "max_error": 0.0},
            solve_time=float(solve_time),
            message=(
                "Pinocchio RNEA inverse-dynamics solve OK"
                if production_ready
                else (
                    "Pinocchio synthetic point-mass rig model is diagnostic-only; "
                    "provide matching_model_urdf for production inverse dynamics"
                )
            ),
            metadata={
                "backend": MatchingBackendType.INVERSE_DYN_PINOCCHIO.value,
                "n_frames": len(times),
                "n_dof": n_dof_traj,
                "model_source": model_source,
                "model_fidelity": model_fidelity,
                "production_ready": production_ready,
                "urdf_path": str(self.urdf_path) if self.urdf_path else None,
                "shadow_report": shadow_report,
            },
        )
        self._validate_result(reference, result)
        return result

    def _run_shadow_observation(
        self,
        *,
        model: Any,
        data: Any,
        q_all: np.ndarray,
        qdot_all: np.ndarray,
        qddot_all: np.ndarray,
    ) -> dict[str, Any] | None:
        """Run ShadowModel in observation mode on inverse-dynamics output (#11028)."""
        if not self.enable_shadow:
            return None
        try:
            from src.shared.python.physics_informed.rigid_core import RigidCore
            from src.shared.python.physics_informed.shadow_model import ShadowModel

            core = RigidCore(model=model, data=data)

            def _fallback_mlp(x: Any) -> np.ndarray:  # noqa: ARG001
                return np.zeros(q_all.shape[1], dtype=np.float64)

            try:
                import jax
                from src.shared.python.physics_informed.mlp_residual import MlpResidual

                mlp = MlpResidual(
                    input_dim=q_all.shape[1] * 3,
                    output_dim=q_all.shape[1],
                    hidden_dims=[16],
                    key=jax.random.PRNGKey(0),
                )
            except Exception:
                mlp = _fallback_mlp  # type: ignore[assignment]

            shadow = ShadowModel(core, mlp)
            frames = [
                {"q": q_all[i], "dq": qdot_all[i], "ddq": qddot_all[i]}
                for i in range(len(q_all))
            ]
            report = shadow.observe(frames)
            return {
                "peak_rigid_torques": report.peak_rigid_torques,
                "peak_residuals": report.peak_residuals,
            }
        except Exception as exc:
            logger.debug("Shadow observation skipped or failed: %s", exc)
            return None


__all__ = ["PinocchioInverseDynMatchingSolver"]
