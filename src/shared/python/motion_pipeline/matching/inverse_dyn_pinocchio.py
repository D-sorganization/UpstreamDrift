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
from typing import Any, Optional

import numpy as np

from ..contracts import (
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)
from .base import (
    BaseMotionMatchingSolver,
    CostWeights,
    MatchingBackendType,
    MotionMatchingRequest,
    MotionMatchingResult,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        """
        Args:
            cost_weights: Cost weights for diagnostics.
            urdf_path: Optional path to a URDF describing the rig.
                If ``None``, a minimal model is built from the
                :class:`SkeletonRig` passed to :meth:`match`.
        """
        super().__init__(cost_weights)
        self.urdf_path = Path(urdf_path) if urdf_path is not None else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finite_difference(
        traj: JointTrajectory,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(times, q, qdot, qddot)`` matrices."""
        if not traj.frames:
            raise ValueError("Trajectory must have at least one frame")
        times = np.asarray([f.timestamp for f in traj.frames], dtype=float)
        q = np.asarray([list(f.q) for f in traj.frames], dtype=float)

        # qdot
        if all(f.qdot is not None for f in traj.frames):
            qdot = np.asarray([list(f.qdot) for f in traj.frames], dtype=float)
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
        if all(f.qddot is not None for f in traj.frames):
            qddot = np.asarray([list(f.qddot) for f in traj.frames], dtype=float)
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
                ax_letter = axis[-1].upper()
                if ax_letter == "X":
                    joint_model = pin.JointModelRX()
                elif ax_letter == "Y":
                    joint_model = pin.JointModelRY()
                else:
                    joint_model = pin.JointModelRZ()
                jid = model.addJoint(
                    current_parent, joint_model, current_placement, jname
                )
                # Body inertia: unit point-mass at the segment offset.
                inertia = pin.Inertia.FromSphere(1.0, 0.05)
                model.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
                current_parent = jid
                current_placement = pin.SE3.Identity()
            joint_to_id[jname] = current_parent
        data = model.createData()
        return model, data

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
        else:
            model, data = self._build_model_from_rig(rig, pin)

        times, q_all, qdot_all, qddot_all = self._finite_difference(reference)
        n_dof_traj = q_all.shape[1]
        if model.nq != n_dof_traj:
            raise ValueError(
                f"Pinocchio model nq={model.nq} does not match "
                f"trajectory DOFs={n_dof_traj}"
            )

        torque_frames: list[JointStateFrame] = []
        for i, t in enumerate(times):
            q = q_all[i]
            v = qdot_all[i]
            a = qddot_all[i]
            tau = pin.rnea(model, data, q, v, a)
            tau_arr = np.asarray(tau, dtype=float).flatten()
            if not np.all(np.isfinite(tau_arr)):
                raise RuntimeError(f"RNEA produced non-finite torques at frame {i}")
            torque_frames.append(
                JointStateFrame(
                    timestamp=float(t),
                    q=tau_arr.tolist(),
                    frame_index=i,
                )
            )

        torque_traj = JointTrajectory(
            id=f"{reference.id}-torques",
            skeleton=rig,
            frames=torque_frames,
            metadata={"semantics": "torques"},
        )

        residual_report = self._compute_residual_report(reference, reference)
        rmse = self._compute_rmse(reference, reference)
        solve_time = time.perf_counter() - t_start

        if not self._validate_result(reference):
            raise RuntimeError("Reference trajectory failed postcondition check")

        return MotionMatchingResult(
            request_id=request_id,
            success=True,
            tracked_trajectory=reference,
            torque_trajectory=torque_traj,
            residual_report=residual_report,
            fit_metrics={"rmse": float(rmse), "max_error": 0.0},
            solve_time=float(solve_time),
            message="Pinocchio RNEA inverse-dynamics solve OK",
            metadata={
                "backend": MatchingBackendType.INVERSE_DYN_PINOCCHIO.value,
                "n_frames": len(times),
                "n_dof": n_dof_traj,
            },
        )


__all__ = ["PinocchioInverseDynMatchingSolver"]
