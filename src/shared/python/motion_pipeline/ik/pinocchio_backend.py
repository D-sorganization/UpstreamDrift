"""
Pinocchio backend for Inverse Kinematics.

Part of issue #4566, implemented under epic #8390 (C1/#8401, closing the
#7046 stub). Builds a ``pin.Model`` from the canonical rig via the shared
URDF bridge and solves multi-marker position IK with a damped
Levenberg-Marquardt iteration — the same pure-pinocchio approach as the
engine's ``diff_ik`` module, deliberately not requiring ``pink`` (#4138:
pin-pink is fragile to install on some platforms). When ``pink`` *is*
importable a task-based QP formulation is available via
``method="pink"``.

Marker convention matches the geometric backend: a marker named after a
rig joint targets that joint's body position.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

import numpy as np

from ..contracts import (
    JointStateFrame,
    JointTrajectory,
    MarkerTrajectory,
    SkeletonRig,
)
from ..model_bridge import rig_joint_link_name, rig_to_pinocchio_model
from .base import BaseIKSolver, IKConfig, MarkerWeights

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "pinocchio is required for the pinocchio IK backend. Install the "
    "pinocchio extra: pip install 'upstream-drift[pinocchio]' "
    "(or use the dependency-free 'geometric' backend)."
)


def _module_available(name: str) -> bool:
    """Mock-tolerant availability probe (spec-less mocks count as absent)."""
    try:
        return find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


class PinocchioIKSolver(BaseIKSolver):
    """
    Pinocchio-based Inverse Kinematics solver.

    Default method is a pure-pinocchio damped-LM multi-marker position
    solve; ``method="pink"`` opts into the task-based QP formulation when
    pin-pink is installed.
    """

    def __init__(
        self,
        config: IKConfig | None = None,
        *,
        method: str = "lm",
        damping: float = 1e-6,
    ):
        """
        Initialize Pinocchio IK solver.

        Args:
            config: Solver configuration.
            method: ``"lm"`` (pure pinocchio, default) or ``"pink"``
                (task-based QP; requires pin-pink).
            damping: Levenberg-Marquardt damping (>= 0).
        """
        super().__init__(config)
        if method not in {"lm", "pink"}:
            raise ValueError(f"method must be 'lm' or 'pink', got {method!r}")
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        self.method = method
        self.damping = damping

    def solve(
        self,
        markers: MarkerTrajectory,
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
        config: IKConfig | None = None,
    ) -> JointTrajectory:
        """
        Solve IK for a marker trajectory using Pinocchio.

        Builds the model once and warm-starts each frame from the previous
        solution.
        """
        config = config or self.config
        pin, model, data = self._build_model(rig)

        q = np.array(pin.neutral(model), dtype=float)
        frames: list[JointStateFrame] = []
        for frame in markers.frames:
            marker_positions = {
                name: (m.x, m.y, m.z) for name, m in frame.markers.items()
            }
            q = self._solve_frame_from(
                pin, model, data, q, marker_positions, rig, weights, config
            )
            frames.append(
                JointStateFrame(
                    timestamp=frame.timestamp,
                    q=[float(v) for v in q],
                    qdot=None,
                    qddot=None,
                    frame_index=frame.frame_index,
                )
            )

        return JointTrajectory(
            id=f"ik-pinocchio-{markers.id}",
            skeleton=rig,
            frames=frames,
            metadata={
                "backend": "pinocchio",
                "method": self.method,
                "config": {
                    "max_iterations": config.max_iterations,
                    "tolerance": config.tolerance,
                },
            },
        )

    def solve_frame(
        self,
        markers: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
    ) -> list[float]:
        """
        Solve IK for a single frame using Pinocchio.

        Args:
            markers: Dict mapping marker names to (x, y, z) positions.
            rig: Scaled skeleton rig.
            weights: Optional per-marker weights.

        Returns:
            Joint angles (q) in rig DOF order.

        Raises:
            ImportError: When the pinocchio bindings are not installed.
            ValueError: When no marker matches a joint in the rig.
        """
        pin, model, data = self._build_model(rig)
        q0 = np.array(pin.neutral(model), dtype=float)
        q = self._solve_frame_from(
            pin, model, data, q0, markers, rig, weights, self.config
        )
        return [float(v) for v in q]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_model(self, rig: SkeletonRig) -> tuple[Any, Any, Any]:
        if not _module_available("pinocchio"):
            raise ImportError(_INSTALL_HINT)
        import pinocchio as pin

        model = rig_to_pinocchio_model(rig)
        if model.nq != rig.num_dofs:
            raise ValueError(
                f"URDF bridge produced nq={model.nq} for a {rig.num_dofs}-DOF rig"
            )
        return pin, model, model.createData()

    def _targets(
        self,
        pin: Any,
        model: Any,
        markers: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
        weights: MarkerWeights | None,
    ) -> list[tuple[int, np.ndarray, float]]:
        """(frame_id, target_xyz, weight) for markers matching rig joints."""
        weights = weights or MarkerWeights()
        targets: list[tuple[int, np.ndarray, float]] = []
        for name, xyz in markers.items():
            if name not in rig.joints:
                continue
            link = rig_joint_link_name(name)
            if not model.existFrame(link):
                continue
            targets.append(
                (
                    model.getFrameId(link),
                    np.asarray(xyz, dtype=float),
                    float(weights.get_weight(name)),
                )
            )
        if not targets:
            raise ValueError(
                "No marker matches a joint in the rig; cannot solve IK frame"
            )
        return targets

    def _solve_frame_from(
        self,
        pin: Any,
        model: Any,
        data: Any,
        q0: np.ndarray,
        markers: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
        weights: MarkerWeights | None,
        config: IKConfig,
    ) -> np.ndarray:
        targets = self._targets(pin, model, markers, rig, weights)
        if self.method == "pink":
            return self._solve_pink(pin, model, data, q0, targets, config)
        return self._solve_lm(pin, model, data, q0, targets, config)

    def _solve_lm(
        self,
        pin: Any,
        model: Any,
        data: Any,
        q0: np.ndarray,
        targets: list[tuple[int, np.ndarray, float]],
        config: IKConfig,
    ) -> np.ndarray:
        """Damped-LM multi-marker position solve (pure pinocchio)."""
        q = q0.copy()
        n_rows = 3 * len(targets)
        err = np.zeros(n_rows)
        jac = np.zeros((n_rows, model.nv))
        for _ in range(config.max_iterations):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            for row, (fid, target, weight) in enumerate(targets):
                err[3 * row : 3 * row + 3] = weight * (
                    target - np.asarray(data.oMf[fid].translation)
                )
            if float(np.linalg.norm(err)) < config.tolerance:
                break
            pin.computeJointJacobians(model, data, q)
            for row, (fid, _target, weight) in enumerate(targets):
                frame_jac = pin.getFrameJacobian(
                    model, data, fid, pin.LOCAL_WORLD_ALIGNED
                )
                jac[3 * row : 3 * row + 3, :] = weight * frame_jac[:3, :]
            # Damped least squares: dq = J^T (J J^T + lambda I)^-1 err
            jjt = jac @ jac.T
            jjt[np.diag_indices_from(jjt)] += max(self.damping, 1e-12)
            dq = jac.T @ np.linalg.solve(jjt, err)
            q = pin.integrate(model, q, dq)
            q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
        return q

    def _solve_pink(
        self,
        pin: Any,
        model: Any,
        data: Any,
        q0: np.ndarray,
        targets: list[tuple[int, np.ndarray, float]],
        config: IKConfig,
    ) -> np.ndarray:
        """Task-based QP solve via pin-pink (optional upgrade path)."""
        if not _module_available("pink"):
            raise ImportError(
                "method='pink' requires pin-pink. Install the pinocchio "
                "extra: pip install 'upstream-drift[pinocchio]' — or use "
                "the default method='lm'."
            )
        import pink
        from pink.tasks import FrameTask

        try:
            import qpsolvers
        except ImportError as exc:  # pragma: no cover - pink requires qpsolvers
            raise ImportError("pink requires qpsolvers") from exc
        if not qpsolvers.available_solvers:
            raise ImportError(
                "pink requires at least one QP solver backend; install "
                "quadprog (pip install quadprog) or another qpsolvers "
                "backend."
            )
        solver = (
            "quadprog"
            if "quadprog" in qpsolvers.available_solvers
            else qpsolvers.available_solvers[0]
        )

        configuration = pink.Configuration(model, data, q0)
        tasks = []
        for fid, target, weight in targets:
            frame_name = model.frames[fid].name
            task = FrameTask(
                frame_name,
                position_cost=weight,
                orientation_cost=0.0,
            )
            desired = pin.SE3.Identity()
            desired.translation = target
            task.set_target(desired)
            tasks.append(task)

        dt = 5e-2
        for _ in range(config.max_iterations):
            velocity = pink.solve_ik(configuration, tasks, dt, solver=solver)
            configuration.integrate_inplace(velocity, dt)
            if float(np.linalg.norm(velocity)) * dt < config.tolerance:
                break
        return np.asarray(configuration.q, dtype=float)
