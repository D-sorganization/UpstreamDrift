"""
Drake trajectory optimization backend for motion matching.

Part of issue #4568 / epic #8390 (B2/#8397). Direct-collocation trajectory
optimization over a ``MultibodyPlant`` built from the pipeline's canonical
``SkeletonRig`` (via :mod:`..model_bridge`), tracking a reference joint
trajectory with running effort cost and per-knot tracking costs, solved
with Drake's default NLP solver selection (SNOPT when licensed, else IPOPT).

pydrake is an optional dependency: when it is absent, :meth:`match` returns
a failed result with an install hint instead of raising at import time.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..contracts import JointStateFrame, JointTrajectory, SkeletonRig
from ..model_bridge import rig_root_link_name, rig_to_urdf
from .base import (
    BaseMotionMatchingSolver,
    CostWeights,
    MotionMatchingRequest,
    MotionMatchingResult,
)

logger = logging.getLogger(__name__)

# Direct collocation cost scales with knot count; long captures are tracked
# at a subsampled knot grid and the solution is reported at those knots.
_MAX_KNOTS = 50
_MIN_KNOTS = 3


def _drake_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("pydrake") is not None
    except (ValueError, ModuleNotFoundError):
        # A spec-less sys.modules entry is a test mock, not a usable
        # install (find_spec raises ValueError on those).
        return False


class DrakeTrajoptMatchingSolver(BaseMotionMatchingSolver):
    """
    Drake trajectory optimization motion matching solver.

    Builds a ``MultibodyPlant`` from the rig URDF, transcribes the tracking
    problem with ``DirectCollocation``, and returns the optimized joint
    trajectory with fit metrics against the reference.
    """

    def __init__(self, cost_weights: CostWeights | None = None):
        """
        Initialize Drake trajectory optimization solver.

        Args:
            cost_weights: Cost function weights
        """
        super().__init__(cost_weights)

    def match(
        self,
        reference: JointTrajectory,
        rig: SkeletonRig,
        request: MotionMatchingRequest | None = None,
    ) -> MotionMatchingResult:
        """
        Solve motion matching using Drake trajectory optimization.

        Args:
            reference: Reference joint trajectory to track
            rig: Scaled skeleton rig
            request: Optional matching request with configuration

        Returns:
            MotionMatchingResult with optimized trajectory
        """
        request_id = request.id if request else f"drake-trajopt-{reference.id}"

        if not _drake_available():
            return MotionMatchingResult(
                request_id=request_id,
                success=False,
                message=(
                    "pydrake is not installed. Install the drake extra: "
                    "pip install 'upstream-drift[drake]'"
                ),
                metadata={
                    "backend": "drake_trajopt",
                    "status": "dependency_missing",
                    "production_ready": False,
                },
            )

        start = time.perf_counter()
        try:
            tracked, solver_name, solver_success = self._solve(reference, rig)
        except (RuntimeError, ValueError) as exc:
            logger.exception("Drake trajopt solve failed")
            return MotionMatchingResult(
                request_id=request_id,
                success=False,
                message=f"Drake trajectory optimization failed: {exc}",
                metadata={
                    "backend": "drake_trajopt",
                    "status": "solver_error",
                    "production_ready": False,
                },
            )
        solve_time = time.perf_counter() - start

        if not solver_success:
            return MotionMatchingResult(
                request_id=request_id,
                success=False,
                message=f"Drake NLP solver ({solver_name}) did not converge",
                solve_time=solve_time,
                metadata={
                    "backend": "drake_trajopt",
                    "status": "not_converged",
                    "solver": solver_name,
                    "production_ready": False,
                },
            )

        knot_reference = self._resample_reference(
            reference, [f.timestamp for f in tracked.frames]
        )
        rmse = self._compute_rmse(knot_reference, tracked)
        max_error = max(
            (
                abs(rq - tq)
                for rf, tf in zip(knot_reference.frames, tracked.frames, strict=True)
                for rq, tq in zip(rf.q, tf.q, strict=True)
            ),
            default=0.0,
        )

        return MotionMatchingResult(
            request_id=request_id,
            success=True,
            tracked_trajectory=tracked,
            fit_metrics={
                "rmse": float(rmse),
                "max_error": float(max_error),
                "num_knots": len(tracked.frames),
            },
            solve_time=solve_time,
            message=f"Drake direct collocation converged via {solver_name}",
            metadata={
                "backend": "drake_trajopt",
                "status": "converged",
                "solver": solver_name,
                "production_ready": True,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _solve(
        self, reference: JointTrajectory, rig: SkeletonRig
    ) -> tuple[JointTrajectory, str, bool]:
        """Transcribe and solve; returns (trajectory, solver name, success)."""
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.planning import DirectCollocation
        from pydrake.solvers import Solve
        from pydrake.trajectories import PiecewisePolynomial

        times, q_ref = self._knot_grid(reference)
        num_knots = len(times)
        nq = q_ref.shape[1]

        plant = MultibodyPlant(time_step=0.0)
        parser = Parser(plant)
        parser.AddModelsFromString(rig_to_urdf(rig), "urdf")
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName(rig_root_link_name(rig)),
        )
        plant.Finalize()
        if plant.num_positions() != nq:
            raise ValueError(
                f"URDF bridge produced {plant.num_positions()} positions "
                f"for a {nq}-DOF rig"
            )

        context = plant.CreateDefaultContext()
        dt = float(times[-1] - times[0]) / max(num_knots - 1, 1)
        dt = max(dt, 1e-3)
        dircol = DirectCollocation(
            plant,
            context,
            num_time_samples=num_knots,
            minimum_time_step=0.5 * dt,
            maximum_time_step=2.0 * dt,
            input_port_index=plant.get_actuation_input_port().get_index(),
        )
        prog = dircol.prog()
        dircol.AddEqualTimeIntervalsConstraints()

        v_ref = np.gradient(q_ref, times, axis=0)
        x_ref = np.hstack([q_ref, v_ref])

        # Per-knot quadratic tracking cost (position-dominant), running
        # effort cost, and a pinned initial state.
        w_track = self.cost_weights.joint_tracking
        w_effort = max(self.cost_weights.effort, 1e-6)
        q_weight = np.diag([w_track] * nq + [0.1 * w_track] * nq)
        for k in range(num_knots):
            prog.AddQuadraticErrorCost(q_weight, x_ref[k], dircol.state(k))
        u = dircol.input()
        dircol.AddRunningCost(w_effort * u.dot(u))
        prog.AddBoundingBoxConstraint(x_ref[0], x_ref[0], dircol.initial_state())

        x_init = PiecewisePolynomial.FirstOrderHold(times, x_ref.T)
        u_init = PiecewisePolynomial.ZeroOrderHold(
            [times[0], times[-1]], np.zeros((plant.num_actuators(), 2))
        )
        dircol.SetInitialTrajectory(u_init, x_init)

        result = Solve(prog)
        solver_name = result.get_solver_id().name()
        if not result.is_success():
            return reference, solver_name, False

        sample_times = dircol.GetSampleTimes(result)
        states = dircol.GetStateSamples(result)  # (2*nq, num_knots)
        frames = [
            JointStateFrame(
                timestamp=float(times[0] + (sample_times[k] - sample_times[0])),
                frame_index=k,
                q=[float(v) for v in states[:nq, k]],
                qd=[float(v) for v in states[nq:, k]],
            )
            for k in range(num_knots)
        ]
        tracked = JointTrajectory(
            id=f"{reference.id}-drake-trajopt",
            skeleton=rig,
            frames=frames,
            metadata={"backend": "drake_trajopt", "solver": solver_name},
        )
        return tracked, solver_name, True

    @staticmethod
    def _knot_grid(reference: JointTrajectory) -> tuple[np.ndarray, np.ndarray]:
        """Knot times and reference q at knots, subsampled to _MAX_KNOTS."""
        all_times = np.array([f.timestamp for f in reference.frames], dtype=float)
        all_q = np.array([f.q for f in reference.frames], dtype=float)
        n = len(all_times)
        if n < _MIN_KNOTS:
            # Direct collocation needs >= 3 samples; pad by linear extension
            # of the last interval (or a nominal 10 ms when only one frame).
            dt = all_times[-1] - all_times[-2] if n > 1 else 0.01
            while len(all_times) < _MIN_KNOTS:
                all_times = np.append(all_times, all_times[-1] + max(dt, 1e-3))
                all_q = np.vstack([all_q, all_q[-1]])
            return all_times, all_q
        if n <= _MAX_KNOTS:
            return all_times, all_q
        idx = np.linspace(0, n - 1, _MAX_KNOTS).round().astype(int)
        return all_times[idx], all_q[idx]

    @staticmethod
    def _resample_reference(
        reference: JointTrajectory, timestamps: list[float]
    ) -> JointTrajectory:
        """Reference trajectory linearly interpolated onto ``timestamps``."""
        ref_t = np.array([f.timestamp for f in reference.frames], dtype=float)
        ref_q = np.array([f.q for f in reference.frames], dtype=float)
        frames = []
        for k, t in enumerate(timestamps):
            q = [float(np.interp(t, ref_t, ref_q[:, d])) for d in range(ref_q.shape[1])]
            frames.append(JointStateFrame(timestamp=float(t), frame_index=k, q=q))
        return JointTrajectory(
            id=f"{reference.id}-knots",
            skeleton=reference.skeleton,
            frames=frames,
        )
