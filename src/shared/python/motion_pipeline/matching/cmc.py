"""
Computed Muscle Control (CMC) backend for motion matching.

Part of issue #4568. OpenSim CMC for muscle-driven matching.

Issue #5254 slice 3: outer-loop acceleration via the
``upstream_pinocchio_id`` Rust crate. The crate is engine-agnostic
despite its name -- it runs a Python callback per frame and amortises
finite-difference + buffer-staging + finiteness validation into native
code. CMC's inner LP/QP for muscle redundancy resolution is a separate,
larger slice; here we wire the *outer* loop and keep the per-frame
callback as a placeholder that returns zero torques when OpenSim is
unavailable.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..contracts import (
    JointTrajectory,
    SkeletonRig,
    TorqueFrame,
    TorqueTrajectory,
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

# Optional Rust acceleration (issue #5254 slice 3). Reuses the
# ``upstream_pinocchio_id`` crate because the outer driver loop is
# engine-agnostic.
try:  # pragma: no cover - exercised conditionally
    import upstream_pinocchio_id as _rust_outer_loop  # type: ignore[import-not-found]

    _HAVE_RUST = True
except Exception:  # pragma: no cover - fallback path  # noqa: BLE001
    _rust_outer_loop = None  # type: ignore[assignment]
    _HAVE_RUST = False

import os


def _use_rust_outer_loop() -> bool:
    return _HAVE_RUST and os.environ.get("RUST_OUTER_LOOP", "1") == "1"


class CMCMatchingSolver(BaseMotionMatchingSolver):
    """
    Computed Muscle Control (CMC) motion matching solver.

    Uses OpenSim's CMC algorithm to compute muscle activations
    that track a reference joint trajectory.

    The current outer loop is accelerated via ``upstream_pinocchio_id``
    when the wheel is installed; the per-frame muscle redundancy
    LP/QP remains a placeholder pending issue #5254 slice 6.
    """

    backend_type = MatchingBackendType.CMC

    def __init__(self, cost_weights: CostWeights | None = None):
        """
        Initialize CMC solver.

        Args:
            cost_weights: Cost function weights
        """
        super().__init__(cost_weights)

    @staticmethod
    def _per_frame_callback(osim_model: object | None, state: object | None):
        """Build a per-frame closure invoked by the Rust outer loop.

        When OpenSim is unavailable (CI, lightweight test runs) the
        closure returns zero torques so the outer loop still exercises
        finite-difference + finiteness + result aggregation. When
        OpenSim is available the closure realises the system to the
        acceleration stage; the muscle redundancy LP/QP is the next
        slice and currently returns zero.
        """

        def _cb(q_row: np.ndarray, v_row: np.ndarray, a_row: np.ndarray) -> np.ndarray:
            if osim_model is None or state is None:
                return np.zeros_like(q_row, dtype=np.float64)
            try:  # pragma: no cover - exercised only when opensim installed
                import opensim as osim  # noqa: F401

                n_coord = osim_model.getNumCoordinates()  # type: ignore[attr-defined]
                if len(q_row) == n_coord:
                    for i in range(n_coord):
                        coord = osim_model.getCoordinateSet().get(i)  # type: ignore[attr-defined]
                        coord.setValue(state, float(q_row[i]))
                        coord.setSpeedValue(state, float(v_row[i]))
                    osim_model.realizeAcceleration(state)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - safety  # noqa: BLE001
                logger.debug("CMC per-frame opensim step failed: %s", exc)
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
        """Pure-Python reference driver loop (preserved verbatim).

        Used when ``upstream_pinocchio_id`` is unavailable and as the
        parity oracle in tests.
        """
        n_frames, n_dof = q_all.shape
        tau_all = np.zeros((n_frames, n_dof), dtype=np.float64)
        for i in range(n_frames):
            tau_all[i] = np.asarray(
                callback(q_all[i], qdot_all[i], qddot_all[i]),
                dtype=np.float64,
            ).flatten()
        if not np.all(np.isfinite(tau_all)):
            bad = int(np.argmax(~np.all(np.isfinite(tau_all), axis=1)))
            raise RuntimeError(f"CMC produced non-finite torques at frame {bad}")
        return tau_all

    @staticmethod
    def _compute_tau_rust(
        times: np.ndarray,
        q_all: np.ndarray,
        qdot_all: np.ndarray,
        qddot_all: np.ndarray,
        callback,
    ) -> np.ndarray:
        """Rust-driven outer loop.

        ``upstream_pinocchio_id.inverse_dynamics`` runs the per-frame
        driver entirely in Rust, calling back into Python only for the
        CMC inner step. Finite-difference, finiteness validation and
        result aggregation all stay native.
        """
        assert _rust_outer_loop is not None  # narrowed by _HAVE_RUST
        q_c = np.ascontiguousarray(q_all, dtype=np.float64)
        v_c = np.ascontiguousarray(qdot_all, dtype=np.float64)
        a_c = np.ascontiguousarray(qddot_all, dtype=np.float64)
        t_c = np.ascontiguousarray(times, dtype=np.float64)

        # The Rust crate expects ``rnea_callback(q, v, a) -> tau``.
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
        Solve motion matching using Computed Muscle Control.

        Args:
            reference: Reference joint trajectory to track
            rig: Scaled skeleton rig
            request: Optional matching request with configuration

        Returns:
            MotionMatchingResult with tracked trajectory and muscle activations
        """
        if reference is None or rig is None:
            raise ValueError("reference and rig must be provided")
        if not reference.frames:
            raise ValueError("reference must have at least one frame")

        request_id = request.id if request else f"cmc-{reference.id}"
        t_start = time.perf_counter()

        # Reuse the shared, Rust-accelerated finite-difference helper.
        times, q_all, qdot_all, qddot_all = (
            PinocchioInverseDynMatchingSolver._finite_difference(reference)
        )
        n_frames, n_dof = q_all.shape

        # Lazily attempt to load OpenSim so the placeholder path still
        # runs on systems without it.
        osim_model = None
        state = None
        try:  # pragma: no cover - exercised when opensim installed
            import opensim as osim

            osim_model = osim.Model()
            state = osim_model.initSystem()
        except Exception:  # noqa: BLE001
            osim_model = None
            state = None

        callback = self._per_frame_callback(osim_model, state)

        if _use_rust_outer_loop():
            try:
                tau_all = self._compute_tau_rust(
                    times, q_all, qdot_all, qddot_all, callback
                )
            except (
                Exception  # noqa: BLE001 - safety fallback to pure-Python CMC
            ) as exc:  # pragma: no cover - safety fallback
                logger.warning(
                    "upstream_pinocchio_id rust path failed (%s); "
                    "falling back to pure-Python CMC outer loop",
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

        rig_joint_names: list[str] = []
        for jname, jdef in rig.joints.items():
            for _ in jdef.axes:
                rig_joint_names.append(jname)

        torque_traj = TorqueTrajectory(
            frames=torque_frames,
            rig_joint_names=rig_joint_names,
            metadata={"semantics": "torques", "source_id": f"{reference.id}-torques"},
        )

        residual_report = self._compute_residual_report(reference, reference)
        solve_time = time.perf_counter() - t_start

        # success=False because muscle redundancy LP/QP is still a
        # placeholder (issue #5269 -> #5254 slice 6). The outer-loop
        # acceleration is the in-scope improvement.
        return MotionMatchingResult(
            request_id=request_id,
            success=False,
            tracked_trajectory=reference,
            torque_trajectory=torque_traj,
            residual_report=residual_report,
            fit_metrics={"rmse": 0.0, "max_error": 0.0},
            solve_time=float(solve_time),
            message="CMC solver - placeholder torques (full CMC not yet implemented)",
            metadata={
                "backend": self.backend_type.value,
                "status": "placeholder",
                "n_frames": int(n_frames),
                "n_dof": int(n_dof),
                "rust_outer_loop": bool(_HAVE_RUST),
            },
        )


__all__ = ["CMCMatchingSolver"]
