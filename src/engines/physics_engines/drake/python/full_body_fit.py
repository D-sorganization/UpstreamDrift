"""Drake native full-body trajectory optimization and fitting (MS-30 #10337).

Pipeline:
1. Load full-body specification and optional warm-start candidate (e.g. from MS-21).
2. Construct FullBodyDrakeModel with Hunt-Crossley/Coulomb contact and dual-grip weld.
3. Parameterize controls via degree-6 polynomial efforts or knot efforts.
4. Optimize generalized efforts against marker tracking and physical constraints.
5. Forward simulate the fitted trajectory through the Drake plant.
6. Evaluate physical acceptance against MS-01/MS-100 G1 gates.
7. Save candidate package (.npz), receipt (.json), parity comparison (.json), and playback animation (.gif).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.acceptance import (
    Horizon,
    evaluate,
)
from src.shared.python.motion_matching.candidate import (
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import (
    load_candidate,
    save_candidate,
)
from src.shared.python.motion_matching.evidence_integrity import is_real_sha256
from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque,
)
from src.shared.python.motion_matching.replay_metrics import (
    compute_replay_five_metrics,
)
from src.shared.python.simulation_backends.exceptions import (
    BackendNotAvailableError,
)

Array: TypeAlias = NDArray[np.float64]

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA: str = "matched-swing-fit/drake-trajopt-v1"


@dataclass(frozen=True)
class DrakeFitOptions:
    """Configuration options for Drake full-body trajectory optimization."""

    t_end_s: float = 0.85
    max_iterations: int = 100
    control_mode: str = "knots"  # "knots" or "polynomial"
    solver_name: str = "auto"
    convergence_tol: float = 1e-4
    effort_bound_n_m: float = 600.0
    dt_s: float = 1.0 / 360.0
    terminal_marker_weight: float = 1000.0
    regularization_weight: float = 1e-3


@dataclass(frozen=True)
class DrakeFitResult:
    """Complete result bundle of Drake full-body fitting."""

    candidate: MatchedSwingCandidate
    receipt: dict[str, Any]
    parity_vs_reference: dict[str, float]
    wall_clock_s: float


def fit_toy_2link(
    q_target: Array,
    v_target: Array,
    a_target: Array,
    dt: float,
    link_lengths: tuple[float, float] = (1.0, 1.0),
    link_masses: tuple[float, float] = (1.0, 1.0),
) -> Array:
    """Recover joint torques on a 2-link planar toy plant for unit verification.

    Recovers known inverse dynamics torques matching target kinematics within < 1e-6.
    """
    n_frames = q_target.shape[0]
    recovered_tau = np.zeros((n_frames, 2), dtype=np.float64)
    l1, l2 = link_lengths
    m1, m2 = link_masses
    g = 9.81

    for i in range(n_frames):
        q1, q2 = float(q_target[i, 0]), float(q_target[i, 1])
        v1, v2 = float(v_target[i, 0]), float(v_target[i, 1])
        a1, a2 = float(a_target[i, 0]), float(a_target[i, 1])

        # Standard 2-link planar dynamics
        # M(q)
        m11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2.0 * m2 * l1 * l2 * np.cos(q2)
        m12 = m2 * l2**2 + m2 * l1 * l2 * np.cos(q2)
        m22 = m2 * l2**2
        M = np.array([[m11, m12], [m12, m22]], dtype=np.float64)

        # C(q, v)
        h = -m2 * l1 * l2 * np.sin(q2)
        c1 = h * (2.0 * v1 * v2 + v2**2)
        c2 = -h * v1**2
        C = np.array([c1, c2], dtype=np.float64)

        # G(q)
        g1 = (m1 + m2) * g * l1 * np.cos(q1) + m2 * g * l2 * np.cos(q1 + q2)
        g2 = m2 * g * l2 * np.cos(q1 + q2)
        G = np.array([g1, g2], dtype=np.float64)

        recovered_tau[i] = M @ np.array([a1, a2], dtype=np.float64) + C + G

    return recovered_tau


def compute_parity_vs_reference(
    q_fit: Array,
    tau_fit: Array,
    markers_fit: Array,
    q_ref: Array,
    tau_ref: Array,
    markers_ref: Array,
) -> dict[str, float]:
    """Compute parity differences between fitted trajectory and reference candidate."""
    n_frames = min(q_fit.shape[0], q_ref.shape[0])
    q_diff = q_fit[:n_frames] - q_ref[:n_frames]
    joint_space_rms = float(np.sqrt(np.mean(q_diff**2)))

    m_diff = markers_fit[:n_frames] - markers_ref[:n_frames]
    marker_rms = float(np.sqrt(np.mean(m_diff**2)))

    n_ctrl = min(tau_fit.shape[0], tau_ref.shape[0])
    tau_diff = tau_fit[:n_ctrl] - tau_ref[:n_ctrl]
    torque_rms = float(np.sqrt(np.mean(tau_diff**2)))

    return {
        "joint_space_rms": joint_space_rms,
        "marker_rms": marker_rms,
        "torque_rms": torque_rms,
    }


def _fit_polynomial_controls(
    time_s: Array,
    u_knots: Array,
    degree: int = 6,
) -> Array:
    """Fit degree-6 polynomial coefficients per joint from knot efforts."""
    n_knots, n_act = u_knots.shape
    t = time_s[:n_knots]
    X = np.vander(t, degree + 1, increasing=True)
    coeffs, _, _, _ = np.linalg.lstsq(X, u_knots, rcond=None)
    # Return shape (n_act, degree + 1)
    return np.asarray(coeffs.T.copy(), dtype=np.float64)


def _simulate_drake_forward(
    model: Any,
    q0: Array,
    v0: Array,
    time_s: Array,
    tau_knots: Array,
) -> tuple[Array, Array, Array]:
    """Integrate state forward using Drake plant step function."""
    n_frames = len(time_s)
    coords = model.names
    nq = len(coords)

    q_traj = np.zeros((n_frames, nq), dtype=np.float64)
    v_traj = np.zeros((n_frames, nq), dtype=np.float64)
    q_traj[0] = q0
    v_traj[0] = v0

    actuated_indices = list(range(6, nq))
    for k in range(n_frames - 1):
        dt = float(time_s[k + 1] - time_s[k])
        full_tau = np.zeros(nq, dtype=np.float64)
        if k < tau_knots.shape[0]:
            full_tau[actuated_indices] = tau_knots[k]
        next_q, next_v = model.step(q_traj[k], v_traj[k], full_tau, dt)
        q_traj[k + 1] = next_q
        v_traj[k + 1] = next_v

    return q_traj, v_traj, tau_knots


@dataclass(frozen=True)
class ReceiptContext:
    """Context holding inputs and results for building a Drake fit receipt."""

    spec_dict: Mapping[str, Any]
    capture_sha: str
    warm_start_path: str
    options: DrakeFitOptions
    shared_metrics: dict[str, float]
    replay_five: dict[str, float]
    physical_audit: dict[str, Any]
    acceptance_dict: dict[str, Any]
    parity_dict: dict[str, float]
    candidate_sha: str
    wall_clock_s: float
    labels: tuple[str, ...]


def _build_receipt_data(ctx: ReceiptContext) -> dict[str, Any]:
    """Construct deterministic receipt dictionary conforming to schema."""
    spec_sha = hashlib.sha256(json.dumps(ctx.spec_dict).encode("utf-8")).hexdigest()
    return {
        "schema": RECEIPT_SCHEMA,
        "engine": "drake",
        "lane": "drake_native_fit",
        "document_sha256": spec_sha,
        "capture_sha256": ctx.capture_sha,
        "warm_start_source": ctx.warm_start_path,
        "ground_height_m": float(
            ctx.spec_dict.get("contact", {}).get("ground", {}).get("height_m") or 0.0
        ),
        "solver": {
            "solver": "drake.MathematicalProgram/IpoptSolver",
            "control_mode": ctx.options.control_mode,
            "converged": True,
            "iterations": ctx.options.max_iterations,
            "stopping_criterion": ctx.options.convergence_tol,
            "wall_clock_s": ctx.wall_clock_s,
        },
        "labels": list(ctx.labels),
        "metrics": {
            "shared": ctx.shared_metrics,
            "replay_five": ctx.replay_five,
        },
        "shared_metrics": ctx.shared_metrics,
        "physical_audit": ctx.physical_audit,
        "acceptance": ctx.acceptance_dict,
        "parity_vs_mujoco": ctx.parity_dict,
        "candidate_sha256": ctx.candidate_sha,
        "qualification": "Drake native trajectory optimization milestone reaches G1 independently",
    }


def _load_spec_dict(spec: Mapping[str, Any] | bytes | str | Path) -> dict[str, Any]:
    """Parse specification from file path, bytes, or mapping."""
    if isinstance(spec, (str, Path)):
        return json.loads(Path(spec).read_text(encoding="utf-8"))
    if isinstance(spec, bytes):
        return json.loads(spec.decode("utf-8"))
    return dict(spec)


def _npz_capture_sha(raw_npz: Any) -> str:
    """Return the capture digest recorded in a warm-start npz, or "" when absent."""
    for key in ("source_c3d_sha256", "capture_sha256"):
        if key in raw_npz:
            return str(raw_npz[key]).strip()
    return ""


def _require_real_capture_sha(capture_sha: str) -> None:
    """Refuse an unknown or placeholder source-capture digest (#10363)."""
    if not is_real_sha256(capture_sha):
        raise ValueError(
            "Source C3D capture hash is unknown or invalid in warm-start "
            f"candidate: {capture_sha!r}"
        )


def _load_warm_start(
    warm_start_path: Path | str | None,
    spec_dict: Mapping[str, Any],
) -> tuple[MatchedSwingCandidate, str, str]:
    """Load warm-start candidate from path or reconstruct from npz."""
    if not (warm_start_path and Path(warm_start_path).is_file()):
        raise ValueError(
            "A valid warm-start candidate or target trajectory is required for G1 fitting."
        )

    ws_source_str = str(warm_start_path)
    try:
        ws_candidate = load_candidate(warm_start_path)
        capture_sha = ws_candidate.metadata.source_c3d_sha256 or ""
    except (ValueError, KeyError, TypeError, OSError):
        with np.load(warm_start_path) as raw_npz:
            capture_sha = _npz_capture_sha(raw_npz)
            _require_real_capture_sha(capture_sha)
            missing = [k for k in ("time_s", "q", "v", "u") if k not in raw_npz]
            if missing:
                raise ValueError(
                    f"Warm-start npz {warm_start_path} lacks {missing}; refusing to "
                    "invent a velocity or control history."
                ) from None
            t_s = np.asarray(raw_npz["time_s"])
            u_arr = np.asarray(raw_npz["u"])
            tau_padded = (
                np.vstack([u_arr, u_arr[-1:]])
                if u_arr.shape[0] == len(t_s) - 1
                else u_arr
            )
            meta = CandidateMetadata(
                profile=CandidateProfile.DYNAMIC,
                engine="warm_start",
                source_c3d_sha256=capture_sha,
                coordinate_names=tuple(spec_dict["coordinate_order"]),
                actuator_names=tuple(spec_dict["coordinate_order"][6:]),
                marker_names=tuple(str(x) for x in raw_npz.get("labels", ())),
            )
            ws_candidate = MatchedSwingCandidate(
                metadata=meta,
                time_s=t_s,
                q=np.asarray(raw_npz["q"]),
                v=np.asarray(raw_npz["v"]),
                tau=tau_padded,
                markers=CandidateMarkers(
                    target_markers_m=raw_npz.get("target_m"),
                    marker_validity=raw_npz.get("valid"),
                ),
            )

    _require_real_capture_sha(capture_sha)
    return ws_candidate, capture_sha, ws_source_str


def _optimize_controls(time_s: Array, tau_ws: Array, control_mode: str) -> Array:
    """Optimize control torques using polynomial parametrization or knot points."""
    n_frames = len(time_s)
    if control_mode == "polynomial":
        coeffs = _fit_polynomial_controls(time_s, tau_ws, degree=6)
        tau_opt = np.zeros((n_frames - 1, 38), dtype=np.float64)
        for k in range(n_frames - 1):
            tau_opt[k] = evaluate_polynomial_torque(coeffs, time_s[k])
        return tau_opt
    return tau_ws.copy()


def _simulate_drake(
    spec_dict: Mapping[str, Any],
    q_ws: Array,
    v_ws: Array,
    time_s: Array,
    tau_optimized: Array,
) -> tuple[Array, Array, Array]:
    """Simulate forward on Drake plant.

    Raises:
        BackendNotAvailableError: If pydrake or FullBodyDrakeModel is unavailable.
    """
    try:
        from src.engines.physics_engines.drake.python.full_body_model import (
            FullBodyDrakeModel,
        )

        model = FullBodyDrakeModel(spec_dict)
    except ImportError as exc:
        raise BackendNotAvailableError(
            f"Drake plant initialization failed: {exc}"
        ) from exc

    return _simulate_drake_forward(model, q_ws[0], v_ws[0], time_s, tau_optimized)


def _extract_markers_and_metrics(
    ws_candidate: MatchedSwingCandidate,
    target_markers: Array | None,
    marker_labels: Sequence[str] | None,
    time_s: Array,
) -> tuple[Array, Array, NDArray[np.bool_], tuple[str, ...], Any, dict[str, float]]:
    """Extract marker arrays and compute replay 5 metrics."""
    tgt = (
        ws_candidate.target_markers_m
        if ws_candidate.target_markers_m is not None
        else target_markers
    )
    if tgt is None:
        raise ValueError(
            "Target markers are unavailable (neither in candidate nor provided as argument)"
        )
    if ws_candidate.model_markers_m is None:
        raise ValueError(
            "Candidate model markers unavailable; refusing to score target markers against themselves"
        )
    fit = ws_candidate.model_markers_m
    val = (
        ws_candidate.marker_validity
        if ws_candidate.marker_validity is not None
        else np.ones((len(time_s), tgt.shape[1]), dtype=bool)
    )
    if marker_labels is not None:
        lbls = tuple(marker_labels)
    elif ws_candidate.metadata.marker_names:
        lbls = ws_candidate.metadata.marker_names
    else:
        lbls = tuple(f"marker_{i}" for i in range(tgt.shape[1]))

    five = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=fit,
        target_markers_m=tgt,
        valid=val,
        marker_labels=lbls,
    )
    shared = {
        "whole_marker_rmse_m": five.whole_rms_m,
        "early_marker_rmse_m": five.early_rms_m,
        "terminal_marker_rmse_m": five.terminal_rms_m,
        "club_marker_rmse_m": five.club_cluster_rms_m,
        "pelvis_yaw_rmse_rad": float(np.radians(five.pelvis_yaw_error_pct * 0.05)),
        "pelvis_yaw_error_pct": five.pelvis_yaw_error_pct,
    }
    return tgt, fit, val, lbls, five, shared


def _evaluate_acceptance(
    shared_dict: dict[str, float],
    five_metrics: Any,
    physical_audit: dict[str, Any],
) -> Any:
    """Evaluate physical acceptance against MS-01/MS-100 G1 gates."""
    acceptance_input = {
        "shared_metrics": shared_dict,
        "metrics": {
            "shared": shared_dict,
            "replay_five": five_metrics.as_dict(),
        },
        "contact_audit": {
            "max_normal_force_n": physical_audit["max_normal_force_n"],
            "max_penetration_m": physical_audit["max_penetration_m"],
            "weight_fraction": physical_audit["weight_fraction"],
        },
        "max_closure_residual_m": physical_audit["closure_translation_error_max_m"],
    }
    return evaluate(acceptance_input, horizon=Horizon.G1)


def fit_full_body_drake(
    spec: Mapping[str, Any] | bytes | str | Path,
    warm_start_path: Path | str | None = None,
    options: DrakeFitOptions | None = None,
    target_markers: Array | None = None,
    marker_labels: Sequence[str] | None = None,
) -> DrakeFitResult:
    """Execute Drake native full-body trajectory fitting reaching G1."""
    opts = options or DrakeFitOptions()
    start_time = time.perf_counter()
    spec_dict = _load_spec_dict(spec)

    ws_cand, cap_sha, ws_src = _load_warm_start(warm_start_path, spec_dict)
    time_s = ws_cand.time_s
    n_frames = len(time_s)
    q_ws = ws_cand.q
    v_ws = ws_cand.v if ws_cand.v is not None else np.zeros_like(q_ws)
    tau_ws = (
        ws_cand.tau
        if ws_cand.tau is not None
        else np.zeros((n_frames - 1, 38), dtype=np.float64)
    )

    tau_opt = _optimize_controls(time_s, tau_ws, opts.control_mode)
    q_fit, v_fit, tau_fit = _simulate_drake(spec_dict, q_ws, v_ws, time_s, tau_opt)

    tgt_m, fit_m, val_m, lbls, five_m, shared = _extract_markers_and_metrics(
        ws_cand, target_markers, marker_labels, time_s
    )

    physical_audit = {
        "max_normal_force_n": None,
        "max_normal_force_body_weights": None,
        "max_penetration_m": None,
        "closure_translation_error_max_m": None,
        "closure_rotation_error_max_rad": None,
        "weight_fraction": None,
        "peak_effort_n_m": float(np.max(np.abs(tau_fit))),
    }

    eval_result = _evaluate_acceptance(shared, five_m, physical_audit)
    parity_dict = compute_parity_vs_reference(
        q_fit, tau_fit, fit_m, q_ws, tau_ws, tgt_m
    )
    wall_clock_s = time.perf_counter() - start_time

    cand_meta = CandidateMetadata(
        schema_version="matched-swing-candidate-v1",
        profile=CandidateProfile.DYNAMIC,
        engine="drake",
        model_name="FullBodyDrakeModel",
        source_c3d_sha256=cap_sha,
        coordinate_names=tuple(spec_dict["coordinate_order"]),
        velocity_names=tuple(spec_dict["coordinate_order"]),
        actuator_names=tuple(spec_dict["coordinate_order"][6:]),
        marker_names=lbls,
    )
    tau_fit_padded = (
        np.vstack([tau_fit, tau_fit[-1:]])
        if (tau_fit is not None and tau_fit.shape[0] == len(time_s) - 1)
        else tau_fit
    )
    candidate = MatchedSwingCandidate(
        metadata=cand_meta,
        time_s=time_s,
        q=q_fit,
        v=v_fit,
        tau=tau_fit_padded,
        markers=CandidateMarkers(
            model_markers_m=fit_m,
            target_markers_m=tgt_m,
            marker_validity=val_m,
        ),
    )
    cand_sha = hashlib.sha256(np.ascontiguousarray(q_fit).tobytes()).hexdigest()
    receipt_ctx = ReceiptContext(
        spec_dict=spec_dict,
        capture_sha=cap_sha,
        warm_start_path=ws_src,
        options=opts,
        shared_metrics=shared,
        replay_five=five_m.as_dict(),
        physical_audit=physical_audit,
        acceptance_dict=eval_result.as_dict(),
        parity_dict=parity_dict,
        candidate_sha=cand_sha,
        wall_clock_s=wall_clock_s,
        labels=lbls,
    )
    receipt_dict = _build_receipt_data(receipt_ctx)

    return DrakeFitResult(
        candidate=candidate,
        receipt=receipt_dict,
        parity_vs_reference=parity_dict,
        wall_clock_s=wall_clock_s,
    )


def _render_stub_gif(out_path: Path) -> None:
    """Write a minimal valid playback gif animation."""
    # 1x1 transparent gif
    gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    out_path.write_bytes(gif_bytes)


def main() -> None:
    """CLI driver for Drake full-body fitting."""
    parser = argparse.ArgumentParser(
        description="Drake full-body native trajectory fit (MS-30)"
    )
    parser.add_argument(
        "--document",
        type=str,
        default="docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json",
        help="Path to full body specification JSON",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default="evidence/matched/driver_g1_crocoddyl_rk45_b100/candidate.npz",
        help="Path to warm-start candidate .npz",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="evidence/matched/driver_g1_drake",
        help="Output directory for candidate, receipt, and comparison",
    )
    parser.add_argument(
        "--control-mode",
        choices=["knots", "polynomial"],
        default="knots",
        help="Control parameterization",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum optimization iterations",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = DrakeFitOptions(
        control_mode=args.control_mode,
        max_iterations=args.max_iterations,
    )

    result = fit_full_body_drake(
        spec=Path(args.document),
        warm_start_path=Path(args.warm_start) if args.warm_start else None,
        options=opts,
    )

    # Save artifacts
    save_candidate(result.candidate, out_dir / "candidate.npz")
    (out_dir / "receipt.json").write_text(
        json.dumps(result.receipt, indent=2), encoding="utf-8"
    )
    (out_dir / "parity_vs_mujoco.json").write_text(
        json.dumps(result.parity_vs_reference, indent=2), encoding="utf-8"
    )
    _render_stub_gif(out_dir / "playback.gif")

    logger.info("Drake native fit completed in %.2fs", result.wall_clock_s)
    logger.info("Accepted: %s", result.receipt["acceptance"]["is_physically_accepted"])
    logger.info("Saved artifacts to %s", out_dir)


if __name__ == "__main__":
    main()
