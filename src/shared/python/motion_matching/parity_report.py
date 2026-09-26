"""Unified Cross-Engine Parity Report generation (MS-70, #10350).

Provides:
1. evaluate_pointwise_trajectory_parity: Pointwise marker distance comparison (rejects aggregate RMSE equivalence).
2. evaluate_pointwise_torque_parity: Pointwise joint torque comparison with absolute floor.
3. build_parity_report: Constructs a complete UnifiedParityReport across all requested engines.
4. run_parity_report_cli: Command-line entry point producing JSON and Markdown reports.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.candidate import (
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import load_candidate
from src.shared.python.motion_matching.parity_schema import (
    PARITY_REPORT_SCHEMA_VERSION,
    ComparisonClass,
    EngineParityRow,
    PointwiseDifference,
    UnifiedParityReport,
)
from src.shared.python.motion_matching.pipeline.plant import (
    EngineUnavailableError,
    MatchingPlant,
    get_plant,
)

logger = logging.getLogger(__name__)

ALL_ENGINES: tuple[str, ...] = (
    "mujoco",
    "drake",
    "pinocchio",
    "opensim",
    "myosuite",
    "simscape",
)

SAME_MODEL_ENGINES: frozenset[str] = frozenset({"mujoco", "drake", "pinocchio"})
NATIVE_MODEL_ENGINES: frozenset[str] = frozenset({"opensim", "myosuite", "simscape"})


@precondition(
    lambda traj_a, traj_b, tolerance_m=0.001: (
        isinstance(traj_a, np.ndarray)
        and isinstance(traj_b, np.ndarray)
        and traj_a.shape == traj_b.shape
        and tolerance_m > 0.0
    ),
    "Trajectories must be numpy arrays of identical shape and tolerance must be positive",
)
@postcondition(
    lambda r: isinstance(r, PointwiseDifference), "Must return PointwiseDifference"
)
def evaluate_pointwise_trajectory_parity(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    tolerance_m: float = 0.001,
) -> PointwiseDifference:
    """Compare two trajectories pointwise, rejecting identical aggregate RMSE on divergent paths.

    Computes:
    - delta(t, m) = ||traj_a(t, m) - traj_b(t, m)||_2
    - max_abs_diff = max(delta)
    - rms_diff = sqrt(mean(delta^2))
    - mean_diff = mean(delta)
    - pass_gate = max_abs_diff <= tolerance_m (default 1.0 mm)
    """
    diff = np.asarray(traj_a, dtype=np.float64) - np.asarray(traj_b, dtype=np.float64)
    # Pointwise Euclidean distance per frame and marker
    if diff.ndim >= 3:
        pointwise_dist = np.sqrt(np.sum(diff**2, axis=-1))
    elif diff.ndim == 2:
        pointwise_dist = np.abs(diff)
    else:
        pointwise_dist = np.abs(diff)

    max_val = float(np.max(pointwise_dist))
    rms_val = float(np.sqrt(np.mean(pointwise_dist**2)))
    mean_val = float(np.mean(pointwise_dist))
    pass_gate = bool(max_val <= tolerance_m)

    return PointwiseDifference(
        metric_name="marker_diff_m",
        max_abs_diff=max_val,
        rms_diff=rms_val,
        mean_diff=mean_val,
        unit="m",
        pass_gate=pass_gate,
    )


@precondition(
    lambda tau_a, tau_b, floor_nm=1.0, tolerance_pct=2.0: (
        isinstance(tau_a, np.ndarray)
        and isinstance(tau_b, np.ndarray)
        and tau_a.shape == tau_b.shape
        and floor_nm > 0.0
        and tolerance_pct > 0.0
    ),
    "Torques must be numpy arrays of identical shape with positive floor and tolerance",
)
@postcondition(
    lambda r: isinstance(r, PointwiseDifference), "Must return PointwiseDifference"
)
def evaluate_pointwise_torque_parity(
    tau_a: np.ndarray,
    tau_b: np.ndarray,
    floor_nm: float = 1.0,
    tolerance_pct: float = 2.0,
) -> PointwiseDifference:
    """Compare joint torques with an absolute floor near zero to prevent divide-by-zero."""
    a = np.asarray(tau_a, dtype=np.float64)
    b = np.asarray(tau_b, dtype=np.float64)
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), floor_nm)
    pct_diff = 100.0 * (abs_diff / denom)

    max_pct = float(np.max(pct_diff))
    rms_pct = float(np.sqrt(np.mean(pct_diff**2)))
    mean_pct = float(np.mean(pct_diff))
    pass_gate = bool(mean_pct <= tolerance_pct)

    return PointwiseDifference(
        metric_name="torque_diff_pct",
        max_abs_diff=max_pct,
        rms_diff=rms_pct,
        mean_diff=mean_pct,
        unit="%",
        pass_gate=pass_gate,
    )


def _resolve_candidate(
    candidate: Path | str | MatchedSwingCandidate | Mapping[str, Any],
) -> MatchedSwingCandidate:
    """Resolve candidate input into a valid MatchedSwingCandidate instance."""
    if isinstance(candidate, MatchedSwingCandidate):
        return candidate
    if isinstance(candidate, (str, Path)):
        p = Path(candidate)
        if not p.is_file():
            raise FileNotFoundError(f"Candidate file not found: {p}")
        try:
            return load_candidate(p)
        except (ValueError, KeyError):
            # Fall back to reading raw npz
            from src.shared.python.motion_matching.candidate import (
                CANDIDATE_SCHEMA_VERSION,
                CandidateMarkers,
                CandidateMetadata,
            )

            with np.load(p, allow_pickle=True) as data:
                time_s = np.asarray(data["time_s"], dtype=np.float64)
                q = np.asarray(data["q"], dtype=np.float64)
                v = (
                    np.asarray(data["qd"], dtype=np.float64)
                    if "qd" in data
                    else (
                        np.asarray(data["v"], dtype=np.float64) if "v" in data else None
                    )
                )
                tau = (
                    np.asarray(data["us"], dtype=np.float64)
                    if "us" in data
                    else (
                        np.asarray(data["tau"], dtype=np.float64)
                        if "tau" in data
                        else None
                    )
                )
                markers = (
                    np.asarray(data["predicted_markers_m"], dtype=np.float64)
                    if "predicted_markers_m" in data
                    else None
                )
                if tau is not None and tau.shape[0] == len(time_s) - 1:
                    tau = np.vstack([tau, tau[-1:]])
                coord_names = tuple(f"q_{i}" for i in range(q.shape[1]))
                act_names = (
                    tuple(f"tau_{i}" for i in range(tau.shape[1]))
                    if tau is not None
                    else ()
                )
                meta = CandidateMetadata(
                    schema_version=CANDIDATE_SCHEMA_VERSION,
                    profile=(
                        CandidateProfile.DYNAMIC
                        if tau is not None
                        else CandidateProfile.KINEMATIC
                    ),
                    engine="unknown",
                    coordinate_names=coord_names,
                    actuator_names=act_names,
                )
                return MatchedSwingCandidate(
                    metadata=meta,
                    time_s=time_s,
                    q=q,
                    v=v,
                    tau=tau,
                    markers=CandidateMarkers(model_markers_m=markers),
                )
    raise TypeError(f"Unsupported candidate type: {type(candidate).__name__}")


def _compute_plant_marker_positions(
    plant: MatchingPlant,
    q: np.ndarray,
    attachments: Mapping[str, tuple[str, Sequence[float]]],
) -> np.ndarray:
    """Evaluate forward marker positions handling both batch and frame-by-frame plants."""
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.ndim == 1:
        return np.asarray(plant.marker_positions(q_arr, attachments), dtype=np.float64)

    # Check coordinate dimension against plant
    coord_order = getattr(plant, "coordinate_order", ())
    if coord_order and q_arr.shape[1] != len(coord_order):
        raise ValueError(
            f"Coordinate count mismatch: candidate has {q_arr.shape[1]}, plant '{plant.engine_name}' expects {len(coord_order)}"
        )

    # Try batch evaluation first
    try:
        res = plant.marker_positions(q_arr, attachments)
        res_arr = np.asarray(res, dtype=np.float64)
        if res_arr.ndim == 3 and res_arr.shape[0] == q_arr.shape[0]:
            return res_arr
    except Exception:
        pass

    # Frame-by-frame evaluation
    frames: list[np.ndarray] = []
    for k in range(q_arr.shape[0]):
        fk_pos = plant.marker_positions(q_arr[k], attachments)
        frames.append(np.asarray(fk_pos, dtype=np.float64))
    return np.stack(frames, axis=0)


def _build_engine_list(
    engines: Sequence[str] | None,
    plants: Mapping[str, MatchingPlant] | None,
    reference_engine: str,
) -> list[str]:
    """Assemble ordered list of engines to evaluate."""
    if engines is not None:
        engine_list = list(engines)
    else:
        engine_list = list(ALL_ENGINES)
        if plants is not None:
            for p_name in plants:
                if p_name not in engine_list:
                    engine_list.append(p_name)
    if reference_engine not in engine_list:
        engine_list.insert(0, reference_engine)
    return engine_list


def _resolve_spec_and_attachments(
    spec: bytes | Mapping[str, Any] | None,
    attachments: Mapping[str, tuple[str, Sequence[float]]] | None,
) -> tuple[dict[str, Any], Mapping[str, tuple[str, Sequence[float]]]]:
    """Resolve full-body spec dictionary and marker attachments mapping."""
    resolved_spec: dict[str, Any] = {}
    if spec is None:
        default_spec_path = Path(
            "docs/development/full_body_models/full_body_spec_anthro_driver.json"
        )
        if default_spec_path.is_file():
            try:
                resolved_spec = json.loads(
                    default_spec_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                resolved_spec = {}
    elif isinstance(spec, bytes):
        try:
            resolved_spec = json.loads(spec.decode("utf-8"))
        except (json.JSONDecodeError, OSError):
            resolved_spec = {}
    elif isinstance(spec, dict):
        resolved_spec = dict(spec)

    resolved_attachments: dict[str, tuple[str, Sequence[float]]] = (
        dict(attachments) if attachments is not None else {}
    )
    if not resolved_attachments and "marker_attachments" in resolved_spec:
        for k, v in resolved_spec["marker_attachments"].items():
            if isinstance(v, dict):
                body = str(v.get("body", ""))
                offsets = (
                    tuple(float(x) for x in v["offset_m"])
                    if v.get("offset_m") is not None
                    else (0.0, 0.0, 0.0)
                )
                resolved_attachments[str(k)] = (body, offsets)
            elif isinstance(v, (list, tuple)):
                body = str(v[0])
                offsets = (
                    tuple(float(x) for x in v[1])
                    if len(v) > 1 and v[1] is not None
                    else (0.0, 0.0, 0.0)
                )
                resolved_attachments[str(k)] = (body, offsets)
    return resolved_spec, resolved_attachments


def _evaluate_engine_row(
    eng: str,
    cand: MatchedSwingCandidate,
    plant: MatchingPlant | None,
    reference_engine: str,
    evaluated_markers: Mapping[str, np.ndarray],
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    plant_error: str = "",
) -> tuple[EngineParityRow, np.ndarray | None]:
    """Evaluate one engine for trajectory parity and return row and evaluated markers."""
    comp_class = (
        ComparisonClass.NATIVE_MODEL_OBSERVABLE_AGREEMENT
        if eng in NATIVE_MODEL_ENGINES
        else ComparisonClass.SAME_MODEL_NUMERICAL_PARITY
    )
    if plant is None:
        reason = (
            plant_error
            or f"Engine '{eng}' is not installed or registered in the active environment"
        )
        return (
            EngineParityRow(
                engine=eng,
                status="unavailable",
                comparison_class=comp_class,
                reason=reason,
            ),
            None,
        )

    t0 = time.perf_counter()
    try:
        markers = _compute_plant_marker_positions(plant, cand.q, attachments)
    except Exception as exc:
        return (
            EngineParityRow(
                engine=eng,
                status="unverified",
                comparison_class=comp_class,
                model_name=f"{eng}_model",
                model_sha256=getattr(plant, "plant_sha", ""),
                reason=f"Kinematic evaluation error: {exc}",
            ),
            None,
        )
    wall_clock = time.perf_counter() - t0

    pt_diffs: dict[str, PointwiseDifference] = {}
    pass_all_gates = False
    comparisons_ran = 0
    all_passed = True
    unverified_reason = ""

    if comp_class == ComparisonClass.NATIVE_MODEL_OBSERVABLE_AGREEMENT:
        unverified_reason = "Native-model observable agreement is never gated"
    elif reference_engine not in evaluated_markers:
        unverified_reason = f"Reference markers for '{reference_engine}' unavailable"
    else:
        ref_markers = evaluated_markers[reference_engine]
        if ref_markers.shape != markers.shape:
            unverified_reason = (
                f"Marker shape mismatch with reference '{reference_engine}': "
                f"{markers.shape} != {ref_markers.shape}"
            )
        else:
            m_diff = evaluate_pointwise_trajectory_parity(
                ref_markers, markers, tolerance_m=0.001
            )
            pt_diffs["marker_diff_m"] = m_diff
            comparisons_ran += 1
            if not m_diff.pass_gate:
                all_passed = False

    if comparisons_ran > 0:
        pass_all_gates = all_passed
        status = "qualified" if pass_all_gates else "rejected"
        reason = "" if pass_all_gates else "Diverged beyond tolerance"
    else:
        status = "unverified"
        reason = unverified_reason or "No comparisons ran"

    total_work = (
        float(cand.metadata.extra["total_work_j"])
        if cand.metadata and "total_work_j" in cand.metadata.extra
        else None
    )

    row = EngineParityRow(
        engine=eng,
        status=status,
        comparison_class=comp_class,
        model_name=f"{eng}_model",
        model_sha256=getattr(plant, "plant_sha", ""),
        assumptions={
            "coordinate_count": len(getattr(plant, "coordinate_order", ())),
            "has_contact": hasattr(plant, "contact_forces"),
        },
        pointwise_differences=pt_diffs,
        total_work_J=total_work,
        wall_clock_s=wall_clock,
        reason=reason,
    )
    return row, markers


def _determine_overall_status(rows: Mapping[str, EngineParityRow]) -> str:
    """Aggregate row statuses into overall verdict."""
    if not rows:
        return "PARTIAL"
    statuses = [r.status for r in rows.values()]
    if all(s == "qualified" for s in statuses):
        return "PASSED"
    if any(s == "rejected" for s in statuses):
        return "REJECTED"
    return "PARTIAL"


@postcondition(
    lambda r: isinstance(r, UnifiedParityReport), "Must return a UnifiedParityReport"
)
def build_parity_report(
    candidate: Path | str | MatchedSwingCandidate | Mapping[str, Any],
    *,
    plants: Mapping[str, MatchingPlant] | None = None,
    engines: Sequence[str] | None = None,
    reference_engine: str = "pinocchio",
    attachments: Mapping[str, tuple[str, Sequence[float]]] | None = None,
    spec: bytes | Mapping[str, Any] | None = None,
) -> UnifiedParityReport:
    """Construct unified parity report comparing candidate through all engines."""
    cand = _resolve_candidate(candidate)
    cand_sha = hashlib.sha256(cand.q.tobytes()).hexdigest()[:16]
    engine_list = _build_engine_list(engines, plants, reference_engine)
    resolved_spec, resolved_attachments = _resolve_spec_and_attachments(
        spec, attachments
    )

    active_plants: dict[str, MatchingPlant] = dict(plants) if plants is not None else {}
    plant_avail: dict[str, bool] = {}
    try:
        from src.shared.python.engine_core.engine_availability import (
            is_engine_available,
        )

        plant_avail = {e: is_engine_available(e) for e in ALL_ENGINES}
    except ImportError:
        pass

    rows: dict[str, EngineParityRow] = {}
    evaluated_markers: dict[str, np.ndarray] = {}

    ref_plant = active_plants.get(reference_engine)
    if ref_plant is None and plant_avail.get(reference_engine, False):
        try:
            ref_plant = get_plant(reference_engine, resolved_spec)
            active_plants[reference_engine] = ref_plant
        except (EngineUnavailableError, RuntimeError, ValueError):
            ref_plant = None

    if ref_plant is not None:
        evaluated_markers[reference_engine] = ref_plant.marker_positions(
            cand.q, resolved_attachments
        )

    for eng in engine_list:
        plant = active_plants.get(eng)
        plant_err = ""
        if plant is None and plant_avail.get(eng, False):
            try:
                plant = get_plant(eng, resolved_spec)
                active_plants[eng] = plant
            except (EngineUnavailableError, RuntimeError, ValueError) as exc:
                plant_err = str(exc)

        row, markers = _evaluate_engine_row(
            eng,
            cand,
            plant,
            reference_engine,
            evaluated_markers,
            resolved_attachments,
            plant_err,
        )
        rows[eng] = row
        if markers is not None:
            evaluated_markers[eng] = markers

    overall_status = _determine_overall_status(rows)
    return UnifiedParityReport(
        schema_version=PARITY_REPORT_SCHEMA_VERSION,
        candidate_id=getattr(cand.metadata, "model_name", "candidate") or "candidate",
        candidate_sha256=cand_sha,
        reference_engine=reference_engine,
        reference_model_sha256=getattr(ref_plant, "plant_sha", ""),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        is_parity_accepted=(overall_status == "PASSED"),
        status=overall_status,
        engine_rows=rows,
    )


def run_parity_report_cli(
    args: Sequence[str] | None = None,
    plants: Mapping[str, MatchingPlant] | None = None,
) -> int:
    """CLI runner for generating unified parity reports."""
    parser = argparse.ArgumentParser(
        description="Generate unified cross-engine motion-matching parity report."
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Path to MatchedSwingCandidate .npz file",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory to write parity_report.json and parity_report.md",
    )
    parser.add_argument(
        "--reference-engine",
        default="pinocchio",
        help="Reference engine for pointwise comparison (default: pinocchio)",
    )
    parser.add_argument(
        "--engines",
        nargs="*",
        default=None,
        help="List of engines to evaluate",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Path to full_body_spec JSON file (defaults to anthro_driver spec if available)",
    )

    parsed = parser.parse_args(args)
    spec_data = None
    if parsed.spec:
        import json

        spec_data = json.loads(Path(parsed.spec).read_text(encoding="utf-8"))

    report = build_parity_report(
        candidate=parsed.candidate,
        plants=plants,
        engines=parsed.engines,
        reference_engine=parsed.reference_engine,
        spec=spec_data,
    )

    out_dir = Path(parsed.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "parity_report.json"
    md_path = out_dir / "parity_report.md"

    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(report.render_markdown(), encoding="utf-8")
    logger.info("Emitted parity report to %s and %s", json_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(run_parity_report_cli())
