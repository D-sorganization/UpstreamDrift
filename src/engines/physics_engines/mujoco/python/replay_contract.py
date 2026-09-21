"""Fail-closed contracts for replaying saved Pinocchio controls (#10336)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import precondition
from src.shared.python.motion_matching.acceptance import AcceptanceGates, evaluate
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES
from src.shared.python.motion_matching.tour_capture_contract import MARKER_SEGMENTS


@dataclass(frozen=True)
class ReplaySettings:
    """Explicit diagnostic settings; never substitutes for source provenance."""

    armature_kg_m2: float = 0.005
    rtol: float = 1e-7
    atol: float = 1e-9
    max_step_s: float = 1 / 1440
    max_evaluations: int = 20000

    def __post_init__(self) -> None:
        values = (self.armature_kg_m2, self.rtol, self.atol, self.max_step_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Replay settings must be finite")
        if self.armature_kg_m2 < 0 or min(values[1:]) <= 0:
            raise ValueError("Armature must be nonnegative; tolerances positive")
        if self.max_evaluations < 1:
            raise ValueError("Positive evaluation budget required")


@precondition(lambda end_s: math.isfinite(end_s), "Finite end time required")
def window_indices(times: np.ndarray, end_s: float) -> np.ndarray:
    """Select original frames through an exact endpoint without interpolation."""
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Time must be finite and strictly increasing")
    matches = np.flatnonzero(np.isclose(times, end_s, atol=1e-10, rtol=0))
    if times.size < 2 or times[0] != 0 or len(matches) != 1:
        raise ValueError("Horizon must start at zero and end at a frame boundary")
    return np.arange(matches[0] + 1)


@precondition(lambda path: path.is_file(), "Candidate file must exist")
def load_candidate(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load non-pickled numeric data, checking identity, names, shapes and finiteness."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Candidate SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    names = tuple(str(x) for x in arrays["coordinate_order"])
    if len(set(names)) != len(names) or not set(ROOT_COORDINATES) <= set(names):
        raise ValueError("Coordinate names must be unique and include all root names")
    times = arrays["time_s"]
    window_indices(times, float(times[-1]))
    shape = (len(times), len(names))
    actuated = np.array([name not in ROOT_COORDINATES for name in names])
    if arrays["actuated"].dtype != bool or not np.array_equal(
        arrays["actuated"], actuated
    ):
        raise ValueError("Actuation mask must exclude exactly the six root coordinates")
    for key, expected in (
        ("q", shape),
        ("v", shape),
    ):
        if arrays[key].shape != expected or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} must be finite with shape {expected}")
    n_act = int(actuated.sum())
    u_shape = arrays["u"].shape
    if (
        u_shape not in ((len(times), n_act), (len(times) - 1, n_act))
        or not np.isfinite(arrays["u"]).all()
    ):
        raise ValueError(
            f"u must be finite with shape (len(times), {n_act}) or (len(times) - 1, {n_act})"
        )
    labels = tuple(str(x) for x in arrays["labels"])
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Marker labels must be unique and nonempty")
    points_shape = (len(times), len(labels), 3)
    valid = arrays["valid"]
    if valid.dtype != bool or valid.shape != points_shape[:2]:
        raise ValueError("Invalid marker validity shape/type")
    for key in ("markers_m", "target_m"):
        if (
            arrays[key].shape != points_shape
            or not np.isfinite(arrays[key][valid]).all()
        ):
            raise ValueError(f"Invalid {key} marker samples")
    return arrays


@precondition(lambda count: count > 0, "Positive frame count required")
def metric_coverage(arrays: dict, count: int) -> bool:
    """Reject empty metric populations rather than accepting shared zero fallbacks."""
    labels = tuple(arrays["labels"])
    club = [i for i, name in enumerate(labels) if name in MARKER_SEGMENTS["club"]]
    if not club or not {"WaistLeft", "WaistRight"} <= set(labels):
        return False
    valid = arrays["valid"][:count]
    left, right = labels.index("WaistLeft"), labels.index("WaistRight")
    times = arrays["time_s"][:count]
    early = valid[times <= 0.6]
    terminal = valid[int(count * 0.9) :]
    return bool(
        valid.any(axis=1).all()
        and early.any()
        and terminal.any()
        and valid[:, club].any()
        and (valid[:, left] & valid[:, right]).any()
    )


@precondition(lambda receipt: isinstance(receipt, dict), "Source receipt required")
def configuration_failures(
    receipt: dict, document: dict, settings: ReplaySettings
) -> list[str]:
    """Return missing/mismatched source settings; no inferred settings pass parity."""
    failures = []
    armature = receipt.get("armature_kg_m2")
    if not isinstance(armature, (float, int)) or armature != settings.armature_kg_m2:
        failures.append("source armature missing or different")
    if receipt.get("contact") != document["contact"]:
        failures.append("source contact configuration missing or different")
    digest = receipt.get("candidate_sha256", "")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        failures.append("source candidate_sha256 missing or invalid")
    if receipt.get("control_interpolation") != "zero_order_hold":
        failures.append("source control interpolation missing or different")
    return failures


@precondition(lambda evidence: isinstance(evidence, dict), "Replay evidence required")
def g1_acceptance(metrics: dict, evidence: dict[str, bool]) -> dict:
    """Reuse G1 thresholds, requiring every measurement and replay prerequisite."""
    finite_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }
    result = evaluate(finite_metrics).as_dict()
    limits = AcceptanceGates()
    required = {
        "whole_marker_rmse_m": limits.g1_whole_rmse_m,
        "early_marker_rmse_m": limits.g1_early_rmse_m,
        "terminal_marker_rmse_m": limits.g1_terminal_rmse_m,
        "club_marker_rmse_m": limits.g1_club_rmse_m,
        "pelvis_yaw_rmse_rad": limits.g1_pelvis_yaw_rmse_rad,
        "max_normal_force_n": limits.nominal_body_mass_kg
        * limits.gravity_m_s2
        * limits.max_normal_force_bw_multiplier,
        "max_penetration_m": limits.max_penetration_m,
        "max_closure_residual_m": limits.max_closure_residual_m,
        "max_closure_residual_rad": limits.max_closure_residual_rad,
    }
    # The older shared evaluator treats some missing measurements as optional.
    # Replay qualification must require finite evidence for every G1 gate.
    for name, threshold in required.items():
        value = metrics.get(name)
        finite = isinstance(value, (int, float)) and math.isfinite(value)
        result["gates"].append(
            {
                "name": f"required_{name}",
                "threshold": threshold,
                "measured": value if finite else None,
                "status": (
                    "passed"
                    if isinstance(value, (int, float))
                    and finite
                    and 0 <= value <= threshold
                    else "failed"
                ),
                "reason": "Required finite G1 measurement",
                "unit": "",
            }
        )
    for name in (
        "parity",
        "complete",
        "converged",
        "root_history",
        "coverage",
        "source_dynamics",
    ):
        passed = evidence.get(name) is True
        result["gates"].append(
            {
                "name": name,
                "status": "passed" if passed else "missing",
                "threshold": 1.0,
                "measured": float(passed),
                "unit": "bool",
                "reason": "Required replay evidence",
            }
        )
    accepted = all(g["status"] == "passed" for g in result["gates"])
    result.update(
        is_physically_accepted=accepted,
        status="PASSED" if accepted else "REJECTED",
        qualification_note="Independent uninterrupted replay; missing evidence fails closed",
    )
    return result
