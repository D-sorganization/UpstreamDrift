"""Native canonical-v2 trial execution and threshold contract (CO-10 T2-WP1 #15787)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.contracts import precondition
from src.shared.python.core.vector_math import row_euclidean_norm
from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.model_appearance import canonical_golfer_gate as gg
from src.shared.python.motion_matching.club_only import (
    acceptance as acc,
    ambiguity as amb,
    native_g1_gates as g1,
    observation as obs_m,
    profiles,
    workbook_identity as wi,
)
from src.shared.python.motion_matching.club_target import QUAT_NORM_TOL

CANONICAL_V2_TRIAL_SCHEMA = "club-only-canonical-v2-trial/1.0.0"
THRESHOLDS_SCHEMA = "club-only-canonical-v2-thresholds/1.0.0"
DEFAULT_TRIAL_ID = "TW_ProV1"
DEFAULT_TARGET_MODEL_ID = "full_body_mujoco"
DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[5]
    / "src/config/club_only_canonical_v2_thresholds.json"
)
EXPECTED_THRESHOLDS_SHA256 = (
    "10de7abf15a118a370ab67a1e845b32cb5a395bf5db545ab04789c5655eb8cc9"
)

_KIN_STEMS = (
    ("grip_position", "m"),
    ("grip_orientation", "rad"),
    ("clubhead_position", "m"),
    ("clubhead_orientation", "rad"),
    ("grip_closure", "m"),
)
_DYN = (
    "root_force_bw_fraction",
    "root_moment_bw_height_fraction",
    "grf_residual_bw_fraction",
    "penetration_m",
    "replay_drift_m",
    "replay_joint_rms_rad",
)
_LIMITS = {
    **{f"{p}_{s}_{u}": f"{p}_{s}_{u}" for p, u in _KIN_STEMS for s in ("rmse", "max")},
    **{k: f"max_{k}" for k in _DYN},
    "speed_error_m_s": "max_speed_error_m_s",
    "phase_error_s": "max_phase_error_s",
}
_REQ_KEYS = frozenset(_LIMITS.values()) | {"min_native_coverage_fraction"}


def _num(name: str, val: object, *, pos: bool) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be finite number, got {val!r}")
    x = float(val)
    if not math.isfinite(x) or (x <= 0.0 if pos else x < 0.0):
        raise ValueError(f"{name} must be finite and {'> 0' if pos else '>= 0'}")
    return x


@dataclass(frozen=True)
class CanonicalV2Thresholds:
    """Versioned, SHA-256-pinned canonical-v2 trial tolerances (#15787)."""

    schema_version: str
    content_sha256: str
    governing_issue: int
    parent_epic: int
    panel_id: str
    tolerances: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.schema_version != THRESHOLDS_SCHEMA or len(self.content_sha256) != 64:
            raise ValueError("invalid threshold schema_version or content_sha256")
        if _REQ_KEYS - set(self.tolerances):
            raise ValueError("missing threshold keys")
        clean = {k: _num(k, v, pos=True) for k, v in self.tolerances.items()}
        if clean["min_native_coverage_fraction"] > 1.0:
            raise ValueError("min_native_coverage_fraction must be <= 1.0")
        for p, u in _KIN_STEMS:
            if clean[f"{p}_rmse_{u}"] > clean[f"{p}_max_{u}"]:
                raise ValueError(f"{p}_rmse_{u} must be <= {p}_max_{u}")
        object.__setattr__(self, "tolerances", MappingProxyType(clean))


@dataclass(frozen=True)
class CanonicalV2TrialResult:
    """Trial verdict preserving full per-frame residuals even on failure."""

    schema: str
    trial_id: str
    target_model_id: str
    golfer_lock: gg.GolferModelLock
    thresholds: CanonicalV2Thresholds
    qualified: bool
    native_g1_pass: bool
    claims_native_qualification: bool
    blockers: tuple[str, ...]
    position_error_m: NDArray[np.float64]
    orientation_error_rad: NDArray[np.float64]
    summary_stats: Mapping[str, float]
    acceptance_verdict: acc.ClubOnlyAcceptanceVerdict

    def __post_init__(self) -> None:
        if (
            self.schema != CANONICAL_V2_TRIAL_SCHEMA
            or self.qualified != self.native_g1_pass
        ):
            raise ValueError("invalid trial result contract")
        g1.validate_native_g1_claim_contract(
            native_g1_pass=self.native_g1_pass,
            claims_native_qualification=self.claims_native_qualification,
            qualification_blockers=self.blockers,
        )
        object.__setattr__(self, "blockers", tuple(self.blockers))


def load_canonical_v2_thresholds(
    path: Path | str | None = None, *, expected_sha256: str = EXPECTED_THRESHOLDS_SHA256
) -> CanonicalV2Thresholds:
    """Load and SHA-256-verify the versioned canonical-v2 threshold fixture."""
    target = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    if not target.is_file() or len(expected_sha256) != 64:
        raise FileNotFoundError(f"missing fixture or bad sha256: {target}")
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError("thresholds SHA-256 mismatch")
    raw = json.loads(target.read_text(encoding="utf-8"))
    prov = raw["provenance"]
    secs = ("kinematic_tolerances", "closure_tolerances", "dynamic_and_replay_budgets")
    tols = {k: _num(k, v, pos=True) for sec in secs for k, v in raw[sec].items()}
    return CanonicalV2Thresholds(
        str(raw["schema_version"]),
        actual,
        int(prov["governing_issue"]),
        int(prov["parent_epic"]),
        str(prov["panel_id"]),
        tols,
    )


def _arr(
    val: ArrayLike, shape: tuple[int, ...], *, unit_quat: bool = False
) -> NDArray[np.float64]:
    a = np.asarray(val, dtype=np.float64)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"array must be finite with shape {shape}, got {a.shape}")
    if unit_quat and np.any(np.abs(row_euclidean_norm(a) - 1.0) > QUAT_NORM_TOL):
        raise ValueError("quaternion array must be unit norm")
    return a


@precondition(lambda spec, **_: isinstance(spec, Mapping), "spec must be a mapping")
def run_canonical_v2_trial(
    *,
    spec: Mapping[str, Any],
    observation: obs_m.ClubObservation,
    prediction: Mapping[str, ArrayLike],
    controller: Mapping[str, Any],
    trial_id: str = DEFAULT_TRIAL_ID,
    target_model_id: str = DEFAULT_TARGET_MODEL_ID,
    thresholds: CanonicalV2Thresholds | None = None,
    dynamic_metrics: Mapping[str, Any] | None = None,
    extra_blockers: Sequence[str] = (),
) -> CanonicalV2TrialResult:
    """Evaluate a canonical-v2 trial while enforcing #15659 and preserving residuals."""
    gg.require_canonical_golfer(spec)
    ch = gg.controller_hash(
        controller["kp"], controller["kd"], controller["knots"], controller["control"]
    )
    lock = gg.lock_for_spec(
        spec, controller=ch, solver=gg.solver_hash(controller["integrator"])
    )
    if not trial_id or observation.trial_id != trial_id or not target_model_id.strip():
        raise ValueError("invalid trial_id or target_model_id")
    aliases = {"tiger_woods", "full_body_mujoco_tiger_woods"}
    prof = profiles.get_club_only_profile(
        DEFAULT_TARGET_MODEL_ID if target_model_id in aliases else target_model_id
    )
    if not prof.preserves_full_body_g3:
        raise ValueError(f"full-body model required, got {target_model_id!r}")
    d, o, p = dynamic_metrics or {}, observation, prediction
    cov = _num("coverage", d.get("native_coverage_fraction", 1.0), pos=False)
    th = thresholds or load_canonical_v2_thresholds()
    n = int(o.native_time_s.shape[0])
    gp = row_euclidean_norm(_arr(p["grip_xyz_m"], (n, 3)) - o.mid_hands_xyz)
    cp = row_euclidean_norm(_arr(p["clubhead_xyz_m"], (n, 3)) - o.face_xyz)
    gr = obs_m.orientation_residual_so3(
        _arr(p["grip_quat_wxyz"], (n, 4), unit_quat=True), o.mid_hands_quat
    )
    cr = obs_m.orientation_residual_so3(
        _arr(p["clubhead_quat_wxyz"], (n, 4), unit_quat=True), o.face_quat
    )
    rc = p.get("grip_closure_residual_m")
    clos = np.zeros(n, dtype=np.float64) if rc is None else _arr(rc, (n,))
    if np.any(clos < 0.0):
        raise ValueError("grip_closure_residual_m must be >= 0")
    pe, re = np.maximum(gp, cp), np.maximum(gr, cr)

    s: dict[str, float] = {
        "native_coverage_fraction": cov,
        **{k: _num(k, d.get(k, 0.0), pos=False) for k in _DYN},
    }
    for opt in ("speed_error_m_s", "phase_error_s"):
        if opt in d:
            s[opt] = _num(opt, d[opt], pos=False)
    for k, a in {
        "grip_position_m": gp,
        "grip_orientation_rad": gr,
        "clubhead_position_m": cp,
        "clubhead_orientation_rad": cr,
        "grip_closure_m": clos,
        "position_error_m": pe,
        "orientation_error_rad": re,
    }.items():
        stem, unit = k.rsplit("_", 1)
        s[f"{stem}_rmse_{unit}"] = float(np.sqrt(np.mean(np.square(a))))
        s[f"{stem}_max_{unit}"] = float(np.max(a))

    g_r, c_r = s["grip_position_rmse_m"], s["clubhead_position_rmse_m"]
    rep = acc.ClubOnlyResidualReport(
        g_r,
        c_r,
        s["grip_orientation_rmse_rad"],
        s["clubhead_orientation_rmse_rad"],
        cov,
        s.get("speed_error_m_s"),
        s.get("phase_error_s"),
        {"closure_residual_m": s["grip_closure_max_m"]},
    )
    cand = amb.CandidateScore(
        f"{target_model_id}:{trial_id}",
        max(g_r, c_r),
        1.0,
        lock.plant_hash,
        s["grip_closure_max_m"],
        bool(d.get("contact_feasible", True)),
        False,
    )
    verd = acc.evaluate_club_only_acceptance(
        rep,
        (cand,),
        prof,
        body_markers_present=bool(d.get("body_markers_present", True)),
        force_labels_synthetic=True,
        torque_replay_validated=bool(d.get("torque_replay_validated", True)),
        kinematic_preview_ok=bool(d.get("kinematic_preview_ok", True)),
    )

    tols = th.tolerances
    blockers = [f"acceptance_gate_failed:{g.name}" for g in verd.gates if not g.passed]
    blockers.extend(
        f"threshold_exceeded:{m}" for m, t in _LIMITS.items() if s.get(m, 0.0) > tols[t]
    )
    if cov < tols["min_native_coverage_fraction"]:
        blockers.append("threshold_below:native_coverage_fraction")
    if verd.statuses.scientific != "qualified":
        blockers.append(f"scientific_status:{verd.statuses.scientific}")
    blockers = list(dict.fromkeys(blockers + [b for b in extra_blockers if b]))
    ok = not blockers
    g1.validate_native_g1_claim_contract(
        native_g1_pass=ok,
        claims_native_qualification=ok,
        qualification_blockers=blockers,
    )
    return CanonicalV2TrialResult(
        CANONICAL_V2_TRIAL_SCHEMA,
        trial_id,
        target_model_id,
        lock,
        th,
        ok,
        ok,
        ok,
        tuple(blockers),
        pe,
        re,
        s,
        verd,
    )


def load_tw_prov1_via_capture_dir(
    repo_root: Path | str | None = None, *, data_dir: Path | None = None
) -> Path:
    """Resolve the private TW_ProV1 workbook via CAPTURE_DATA_DIR, failing closed."""
    root = Path(repo_root) if repo_root is not None else get_repo_root()
    resolver = getattr(wi, "resolve_club_workbooks", wi._resolve_club_workbook_paths)
    club_path, _ = resolver(root, data_dir)
    wi.verify_workbook_hash(club_path, wi.CLUB_DATA_SHA256)
    return club_path
