"""Track C shared sliding-window MuJoCo trajectory fitter (Refs #15837)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from scipy.optimize import least_squares

from . import _validation as val

FitMode = Literal["A_only", "B_only", "fused_AB"]
ClaimLabel = Literal["kinematic_2d_match", "kinematic_3d_plausible", "below_gate"]
SCORECARD_SCHEMA: Final[str] = "markerless-scorecard/1.0.0"
BRUNO_LUMBAR: Final[tuple[float, float, float]] = (0.51, 0.27, 0.17)
BRUNO_THORACIC: Final[tuple[float, float, float]] = (0.49, 0.73, 0.83)
_LATE = frozenset(("downswing", "impact", "follow_through", "finish"))
_PLANTED = frozenset(("address", "backswing", "top"))
_MODES = frozenset(("A_only", "B_only", "fused_AB"))
_DROP = frozenset(
    ("frame_ids", "phases", "fitted_angles", "lumbar_angles", "thoracic_angles")
)
_N_DOF: Final[int] = 12
_LB = np.array([-0.5, -0.8, -1.5, -2.2, 0.0, -2.2, 0.0, -0.5, -0.5, -0.5, -0.5, 0.0])
_UB = np.array([1.2, 0.8, 1.5, 2.2, 2.5, 2.2, 2.5, 0.5, 0.5, 0.5, 0.5, 0.25])


def _sha256(payload: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hex digest over canonical JSON."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tup(arr: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """Convert a 2D float array to nested tuples."""
    return tuple(tuple(float(x) for x in row) for row in arr)


def _pin_hash(obj: object, attr: str, payload: dict[str, Any]) -> None:
    """Validate or assign deterministic SHA-256 digest on a frozen dataclass."""
    exp, cur = _sha256(payload), getattr(obj, attr)
    if cur and val.check_sha256(cur, attr) != exp:
        raise ValueError(f"{attr} mismatch")
    if not cur:
        object.__setattr__(obj, attr, exp)


@dataclass(frozen=True, slots=True)
class SlidingWindowFitterConfig:
    """Immutable solver configuration with ADR-0059 coupled trunk default."""

    trunk_coupled: bool = True
    model_id: str = "mujoco-tiger-woods-2000"
    window_size: int = 5
    stride: int = 2
    smoothness_weight: float = 0.05
    grip_tol_m: float = 0.015
    foot_slip_tol_m: float = 0.02
    max_trail_heel_lift_m: float = 0.12
    reprojection_gate: float = 0.08
    max_nfev: int = 40
    lumbar_split_ratios: tuple[float, float, float] = BRUNO_LUMBAR
    thoracic_split_ratios: tuple[float, float, float] = BRUNO_THORACIC
    config_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate solver parameters and pin deterministic config SHA-256."""
        val.check_bool(self.trunk_coupled, "trunk_coupled")
        val.check_str(self.model_id, "model_id")
        for fn in ("window_size", "stride", "max_nfev"):
            val.check_pos_int(getattr(self, fn), fn)
        for fn in ("grip_tol_m", "foot_slip_tol_m", "reprojection_gate"):
            val.check_pos_float(getattr(self, fn), fn)
        val.check_nonneg_float(self.smoothness_weight, "smoothness_weight")
        val.check_nonneg_float(self.max_trail_heel_lift_m, "max_trail_heel_lift_m")
        for r in (self.lumbar_split_ratios, self.thoracic_split_ratios):
            if len(r) != 3 or any(val.check_nonneg_float(x, "ratio") < 0 for x in r):
                raise ValueError("split ratios must have 3 non-negative elements")
        d = asdict(self)
        d.pop("config_sha256", None)
        _pin_hash(self, "config_sha256", d)


@dataclass(frozen=True, slots=True)
class FitterObservationInput:
    """Observation bundle for A_only, B_only, or fused_AB fitting modes."""

    mode: FitMode
    frame_ids: tuple[str, ...]
    phases: tuple[str, ...]
    target_joints: tuple[tuple[float, ...], ...]
    observation_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate observation dimensions and swing phases."""
        if self.mode not in _MODES:
            raise ValueError(f"Unsupported mode {self.mode!r}")
        n = len(self.frame_ids)
        if n == 0 or len(self.phases) != n or len(self.target_joints) != n:
            raise ValueError("frame_ids, phases, target_joints must align")
        for fid, ph, row in zip(
            self.frame_ids, self.phases, self.target_joints, strict=True
        ):
            val.check_str(fid, "frame_id")
            if ph not in _PLANTED and ph not in _LATE:
                raise ValueError(f"Unsupported phase {ph!r}")
            if len(row) != _N_DOF or not all(math.isfinite(float(v)) for v in row):
                raise ValueError(f"target_joints row must have {_N_DOF} finite values")
        w = self.observation_weights or tuple(1.0 for _ in range(n))
        if len(w) != n:
            raise ValueError("observation_weights length mismatch")
        wt = tuple(val.check_nonneg_float(x, "w") for x in w)
        object.__setattr__(self, "observation_weights", wt)


@dataclass(frozen=True, slots=True)
class TigerFitResult:
    """Qualified trajectory fit output and scorecard record."""

    mode: FitMode
    model_id: str
    config_sha256: str
    trunk_coupled: bool
    frame_ids: tuple[str, ...]
    phases: tuple[str, ...]
    fitted_angles: tuple[tuple[float, ...], ...]
    lumbar_angles: tuple[tuple[float, ...], ...]
    thoracic_angles: tuple[tuple[float, ...], ...]
    max_grip_separation_m: float
    grip_gate_passed: bool
    max_planted_foot_slip_m: float
    max_trail_heel_lift_m: float
    foot_contact_gate_passed: bool
    mean_2d_reprojection_rmse: float
    claim_label: ClaimLabel
    timebase: str = "unverified"
    physical_time_verified: bool = False
    qualification: str = "image_observations_only"
    scorecard_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate fitted result invariants and pinned scorecard digest."""
        val.check_sha256(self.config_sha256, "config_sha256")
        val.check_nonneg_float(self.max_grip_separation_m, "max_grip_separation_m")
        val.check_nonneg_float(self.max_planted_foot_slip_m, "max_planted_foot_slip_m")
        val.check_nonneg_float(self.max_trail_heel_lift_m, "max_trail_heel_lift_m")
        val.check_nonneg_float(
            self.mean_2d_reprojection_rmse, "mean_2d_reprojection_rmse"
        )
        _pin_hash(self, "scorecard_sha256", self.to_scorecard_dict(include_hash=False))

    def to_scorecard_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        """Return schema-compatible markerless-scorecard/1.0.0 dictionary."""
        out = {k: v for k, v in asdict(self).items() if k not in _DROP}
        out["schema_version"] = SCORECARD_SCHEMA
        out["source_schema_version"] = "shadow-tracker/source/1.0.0"
        out["capture_schema_version"] = "historical-capture/1.0.0"
        if not include_hash:
            out.pop("scorecard_sha256", None)
        return out


def _rank_key(r: TigerFitResult) -> tuple[bool, float]:
    """Order fit results with 3D-plausible fits ahead of 2D/below-gate fits."""
    return (r.claim_label != "kinematic_3d_plausible", r.mean_2d_reprojection_rmse)


@dataclass(frozen=True, slots=True)
class AblationComparisonReport:
    """Comparable A_only, B_only, and fused_AB fit outcomes sharing one config."""

    model_id: str
    config_sha256: str
    a_only: TigerFitResult
    b_only: TigerFitResult
    fused_ab: TigerFitResult
    best_mode: FitMode = "fused_AB"
    scorecard_sha256: str = ""

    def __post_init__(self) -> None:
        """Enforce identical config_sha256 and compute best mode and digest."""
        rs = (self.a_only, self.b_only, self.fused_ab)
        if {r.config_sha256 for r in rs} != {self.config_sha256}:
            raise ValueError("All ablation modes must share identical config_sha256")
        object.__setattr__(self, "best_mode", min(rs, key=_rank_key).mode)
        dig = {"c": self.config_sha256, "h": [r.scorecard_sha256 for r in rs]}
        _pin_hash(self, "scorecard_sha256", dig)


@dataclass(frozen=True, slots=True)
class SideBySideOverlayManifest:
    """Synchronized frame manifest aligning source observations and fitted renders."""

    schema_version: str
    model_id: str
    config_sha256: str
    frame_entries: tuple[dict[str, Any], ...]
    timebase: str = "unverified"
    physical_time_verified: bool = False
    qualification: str = "image_observations_only"
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate overlay manifest provenance and SHA-256 digest."""
        val.check_sha256(self.config_sha256, "config_sha256")
        val.check_bool(self.physical_time_verified, "physical_time_verified")
        p = {"c": self.config_sha256, "f": list(self.frame_entries)}
        _pin_hash(self, "manifest_sha256", p)


def _hand_grip(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D model-space lead_hand_grip and trail_hand_grip coordinates."""
    z0 = np.full(q.shape[0], 1.35)
    th = np.stack([0.05 * np.sin(q[:, 1]), 0.05 * np.sin(q[:, 0]), z0], axis=1)
    res = []
    for s, e in ((3, 4), (5, 6)):
        a, b = q[:, s], q[:, e]
        x = 0.58 * np.sin(a) + 0.28 * np.sin(a + 0.2 * b)
        y = 0.58 * np.cos(a) + 0.1 * q[:, 2]
        res.append(th + np.stack([x, y, -0.25 * np.cos(b)], axis=1))
    return res[0], res[1]


def evaluate_trajectory_gates(
    angles: np.ndarray | Sequence[Sequence[float]],
    phases: Sequence[str],
    cfg: SlidingWindowFitterConfig,
) -> dict[str, Any]:
    """Evaluate rigid-club grip closure and phase-specific foot contact gates."""
    q = np.asarray(angles, dtype=np.float64)
    lead, trail = _hand_grip(q)
    grip = float(np.max(np.linalg.norm(lead - trail, axis=1)))
    s1 = np.max(np.linalg.norm(q[:, 7:9] - q[0:1, 7:9], axis=1))
    s2 = np.max(np.linalg.norm(q[:, 9:11] - q[0:1, 9:11], axis=1))
    slip, hz = float(max(s1, s2)), q[:, 11]
    bad = any(
        (p in _PLANTED and abs(float(z)) > cfg.foot_slip_tol_m)
        or (p in _LATE and not 0 <= float(z) <= cfg.max_trail_heel_lift_m)
        for z, p in zip(hz, phases, strict=True)
    )
    return {
        "max_grip_separation_m": grip,
        "grip_gate_passed": grip <= cfg.grip_tol_m,
        "max_planted_foot_slip_m": slip,
        "max_trail_heel_lift_m": float(np.max(hz)),
        "foot_contact_gate_passed": (slip <= cfg.foot_slip_tol_m) and not bad,
    }


def _residuals(
    x: np.ndarray,
    tgt: np.ndarray,
    wts: np.ndarray,
    ph: Sequence[str],
    anc: np.ndarray,
    cfg: SlidingWindowFitterConfig,
) -> np.ndarray:
    """Compute sliding-window tracking, smoothness, grip, and foot residuals."""
    n = tgt.shape[0]
    q = x.reshape(n, _N_DOF)
    track = ((q - tgt) * wts[:, None]).ravel()
    ws = cfg.smoothness_weight
    sm = math.sqrt(ws) * (q[1:] - q[:-1]).ravel() if n > 1 and ws > 0 else np.zeros(0)
    lead, trail = _hand_grip(q)
    grip = ((lead - trail) / max(cfg.grip_tol_m, 1e-4)).ravel()
    sc, hi = max(cfg.foot_slip_tol_m, 1e-4), cfg.max_trail_heel_lift_m
    sl, st = (
        ((q[:, 7:9] - anc[0:1]) / sc).ravel(),
        ((q[:, 9:11] - anc[1:2]) / sc).ravel(),
    )
    hv = [
        (float(z) if p in _PLANTED else max(0.0, float(z) - hi)) / sc
        for z, p in zip(q[:, 11], ph, strict=True)
    ]
    return np.concatenate([track, sm, grip, sl, st, np.array(hv)])


def _solve(
    tgt: np.ndarray, wts: np.ndarray, ph: Sequence[str], cfg: SlidingWindowFitterConfig
) -> np.ndarray:
    """Solve overlapping temporal windows via bounded SciPy least_squares."""
    n = tgt.shape[0]
    win = min(cfg.window_size, n)
    starts = list(range(0, max(1, n - win + 1), cfg.stride))
    if starts[-1] + win < n:
        starts.append(n - win)
    acc, cnt = np.zeros_like(tgt), np.zeros((n, 1), dtype=np.float64)
    anc = np.stack([tgt[0, 7:9], tgt[0, 9:11]], axis=0)
    bnds, mx = (np.tile(_LB, win), np.tile(_UB, win)), cfg.max_nfev
    for s in starts:
        e = s + win
        x0 = np.clip(tgt[s:e], _LB, _UB).ravel()
        args = (tgt[s:e], wts[s:e], ph[s:e], anc, cfg)
        sol = least_squares(_residuals, x0, bounds=bnds, max_nfev=mx, args=args)
        acc[s:e] += sol.x.reshape(win, _N_DOF)
        cnt[s:e] += 1.0
    return acc / cnt


def fit_tiger_trajectory(
    obs: FitterObservationInput, config: SlidingWindowFitterConfig | None = None
) -> TigerFitResult:
    """Run sliding-window kinematic fit and return hash-pinned TigerFitResult."""
    cfg = config or SlidingWindowFitterConfig()
    tgt = np.asarray(obs.target_joints, dtype=np.float64)
    wts = np.asarray(obs.observation_weights, dtype=np.float64)
    q_fit = _solve(tgt, wts, obs.phases, cfg)
    lr = np.asarray(cfg.lumbar_split_ratios if cfg.trunk_coupled else (0.5, 0.5, 0.5))
    tr = np.asarray(cfg.thoracic_split_ratios if cfg.trunk_coupled else (0.5, 0.5, 0.5))
    g = evaluate_trajectory_gates(q_fit, obs.phases, cfg)
    rmse = float(np.sqrt(np.mean((q_fit - tgt) ** 2)))
    ok = bool(g["grip_gate_passed"]) and bool(g["foot_contact_gate_passed"])
    c_ok: ClaimLabel = "kinematic_3d_plausible" if ok else "kinematic_2d_match"
    claim: ClaimLabel = "below_gate" if rmse > cfg.reprojection_gate else c_ok
    return TigerFitResult(
        obs.mode,
        cfg.model_id,
        cfg.config_sha256,
        cfg.trunk_coupled,
        obs.frame_ids,
        obs.phases,
        _tup(q_fit),
        _tup(q_fit[:, :3] * lr),
        _tup(q_fit[:, :3] * tr),
        float(g["max_grip_separation_m"]),
        bool(g["grip_gate_passed"]),
        float(g["max_planted_foot_slip_m"]),
        float(g["max_trail_heel_lift_m"]),
        bool(g["foot_contact_gate_passed"]),
        rmse,
        claim,
    )


def run_ablation_comparison(
    obs_a: FitterObservationInput,
    obs_b: FitterObservationInput,
    config: SlidingWindowFitterConfig | None = None,
) -> AblationComparisonReport:
    """Compare A_only, B_only, and fused_AB fits under identical config_sha256."""
    if obs_a.mode != "A_only" or obs_b.mode != "B_only":
        raise ValueError("obs_a must be 'A_only' and obs_b must be 'B_only'")
    fids, phs = obs_a.frame_ids, obs_a.phases
    if fids != obs_b.frame_ids or phs != obs_b.phases:
        raise ValueError("A_only and B_only inputs must share frame_ids and phases")
    cfg = config or SlidingWindowFitterConfig()
    wa = np.asarray(obs_a.observation_weights)[:, None]
    wb = np.asarray(obs_b.observation_weights)[:, None]
    qa, qb = np.asarray(obs_a.target_joints), np.asarray(obs_b.target_joints)
    fq = (qa * wa + qb * wb) / np.maximum(wa + wb, 1e-8)
    fw = tuple(float(x) for x in (0.5 * (wa[:, 0] + wb[:, 0])))
    obs_f = FitterObservationInput("fused_AB", fids, phs, _tup(fq), fw)
    ra, rb, rf = (fit_tiger_trajectory(o, cfg) for o in (obs_a, obs_b, obs_f))
    return AblationComparisonReport(cfg.model_id, cfg.config_sha256, ra, rb, rf)


def build_side_by_side_overlay_manifest(
    report: AblationComparisonReport, source_refs: Sequence[str] | None = None
) -> SideBySideOverlayManifest:
    """Build synchronized source-vs-model side-by-side overlay manifest."""
    f, mid, ch = report.fused_ab, report.model_id, report.config_sha256
    refs = tuple(source_refs or (f"source://{fid}" for fid in f.frame_ids))
    if len(refs) != len(f.frame_ids):
        raise ValueError("source_refs length must match frame count")
    cl = {
        "A_only": report.a_only.claim_label,
        "B_only": report.b_only.claim_label,
        "fused_AB": f.claim_label,
    }
    ent = tuple(
        {
            "i": i,
            "id": fid,
            "phase": f.phases[i],
            "src": refs[i],
            "q": list(f.fitted_angles[i]),
            "claims": cl,
        }
        for i, fid in enumerate(f.frame_ids)
    )
    return SideBySideOverlayManifest(SCORECARD_SCHEMA, mid, ch, ent)


def export_scorecard_json(result: TigerFitResult, destination: Path) -> dict[str, Any]:
    """Write hash-pinned JSON scorecard to destination and return payload."""
    payload = result.to_scorecard_dict()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    return payload
