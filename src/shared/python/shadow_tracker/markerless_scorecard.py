"""Frozen Tiger 2000 markerless scorecard, phase gates, and null baseline (#15837)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Literal

from . import _validation as _v

SCORECARD_GATES_SCHEMA_VERSION: Final[str] = "markerless-scorecard-gates/1.0.0"
SCORECARD_SCHEMA_VERSION: Final[str] = "markerless-scorecard/1.0.0"
NULL_TEMPLATE_SCHEMA_VERSION: Final[str] = "markerless-null-template/1.0.0"
_CFG: Final[Path] = Path(__file__).resolve().parents[3] / "config"
DEFAULT_GATES_PATH: Final[Path] = _CFG / "tiger_2000_scorecard_gates.json"
ScorecardStatus = Literal["passed", "below_gate"]
MatchKind = Literal["kinematic_2d_match", "kinematic_3d_plausible", "below_gate"]
_KP = dict[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class ScorecardGates:
    """Versioned frozen gates for Tiger 2000 markerless evaluation."""

    schema_version: str
    required_phases: tuple[str, ...]
    annotator_noise_floors: dict[str, float]
    noise_floor_multiplier: float
    min_phase_coverage: float
    null_baseline_margins: dict[str, float]
    pinned_reference: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate schema version, phase list, noise floors, and pinned reference."""
        _v.check_schema_version(self.schema_version, SCORECARD_GATES_SCHEMA_VERSION)
        phases = tuple(_v.check_str(p, "phase") for p in self.required_phases)
        if not phases or len(set(phases)) != len(phases):
            raise ValueError("required_phases must be non-empty and unique")
        object.__setattr__(self, "required_phases", phases)
        _v.check_pos_float(self.noise_floor_multiplier, "noise_floor_multiplier")
        if _v.check_pos_float(self.min_phase_coverage, "min_phase_coverage") > 1.0:
            raise ValueError("min_phase_coverage must be <= 1.0")
        for key in ("keypoint_rmse_px", "shaft_angle_deg"):
            _v.check_pos_float(self.annotator_noise_floors[key], key)
            _v.check_nonneg_float(self.null_baseline_margins[key], key)
        ref = self.pinned_reference
        _v.check_pos_int(ref.get("epic_issue"), "epic_issue")
        _v.check_str(ref.get("video_id"), "video_id")
        if (
            ref.get("timebase_status") != "unverified"
            or ref.get("physical_time_verified") is not False
        ):
            raise ValueError("pinned_reference must retain unverified timebase")
        if ref.get("qualification_floor") != "image_observations_only":
            raise ValueError("pinned_reference qualification_floor mismatch")

    def ceiling(self, key: str) -> float:
        """Return the noise-floor ceiling for ``key``."""
        return float(self.noise_floor_multiplier * self.annotator_noise_floors[key])


@dataclass(frozen=True, slots=True)
class FramePoseRecord:
    """2D keypoints and shaft angle for a frame and swing phase."""

    frame_index: int
    phase: str
    keypoints_px: _KP
    shaft_angle_deg: float

    def __post_init__(self) -> None:
        """Validate frame index, phase name, shaft angle, and 2D keypoints."""
        if _v.check_int(self.frame_index, "frame_index") < 0:
            raise ValueError("frame_index must be >= 0")
        _v.check_str(self.phase, "phase")
        _v.check_float(self.shaft_angle_deg, "shaft_angle_deg")
        if not isinstance(self.keypoints_px, Mapping) or not self.keypoints_px:
            raise ValueError("keypoints_px must be a non-empty mapping")
        cleaned = {}
        for name, xy in self.keypoints_px.items():
            if not isinstance(xy, (tuple, list)) or len(xy) != 2:
                raise ValueError(f"keypoints_px[{name}] must be a (x, y) pair")
            cleaned[_v.check_str(name, "kp")] = (
                _v.check_float(xy[0], "x"),
                _v.check_float(xy[1], "y"),
            )
        object.__setattr__(self, "keypoints_px", cleaned)


PhaseAnnotation = FramePoseRecord
CandidateFramePrediction = FramePoseRecord


@dataclass(frozen=True, slots=True)
class PhaseAlignedNullTemplate:
    """Phase-aligned static Tiger Woods null baseline template."""

    schema_version: str
    template_id: str
    pinned_reference: dict[str, Any]
    phase_poses: dict[str, FramePoseRecord]

    def predict(self, anns: Sequence[FramePoseRecord]) -> tuple[FramePoseRecord, ...]:
        """Broadcast each phase's canonical Tiger pose onto annotated frames."""
        out: list[FramePoseRecord] = []
        for a in anns:
            pose = self.phase_poses.get(a.phase)
            if pose is None:
                raise ValueError(f"Null template missing phase {a.phase!r}")
            kp = dict(pose.keypoints_px)
            out.append(
                FramePoseRecord(a.frame_index, a.phase, kp, pose.shaft_angle_deg)
            )
        return tuple(out)


@dataclass(frozen=True, slots=True)
class ModelSpaceChecks:
    """Model-space 3D plausibility checks (not physical time or SI kinetics)."""

    bone_lengths_ok: bool = False
    club_geometry_ok: bool = False
    grip_closure_ok: bool = False
    joint_limits_ok: bool = False
    ground_contact_ok: bool = False

    def __post_init__(self) -> None:
        """Validate that all model-space check flags are booleans."""
        if not all(isinstance(v, bool) for v in asdict(self).values()):
            raise TypeError("ModelSpaceChecks flags must be bool")

    @property
    def all_passed(self) -> bool:
        """Return True iff every model-space plausibility check passed."""
        return all(asdict(self).values())


@dataclass(frozen=True, slots=True)
class MarkerlessScorecard:
    """Immutable evaluation scorecard with deterministic SHA-256 content hash."""

    schema_version: str
    candidate_id: str
    status: ScorecardStatus
    match_classification: MatchKind
    metrics: dict[str, float | None]
    phase_results: tuple[dict[str, Any], ...]
    failing_phases: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    model_space_checks: ModelSpaceChecks | None
    pinned_reference: dict[str, Any]
    timebase: Literal["unverified"] = "unverified"
    physical_time_verified: Literal[False] = False
    qualification: Literal["image_observations_only"] = "image_observations_only"
    content_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate scorecard invariants and compute or verify content_sha256."""
        _v.check_schema_version(self.schema_version, SCORECARD_SCHEMA_VERSION)
        _v.check_id(self.candidate_id, "candidate_id")
        if self.timebase != "unverified" or self.physical_time_verified is not False:
            raise ValueError("Scorecard timebase must remain unverified")
        if self.qualification != "image_observations_only":
            raise ValueError("qualification must remain 'image_observations_only'")
        if self.status == "passed" and (self.failing_phases or self.failure_reasons):
            raise ValueError("Passed scorecard cannot contain failure fields")
        if self.status == "below_gate" and not self.failure_reasons:
            raise ValueError("below_gate scorecard requires failure_reasons")
        pl = self.to_payload_dict()
        raw = json.dumps(pl, sort_keys=True, allow_nan=False, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", digest)
        elif self.content_sha256 != digest:
            raise ValueError("content_sha256 mismatch")

    def to_payload_dict(self) -> dict[str, Any]:
        """Serialize canonical scorecard fields excluding content_sha256."""
        return {k: v for k, v in asdict(self).items() if k != "content_sha256"}

    def to_json(self) -> str:
        """Serialize the scorecard as deterministic indented JSON."""
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False, indent=2)


def load_scorecard_gates(path: Path | None = None) -> ScorecardGates:
    """Load and validate the frozen Tiger 2000 markerless scorecard gate config."""
    raw = json.loads((path or DEFAULT_GATES_PATH).read_text(encoding="utf-8"))
    return ScorecardGates(**raw)


def build_phase_aligned_null_template(
    gates: ScorecardGates | None = None,
    *,
    phase_poses: Mapping[str, FramePoseRecord] | None = None,
) -> PhaseAlignedNullTemplate:
    """Construct the phase-aligned static Tiger Woods null baseline template."""
    g = gates or load_scorecard_gates()
    specs = (
        ("address", (625.0, 390.0), (595.0, 560.0), -58.0),
        ("top", (685.0, 175.0), (540.0, 150.0), 15.0),
        ("downswing", (640.0, 330.0), (720.0, 265.0), -32.0),
        ("impact", (618.0, 388.0), (598.0, 558.0), -60.0),
        ("finish", (575.0, 185.0), (705.0, 195.0), 145.0),
    )
    if phase_poses is not None:
        poses = dict(phase_poses)
    else:
        poses = {
            p: FramePoseRecord(0, p, {"left_wrist": w, "club_head": c}, ang)
            for p, w, c, ang in specs
        }
    if [p for p in g.required_phases if p not in poses]:
        raise ValueError("Null template missing required phases")
    return PhaseAlignedNullTemplate(
        NULL_TEMPLATE_SCHEMA_VERSION, "tiger-2000-null-v1", g.pinned_reference, poses
    )


def _pair_rmse(
    pairs: Sequence[tuple[FramePoseRecord, FramePoseRecord]],
) -> tuple[float | None, float | None]:
    """Compute keypoint pixel RMSE and wrapped shaft-angle RMSE over record pairs."""
    if not pairs:
        return None, None
    kp_sq, ang_sq = list[float](), list[float]()
    for ann, pred in pairs:
        for name, (ax, ay) in ann.keypoints_px.items():
            px, py = pred.keypoints_px[name]
            kp_sq.append((px - ax) ** 2 + (py - ay) ** 2)
        err = abs((pred.shaft_angle_deg - ann.shaft_angle_deg + 180.0) % 360.0 - 180.0)
        ang_sq.append(err * err)
    return math.sqrt(sum(kp_sq) / len(kp_sq)), math.sqrt(sum(ang_sq) / len(ang_sq))


def _ceil_reasons(
    kp_r: float | None, ang_r: float | None, g: ScorecardGates, pfx: str
) -> list[str]:
    """Return prefixed failure reasons when RMSE metrics exceed gate ceilings."""
    out: list[str] = []
    if kp_r is not None and kp_r > g.ceiling("keypoint_rmse_px"):
        out.append(f"{pfx}keypoint_noise_ceiling_exceeded")
    if ang_r is not None and ang_r > g.ceiling("shaft_angle_deg"):
        out.append(f"{pfx}shaft_angle_noise_ceiling_exceeded")
    return out


def evaluate_markerless_scorecard(
    candidate_id: str,
    annotations: Sequence[FramePoseRecord],
    predictions: Sequence[FramePoseRecord],
    *,
    gates: ScorecardGates | None = None,
    null_template: PhaseAlignedNullTemplate | None = None,
    model_space_checks: ModelSpaceChecks | None = None,
) -> MarkerlessScorecard:
    """Evaluate a candidate against frozen Tiger 2000 2D gates and null baseline."""
    _v.check_id(candidate_id, "candidate_id")
    if not annotations:
        raise ValueError("annotations cannot be empty")
    g = gates or load_scorecard_gates()
    unknown = sorted({a.phase for a in annotations} - set(g.required_phases))
    if unknown:
        raise ValueError(f"annotation phases outside required_phases: {unknown}")
    tpl = null_template or build_phase_aligned_null_template(g)
    if len({(a.frame_index, a.phase) for a in annotations}) != len(annotations):
        raise ValueError("Duplicate annotation frame/phase")
    pred_map = {(p.frame_index, p.phase): p for p in predictions}
    if len(pred_map) != len(predictions):
        raise ValueError("Duplicate prediction frame/phase")
    null_map = {(n.frame_index, n.phase): n for n in tpl.predict(annotations)}
    p_res, failing, reasons, all_cand = [], [], [], []
    for phase in g.required_phases:
        anns = [a for a in annotations if a.phase == phase]
        c_pairs: list[tuple[FramePoseRecord, FramePoseRecord]] = []
        for a in anns:
            k = (a.frame_index, a.phase)
            if not set(a.keypoints_px).issubset(null_map[k].keypoints_px):
                raise ValueError(f"Null template missing keypoints in phase {phase!r}")
            p = pred_map.get(k)
            if p is not None and set(a.keypoints_px).issubset(p.keypoints_px):
                c_pairs.append((a, p))
        all_cand.extend(c_pairs)
        cov = len(c_pairs) / len(anns) if anns else 0.0
        kp_r, ang_r = _pair_rmse(c_pairs)
        pr = [f"missing_phase:{phase}"] if not anns else []
        if anns and cov < g.min_phase_coverage:
            pr.append(f"phase_coverage_below_gate:{phase}")
        pr.extend(f"{r}:{phase}" for r in _ceil_reasons(kp_r, ang_r, g, ""))
        if pr:
            failing.append(phase)
            reasons.extend(pr)
        p_res.append({"phase": phase, "coverage": cov, "passed": not pr})
    okp, oang = _pair_rmse(all_cand)
    # Candidate and null must be scored on one identical population: frames the
    # candidate was never asked to predict inflate the baseline and can turn a
    # candidate that merely re-emits the null pose into a false "passed".
    nkp, nang = _pair_rmse(
        [(a, null_map[(a.frame_index, a.phase)]) for a, _ in all_cand]
    )
    margins = g.null_baseline_margins
    if okp is None or oang is None:
        reasons.append("no_covered_observations")
    else:
        reasons.extend(_ceil_reasons(okp, oang, g, "overall_"))
        if nkp is None or (nkp - okp) < margins["keypoint_rmse_px"]:
            reasons.append("null_baseline_keypoint_margin_not_met")
        if nang is None or (nang - oang) < margins["shaft_angle_deg"]:
            reasons.append("null_baseline_shaft_angle_margin_not_met")
    uniq = tuple(dict.fromkeys(reasons))
    status: ScorecardStatus = "below_gate" if (failing or uniq) else "passed"
    ok_3d = bool(model_space_checks and model_space_checks.all_passed)
    cls: MatchKind = "below_gate"
    if status != "below_gate":
        cls = "kinematic_3d_plausible" if ok_3d else "kinematic_2d_match"
    m = {"kp_px": okp, "ang_deg": oang, "null_kp_px": nkp, "null_ang_deg": nang}
    return MarkerlessScorecard(
        SCORECARD_SCHEMA_VERSION,
        candidate_id,
        status,
        cls,
        m,
        tuple(p_res),
        tuple(failing),
        uniq,
        model_space_checks,
        g.pinned_reference,
    )
