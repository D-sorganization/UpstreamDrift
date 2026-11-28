"""Canonical-v2 workbook matrix, #15725 package, #15837 cross-check (T2-WP7, #15787)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from openpyxl import load_workbook

from src.shared.python.contracts import precondition
from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.model_appearance.player_models import player_profile
from src.shared.python.motion_matching.club_only import canonical_v2_trial as cv2
from src.shared.python.motion_matching.club_only import workbook_identity as wi

MATRIX_BUNDLE_SCHEMA = "canonical-v2-matrix-bundle/1.0.0"
COMPARISON_PACKAGE_SCHEMA = "swing-report-comparison-package/1.0.0"
PACKAGE_REF, CROSSCHECK_ISSUE_REF, SKIN_ISSUE_REF = "#15725", "#15837", "#15814"
FLAGSHIP_TRIAL_ID = "TW_ProV1"
REPLAY_BADGES = frozenset(
    {"pre-impact-replayed", "full-swing-replayed", "kinematic-only"}
)
COMPARISON_ARMS = ("Wiffle", "ProV1", "tour_baseline")
_TW, _FB = "full_body_mujoco_tiger_woods", "full_body_mujoco"
_META = {
    "TW_wiffle": ("TW", "Wiffle", "rob_neal", _TW, "tiger_woods", "#15814", False),
    "TW_ProV1": ("TW", "ProV1", "rob_neal", _TW, "tiger_woods", "#15814", True),
    "GW_wiffle": ("GW", "Wiffle", "rob_neal", _FB, None, None, False),
    "GW_ProV11": ("GW", "ProV1", "rob_neal", _FB, None, None, False),
}
_KEYS = (
    "grip_position_rmse_m",
    "clubhead_position_rmse_m",
    "grip_orientation_rmse_rad",
    "clubhead_orientation_rmse_rad",
    "grip_closure_rmse_m",
    "grip_closure_max_m",
    "root_force_bw_fraction",
    "root_moment_bw_height_fraction",
    "grf_residual_bw_fraction",
)


def _ser(v: Any) -> Any:
    """Recursively convert dataclasses and containers into JSON-safe values."""
    if hasattr(v, "as_dict"):
        return v.as_dict()
    if isinstance(v, Mapping):
        return {k: _ser(x) for k, x in v.items()}
    return [_ser(x) for x in v] if isinstance(v, (tuple, list)) else v


class _AsDict:
    """Shared dataclass serialization mixin."""

    def as_dict(self) -> dict[str, Any]:
        """Serialize dataclass fields to a JSON-compatible dictionary."""
        return {f.name: _ser(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]


def _num(name: str, value: object, *, neg: bool = False) -> float:
    """Validate and coerce a numeric scalar to finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or (not neg and out < 0.0):
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return out


@dataclass(frozen=True)
class MatrixTrialSummary(_AsDict):
    """Qualification summary for one canonical workbook matrix trial cell."""

    trial_id: str
    subject_id: str
    ball_type: str
    source_dataset: str
    target_model_id: str
    skin_id: str | None
    skin_issue_ref: str | None
    is_flagship: bool
    replay_badge: str
    qualified: bool
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate trial identity, replay badge, and qualification blockers."""
        ok_id = self.trial_id in wi.CANONICAL_TRIAL_SHEETS
        ok_rb = self.replay_badge in REPLAY_BADGES
        if not (ok_id and ok_rb) or self.qualified == bool(self.blockers):
            raise ValueError(f"invalid trial summary: {self.trial_id!r}")


@dataclass(frozen=True)
class ComparisonCellMetrics(_AsDict):
    """Side-by-side metrics for one Wiffle, ProV1, or tour_baseline cell."""

    arm: str
    trial_id: str
    subject_id: str
    replay_badge: str
    metrics: Mapping[str, float]
    cop_feasible: bool
    qualified: bool
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate arm label, replay badge, and numeric metric domains."""
        if self.arm not in COMPARISON_ARMS or self.replay_badge not in REPLAY_BADGES:
            raise ValueError(f"invalid arm={self.arm!r} or badge={self.replay_badge!r}")
        if not isinstance(self.cop_feasible, bool):
            raise TypeError("cop_feasible must be bool")
        for key in _KEYS:
            _num(key, self.metrics[key])
        hr = self.metrics["torque_headroom_fraction"]
        _num("torque_headroom_fraction", hr, neg=True)

    def as_dict(self) -> dict[str, Any]:
        """Serialize comparison cell metrics with flattened #15725 metric fields."""
        base = super().as_dict()
        return {**{k: v for k, v in base.items() if k != "metrics"}, **self.metrics}


@dataclass(frozen=True)
class ShaftShapeCrossCheck(_AsDict):
    """Non-gating #15837 phase-normalized shaft-shape cross-check telemetry."""

    trial_id: str
    grid_points: int
    shaft_angle_rmse_deg: float
    shaft_angle_max_diff_deg: float
    affects_pass_criteria: bool = False
    issue_ref: str = CROSSCHECK_ISSUE_REF
    notes: str = "Phase-normalized #15837 cross-check; non-gating telemetry only."

    def __post_init__(self) -> None:
        """Enforce non-gating contract so #15837 never affects pass criteria."""
        if self.affects_pass_criteria is not False:
            raise ValueError("non-gating cross-check cannot affect pass criteria")
        bad = self.issue_ref != CROSSCHECK_ISSUE_REF or not self.trial_id
        if bad or self.grid_points < 2:
            raise ValueError("invalid cross-check metadata or grid_points")
        _num("shaft_angle_rmse_deg", self.shaft_angle_rmse_deg)
        _num("shaft_angle_max_diff_deg", self.shaft_angle_max_diff_deg)


_DEF_CAT = {
    "render_engine": "mujoco",
    "views": ["face_on", "down_the_line"],
    "speeds": [0.25],
    "composite_items": ["case_grid", "phase_sheet"],
}


@dataclass(frozen=True)
class SideBySideComparisonPackage(_AsDict):
    """#15725-compatible side-by-side comparison payload."""

    cells: tuple[ComparisonCellMetrics, ...]
    arm_summaries: Mapping[str, Mapping[str, Any]]
    flagship_model_binding: Mapping[str, Any]
    schema_version: str = COMPARISON_PACKAGE_SCHEMA
    package_ref: str = PACKAGE_REF
    catalog: Mapping[str, Any] = field(default_factory=lambda: dict(_DEF_CAT))
    arms: tuple[str, ...] = COMPARISON_ARMS
    non_gating_cross_check: ShaftShapeCrossCheck | None = None
    package_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate schema/arms and compute deterministic package SHA-256."""
        miss = set(self.arms) - {c.arm for c in self.cells}
        if self.schema_version != COMPARISON_PACKAGE_SCHEMA or (
            self.package_ref != PACKAGE_REF or miss
        ):
            raise ValueError("invalid comparison package schema or missing arms")
        cc = self.non_gating_cross_check
        if cc is not None and cc.affects_pass_criteria:
            raise ValueError("non_gating_cross_check cannot affect pass criteria")
        p = self.as_dict()
        p.pop("package_sha256", None)
        raw = json.dumps(p, sort_keys=True, allow_nan=False, separators=(",", ":"))
        d = hashlib.sha256(raw.encode()).hexdigest()
        if not self.package_sha256:
            object.__setattr__(self, "package_sha256", d)
        elif self.package_sha256 != d:
            raise ValueError("package_sha256 mismatch")


@dataclass(frozen=True)
class CanonicalV2MatrixBundle(_AsDict):
    """Full canonical-v2 workbook matrix qualification and package bundle."""

    workbook_paths: Mapping[str, str]
    canonical_sheets: tuple[str, ...]
    flagship_trial_id: str
    flagship_model_binding: Mapping[str, Any]
    trials: tuple[MatrixTrialSummary, ...]
    comparison_package: SideBySideComparisonPackage
    qualified: bool
    blockers: tuple[str, ...] = ()
    schema_version: str = MATRIX_BUNDLE_SCHEMA
    governing_issue: int = 15787

    def __post_init__(self) -> None:
        """Validate schema and blocker consistency."""
        bad_ok = self.qualified == bool(self.blockers)
        if self.schema_version != MATRIX_BUNDLE_SCHEMA or bad_ok:
            raise ValueError("invalid matrix bundle schema or qualified/blockers state")


def resolve_matrix_workbook_paths(
    data_dir: Path | str | None = None, *, repo_root: Path | str | None = None
) -> dict[str, Path]:
    """Resolve Club_Data and Wiffle_ProV1 workbooks exclusively via CAPTURE_DATA_DIR."""
    root = Path(repo_root) if repo_root is not None else get_repo_root()
    resolver = getattr(wi, "resolve_club_workbooks", wi._resolve_club_workbook_paths)
    club_p, wiffle_p = resolver(root, Path(data_dir) if data_dir is not None else None)
    wi.verify_workbook_hash(club_p, wi.CLUB_DATA_SHA256)
    wi.verify_workbook_hash(wiffle_p, wi.WIFFLE_PROV1_SHA256)
    for path in (club_p, wiffle_p):
        wb = load_workbook(path, read_only=True, data_only=True)
        names = set(wb.sheetnames)
        wb.close()
        if miss := [s for s in wi.CANONICAL_TRIAL_SHEETS if s not in names]:
            raise ValueError(f"workbook {path.name} missing sheets: {miss}")
    return {"club-workbook-main": club_p, "club-workbook-wiffle": wiffle_p}


def compute_phase_normalized_shaft_crosscheck(
    club_only_angles_deg: ArrayLike,
    markerless_angles_deg: ArrayLike,
    *,
    club_only_phases: ArrayLike | None = None,
    markerless_phases: ArrayLike | None = None,
    trial_id: str = FLAGSHIP_TRIAL_ID,
    grid_points: int = 101,
) -> ShaftShapeCrossCheck:
    """Compute non-gating #15837 phase-normalized shaft-angle cross-check metrics."""
    if grid_points < 2:
        raise ValueError("grid_points must be >= 2")
    grid = np.linspace(0.0, 1.0, grid_points)
    series: list[np.ndarray] = []
    for raw_ang, raw_ph in (
        (club_only_angles_deg, club_only_phases),
        (markerless_angles_deg, markerless_phases),
    ):
        ang = np.asarray(raw_ang, dtype=np.float64)
        if ang.ndim != 1 or ang.size < 2 or not np.all(np.isfinite(ang)):
            raise ValueError("angles must be finite 1-D with >= 2 samples")
        dflt = np.linspace(0.0, 1.0, ang.size)
        ph = dflt if raw_ph is None else np.asarray(raw_ph, dtype=np.float64)
        ph_ok = ph.shape == ang.shape and np.all(np.isfinite(ph))
        if not ph_ok or not np.all(np.diff(ph) > 0):
            raise ValueError("phases must be finite and strictly increasing")
        ph = (ph - ph[0]) / float(ph[-1] - ph[0])
        series.append(np.interp(grid, ph, np.rad2deg(np.unwrap(np.deg2rad(ang)))))
    diff = ((series[0] - series[1] + 180.0) % 360.0) - 180.0
    rmse, mx = float(np.sqrt(np.mean(diff**2))), float(np.max(np.abs(diff)))
    return ShaftShapeCrossCheck(trial_id, grid_points, rmse, mx)


def _cell(
    arm: str, tid: str, subj: str, raw: Mapping[str, Any], tols: Mapping[str, float]
) -> ComparisonCellMetrics:
    """Validate thresholds/feasibility and construct ComparisonCellMetrics."""
    blks = [
        f"{tid}:threshold_exceeded:{m}"
        for m in _KEYS
        if _num(m, raw[m]) > tols[cv2._LIMITS[m]]
    ]
    hr = _num("torque_headroom_fraction", raw["torque_headroom_fraction"], neg=True)
    if hr < 0.0:
        blks.append(f"{tid}:torque_headroom_exhausted")
    cop = bool(raw["cop_feasible"])
    if not cop:
        blks.append(f"{tid}:cop_infeasible")
    metrics = {k: _num(k, raw[k]) for k in _KEYS} | {"torque_headroom_fraction": hr}
    rb, ok = str(raw["replay_badge"]), not blks
    return ComparisonCellMetrics(arm, tid, subj, rb, metrics, cop, ok, tuple(blks))


def build_side_by_side_comparison_package(
    cells: Sequence[ComparisonCellMetrics],
    flagship_model_binding: Mapping[str, Any],
    *,
    non_gating_cross_check: ShaftShapeCrossCheck | None = None,
    repo_root: Path | str | None = None,
) -> SideBySideComparisonPackage:
    """Assemble a deterministic #15725 side-by-side comparison package."""
    root = Path(repo_root) if repo_root is not None else get_repo_root()
    cat_p = root / "src/tools/swing_report_package/package_catalog.json"
    cat = dict(_DEF_CAT)
    if cat_p.is_file():
        doc = json.loads(cat_p.read_text(encoding="utf-8"))
        eng = str(doc.get("render_defaults", {}).get("engines", "mujoco"))
        cat |= {"render_engine": eng, "views": list(doc.get("views", cat["views"]))}
        cat["speeds"] = [float(s) for s in doc.get("speeds", cat["speeds"])]
    sums: dict[str, dict[str, Any]] = {}
    gp, cp, th = _KEYS[0], _KEYS[1], "torque_headroom_fraction"
    for arm in COMPARISON_ARMS:
        if g := [c for c in cells if c.arm == arm]:
            n = float(len(g))
            sums[arm] = {
                "cell_count": len(g),
                "qualified": all(c.qualified for c in g),
                "replay_badges": sorted({c.replay_badge for c in g}),
                f"mean_{gp}": sum(c.metrics[gp] for c in g) / n,
                f"mean_{cp}": sum(c.metrics[cp] for c in g) / n,
                f"min_{th}": min(c.metrics[th] for c in g),
                "all_cop_feasible": all(c.cop_feasible for c in g),
            }
    return SideBySideComparisonPackage(
        tuple(cells),
        sums,
        flagship_model_binding,
        catalog=cat,
        non_gating_cross_check=non_gating_cross_check,
    )


@precondition(lambda cell_metrics, **_: isinstance(cell_metrics, Mapping), "mapping")
def qualify_canonical_v2_matrix(
    *,
    cell_metrics: Mapping[str, Mapping[str, Any]],
    tour_baseline_metrics: Mapping[str, Any],
    data_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
    shaft_crosscheck: ShaftShapeCrossCheck | None = None,
    thresholds: cv2.CanonicalV2Thresholds | None = None,
) -> CanonicalV2MatrixBundle:
    """Qualify the 4-cell TW/GW x ProV1/Wiffle matrix and build #15725 package."""
    resolved = resolve_matrix_workbook_paths(data_dir=data_dir, repo_root=repo_root)
    sheets = wi.CANONICAL_TRIAL_SHEETS
    if miss := [s for s in sheets if s not in cell_metrics]:
        raise ValueError(f"missing canonical matrix trial metrics: {miss}")
    tols = (thresholds or cv2.load_canonical_v2_thresholds()).tolerances
    cells = [_cell(_META[t][1], t, _META[t][0], cell_metrics[t], tols) for t in sheets]
    _m = MatrixTrialSummary
    trials = tuple(
        _m(c.trial_id, *_META[c.trial_id], c.replay_badge, c.qualified, c.blockers)
        for c in cells
    )
    tb = "tour_baseline"
    cells.append(_cell(tb, tb, tb, tour_baseline_metrics, tols))
    bind = {
        "trial_id": FLAGSHIP_TRIAL_ID,
        "subject_label": "Rob Neal TW",
        "source_dataset": "rob_neal",
        "engine": "mujoco",
        "target_model_id": _TW,
        "skin_id": "tiger_woods",
        "skin_name": str(player_profile("tiger_woods")["appearance"]["name"]),
        "skin_issue_ref": SKIN_ISSUE_REF,
    }
    pkg = build_side_by_side_comparison_package(
        cells, bind, non_gating_cross_check=shaft_crosscheck, repo_root=repo_root
    )
    blks = tuple(dict.fromkeys(b for c in cells for b in c.blockers))
    paths = {k: str(v) for k, v in resolved.items()}
    return CanonicalV2MatrixBundle(
        paths, sheets, FLAGSHIP_TRIAL_ID, bind, trials, pkg, not blks, blks
    )
