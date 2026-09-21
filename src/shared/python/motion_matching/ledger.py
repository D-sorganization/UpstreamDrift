"""Matched-swing run ledger and evidence scanner (MS-02, #10323).

Discovers, classifies, and indexes all execution receipts across evidence trees
into a single browsable index saved at ``reports/matched_swing_ledger.json``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.shared.python.contracts import postcondition, precondition
from .ledger_schema import ArtefactPaths, Ledger, LedgerRow, SharedMetrics

__all__ = [
    "DEFAULT_ROOTS",
    "Ledger",
    "LedgerRow",
    "SharedMetrics",
    "ArtefactPaths",
    "classify_receipt",
    "default_ledger_path",
    "extract_acceptance",
    "extract_artefacts",
    "extract_candidate_sha",
    "extract_horizon_s",
    "extract_metrics",
    "scan",
]

DEFAULT_ROOTS: tuple[Path, ...] = (
    Path("docs/development/full_body_models/evidence"),
    Path("docs/development/simscape_tour_matching/native_evidence"),
    Path("docs/development/opensim_tour_matching/evidence"),
    Path("evidence"),
)


def default_ledger_path(repo_root: Path | None = None) -> Path:
    """Return canonical path for reports/matched_swing_ledger.json."""
    root = repo_root or _find_repo_root()
    return root / "reports" / "matched_swing_ledger.json"


def _find_repo_root(start: Path | None = None) -> Path:
    """Find the root directory of the repository."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists() or (
            (parent / "docs").is_dir()
            and (parent / "src").is_dir()
            and (parent / "pyproject.toml").is_file()
        ):
            return parent
    return Path.cwd()


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_metric_value(data: Mapping[str, Any], *keys: str) -> float | None:
    """Recursively search for a numeric metric across known receipt blocks."""
    for key in keys:
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)) and not math.isnan(val):
                return float(val)
    for sub in (
        "shared_metrics",
        "restart_uninterrupted_metrics",
        "zero_displacement_parity",
        "dynamics",
        "ik",
        "forward_rollout",
        "calibrated",
    ):
        nested = data.get(sub)
        if isinstance(nested, Mapping):
            res = _extract_metric_value(nested, *keys)
            if res is not None:
                return res
    return None


def extract_metrics(data: Mapping[str, Any]) -> SharedMetrics:
    """Extract standard five comparison metrics from receipt data."""
    whole = _extract_metric_value(
        data, "whole_marker_rmse_m", "whole_rms_m", "marker_rms_m"
    )
    early = _extract_metric_value(data, "early_marker_rmse_m", "early_rms_m")
    terminal = _extract_metric_value(data, "terminal_marker_rmse_m", "terminal_rms_m")
    club = _extract_metric_value(
        data, "club_marker_rmse_m", "club_cluster_rms_m", "clubhead_terminal_rmse_m"
    )

    yaw = _extract_metric_value(data, "pelvis_yaw_rmse_rad")
    if yaw is None:
        deg = _extract_metric_value(data, "pelvis_yaw_diff_deg")
        if deg is not None:
            yaw = math.radians(abs(deg))
    return SharedMetrics(
        whole_marker_rmse_m=whole,
        early_marker_rmse_m=early,
        terminal_marker_rmse_m=terminal,
        club_marker_rmse_m=club,
        pelvis_yaw_rmse_rad=yaw,
    )


def extract_candidate_sha(data: Mapping[str, Any]) -> str | None:
    """Extract candidate SHA or commit hash identifying the model/fit run."""
    for key in (
        "candidate_sha256",
        "candidate_sha",
        "returned_sha256",
        "config_sha256",
        "bundle_sha256",
        "calibrated_model_sha256",
        "scaled_model_sha256",
        "model_sha256",
        "spec_sha256",
        "git_commit",
        "commit",
    ):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    outputs = data.get("outputs")
    if isinstance(outputs, Mapping):
        for k in ("calibrated_model_sha256", "scaled_model_sha256", "ik_npz_sha256"):
            v = outputs.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def extract_horizon_s(data: Mapping[str, Any]) -> float | None:
    """Extract evaluation horizon or trajectory duration in seconds."""
    for key in ("horizon_s", "duration_s", "elapsed_s"):
        val = data.get(key)
        if isinstance(val, (int, float)) and not math.isnan(val):
            return float(val)
    for sub in ("dynamics", "forward_rollout"):
        nested = data.get(sub)
        if isinstance(nested, Mapping):
            for k in ("duration_s", "horizon_s", "sim_duration_s"):
                v = nested.get(k)
                if isinstance(v, (int, float)) and not math.isnan(v):
                    return float(v)
    frames = None
    if isinstance(data.get("ik"), Mapping) and "frames" in data["ik"]:
        frames = data["ik"]["frames"]
    elif "total_frames" in data:
        frames = data["total_frames"]
    elif "frames" in data and isinstance(data["frames"], (int, float)):
        frames = data["frames"]
    if frames and isinstance(frames, (int, float)) and frames > 1:
        rate = data.get("rate_hz", 360.0)
        return round((float(frames) - 1.0) / float(rate), 4)
    return None


def classify_receipt(
    rel_path: str,
    data: Mapping[str, Any],
) -> tuple[str, str, str | None, str | None]:
    """Pure classification function returning (engine, lane, capture, reason)."""
    p = rel_path.replace("\\", "/")
    engine = "unknown"
    lane = "unclassified"
    capture: str | None = None
    reason: str | None = None

    if "simscape_tour_matching" in p:
        engine, lane, capture = "simscape", "native", "driver"
    elif "opensim_tour_matching" in p:
        engine, lane, capture = "opensim", "tour_matching", "driver"
    elif "ground_support" in p:
        lane = "ground_support"
        engine = str(data.get("backend", "mujoco")).lower()
        capture = "driver" if "driver" in p else ("iron" if "iron" in p else None)
    elif "fb4_calibration" in p:
        lane = "fb4_calibration"
        capture = "driver"
        for eng in ("mujoco", "drake", "pinocchio", "simscape", "opensim"):
            if eng in p:
                engine = eng
                break
    elif "fb6_parity" in p:
        lane = "fb6_parity"
        capture = "driver"
        for eng in ("mujoco", "drake", "pinocchio", "simscape", "opensim"):
            if eng in p:
                engine = eng
                break
    elif "fb5_matching" in p:
        lane = "fb5_matching"
        capture = "driver"
        for eng in ("mujoco", "drake", "pinocchio", "simscape", "opensim"):
            if eng in p:
                engine = eng
                break
    elif "replays" in p:
        lane = "replays"
        capture = "driver"
        for eng in ("mujoco", "drake", "pinocchio", "simscape", "opensim"):
            if eng in p:
                engine = eng
                break
    elif "fb3_" in p:
        lane = "fb3_kinematics"
        capture = "driver"
        for eng in ("mujoco", "drake", "pinocchio", "simscape", "opensim"):
            if eng in p:
                engine = eng
                break
    elif "setup_parity" in p:
        lane, engine = "setup_parity", "mujoco"
        capture = "driver" if "driver" in p else ("iron" if "iron" in p else None)
    elif "anthropometry" in p:
        lane, engine, capture = "anthropometry", "mujoco", "driver"
    elif "visual_layer" in p:
        lane, engine = "visual_layer", "mujoco"
        capture = "driver" if "driver" in p else ("iron" if "iron" in p else None)
    elif "viewer" in p:
        lane, engine, capture = "viewer", "mujoco", "driver"
    elif "evidence/matched" in p or "work_package" in data:
        lane = "matched"
        engine = str(data.get("engine", "unknown")).lower()
        capture = "driver" if "driver" in p else ("iron" if "iron" in p else None)
    else:
        lane = "unclassified"
        reason = "Unrecognized directory path structure"

    if "engine" in data and isinstance(data["engine"], str) and data["engine"].strip():
        engine = data["engine"].strip().lower()
    elif (
        "backend" in data
        and isinstance(data["backend"], str)
        and data["backend"].strip()
    ):
        engine = data["backend"].strip().lower()

    if (
        "capture" in data
        and isinstance(data["capture"], str)
        and data["capture"].strip()
    ):
        capture = data["capture"].strip().lower()

    return engine, lane, capture, reason


def extract_artefacts(
    receipt_path: Path,
    data: Mapping[str, Any],
    repo_root: Path,
) -> ArtefactPaths:
    """Discover and resolve associated npz, gif, and mot artefact paths."""
    rdir = receipt_path.parent
    npz_path: str | None = None
    gif_path: str | None = None
    mot_path: str | None = None

    # 1. Search embedded paths in receipt
    for block_name in ("artifacts", "outputs"):
        block = data.get(block_name)
        if isinstance(block, Mapping):
            for k, val in block.items():
                if isinstance(val, str):
                    if val.endswith(".npz") or "npz" in k:
                        npz_path = (
                            (rdir / val).resolve().relative_to(repo_root).as_posix()
                            if (rdir / val).is_file()
                            else None
                        )
                    elif val.endswith(".gif") or "gif" in k:
                        gif_path = (
                            (rdir / val).resolve().relative_to(repo_root).as_posix()
                            if (rdir / val).is_file()
                            else None
                        )
                    elif val.endswith(".mot") or "mot" in k:
                        mot_path = (
                            (rdir / val).resolve().relative_to(repo_root).as_posix()
                            if (rdir / val).is_file()
                            else None
                        )

    # 2. Filesystem search adjacent to receipt if not found
    if npz_path is None:
        npzs = sorted(rdir.glob("*.npz"))
        if npzs:
            npz_path = npzs[0].relative_to(repo_root).as_posix()
    if gif_path is None:
        gifs = sorted(rdir.glob("*.gif"))
        if gifs:
            gif_path = gifs[0].relative_to(repo_root).as_posix()
    if mot_path is None:
        mots = sorted(rdir.glob("*.mot"))
        if mots:
            mot_path = mots[0].relative_to(repo_root).as_posix()

    return ArtefactPaths(npz=npz_path, gif=gif_path, mot=mot_path)


def _load_verdicts_map(repo_root: Path) -> dict[str, Any]:
    """Load pre-evaluated acceptance verdicts from MS-01 if present."""
    path = (
        repo_root
        / "docs"
        / "development"
        / "full_body_models"
        / "evidence"
        / "acceptance"
        / "verdicts_2026-09.json"
    )
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def extract_acceptance(
    rel_path: str,
    data: Mapping[str, Any],
    verdicts_map: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve the acceptance block from the MS-01 verdict registry or the receipt.

    Fail-closed (MS-100): only a verdict produced by ``acceptance.evaluate``
    (recognisable by its non-empty ``gates`` list) can mark a row accepted.
    A receipt's self-declared ``accepted`` flag, or an ``acceptance`` block
    without gate results, is reported as ``UNVERIFIED`` and never as PASSED.
    """
    if verdicts_map and rel_path in verdicts_map:
        return dict(verdicts_map[rel_path])
    if "acceptance" in data and isinstance(data["acceptance"], Mapping):
        block = dict(data["acceptance"])
        if block.get("gates"):
            return block
        return _unverified(
            block.get("horizon", "G1"), "acceptance block carries no gate results"
        )
    if "receipt" in data and isinstance(data["receipt"], Mapping):
        inner = data["receipt"]
        if "accepted" in inner:
            return _unverified(
                "G1",
                f"self-reported accepted={bool(inner.get('accepted'))}; "
                "not evaluated by acceptance.py",
            )
    return None


def _unverified(horizon: Any, note: str) -> dict[str, Any]:
    return {
        "horizon": str(horizon),
        "is_physically_accepted": False,
        "status": "UNVERIFIED",
        "qualification_note": note,
    }


@precondition(
    lambda roots=None, repo_root=None: (
        roots is None or isinstance(roots, (list, tuple))
    ),
    "roots must be None or Sequence[Path]",
)
@postcondition(
    lambda result: all(
        not Path(r.receipt_path).is_absolute() and len(r.sha256) == 64
        for r in result.rows
    ),
    "every row has receipt_path relative to repo root and sha256 of the receipt file",
)
def scan(
    roots: Sequence[Path] | None = None,
    repo_root: Path | None = None,
) -> Ledger:
    """Discover, classify, and index every committed execution receipt into a Ledger."""
    root = repo_root or _find_repo_root()
    search_roots = roots if roots is not None else [root / r for r in DEFAULT_ROOTS]

    verdicts = _load_verdicts_map(root)
    discovered_files: set[Path] = set()
    for s_root in search_roots:
        resolved_s = Path(s_root).resolve()
        if resolved_s.is_dir():
            for f in resolved_s.rglob("*.json"):
                if f.is_file() and "receipt" in f.name.lower():
                    discovered_files.add(f.resolve())

    rows: list[LedgerRow] = []
    for f in sorted(discovered_files):
        rel_str: str
        try:
            rel_str = f.relative_to(root).as_posix()
        except ValueError:
            matched_rel: str | None = None
            for s_root in search_roots:
                try:
                    matched_rel = f.relative_to(Path(s_root).resolve()).as_posix()
                    break
                except ValueError:
                    continue
            rel_str = matched_rel if matched_rel is not None else f.name

        sha = _compute_sha256(f)
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                data = {}
        except Exception as exc:
            rows.append(
                LedgerRow(
                    receipt_path=rel_str,
                    sha256=sha,
                    engine="unknown",
                    lane="unclassified",
                    capture=None,
                    candidate_sha=None,
                    horizon_s=None,
                    metrics=SharedMetrics(),
                    acceptance=None,
                    artefacts=ArtefactPaths(),
                    reason=f"Failed to parse receipt JSON: {exc}",
                )
            )
            continue

        engine, lane, capture, reason = classify_receipt(rel_str, data)
        metrics = extract_metrics(data)
        cand_sha = extract_candidate_sha(data)
        horizon = extract_horizon_s(data)
        artefacts = extract_artefacts(f, data, root)
        acceptance = extract_acceptance(rel_str, data, verdicts)

        rows.append(
            LedgerRow(
                receipt_path=rel_str,
                sha256=sha,
                engine=engine,
                lane=lane,
                capture=capture,
                candidate_sha=cand_sha,
                horizon_s=horizon,
                metrics=metrics,
                acceptance=acceptance,
                artefacts=artefacts,
                reason=reason,
            )
        )

    # Deterministic sorting by receipt_path
    rows.sort(key=lambda r: r.receipt_path)

    return Ledger(
        schema_version="1.0.0",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_receipts=len(rows),
        rows=rows,
    )
