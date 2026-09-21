"""Matched-swing ledger API service (MS-85, #10358).

Read-only access to the matched-swing run ledger, receipts, candidate packages,
parity reports, and animation artefacts. Reuses the MS-02 ledger scanner and
MS-80 browser resolution rules without exposing absolute filesystem paths in
public responses (run ids are receipt SHA-256 digests from the ledger).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.ledger import (
    default_ledger_path,
    find_repo_root,
)
from src.shared.python.motion_matching.ledger_schema import Ledger, LedgerRow
from src.tools.matched_swing_browser.model import MatchedSwingBrowserModel

__all__ = [
    "MatchedSwingJobError",
    "MatchedSwingsService",
    "RunCapabilities",
    "RunSummary",
]


@dataclass(frozen=True)
class MatchedSwingJobError:
    """Typed error surfaced to API clients when an artefact is unavailable."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class RunCapabilities:
    """Explicit model/horizon capabilities advertised for a ledger row."""

    has_candidate_npz: bool
    has_animation_gif: bool
    has_parity_report: bool
    candidate_profile: str | None
    horizon_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_candidate_npz": self.has_candidate_npz,
            "has_animation_gif": self.has_animation_gif,
            "has_parity_report": self.has_parity_report,
            "candidate_profile": self.candidate_profile,
            "horizon_s": self.horizon_s,
        }


@dataclass(frozen=True)
class RunSummary:
    """Public ledger row summary — no absolute paths."""

    id: str
    engine: str
    lane: str
    capture: str | None
    candidate_sha256: str | None
    receipt_sha256: str
    horizon_s: float | None
    verdict: str
    metrics: dict[str, float | None]
    capabilities: RunCapabilities
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "engine": self.engine,
            "lane": self.lane,
            "capture": self.capture,
            "candidate_sha256": self.candidate_sha256,
            "receipt_sha256": self.receipt_sha256,
            "horizon_s": self.horizon_s,
            "verdict": self.verdict,
            "metrics": self.metrics,
            "capabilities": self.capabilities.to_dict(),
            "reason": self.reason,
        }


class MatchedSwingsService:
    """Resolve matched-swing ledger rows and artefacts for HTTP handlers."""

    def __init__(
        self,
        repo_root: Path | str | None = None,
        ledger_path: Path | str | None = None,
    ) -> None:
        self._repo_root = Path(repo_root or find_repo_root()).resolve()
        self._ledger_path = (
            Path(ledger_path) if ledger_path else default_ledger_path(self._repo_root)
        )
        if not self._ledger_path.is_absolute():
            self._ledger_path = (self._repo_root / self._ledger_path).resolve()
        self._browser = MatchedSwingBrowserModel(self._repo_root)

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @precondition(lambda self: True)
    @postcondition(lambda result: isinstance(result, list))
    def list_runs(self) -> list[RunSummary]:
        """Return all ledger rows as public summaries keyed by receipt SHA-256."""
        rows = self._load_rows()
        return [self._to_summary(row) for row in rows]

    @precondition(lambda self, run_id: isinstance(run_id, str) and bool(run_id.strip()))
    def get_row(self, run_id: str) -> LedgerRow:
        """Look up a ledger row by receipt SHA-256 id."""
        normalized = run_id.strip().lower()
        for row in self._load_rows():
            if row.sha256.lower() == normalized:
                return row
        raise KeyError(f"Unknown matched-swing run id: {run_id}")

    @precondition(lambda self, run_id: isinstance(run_id, str) and bool(run_id.strip()))
    def get_run_summary(self, run_id: str) -> RunSummary:
        """Return the public summary for a single run."""
        return self._to_summary(self.get_row(run_id))

    @precondition(lambda self, run_id: isinstance(run_id, str) and bool(run_id.strip()))
    def get_receipt(self, run_id: str) -> dict[str, Any]:
        """Load the receipt JSON for a run."""
        row = self.get_row(run_id)
        receipt_path = self._browser.resolve_artifact_path(row, "receipt")
        if receipt_path is None or not receipt_path.is_file():
            raise FileNotFoundError("Receipt file missing on disk")
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Receipt JSON must be an object")
        return data

    @precondition(
        lambda self, run_id, artifact: (
            isinstance(run_id, str)
            and bool(run_id.strip())
            and artifact in {"candidate", "gif", "parity"}
        )
    )
    def resolve_artifact_path(
        self,
        run_id: str,
        artifact: Literal["candidate", "gif", "parity"],
    ) -> Path:
        """Resolve an on-disk artefact path for streaming handlers."""
        row = self.get_row(run_id)
        if artifact == "candidate":
            path = self._browser.resolve_artifact_path(row, "npz")
        elif artifact == "gif":
            path = self._browser.resolve_artifact_path(row, "gif")
        else:
            path = self._browser.resolve_artifact_path(row, "parity")
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{artifact} artefact missing for run {run_id}")
        return path

    @precondition(
        lambda self, run_id, frame_index: isinstance(run_id, str) and frame_index >= 0
    )
    @postcondition(lambda result: isinstance(result, list))
    def candidate_preview_joints(
        self, run_id: str, frame_index: int = 0
    ) -> list[dict[str, Any]]:
        """Build MocapSkeleton3D-compatible joints from candidate marker positions."""
        from src.shared.python.motion_matching.candidate_io import load_candidate

        npz_path = self.resolve_artifact_path(run_id, "candidate")
        candidate = load_candidate(npz_path, validate_checksums=False)
        markers = candidate.markers.model_markers_m
        if markers is None:
            raise ValueError("Candidate package has no model_markers_m")
        if frame_index >= markers.shape[0]:
            raise IndexError(f"Frame index {frame_index} out of range")
        names = candidate.metadata.marker_names or tuple(
            f"marker_{idx}" for idx in range(markers.shape[1])
        )
        joints: list[dict[str, Any]] = []
        for idx, name in enumerate(names):
            pos = markers[frame_index, idx]
            joints.append(
                {
                    "name": name,
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "confidence": 1.0,
                    "parent": None,
                }
            )
        return joints

    @precondition(lambda self, run_id: isinstance(run_id, str) and bool(run_id.strip()))
    def candidate_frame_count(self, run_id: str) -> int:
        """Return the number of preview frames available in the candidate package."""
        npz_path = self.resolve_artifact_path(run_id, "candidate")
        with np.load(npz_path, allow_pickle=False) as data:
            markers = data.get("model_markers_m")
            if markers is None:
                raise ValueError("Candidate package has no model_markers_m")
            return int(markers.shape[0])

    def _load_rows(self) -> list[LedgerRow]:
        rows = self._browser.load_ledger(self._ledger_path)
        return rows

    def _to_summary(self, row: LedgerRow) -> RunSummary:
        verdict = MatchedSwingBrowserModel.extract_verdict_string(row)
        caps = RunCapabilities(
            has_candidate_npz=bool(row.artefacts.npz),
            has_animation_gif=bool(row.artefacts.gif),
            has_parity_report=self._browser.resolve_artifact_path(row, "parity")
            is not None,
            candidate_profile=self._candidate_profile(row),
            horizon_s=row.horizon_s,
        )
        metrics = {
            "whole_marker_rmse_m": row.metrics.whole_marker_rmse_m,
            "early_marker_rmse_m": row.metrics.early_marker_rmse_m,
            "terminal_marker_rmse_m": row.metrics.terminal_marker_rmse_m,
            "club_marker_rmse_m": row.metrics.club_marker_rmse_m,
            "pelvis_yaw_rmse_rad": row.metrics.pelvis_yaw_rmse_rad,
        }
        return RunSummary(
            id=row.sha256,
            engine=row.engine,
            lane=row.lane,
            capture=row.capture,
            candidate_sha256=row.candidate_sha,
            receipt_sha256=row.sha256,
            horizon_s=row.horizon_s,
            verdict=verdict,
            metrics=metrics,
            capabilities=caps,
            reason=row.reason,
        )

    @staticmethod
    def _candidate_profile(row: LedgerRow) -> str | None:
        acceptance = row.acceptance or {}
        profile = acceptance.get("candidate_profile")
        if isinstance(profile, str) and profile.strip():
            return profile.strip()
        return None

    @classmethod
    def from_ledger_file(
        cls, ledger_path: Path, repo_root: Path
    ) -> MatchedSwingsService:
        """Construct a service bound to an explicit ledger file (tests)."""
        return cls(repo_root=repo_root, ledger_path=ledger_path)
