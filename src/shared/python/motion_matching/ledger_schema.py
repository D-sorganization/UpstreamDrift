"""Pydantic v2 schema for the matched-swing run ledger (MS-02, #10323).

Defines the serialized index structure for all execution receipts, metrics,
and artefacts discovered across evidence trees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ArtefactPaths",
    "Ledger",
    "LedgerRow",
    "SharedMetrics",
]


class SharedMetrics(BaseModel):
    """Five standardized cross-engine comparison metrics (metres or radians)."""

    model_config = ConfigDict(extra="ignore")

    whole_marker_rmse_m: float | None = Field(
        default=None,
        description="Whole-capture marker tracking root-mean-square error in metres",
    )
    early_marker_rmse_m: float | None = Field(
        default=None,
        description="Early backswing / phase marker tracking RMSE in metres",
    )
    terminal_marker_rmse_m: float | None = Field(
        default=None,
        description="Terminal / downswing marker tracking RMSE in metres",
    )
    club_marker_rmse_m: float | None = Field(
        default=None,
        description="Clubhead cluster marker tracking RMSE in metres",
    )
    pelvis_yaw_rmse_rad: float | None = Field(
        default=None,
        description="Pelvis yaw tracking RMSE in radians or angle difference",
    )


class ArtefactPaths(BaseModel):
    """Paths to primary matched-swing visual and kinematic artefacts."""

    model_config = ConfigDict(extra="ignore")

    npz: str | None = Field(
        default=None,
        description="Path to NumPy trajectory or replay record (.npz)",
    )
    gif: str | None = Field(
        default=None,
        description="Path to visual playback animation (.gif)",
    )
    mot: str | None = Field(
        default=None,
        description="Path to OpenSim motion file (.mot)",
    )
    baseline_package: str | None = Field(
        default=None,
        description="Path to versioned tour baseline package directory",
    )


class LedgerRow(BaseModel):
    """Indexed and classified record for a single execution receipt on disk."""

    model_config = ConfigDict(extra="ignore")

    receipt_path: str = Field(
        ..., description="POSIX path to receipt relative to repository root"
    )
    sha256: str = Field(..., description="SHA-256 hash of the receipt JSON content")
    engine: str = Field(
        ...,
        description="Simulation engine (mujoco, simscape, opensim, drake, pinocchio, unknown)",
    )
    lane: str = Field(
        ...,
        description="Execution lane (ground_support, native, tour_matching, fb4_calibration, fb6_parity, replays, matched, unclassified)",
    )
    capture: str | None = Field(
        default=None, description="Tour capture identity (driver, iron, or None)"
    )
    candidate_sha: str | None = Field(
        default=None, description="Unique candidate SHA or commit hash"
    )
    horizon_s: float | None = Field(
        default=None, description="Evaluation horizon or duration in seconds"
    )
    metrics: SharedMetrics = Field(
        default_factory=lambda: SharedMetrics(),
        description="Standardized shared metrics",
    )
    acceptance: dict[str, Any] | None = Field(
        default=None,
        description="Acceptance verdict block from MS-01 when available",
    )
    artefacts: ArtefactPaths = Field(
        default_factory=lambda: ArtefactPaths(),
        description="Associated file artefacts",
    )
    reason: str | None = Field(
        default=None,
        description="Diagnostic classification reason or fallback note",
    )


class Ledger(BaseModel):
    """Complete collection of indexed matched-swing runs and receipts."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field("1.0.0", description="Ledger schema version")
    generated_at: str = Field(
        ..., description="UTC timestamp of ledger generation (ISO 8601)"
    )
    total_receipts: int = Field(
        ..., description="Total receipts indexed in this ledger"
    )
    rows: list[LedgerRow] = Field(
        default_factory=list, description="Sorted list of classified receipts"
    )

    def to_json(self) -> str:
        """Serialize the ledger deterministically to JSON format."""
        data = self.model_dump(mode="json", exclude_none=False)
        return json.dumps(data, indent=2, sort_keys=True) + "\n"

    def write_json(self, path: Path) -> Path:
        """Write the serialized ledger to a file path deterministically."""
        resolved = Path(path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(self.to_json(), encoding="utf-8")
        return resolved
