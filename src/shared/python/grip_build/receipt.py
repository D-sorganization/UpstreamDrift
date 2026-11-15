"""Versioned grip build receipt schema (HRG-GB2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

BUILD_RECEIPT_SCHEMA_VERSION: Final[int] = 1


class BuildStatus(str, Enum):
    OK = "OK"
    GAP = "GAP"
    JOINT_LIMIT = "JOINT_LIMIT"
    INFEASIBLE = "INFEASIBLE"


@dataclass(frozen=True)
class BuildStageRecord:
    """One recorded build stage for GB-4 animation replay."""

    stage_id: str
    finger: str
    joint: str
    flexion_rad: float
    status: BuildStatus
    min_clearance_m: float
    target_tolerance_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "finger": self.finger,
            "joint": self.joint,
            "flexion_rad": self.flexion_rad,
            "status": self.status.value,
            "min_clearance_m": self.min_clearance_m,
            "target_tolerance_m": self.target_tolerance_m,
        }


@dataclass(frozen=True)
class BuildReceipt:
    """Versioned, deterministic build metadata."""

    schema_version: int
    builder_version: str
    spec_sha256: str
    curl_order: tuple[str, ...]
    handedness: str
    bone_tolerance_m: float
    pad_tolerance_m: float
    stages: tuple[BuildStageRecord, ...]
    overall_status: BuildStatus
    gaps_m: tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "builder_version": self.builder_version,
            "spec_sha256": self.spec_sha256,
            "curl_order": list(self.curl_order),
            "handedness": self.handedness,
            "bone_tolerance_m": self.bone_tolerance_m,
            "pad_tolerance_m": self.pad_tolerance_m,
            "overall_status": self.overall_status.value,
            "gaps_m": list(self.gaps_m),
            "stages": [s.to_dict() for s in self.stages],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class GripBuildResult:
    """Builder output: final chain state plus receipt."""

    receipt: BuildReceipt
    joint_angles_rad: dict[str, dict[str, float]]
    status: BuildStatus
