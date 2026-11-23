"""Export manifest of a synchronized strategy-replay bundle (#15494, Slice C).

Links the admitted baseline, each candidate, its residuals, the assumptions and
the SHA-256 of every video and series file, so a reviewer can re-hash the bundle.
A candidate is ``REPLAYED``, ``ALLOCATION_ONLY`` (may not carry a video) or
``UNAVAILABLE`` (reason mandatory, nothing attached), so missing data stays
visibly missing.  One :class:`RenderSettings` serves the whole bundle.  Pure
stdlib; the JSON form is canonical, so the manifest digest is stable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any

__all__ = [
    "MANIFEST_SCHEMA",
    "ArtifactRef",
    "ArtifactRole",
    "BundleCandidate",
    "RenderSettings",
    "ReplayBundleManifest",
    "ReplayStatus",
    "verify_against_dir",
]

MANIFEST_SCHEMA = "nullspace-replay-bundle-manifest/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArtifactRole(str, Enum):
    """What an artifact is within the bundle."""

    BASELINE = "baseline"
    VIDEO = "video"
    TIME_SERIES = "time_series"
    RESIDUALS = "residuals"


class ReplayStatus(str, Enum):
    """How far a candidate got; never upgraded implicitly."""

    REPLAYED = "replayed"
    ALLOCATION_ONLY = "allocation_only"
    UNAVAILABLE = "unavailable"


def _is_safe_relative_name(name: str) -> bool:
    """True only for a strict, platform-independent bundle-relative POSIX path.

    Rejects empty names, backslashes, leading separators (root, UNC), Windows
    drives, any colon (NTFS streams), NUL bytes and ``.``/``..``/empty parts.
    """
    if not name or "\\" in name or "\x00" in name or ":" in name:
        return False
    if name.startswith("/") or PureWindowsPath(name).drive:
        return False
    return not any(p in ("", "..", ".") for p in name.split("/"))


def _positive(value: float, label: str) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")


@dataclass(frozen=True)
class ArtifactRef:
    """A file of the bundle: relative POSIX name, SHA-256 and size."""

    role: ArtifactRole
    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not _is_safe_relative_name(self.name):
            raise ValueError(f"unsafe artifact name: {self.name!r}")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")

    @classmethod
    def from_bytes(cls, role: ArtifactRole, name: str, payload: bytes) -> ArtifactRef:
        """Build a reference from file content."""
        return cls(role, name, hashlib.sha256(payload).hexdigest(), len(payload))


@dataclass(frozen=True)
class RenderSettings:
    """The single camera, timebase and vector scales shared by every video."""

    camera_id: str
    timebase_s: float
    force_scale_m_per_n: float
    torque_scale_m_per_nm: float

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")
        _positive(self.timebase_s, "timebase_s")
        _positive(self.force_scale_m_per_n, "force_scale_m_per_n")
        _positive(self.torque_scale_m_per_nm, "torque_scale_m_per_nm")


@dataclass(frozen=True)
class BundleCandidate:
    """One strategy in the bundle with its status, residual and artifacts."""

    candidate_id: str
    strategy: str
    status: ReplayStatus
    max_residual: float | None = None
    unavailable_reason: str = ""
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.strategy:
            raise ValueError("candidate_id and strategy must be non-empty")
        roles = {a.role for a in self.artifacts}
        if self.status is ReplayStatus.UNAVAILABLE:
            if not self.unavailable_reason:
                raise ValueError("an unavailable candidate needs a reason")
            if self.artifacts or self.max_residual is not None:
                raise ValueError("unavailable candidate carries no artifacts")
            return
        if self.unavailable_reason:
            raise ValueError("reason only applies to unavailable candidates")
        res = self.max_residual
        if res is None or not math.isfinite(res) or res < 0:
            raise ValueError("max_residual must be finite and >= 0")
        if ArtifactRole.RESIDUALS not in roles:
            raise ValueError("candidate needs a residuals artifact")
        if self.status is ReplayStatus.REPLAYED and ArtifactRole.VIDEO not in roles:
            raise ValueError("a replayed candidate needs a video artifact")
        if self.status is ReplayStatus.ALLOCATION_ONLY and ArtifactRole.VIDEO in roles:
            raise ValueError("an allocation_only candidate must not carry a video")


@dataclass(frozen=True)
class ReplayBundleManifest:
    """Baseline, shared render settings, assumptions and candidates."""

    baseline: ArtifactRef
    render: RenderSettings
    assumptions: tuple[str, ...]
    candidates: tuple[BundleCandidate, ...]

    def __post_init__(self) -> None:
        if self.baseline.role is not ArtifactRole.BASELINE:
            raise ValueError("baseline artifact must have role 'baseline'")
        if not self.assumptions or not all(self.assumptions):
            raise ValueError("at least one non-empty assumption is required")
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate candidate_id in manifest")
        names = [a.name for a in self.all_artifacts()]
        if len(set(names)) != len(names):
            raise ValueError("duplicate artifact name in manifest")

    def all_artifacts(self) -> tuple[ArtifactRef, ...]:
        """Baseline first, then every candidate artifact in candidate-id order."""
        ordered = sorted(self.candidates, key=lambda c: c.candidate_id)
        return (self.baseline, *(a for c in ordered for a in c.artifacts))

    def to_dict(self) -> dict[str, Any]:
        """Canonical dictionary form (candidates sorted by id)."""
        return {
            "schema": MANIFEST_SCHEMA,
            "baseline": _ref_dict(self.baseline),
            "render": asdict(self.render),
            "assumptions": list(self.assumptions),
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "strategy": c.strategy,
                    "status": c.status.value,
                    "max_residual": c.max_residual,
                    "unavailable_reason": c.unavailable_reason,
                    "artifacts": [_ref_dict(a) for a in c.artifacts],
                }
                for c in sorted(self.candidates, key=lambda c: c.candidate_id)
            ],
        }

    def to_json(self) -> str:
        """Canonical JSON text."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def digest(self) -> str:
        """SHA-256 of the canonical JSON text."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> ReplayBundleManifest:
        """Parse and re-validate a manifest; rejects a foreign schema."""
        data = json.loads(text)
        if data.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported manifest schema: {data.get('schema')!r}")
        return cls(
            baseline=_ref_from(data["baseline"]),
            render=RenderSettings(**data["render"]),
            assumptions=tuple(data["assumptions"]),
            candidates=tuple(
                BundleCandidate(
                    **{
                        **c,
                        "status": ReplayStatus(c["status"]),
                        "artifacts": tuple(_ref_from(a) for a in c["artifacts"]),
                    }
                )
                for c in data["candidates"]
            ),
        )


def _ref_dict(ref: ArtifactRef) -> dict[str, Any]:
    return {**asdict(ref), "role": ref.role.value}


def _ref_from(d: dict[str, Any]) -> ArtifactRef:
    return ArtifactRef(**{**d, "role": ArtifactRole(d["role"])})


def verify_against_dir(manifest: ReplayBundleManifest, root: Path) -> tuple[str, ...]:
    """Re-hash every artifact under ``root``; return one message per problem.

    Postcondition: an empty tuple means every file exists with the recorded
    SHA-256 and size.
    """
    problems: list[str] = []
    for ref in manifest.all_artifacts():
        path = root / ref.name
        if not _is_safe_relative_name(ref.name) or not path.resolve().is_relative_to(
            root.resolve()
        ):
            problems.append(f"{ref.name}: outside bundle")
            continue
        if not path.is_file():
            problems.append(f"{ref.name}: missing")
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != ref.sha256:
            problems.append(f"{ref.name}: sha256 mismatch")
        elif len(payload) != ref.size_bytes:
            problems.append(f"{ref.name}: size mismatch")
    return tuple(problems)
