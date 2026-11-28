"""Frozen MuJoCo canonical-v2 / capture-A reference fixture for OpenSim (#15793 A1).

Pure Python + NumPy (no ``import opensim`` required).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.canonical_golfer_gate import (
    plant_hash as compute_plant_hash,
    require_canonical_golfer,
)
from src.shared.python.motion_matching.evidence_integrity import is_real_sha256

SCHEMA_VERSION: Final[str] = "opensim-canonical-v2-fixture/1.0.0"
DEFAULT_CAPTURE_ID: Final[str] = "capture-A"
ORTHOGONAL_TOL: Final[float] = 1e-9
TIMEBASE_DT_TOL: Final[float] = 1e-9
RotationTuple = tuple[tuple[float, float, float], ...]


class FixtureIntegrityError(ValueError):
    """Raised when a canonical-v2 fixture fails schema or SHA-256 checks."""


def _req_str(value: object, name: str) -> str:
    """Validate that ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _req_sha256(value: object, name: str) -> str:
    """Validate and normalize a real 64-hex SHA-256 digest."""
    if not is_real_sha256(value):
        raise FixtureIntegrityError(f"{name} must be a real 64-hex SHA-256")
    return str(value).strip().lower()


def _req_labels(values: object, name: str) -> tuple[str, ...]:
    """Validate a non-empty sequence of unique non-empty label strings."""
    if isinstance(values, (str, bytes)) or not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of strings")
    labels = tuple(_req_str(item, f"{name}[]") for item in values)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError(f"{name} must be non-empty and contain unique labels")
    return labels


def _frozen_array(
    values: object, shape: tuple[int, ...], name: str
) -> NDArray[np.float64]:
    """Convert ``values`` to a read-only finite float64 array with exact shape."""
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    arr.setflags(write=False)
    return arr


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 hex digest of a canonicalized JSON mapping."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrameTransform3D:
    """Proper-orthogonal SO(3) + translation transform between two 3D frames."""

    source_frame: str
    target_frame: str
    rotation_matrix: RotationTuple
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Enforce frame names, SO(3) orthogonality, det == +1, and immutability."""
        src, dst = (
            _req_str(self.source_frame, "source_frame"),
            _req_str(self.target_frame, "target_frame"),
        )
        rot = np.asarray(self.rotation_matrix, dtype=np.float64)
        if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
            raise ValueError("rotation_matrix must be a finite (3, 3) matrix")
        if not np.allclose(rot.T @ rot, np.eye(3), atol=ORTHOGONAL_TOL, rtol=0.0):
            raise ValueError("rotation_matrix must be orthogonal (R.T @ R == I)")
        if abs(float(np.linalg.det(rot)) - 1.0) > ORTHOGONAL_TOL:
            raise ValueError("rotation_matrix must have det == +1")
        t = np.asarray(self.translation_m, dtype=np.float64)
        if t.shape != (3,) or not np.all(np.isfinite(t)):
            raise ValueError("translation_m must be a finite 3-vector")
        rot_t: RotationTuple = tuple(tuple(map(float, rot[i])) for i in range(3))  # type: ignore[misc]
        object.__setattr__(self, "source_frame", src)
        object.__setattr__(self, "target_frame", dst)
        object.__setattr__(self, "rotation_matrix", rot_t)
        object.__setattr__(
            self, "translation_m", (float(t[0]), float(t[1]), float(t[2]))
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the 3D frame transform to a JSON-compatible dictionary."""
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "rotation_matrix": [list(r) for r in self.rotation_matrix],
            "translation_m": list(self.translation_m),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameTransform3D:
        """Deserialize a ``FrameTransform3D`` from a JSON mapping."""
        r_raw, t_raw = data["rotation_matrix"], data["translation_m"]
        rot: RotationTuple = tuple(
            (float(r[0]), float(r[1]), float(r[2])) for r in r_raw
        )
        return cls(
            str(data["source_frame"]),
            str(data["target_frame"]),
            rot,
            (float(t_raw[0]), float(t_raw[1]), float(t_raw[2])),
        )


@dataclass(frozen=True)
class CanonicalV2ReferenceTrajectory:
    """Immutable reference joint states and 3D marker trajectories."""

    times_s: NDArray[np.float64]
    dt_s: float
    coordinate_order: tuple[str, ...]
    q: NDArray[np.float64]
    qd: NDArray[np.float64]
    marker_labels: tuple[str, ...]
    marker_positions_m: NDArray[np.float64]
    tau_ff: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate timebase, vocabularies, shapes, and finiteness."""
        if isinstance(self.dt_s, bool) or not isinstance(self.dt_s, (int, float)):
            raise ValueError(f"dt_s must be finite > 0, got {self.dt_s!r}")
        dt, t = float(self.dt_s), np.array(self.times_s, dtype=np.float64, copy=True)
        if (
            not math.isfinite(dt)
            or dt <= 0.0
            or t.ndim != 1
            or t.shape[0] < 2
            or not np.all(np.isfinite(t))
        ):
            raise ValueError(
                "times_s and dt_s must be finite with dt_s > 0 and len >= 2"
            )
        diffs = np.diff(t)
        if not np.all(diffs > 0.0) or not np.allclose(
            diffs, dt, rtol=1e-6, atol=TIMEBASE_DT_TOL
        ):
            raise ValueError("times_s must be strictly increasing and match dt_s")
        t.setflags(write=False)
        cs = _req_labels(tuple(self.coordinate_order), "coordinate_order")
        ms = _req_labels(tuple(self.marker_labels), "marker_labels")
        n, d, m = int(t.shape[0]), len(cs), len(ms)
        q, qd = _frozen_array(self.q, (n, d), "q"), _frozen_array(self.qd, (n, d), "qd")
        pos = _frozen_array(self.marker_positions_m, (n, m, 3), "marker_positions_m")
        tau = (
            None
            if self.tau_ff is None
            else _frozen_array(self.tau_ff, (n, d), "tau_ff")
        )
        names = (
            "times_s",
            "dt_s",
            "coordinate_order",
            "q",
            "qd",
            "marker_labels",
            "marker_positions_m",
            "tau_ff",
        )
        for k, v in zip(names, (t, dt, cs, q, qd, ms, pos, tau), strict=True):
            object.__setattr__(self, k, v)

    def __eq__(self, other: object) -> bool:
        """Compare trajectories by scalar metadata and exact array equality."""
        if not isinstance(other, CanonicalV2ReferenceTrajectory):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Serialize trajectory arrays and vocabularies to JSON primitives."""
        out: dict[str, Any] = {
            "times_s": self.times_s.tolist(),
            "dt_s": self.dt_s,
            "coordinate_order": list(self.coordinate_order),
            "q": self.q.tolist(),
            "qd": self.qd.tolist(),
            "marker_labels": list(self.marker_labels),
            "marker_positions_m": self.marker_positions_m.tolist(),
        }
        if self.tau_ff is not None:
            out["tau_ff"] = self.tau_ff.tolist()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CanonicalV2ReferenceTrajectory:
        """Deserialize a ``CanonicalV2ReferenceTrajectory`` from a JSON mapping."""
        a, t_raw = np.asarray, data.get("tau_ff")
        cs = tuple(str(x) for x in data["coordinate_order"])
        ms = tuple(str(x) for x in data["marker_labels"])
        tau = None if t_raw is None else a(t_raw, dtype=np.float64)
        return cls(
            a(data["times_s"], dtype=np.float64),
            float(data["dt_s"]),
            cs,
            a(data["q"], dtype=np.float64),
            a(data["qd"], dtype=np.float64),
            ms,
            a(data["marker_positions_m"], dtype=np.float64),
            tau,
        )


@dataclass(frozen=True)
class CanonicalV2ReferenceFixture:
    """Frozen, SHA-256-verified canonical-v2 / capture-A reference fixture."""

    schema_version: str
    capture_id: str
    capture_sha256: str
    model_version: str
    spec_revision: str
    plant_hash: str
    mujoco_to_opensim: FrameTransform3D
    trajectory: CanonicalV2ReferenceTrajectory
    fixture_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate schema version, hashes, nested contracts, and freeze metadata."""
        if self.schema_version != SCHEMA_VERSION:
            raise FixtureIntegrityError(f"bad schema_version {self.schema_version!r}")
        if not isinstance(self.mujoco_to_opensim, FrameTransform3D) or not isinstance(
            self.trajectory, CanonicalV2ReferenceTrajectory
        ):
            raise TypeError("invalid mujoco_to_opensim or trajectory type")
        for k in ("capture_id", "model_version", "spec_revision"):
            object.__setattr__(self, k, _req_str(getattr(self, k), k))
        for k in ("capture_sha256", "plant_hash", "fixture_sha256"):
            object.__setattr__(self, k, _req_sha256(getattr(self, k), k))
        object.__setattr__(
            self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata)))
        )

    def to_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        """Serialize fixture to a JSON dictionary (optionally omitting digest)."""
        doc: dict[str, Any] = {
            "schema_version": self.schema_version,
            "capture_id": self.capture_id,
            "capture_sha256": self.capture_sha256,
            "model_version": self.model_version,
            "spec_revision": self.spec_revision,
            "plant_hash": self.plant_hash,
            "mujoco_to_opensim": self.mujoco_to_opensim.to_dict(),
            "trajectory": self.trajectory.to_dict(),
            "metadata": dict(self.metadata),
        }
        if include_sha:
            doc["fixture_sha256"] = self.fixture_sha256
        return doc

    def to_json(self) -> str:
        """Serialize the fixture deterministically to canonical JSON text."""
        return (
            json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> CanonicalV2ReferenceFixture:
        """Deserialize and cryptographically verify a fixture from JSON text."""
        try:
            loaded = json.loads(raw)
            if not isinstance(loaded, dict):
                raise FixtureIntegrityError("fixture JSON root must be an object")
            recorded = _req_sha256(loaded.get("fixture_sha256"), "fixture_sha256")
            payload = {k: v for k, v in loaded.items() if k != "fixture_sha256"}
            if _canonical_sha256(payload) != recorded:
                raise FixtureIntegrityError("fixture_sha256 mismatch")
            g = loaded.__getitem__
            tx = FrameTransform3D.from_dict(g("mujoco_to_opensim"))
            traj = CanonicalV2ReferenceTrajectory.from_dict(g("trajectory"))
            fixture = cls(
                str(g("schema_version")),
                str(g("capture_id")),
                str(g("capture_sha256")),
                str(g("model_version")),
                str(g("spec_revision")),
                str(g("plant_hash")),
                tx,
                traj,
                recorded,
                dict(loaded.get("metadata", {})),
            )
        except FixtureIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise FixtureIntegrityError(f"malformed fixture: {exc}") from exc
        verify_fixture_integrity(fixture)
        return fixture


def verify_fixture_integrity(fixture: CanonicalV2ReferenceFixture) -> None:
    """Verify schema version and ``fixture_sha256`` against ``fixture``'s payload."""
    if fixture.schema_version != SCHEMA_VERSION:
        raise FixtureIntegrityError(f"bad schema_version {fixture.schema_version!r}")
    if fixture.fixture_sha256 != _canonical_sha256(fixture.to_dict(include_sha=False)):
        raise FixtureIntegrityError("fixture_sha256 mismatch")


def build_canonical_v2_reference_fixture(
    spec: Mapping[str, Any],
    trajectory: CanonicalV2ReferenceTrajectory,
    mujoco_to_opensim: FrameTransform3D,
    capture_sha256: str,
    *,
    capture_id: str = DEFAULT_CAPTURE_ID,
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalV2ReferenceFixture:
    """Build and SHA-256-seal a canonical-v2 / capture-A reference fixture."""
    cur = require_canonical_golfer(spec)
    meta = dict(metadata) if metadata is not None else {}
    seed = CanonicalV2ReferenceFixture(
        SCHEMA_VERSION,
        capture_id,
        capture_sha256,
        cur.version,
        cur.spec_revision_configuration,
        compute_plant_hash(spec),
        mujoco_to_opensim,
        trajectory,
        "a" * 64,
        meta,
    )
    fixture = replace(
        seed, fixture_sha256=_canonical_sha256(seed.to_dict(include_sha=False))
    )
    verify_fixture_integrity(fixture)
    return fixture
