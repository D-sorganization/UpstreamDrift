"""Canonical club observation contracts with masks and native clock (CO-01 #10605).

Extends the measured-club surface beyond legacy :class:`ClubTarget` without
replacing it. Mid-hands and face frames are explicit; missing orientation or
twist is unobserved (NaN), never identity. Resampling is a recorded operation;
the native 240 Hz clock remains available for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.core.vector_math import row_euclidean_norm
from src.shared.python.math_utils.quaternion import rotmat_to_quat, slerp
from src.shared.python.motion_matching._geodesic import quaternion_geodesic_angles
from src.shared.python.motion_matching.club_models import CLUBS
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
    EXPECTED_EVENT_SAMPLES,
    EXPECTED_SAMPLE_COUNTS,
    NATIVE_SAMPLE_RATE_HZ,
)
from src.shared.python.motion_matching.club_target import (
    MAX_POSITION_NORM_M,
    QUAT_NORM_TOL,
    SourceProvenance,
)

OBSERVATION_SCHEMA = "club-observation-contracts/1.0.0"
_ROT_DET_TOL = 1.0e-3
_ROT_ORTHO_TOL = 1.0e-3


class ComponentStatus(str, Enum):
    """Whether a component is measured, derived, or unobserved."""

    MEASURED = "measured"
    DERIVED = "derived_not_measured"
    UNOBSERVED = "unobserved"


@dataclass(frozen=True)
class ComponentMask:
    """Per-component observation status for a club trial."""

    mid_hands_position: ComponentStatus
    face_position: ComponentStatus
    mid_hands_orientation: ComponentStatus
    face_orientation: ComponentStatus
    twist: ComponentStatus

    def as_dict(self) -> dict[str, str]:
        return {
            "mid_hands_position": self.mid_hands_position.value,
            "face_position": self.face_position.value,
            "mid_hands_orientation": self.mid_hands_orientation.value,
            "face_orientation": self.face_orientation.value,
            "twist": self.twist.value,
        }


@dataclass(frozen=True)
class DerivationMetadata:
    """Provenance for derived axes and degeneracy flags."""

    orientation_axis_status: str
    degenerate_axes: bool
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "orientation_axis_status": self.orientation_axis_status,
            "degenerate_axes": self.degenerate_axes,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class UncertaintyMetadata:
    """Scalar uncertainty metadata attached to an observation package."""

    position_sigma_m: float
    orientation_sigma_rad: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_sigma_m) or self.position_sigma_m < 0.0:
            raise ValueError("position_sigma_m must be finite and >= 0")
        if (
            not np.isfinite(self.orientation_sigma_rad)
            or self.orientation_sigma_rad < 0.0
        ):
            raise ValueError("orientation_sigma_rad must be finite and >= 0")

    def as_dict(self) -> dict[str, float]:
        return {
            "position_sigma_m": float(self.position_sigma_m),
            "orientation_sigma_rad": float(self.orientation_sigma_rad),
        }


@dataclass(frozen=True)
class ObservationEvent:
    """Named event on the native observation clock."""

    label: str
    sample_index: int
    time_s: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ObservationEvent.label must be non-empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be >= 0")


@dataclass(frozen=True)
class ResampleRecord:
    """Saved resampling operation; native clock remains authoritative."""

    method: str
    source_rate_hz: float
    target_rate_hz: float
    params_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "source_rate_hz": self.source_rate_hz,
            "target_rate_hz": self.target_rate_hz,
            "params_hash": self.params_hash,
        }


@dataclass(frozen=True)
class ClubObservationKinematics:
    """Frame arrays used to construct a :class:`ClubObservation`."""

    native_time_s: np.ndarray
    mid_hands_xyz: np.ndarray
    face_xyz: np.ndarray
    mid_hands_rotmats: np.ndarray
    face_rotmats: np.ndarray


@dataclass(frozen=True)
class ClubObservationProvenance:
    """Identity, masks, and catalog metadata for a club observation."""

    mask: ComponentMask
    derivation: DerivationMetadata
    uncertainty: UncertaintyMetadata
    events: tuple[ObservationEvent, ...]
    sample_rate_hz: float
    club_type: str
    catalog_length_m: float
    source: SourceProvenance
    trial_id: str


@dataclass(frozen=True)
class ClubObservation:
    """Canonical club observation with explicit mid-hands/face frames.

    Positions are metres. Orientations are unit quaternions ``[w,x,y,z]`` when
    observed; unobserved orientation rows are all-NaN and must not be treated
    as identity. ``native_time_s`` preserves the source clock (typically
    impact-relative 240 Hz).
    """

    native_time_s: np.ndarray
    mid_hands_xyz: np.ndarray
    face_xyz: np.ndarray
    mid_hands_quat: np.ndarray
    face_quat: np.ndarray
    mask: ComponentMask
    derivation: DerivationMetadata
    uncertainty: UncertaintyMetadata
    events: tuple[ObservationEvent, ...]
    sample_rate_hz: float
    club_type: str
    catalog_length_m: float
    source: SourceProvenance
    trial_id: str
    resample: ResampleRecord | None = None
    schema: str = OBSERVATION_SCHEMA
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_observation(self)

    @classmethod
    @precondition(
        lambda cls, kinematics, provenance: isinstance(
            kinematics, ClubObservationKinematics
        ),
        "kinematics must be ClubObservationKinematics",
    )
    @precondition(
        lambda cls, kinematics, provenance: isinstance(
            provenance, ClubObservationProvenance
        ),
        "provenance must be ClubObservationProvenance",
    )
    @postcondition(
        lambda r: isinstance(r, ClubObservation), "must return ClubObservation"
    )
    def from_frames(
        cls,
        kinematics: ClubObservationKinematics,
        provenance: ClubObservationProvenance,
    ) -> ClubObservation:
        """Build an observation from rotation matrices with SO(3) validation."""
        mid_r = _validate_rotation_stack(kinematics.mid_hands_rotmats, "mid_hands")
        face_r = _validate_rotation_stack(kinematics.face_rotmats, "face")
        return cls(
            native_time_s=kinematics.native_time_s,
            mid_hands_xyz=kinematics.mid_hands_xyz,
            face_xyz=kinematics.face_xyz,
            mid_hands_quat=rotmat_to_quat(mid_r),
            face_quat=rotmat_to_quat(face_r),
            mask=provenance.mask,
            derivation=provenance.derivation,
            uncertainty=provenance.uncertainty,
            events=provenance.events,
            sample_rate_hz=provenance.sample_rate_hz,
            club_type=provenance.club_type,
            catalog_length_m=provenance.catalog_length_m,
            source=provenance.source,
            trial_id=provenance.trial_id,
        )


def _validate_rotation_stack(rotmats: np.ndarray, name: str) -> np.ndarray:
    """Reject non-proper rotations (det != +1 or non-orthonormal)."""
    arr = np.asarray(rotmats, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"{name} rotmats must have shape (N, 3, 3), got {arr.shape}")
    for i, r in enumerate(arr):
        if not np.all(np.isfinite(r)):
            raise ValueError(f"{name} rotmat[{i}] contains non-finite values")
        det = float(np.linalg.det(r))
        if abs(det - 1.0) > _ROT_DET_TOL:
            raise ValueError(
                f"{name} rotmat[{i}] is not a proper SO(3) rotation "
                f"(determinant {det:.6f})"
            )
        ortho = r @ r.T
        if not np.allclose(ortho, np.eye(3), atol=_ROT_ORTHO_TOL):
            raise ValueError(f"{name} rotmat[{i}] is not orthonormal")
    return arr


def _validate_time(time: np.ndarray) -> int:
    if not isinstance(time, np.ndarray) or time.ndim != 1:
        raise ValueError("native_time_s must be a 1-D ndarray")
    n = int(time.shape[0])
    if n < 2:
        raise ValueError(f"native_time_s must have at least 2 samples (got {n})")
    if not np.all(np.isfinite(time)):
        raise ValueError("native_time_s must be finite")
    if not np.all(np.diff(time) > 0):
        raise ValueError("native_time_s must be strictly increasing")
    return n


def _validate_positions(
    name: str, arr: np.ndarray, n: int, status: ComponentStatus
) -> None:
    if arr.shape != (n, 3):
        raise ValueError(f"{name} must have shape ({n}, 3), got {arr.shape}")
    if status is ComponentStatus.UNOBSERVED:
        if not np.all(np.isnan(arr)):
            raise ValueError(f"{name} marked unobserved must be all-NaN")
        return
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    norms = row_euclidean_norm(arr)
    if np.any(norms >= MAX_POSITION_NORM_M):
        raise ValueError(
            f"{name} has |r| >= {MAX_POSITION_NORM_M} m (max {float(norms.max()):.3f})"
        )


def _validate_quats(
    name: str, arr: np.ndarray, n: int, status: ComponentStatus
) -> None:
    if arr.shape != (n, 4):
        raise ValueError(f"{name} must have shape ({n}, 4), got {arr.shape}")
    if status is ComponentStatus.UNOBSERVED:
        if np.any(np.isfinite(arr)):
            raise ValueError(
                f"{name} marked unobserved must be all-NaN "
                "(missing orientation is not identity)"
            )
        return
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf while marked {status.value}")
    qnorms = row_euclidean_norm(arr)
    if np.any(np.abs(qnorms - 1.0) > QUAT_NORM_TOL):
        max_dev = float(np.abs(qnorms - 1.0).max())
        raise ValueError(
            f"{name} rows must be unit-norm to within {QUAT_NORM_TOL} "
            f"(max deviation {max_dev:.2e})"
        )


def _validate_observation(obs: ClubObservation) -> None:
    if obs.schema != OBSERVATION_SCHEMA:
        raise ValueError(f"unsupported observation schema {obs.schema!r}")
    if not isinstance(obs.source, SourceProvenance):
        raise TypeError("source must be a SourceProvenance instance")
    if not isinstance(obs.mask, ComponentMask):
        raise TypeError("mask must be a ComponentMask")
    if obs.sample_rate_hz <= 0 or not np.isfinite(obs.sample_rate_hz):
        raise ValueError("sample_rate_hz must be finite and > 0")
    if obs.catalog_length_m <= 0 or not np.isfinite(obs.catalog_length_m):
        raise ValueError("catalog_length_m must be finite and > 0")
    n = _validate_time(obs.native_time_s)
    _validate_positions(
        "mid_hands_xyz", obs.mid_hands_xyz, n, obs.mask.mid_hands_position
    )
    _validate_positions("face_xyz", obs.face_xyz, n, obs.mask.face_position)
    _validate_quats(
        "mid_hands_quat", obs.mid_hands_quat, n, obs.mask.mid_hands_orientation
    )
    _validate_quats("face_quat", obs.face_quat, n, obs.mask.face_orientation)
    for event in obs.events:
        if event.sample_index >= n:
            raise ValueError(
                f"event {event.label!r} sample_index {event.sample_index} "
                f"out of range for N={n}"
            )


@precondition(
    lambda q_a, q_b: np.asarray(q_a).shape == np.asarray(q_b).shape,
    "quaternion arrays must share shape",
)
@postcondition(
    lambda result: isinstance(result, np.ndarray) and result.ndim == 1,
    "SO(3) residual must be a 1-D ndarray",
)
def orientation_residual_so3(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Sign-invariant geodesic residual between quaternion rows (radians)."""
    return quaternion_geodesic_angles(
        np.asarray(q_a, dtype=np.float64),
        np.asarray(q_b, dtype=np.float64),
    )


@precondition(
    lambda obs: isinstance(obs, ClubObservation),
    "obs must be a ClubObservation",
)
@postcondition(
    lambda result: isinstance(result, frozenset),
    "scored subset must be a frozenset",
)
def scored_component_subset(obs: ClubObservation) -> frozenset[str]:
    """Return the component names a provider may score (measured or derived)."""
    scored: set[str] = set()
    mapping = {
        "mid_hands_position": obs.mask.mid_hands_position,
        "face_position": obs.mask.face_position,
        "mid_hands_orientation": obs.mask.mid_hands_orientation,
        "face_orientation": obs.mask.face_orientation,
        "twist": obs.mask.twist,
    }
    for name, status in mapping.items():
        if status is not ComponentStatus.UNOBSERVED:
            scored.add(name)
    return frozenset(scored)


def _slerp_series(
    query_t: np.ndarray, raw_t: np.ndarray, raw_q: np.ndarray
) -> np.ndarray:
    out = np.empty((query_t.shape[0], 4), dtype=np.float64)
    last = raw_t.shape[0] - 1
    for i, t in enumerate(query_t):
        if t <= raw_t[0]:
            out[i] = raw_q[0]
            continue
        if t >= raw_t[last]:
            out[i] = raw_q[last]
            continue
        j = int(np.searchsorted(raw_t, t)) - 1
        j = max(0, min(j, last - 1))
        span = raw_t[j + 1] - raw_t[j]
        alpha = 0.0 if span == 0.0 else (t - raw_t[j]) / span
        out[i] = slerp(raw_q[j], raw_q[j + 1], float(alpha))
    norms = np.sqrt(np.einsum("ij,ij->i", out, out))[:, np.newaxis]
    norms[norms == 0.0] = 1.0
    return out / norms


def _interp_xyz(
    query_t: np.ndarray, raw_t: np.ndarray, raw_xyz: np.ndarray
) -> np.ndarray:
    out = np.empty((query_t.shape[0], 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(query_t, raw_t, raw_xyz[:, k])
    return out


@precondition(
    lambda obs, sample_rate_hz: sample_rate_hz > 0,
    "sample_rate_hz must be > 0",
)
@postcondition(
    lambda result: isinstance(result, ClubObservation),
    "interpolation must return ClubObservation",
)
def interpolate_observation(
    obs: ClubObservation, sample_rate_hz: float
) -> ClubObservation:
    """SO(3)-appropriate densification; records a resample operation."""
    if obs.mask.face_orientation is ComponentStatus.UNOBSERVED:
        raise ValueError("cannot interpolate unobserved face orientation")
    if obs.mask.mid_hands_orientation is ComponentStatus.UNOBSERVED:
        raise ValueError("cannot interpolate unobserved mid-hands orientation")
    t0 = float(obs.native_time_s[0])
    t1 = float(obs.native_time_s[-1])
    n_out = int(round((t1 - t0) * sample_rate_hz)) + 1
    query_t = t0 + np.arange(n_out, dtype=np.float64) / float(sample_rate_hz)
    query_t[-1] = t1
    mid_q = _slerp_series(query_t, obs.native_time_s, obs.mid_hands_quat)
    face_q = _slerp_series(query_t, obs.native_time_s, obs.face_quat)
    params = (
        f"slerp|{obs.sample_rate_hz}->{sample_rate_hz}|"
        f"{obs.native_time_s.shape[0]}->{n_out}"
    )
    return ClubObservation(
        native_time_s=query_t,
        mid_hands_xyz=_interp_xyz(query_t, obs.native_time_s, obs.mid_hands_xyz),
        face_xyz=_interp_xyz(query_t, obs.native_time_s, obs.face_xyz),
        mid_hands_quat=mid_q,
        face_quat=face_q,
        mask=obs.mask,
        derivation=obs.derivation,
        uncertainty=obs.uncertainty,
        events=obs.events,
        sample_rate_hz=float(sample_rate_hz),
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        source=obs.source,
        trial_id=obs.trial_id,
        resample=ResampleRecord(
            method="slerp_positions_linear",
            source_rate_hz=obs.sample_rate_hz,
            target_rate_hz=float(sample_rate_hz),
            params_hash=str(abs(hash(params))),
        ),
    )


@precondition(
    lambda path: isinstance(path, (str, Path)),
    "path must be a filesystem path",
)
@postcondition(
    lambda result: result.get("schema") == OBSERVATION_SCHEMA,
    "fixture pack must declare club-observation-contracts/1.0.0",
)
def load_observation_fixture_pack(path: Path | str) -> dict[str, Any]:
    """Load the versioned four-trial observation fixture pack."""
    pack_path = Path(path)
    if not pack_path.is_file():
        raise FileNotFoundError(f"observation fixture pack not found: {pack_path}")
    with pack_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("fixture pack must be a JSON object")
    return data


def _synthetic_series(
    n: int, length_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = (np.arange(n, dtype=np.float64) - (n // 2)) / NATIVE_SAMPLE_RATE_HZ
    mid = np.zeros((n, 3), dtype=np.float64)
    mid[:, 0] = 0.01 * np.sin(np.linspace(0.0, np.pi, n))
    face = mid.copy()
    face[:, 1] = length_m
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return time, mid, face, quat


@precondition(
    lambda trial_id: trial_id in CANONICAL_TRIAL_SHEETS,
    "trial_id must be one of the four canonical club-only sheets",
)
@postcondition(
    lambda result: (
        isinstance(result, ClubObservation)
        and result.sample_rate_hz == NATIVE_SAMPLE_RATE_HZ
    ),
    "fixture must be ClubObservation at native 240 Hz",
)
def build_calibrated_observation_fixture(trial_id: str) -> ClubObservation:
    """Synthetic calibrated fixture for ``trial_id`` (software contract only)."""
    spec = CLUBS["driver"]
    n = min(32, int(EXPECTED_SAMPLE_COUNTS[trial_id]))
    time, mid, face, quat = _synthetic_series(n, spec.length_m)
    event_samples = EXPECTED_EVENT_SAMPLES[trial_id]
    full_n = float(EXPECTED_SAMPLE_COUNTS[trial_id])
    events: list[ObservationEvent] = []
    for label, sample in event_samples.items():
        if label == "CHS":
            continue
        idx = int(round((float(sample) - 1.0) / (full_n - 1.0) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        events.append(
            ObservationEvent(label=label, sample_index=idx, time_s=float(time[idx]))
        )
    return ClubObservation(
        native_time_s=time,
        mid_hands_xyz=mid,
        face_xyz=face,
        mid_hands_quat=quat,
        face_quat=quat.copy(),
        mask=ComponentMask(
            mid_hands_position=ComponentStatus.MEASURED,
            face_position=ComponentStatus.MEASURED,
            mid_hands_orientation=ComponentStatus.MEASURED,
            face_orientation=ComponentStatus.DERIVED,
            twist=ComponentStatus.UNOBSERVED,
        ),
        derivation=DerivationMetadata(
            orientation_axis_status="derived_not_measured",
            degenerate_axes=False,
            notes=("synthetic fixture for CO-01 software contracts",),
        ),
        uncertainty=UncertaintyMetadata(
            position_sigma_m=0.001,
            orientation_sigma_rad=0.02,
        ),
        events=tuple(events),
        sample_rate_hz=NATIVE_SAMPLE_RATE_HZ,
        club_type=spec.name,
        catalog_length_m=spec.length_m,
        source=SourceProvenance(
            filename="club_observation_contracts.json",
            format="synthetic",
            subject_id=trial_id.split("_")[0],
            trial_id=trial_id,
            sha256="synthetic-fixture",
        ),
        trial_id=trial_id,
    )


def build_four_trial_fixture_pack() -> dict[str, Any]:
    """Versioned operator inputs declaring coverage for all four trials."""
    trials: dict[str, Any] = {}
    for trial_id in CANONICAL_TRIAL_SHEETS:
        obs = build_calibrated_observation_fixture(trial_id)
        trials[trial_id] = {
            "trial_id": trial_id,
            "sample_rate_hz": obs.sample_rate_hz,
            "catalog_club_type": obs.club_type,
            "catalog_length_m": obs.catalog_length_m,
            "events": {e.label: e.sample_index for e in obs.events},
            "component_coverage": obs.mask.as_dict(),
            "derivation": obs.derivation.as_dict(),
            "uncertainty": obs.uncertainty.as_dict(),
            "native_clock": "impact-relative 240 Hz (synthetic densified window)",
            "note": (
                "Synthetic calibrated fixture for software contracts; "
                "not native physical evidence."
            ),
        }
    return {
        "schema": OBSERVATION_SCHEMA,
        "native_sample_rate_hz": NATIVE_SAMPLE_RATE_HZ,
        "governing_issue": 10605,
        "trials": trials,
    }


def write_observation_fixture_pack(path: Path | str) -> Path:
    """Write the four-trial fixture pack to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_four_trial_fixture_pack()
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out
