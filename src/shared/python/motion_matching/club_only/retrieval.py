"""Reference-library retrieval for club-only starting guesses (CO-03 #10607).

Indexes library descriptors by model/geometry and observable club motion.
Applies one saved rigid placement and bounded event-aware initialization.
Body motion remains a named prior; no per-frame registration or time warp.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_only.observation import ClubObservation
from src.shared.python.motion_matching.club_only.profiles import ClubOnlyProfile

__all__ = [
    "LibraryEntry",
    "ObservableClubDescriptor",
    "RigidPlacement",
    "build_observable_descriptor",
    "retrieve_starting_seeds",
    "validate_library_not_query_trial",
]


@dataclass(frozen=True)
class RigidPlacement:
    """One fixed proper-rigid placement applied for a whole retrieval seed."""

    rotation: NDArray[np.floating]
    translation_m: NDArray[np.floating]
    is_single_rigid: bool = True

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must be (3, 3), got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"translation_m must be (3,), got {translation.shape}")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("placement values must be finite")
        det = float(np.linalg.det(rotation))
        if abs(det - 1.0) > 1.0e-5:
            raise ValueError(f"rotation must be proper SO(3) (det={det})")
        if not self.is_single_rigid:
            raise ValueError("retrieval placement must be a single rigid transform")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)

    @classmethod
    def identity(cls) -> RigidPlacement:
        return cls(rotation=np.eye(3), translation_m=np.zeros(3), is_single_rigid=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rotation": self.rotation.tolist(),
            "translation_m": self.translation_m.tolist(),
            "is_single_rigid": self.is_single_rigid,
        }


@dataclass(frozen=True)
class ObservableClubDescriptor:
    """Indexed club-motion descriptor for library retrieval."""

    trial_id: str
    model_id: str
    geometry_hash: str
    event_phase_key: str
    mid_hands_centroid_m: tuple[float, float, float]
    face_centroid_m: tuple[float, float, float]
    duration_s: float
    sample_rate_hz: float

    def __post_init__(self) -> None:
        if not self.trial_id or not self.model_id or not self.geometry_hash:
            raise ValueError("trial_id, model_id, geometry_hash must be non-empty")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and > 0")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and > 0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "model_id": self.model_id,
            "geometry_hash": self.geometry_hash,
            "event_phase_key": self.event_phase_key,
            "mid_hands_centroid_m": list(self.mid_hands_centroid_m),
            "face_centroid_m": list(self.face_centroid_m),
            "duration_s": self.duration_s,
            "sample_rate_hz": self.sample_rate_hz,
        }


@dataclass(frozen=True)
class LibraryEntry:
    """One qualified reference entry; body configuration is a prior only."""

    reference_id: str
    descriptor: ObservableClubDescriptor
    body_q0: NDArray[np.floating]
    rigid_placement: RigidPlacement
    source_clock_times_s: NDArray[np.floating]
    body_is_prior: bool = True

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError("reference_id must be non-empty")
        if not self.body_is_prior:
            raise ValueError("retrieved body motion must remain a prior")
        q0 = np.asarray(self.body_q0, dtype=np.float64)
        times = np.asarray(self.source_clock_times_s, dtype=np.float64)
        if q0.ndim != 1 or q0.size == 0 or not np.all(np.isfinite(q0)):
            raise ValueError("body_q0 must be a non-empty finite 1-D array")
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("source_clock_times_s must be finite 1-D")
        if times.size >= 2 and np.any(np.diff(times) <= 0.0):
            raise ValueError("source clock must be strictly increasing")
        object.__setattr__(self, "body_q0", q0)
        object.__setattr__(self, "source_clock_times_s", times)


def _event_phase_key(obs: ClubObservation) -> str:
    if not obs.events:
        return "no_events"
    labels = sorted({event.label for event in obs.events})
    return "+".join(labels)


@precondition(
    lambda obs, model_id, geometry_hash: (
        isinstance(obs, ClubObservation) and bool(model_id) and bool(geometry_hash)
    ),
    "ClubObservation, model_id, and geometry_hash required",
)
@postcondition(
    lambda result: isinstance(result, ObservableClubDescriptor),
    "must return ObservableClubDescriptor",
)
def build_observable_descriptor(
    obs: ClubObservation,
    *,
    model_id: str,
    geometry_hash: str,
) -> ObservableClubDescriptor:
    """Build a retrieval descriptor from measured club kinematics only."""
    mid = np.asarray(obs.mid_hands_xyz, dtype=np.float64)
    face = np.asarray(obs.face_xyz, dtype=np.float64)
    if mid.ndim != 2 or mid.shape[1] != 3 or face.shape != mid.shape:
        raise ValueError("mid_hands_xyz and face_xyz must share shape (N, 3)")
    if not np.all(np.isfinite(mid)) or not np.all(np.isfinite(face)):
        raise ValueError("club positions must be finite for descriptor build")
    times = np.asarray(obs.native_time_s, dtype=np.float64)
    duration = float(times[-1] - times[0]) if times.size else 0.0
    mid_c = (
        float(np.mean(mid[:, 0])),
        float(np.mean(mid[:, 1])),
        float(np.mean(mid[:, 2])),
    )
    face_c = (
        float(np.mean(face[:, 0])),
        float(np.mean(face[:, 1])),
        float(np.mean(face[:, 2])),
    )
    return ObservableClubDescriptor(
        trial_id=obs.trial_id,
        model_id=model_id,
        geometry_hash=geometry_hash,
        event_phase_key=_event_phase_key(obs),
        mid_hands_centroid_m=mid_c,
        face_centroid_m=face_c,
        duration_s=duration,
        sample_rate_hz=float(obs.sample_rate_hz),
    )


def _descriptor_distance(
    left: ObservableClubDescriptor, right: ObservableClubDescriptor
) -> float:
    if left.model_id != right.model_id or left.geometry_hash != right.geometry_hash:
        return float("inf")
    mid = np.asarray(left.mid_hands_centroid_m) - np.asarray(right.mid_hands_centroid_m)
    face = np.asarray(left.face_centroid_m) - np.asarray(right.face_centroid_m)
    duration_pen = abs(left.duration_s - right.duration_s)
    phase_pen = 0.0 if left.event_phase_key == right.event_phase_key else 1.0
    return float(np.linalg.norm(mid) + np.linalg.norm(face) + duration_pen + phase_pen)


def _body_hash(q: NDArray[np.floating]) -> str:
    return hashlib.sha256(np.asarray(q, dtype=np.float64).tobytes()).hexdigest()


def validate_library_not_query_trial(
    library: Sequence[LibraryEntry],
    query_trial_id: str,
) -> None:
    """Raise ``ValueError`` if any library entry was built from the query trial.

    A seed retrieved from the query's own descriptor is the target copied as a
    fit (#10960 P1-2), so every retrieval entry point calls this first.
    """
    if not query_trial_id:
        raise ValueError("query trial_id must be non-empty")
    for entry in library:
        if entry.descriptor.trial_id == query_trial_id:
            raise ValueError(
                f"library entry {entry.reference_id!r} was built from the query "
                f"trial {query_trial_id!r} (trial leakage)"
            )


@precondition(
    lambda observation, profile, library, geometry_hash, profile_hash, max_seeds=2: (
        isinstance(observation, ClubObservation)
        and isinstance(profile, ClubOnlyProfile)
        and max_seeds >= 1
    ),
    "observation, profile, and max_seeds>=1 required",
)
@postcondition(
    lambda result: isinstance(result, tuple) and len(result) >= 1,
    "must return non-empty seed tuple",
)
def retrieve_starting_seeds(
    *,
    observation: ClubObservation,
    profile: ClubOnlyProfile,
    library: Sequence[LibraryEntry],
    geometry_hash: str,
    profile_hash: str,
    max_seeds: int = 2,
) -> tuple[Any, ...]:
    """Return bounded retrieval-only seeds with one rigid placement each."""
    # Local import avoids circular init with seeds.py.
    from src.shared.python.motion_matching.club_only.seeds import CandidateSeed

    if not library:
        raise ValueError("library must be non-empty")
    validate_library_not_query_trial(library, observation.trial_id)
    query = build_observable_descriptor(
        observation,
        model_id=profile.model_id,
        geometry_hash=geometry_hash,
    )
    ranked = sorted(
        library,
        key=lambda entry: _descriptor_distance(query, entry.descriptor),
    )
    seeds: list[Any] = []
    for entry in ranked[:max_seeds]:
        if entry.descriptor.model_id != profile.model_id:
            continue
        if entry.descriptor.geometry_hash != geometry_hash:
            continue
        if entry.source_clock_times_s.shape != observation.native_time_s.shape:
            raise ValueError(
                "library entry clock length must match observation native clock; "
                "time warp is not authorized"
            )
        if not np.allclose(entry.source_clock_times_s, observation.native_time_s):
            raise ValueError(
                "library entry timestamps must equal observation native clock; "
                "hidden alignment is not authorized"
            )
        residual = _descriptor_distance(query, entry.descriptor)
        if not np.isfinite(residual):
            continue
        seeds.append(
            CandidateSeed(
                seed_id=f"retrieval:{entry.reference_id}",
                trial_id=observation.trial_id,
                model_id=profile.model_id,
                source="retrieval",
                q=entry.body_q0.copy(),
                body_configuration_hash=_body_hash(entry.body_q0),
                observed_residual_m=float(residual),
                prior_score=0.6,
                feasibility_reasons=(
                    "single_rigid_placement",
                    "event_aware_descriptor_match",
                    "body_motion_is_prior",
                ),
                timestamps_s=observation.native_time_s.copy(),
                geometry_hash=geometry_hash,
                profile_hash=profile_hash,
                placement=entry.rigid_placement,
                body_is_prior=True,
                is_kinematic_preview=True,
                claims_torque=False,
                claims_physiological_inference=False,
            )
        )
    if not seeds:
        raise ValueError("no library entries matched model/geometry for retrieval")
    return tuple(seeds)
