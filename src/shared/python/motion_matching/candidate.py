"""Unified MatchedSwingCandidate versioned package contract (MS-15, #10334).

Provides:
1. CandidateProfile: Kinematic vs Dynamic candidate profiles.
2. CandidateMetadata: Provenance, hashing, coordinate/effort conventions, SI units.
3. MatchedSwingCandidate: Immutable, checksummed, multi-engine candidate representation.
4. Virtual-work consistency verifier across generalized coordinates and actuator mappings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import logging
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition

logger = logging.getLogger(__name__)

CANDIDATE_SCHEMA_VERSION = "matched-swing-candidate-v1"

DEFAULT_UNITS: dict[str, str] = {
    "time": "s",
    "position": "m",
    "angle": "rad",
    "linear_velocity": "m/s",
    "angular_velocity": "rad/s",
    "force": "N",
    "torque": "N*m",
}


class CandidateProfile(str, Enum):
    """Candidate representation profile.

    - KINEMATIC: Trajectory of coordinates (and optionally velocities/markers).
      Does not contain actuator efforts; rejected for dynamic full-swing acceptance.
    - DYNAMIC: Fully forward-replayable trajectory containing initial states,
      complete controls/efforts, solver configuration, contact and closure parameters.
    """

    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"


def _array_sha256(arr: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash of a numpy array's contiguous byte representation."""
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


@dataclass(frozen=True)
class CandidateMetadata:
    """Immutable provenance and physical convention metadata for a candidate package."""

    schema_version: str = CANDIDATE_SCHEMA_VERSION
    profile: CandidateProfile = CandidateProfile.KINEMATIC
    engine: str = "unknown"
    model_name: str = ""
    model_sha256: str = ""
    source_c3d_sha256: str | None = None
    document_sha256: str | None = None
    coordinate_names: tuple[str, ...] = ()
    velocity_names: tuple[str, ...] = ()
    actuator_names: tuple[str, ...] = ()
    marker_names: tuple[str, ...] = ()
    units: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UNITS))
    frame_convention: str = "z_up_y_forward"
    interpolation: str = "cubic_spline"
    event_indices: dict[str, int] = field(default_factory=dict)
    coverage_mask: dict[str, Any] = field(default_factory=dict)
    solver_settings: dict[str, Any] = field(default_factory=dict)
    contact_parameters: dict[str, Any] = field(default_factory=dict)
    closure_parameters: dict[str, Any] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {self.schema_version!r} "
                f"(expected {CANDIDATE_SCHEMA_VERSION!r})"
            )
        if isinstance(self.profile, str):
            object.__setattr__(self, "profile", CandidateProfile(self.profile))

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a clean JSON-serializable dictionary."""
        d = asdict(self)
        d["profile"] = self.profile.value
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateMetadata:
        """Construct CandidateMetadata from a dictionary."""
        d = dict(data)
        if "profile" in d and isinstance(d["profile"], str):
            d["profile"] = CandidateProfile(d["profile"])
        if "coordinate_names" in d:
            d["coordinate_names"] = tuple(d["coordinate_names"])
        if "velocity_names" in d:
            d["velocity_names"] = tuple(d["velocity_names"])
        if "actuator_names" in d:
            d["actuator_names"] = tuple(d["actuator_names"])
        if "marker_names" in d:
            d["marker_names"] = tuple(d["marker_names"])
        return cls(**d)


class MatchedSwingCandidate:
    """Unified, immutable, and checksum-verified matched swing candidate."""

    def __init__(
        self,
        metadata: CandidateMetadata,
        time_s: np.ndarray | Sequence[float],
        q: np.ndarray,
        v: np.ndarray | None = None,
        tau: np.ndarray | None = None,
        actuator_states: np.ndarray | None = None,
        model_markers_m: np.ndarray | None = None,
        target_markers_m: np.ndarray | None = None,
        marker_validity: np.ndarray | None = None,
        external_forces: np.ndarray | None = None,
        *,
        compute_checksums: bool = True,
    ) -> None:
        if metadata.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {metadata.schema_version!r} "
                f"(expected {CANDIDATE_SCHEMA_VERSION!r})"
            )

        t_arr = np.asarray(time_s, dtype=np.float64)
        if t_arr.ndim != 1:
            raise ValueError(f"time_s must be 1D array, got ndim={t_arr.ndim}")
        n_frames = len(t_arr)
        if n_frames < 2:
            raise ValueError(f"Candidate requires at least 2 frames, got {n_frames}")
        if not np.all(np.isfinite(t_arr)):
            raise ValueError("time_s contains non-finite (NaN or Inf) values")
        if t_arr[0] < 0.0:
            raise ValueError(f"Initial time must be non-negative, got {t_arr[0]}")
        if np.any(np.diff(t_arr) <= 0.0):
            raise ValueError("time_s must be strictly monotonically increasing")

        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.ndim != 2:
            raise ValueError(
                f"q must be 2D array of shape (N, nq), got shape {q_arr.shape}"
            )
        if q_arr.shape[0] != n_frames:
            raise ValueError(
                f"length mismatch: q frames {q_arr.shape[0]} != time_s frames {n_frames}"
            )
        if metadata.coordinate_names and q_arr.shape[1] != len(
            metadata.coordinate_names
        ):
            raise ValueError(
                f"q column count {q_arr.shape[1]} != coordinate_names count {len(metadata.coordinate_names)}"
            )

        v_arr: np.ndarray | None = None
        if v is not None:
            v_arr = np.asarray(v, dtype=np.float64)
            if v_arr.ndim != 2 or v_arr.shape[0] != n_frames:
                raise ValueError(
                    f"v must be 2D array with {n_frames} frames, got shape {v_arr.shape}"
                )
            if metadata.velocity_names and v_arr.shape[1] != len(
                metadata.velocity_names
            ):
                raise ValueError(
                    f"v column count {v_arr.shape[1]} != velocity_names count {len(metadata.velocity_names)}"
                )

        tau_arr: np.ndarray | None = None
        if tau is not None:
            tau_arr = np.asarray(tau, dtype=np.float64)
            if tau_arr.ndim != 2 or tau_arr.shape[0] != n_frames:
                raise ValueError(
                    f"tau must be 2D array with {n_frames} frames, got shape {tau_arr.shape}"
                )
            if metadata.actuator_names and tau_arr.shape[1] != len(
                metadata.actuator_names
            ):
                raise ValueError(
                    f"tau column count {tau_arr.shape[1]} != actuator_names count {len(metadata.actuator_names)}"
                )

        # Profile checks
        if metadata.profile == CandidateProfile.DYNAMIC:
            if v_arr is None:
                raise ValueError("Dynamic candidate profile requires velocity array v")
            if tau_arr is None:
                raise ValueError(
                    "Dynamic candidate profile requires actuator effort array tau"
                )
        elif metadata.profile == CandidateProfile.KINEMATIC:
            if tau_arr is not None:
                raise ValueError(
                    "Kinematic candidate profile must not carry actuator effort tau"
                )

        # Marker checks
        m_arr: np.ndarray | None = None
        if model_markers_m is not None:
            m_arr = np.asarray(model_markers_m, dtype=np.float64)
            if m_arr.ndim != 3 or m_arr.shape[0] != n_frames or m_arr.shape[2] != 3:
                raise ValueError(
                    f"model_markers_m must be (N, M, 3), got {m_arr.shape}"
                )

        tgt_arr: np.ndarray | None = None
        if target_markers_m is not None:
            tgt_arr = np.asarray(target_markers_m, dtype=np.float64)
            if (
                tgt_arr.ndim != 3
                or tgt_arr.shape[0] != n_frames
                or tgt_arr.shape[2] != 3
            ):
                raise ValueError(
                    f"target_markers_m must be (N, M, 3), got {tgt_arr.shape}"
                )

        val_arr: np.ndarray | None = None
        if marker_validity is not None:
            val_arr = np.asarray(marker_validity, dtype=bool)
            if val_arr.ndim != 2 or val_arr.shape[0] != n_frames:
                raise ValueError(f"marker_validity must be (N, M), got {val_arr.shape}")

        act_states_arr: np.ndarray | None = None
        if actuator_states is not None:
            act_states_arr = np.asarray(actuator_states, dtype=np.float64)
            if act_states_arr.ndim != 2 or act_states_arr.shape[0] != n_frames:
                raise ValueError(
                    f"actuator_states must be (N, K), got {act_states_arr.shape}"
                )

        forces_arr: np.ndarray | None = None
        if external_forces is not None:
            forces_arr = np.asarray(external_forces, dtype=np.float64)
            if forces_arr.ndim != 2 or forces_arr.shape[0] != n_frames:
                raise ValueError(
                    f"external_forces must be (N, F), got {forces_arr.shape}"
                )

        # Set read-only flags to guarantee immutability
        t_arr.flags.writeable = False
        q_arr.flags.writeable = False
        if v_arr is not None:
            v_arr.flags.writeable = False
        if tau_arr is not None:
            tau_arr.flags.writeable = False
        if m_arr is not None:
            m_arr.flags.writeable = False
        if tgt_arr is not None:
            tgt_arr.flags.writeable = False
        if val_arr is not None:
            val_arr.flags.writeable = False
        if act_states_arr is not None:
            act_states_arr.flags.writeable = False
        if forces_arr is not None:
            forces_arr.flags.writeable = False

        self._time_s = t_arr
        self._q = q_arr
        self._v = v_arr
        self._tau = tau_arr
        self._actuator_states = act_states_arr
        self._model_markers_m = m_arr
        self._target_markers_m = tgt_arr
        self._marker_validity = val_arr
        self._external_forces = forces_arr

        # Checksums
        if compute_checksums or not metadata.checksums:
            calculated = self.compute_checksums()
            meta_dict = metadata.to_dict()
            meta_dict["checksums"] = calculated
            self._metadata = CandidateMetadata.from_dict(meta_dict)
        else:
            self._metadata = metadata

    @property
    def metadata(self) -> CandidateMetadata:
        return self._metadata

    @property
    def time_s(self) -> np.ndarray:
        return self._time_s

    @property
    def q(self) -> np.ndarray:
        return self._q

    @property
    def v(self) -> np.ndarray | None:
        return self._v

    @property
    def tau(self) -> np.ndarray | None:
        return self._tau

    @property
    def actuator_states(self) -> np.ndarray | None:
        return self._actuator_states

    @property
    def model_markers_m(self) -> np.ndarray | None:
        return self._model_markers_m

    @property
    def target_markers_m(self) -> np.ndarray | None:
        return self._target_markers_m

    @property
    def marker_validity(self) -> np.ndarray | None:
        return self._marker_validity

    @property
    def external_forces(self) -> np.ndarray | None:
        return self._external_forces

    @property
    def n_frames(self) -> int:
        return len(self._time_s)

    @property
    def nq(self) -> int:
        return self._q.shape[1]

    @property
    def nv(self) -> int:
        return self._v.shape[1] if self._v is not None else self.nq

    @property
    def nu(self) -> int:
        return self._tau.shape[1] if self._tau is not None else 0

    def compute_checksums(self) -> dict[str, str]:
        """Compute SHA-256 checksums for all present data arrays."""
        res: dict[str, str] = {
            "time_s": _array_sha256(self._time_s),
            "q": _array_sha256(self._q),
        }
        if self._v is not None:
            res["v"] = _array_sha256(self._v)
        if self._tau is not None:
            res["tau"] = _array_sha256(self._tau)
        if self._actuator_states is not None:
            res["actuator_states"] = _array_sha256(self._actuator_states)
        if self._model_markers_m is not None:
            res["model_markers_m"] = _array_sha256(self._model_markers_m)
        if self._target_markers_m is not None:
            res["target_markers_m"] = _array_sha256(self._target_markers_m)
        if self._marker_validity is not None:
            res["marker_validity"] = _array_sha256(self._marker_validity)
        if self._external_forces is not None:
            res["external_forces"] = _array_sha256(self._external_forces)
        return res

    def verify_checksums(self) -> None:
        """Verify that arrays match the metadata checksums. Fail closed on mismatch."""
        expected = self._metadata.checksums
        if not expected:
            raise ValueError(
                "Candidate metadata contains no checksums for verification"
            )

        current = self.compute_checksums()
        for name, exp_hash in expected.items():
            if name not in current:
                raise ValueError(
                    f"Tampering detected: missing array {name!r} expected by checksums"
                )
            if current[name] != exp_hash:
                raise ValueError(
                    f"Tampering detected: checksum mismatch for {name!r} "
                    f"(expected {exp_hash}, got {current[name]})"
                )


@precondition(
    lambda generalized_velocity, generalized_effort, actuator_transmission, actuator_effort, tolerance=1e-6: (
        isinstance(tolerance, float) and tolerance > 0.0
    ),
    "tolerance must be strictly positive float",
)
def check_virtual_work_consistency(
    generalized_velocity: np.ndarray,
    generalized_effort: np.ndarray,
    actuator_transmission: np.ndarray,
    actuator_effort: np.ndarray,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Verify consistency of generalized effort with actuator transmission by virtual work principle.

    Power in generalized coordinates: P_q = tau_q^T * v
    Power in actuator space: P_act = tau_act^T * (B^T * v)
    Since tau_q = B * tau_act:
    | tau_q^T * v - tau_act^T * (B^T * v) | == 0 for consistent power transmission.
    """
    v = np.asarray(generalized_velocity, dtype=np.float64)
    tau_q = np.asarray(generalized_effort, dtype=np.float64)
    B = np.asarray(actuator_transmission, dtype=np.float64)
    tau_act = np.asarray(actuator_effort, dtype=np.float64)

    power_gen = float(np.dot(tau_q, v))
    qd_act = B.T @ v
    power_act = float(np.dot(tau_act, qd_act))

    diff = abs(power_gen - power_act)
    return bool(diff <= tolerance), diff
