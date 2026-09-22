"""Immutable episode records for neural motion matching (NM-03 #10618)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    N_JOINTS,
)

__all__ = [
    "EPISODE_STORE_SCHEMA",
    "ALLOWED_CONTROL_BASES",
    "ALLOWED_UNITS",
    "EpisodeRecord",
    "require_finite_or_none",
]

EPISODE_STORE_SCHEMA = "neural-episode-store/1.0.0"

ALLOWED_UNITS: frozenset[str] = frozenset({"SI"})
ALLOWED_CONTROL_BASES: frozenset[str] = frozenset(
    {
        "joint_torque",
        "polynomial_coefficient",
        "muscle_excitation",
        "muscle_activation",
    }
)

_IDENTITY = ("q", "v", "u", "a_native")
_PREDICTIVE = ("q_next",)


def require_finite_or_none(
    name: str,
    values: np.ndarray | None,
    *,
    required: bool,
) -> np.ndarray | None:
    """Reject non-finite arrays; allow None only when channel is unavailable."""
    if values is None:
        if required:
            raise ValueError(f"{name} must not be None when available")
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} values must be finite")
    return arr


def _is_uniform_clock(times: np.ndarray) -> bool:
    if times.size < 2:
        return True
    diffs = np.diff(times)
    return bool(np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9))


@dataclass(frozen=True)
class EpisodeRecord:
    """One immutable native or adapted episode.

    Design by Contract:
    - ``trial_id`` / ``family_id`` / ``model_id`` are non-empty.
    - Units and control basis must be in the allow-lists (reject wrong basis).
    - Joint order must match ``CANONICAL_JOINTS`` length and names when
      compact-compatible; coefficient vectors are exactly ``N_COEFFS`` when set.
    - Available channels are finite; unavailable channels are ``None`` (never
      silent zeros-as-measurements).
    - ``q_next`` is a predictive target, not an identity channel.
    """

    trial_id: str
    family_id: str
    model_id: str
    control_basis: str
    units: str
    joint_names: tuple[str, ...]
    coefficient_letters: tuple[str, ...]
    schema_version: str
    sample_times_s: np.ndarray
    q: np.ndarray
    v: np.ndarray | None
    u: np.ndarray | None
    a_native: np.ndarray | None
    q_next: np.ndarray | None
    channel_availability: Mapping[str, str]
    ancestry: tuple[str, ...]
    geometry_stratum: str
    contact_stratum: str
    club_stratum: str
    coefficients: np.ndarray | None = None
    source_schema: str | None = None
    episode_id: str = ""
    content_sha256: str = ""
    _payload_digest: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_basis_and_units()
        self._validate_layout()
        if self.source_schema == "":
            object.__setattr__(self, "source_schema", None)
        times = require_finite_or_none(
            "sample_times_s", self.sample_times_s, required=True
        )
        assert times is not None
        object.__setattr__(self, "sample_times_s", times)
        if times.size < 1 or float(times[0]) < -1e-12:
            raise ValueError("sample_times_s must start at or after 0")
        if times.size >= 2 and bool(np.any(np.diff(times) < -1e-12)):
            raise ValueError("sample_times_s must be monotonic non-decreasing")

        n_t = int(times.shape[0])
        object.__setattr__(
            self, "q", self._require_channel("q", self.q, n_t, required=True)
        )
        object.__setattr__(
            self, "v", self._require_channel("v", self.v, n_t, required=False)
        )
        object.__setattr__(
            self, "u", self._require_channel("u", self.u, n_t, required=False)
        )
        object.__setattr__(
            self,
            "a_native",
            self._require_channel("a_native", self.a_native, n_t, required=False),
        )
        object.__setattr__(
            self,
            "q_next",
            self._require_channel("q_next", self.q_next, n_t, required=False),
        )
        if self.coefficients is not None:
            coeffs = require_finite_or_none(
                "coefficients", self.coefficients, required=True
            )
            assert coeffs is not None
            if coeffs.shape != (N_COEFFS,):
                raise ValueError(
                    f"coefficients must have shape ({N_COEFFS},), got {coeffs.shape}"
                )
            object.__setattr__(self, "coefficients", coeffs)

        digest = self._compute_payload_digest()
        object.__setattr__(self, "_payload_digest", digest)
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", digest)
        if not self.episode_id:
            object.__setattr__(self, "episode_id", digest[:16])

    def _validate_identity(self) -> None:
        for name, value in (
            ("trial_id", self.trial_id),
            ("family_id", self.family_id),
            ("model_id", self.model_id),
            ("geometry_stratum", self.geometry_stratum),
            ("contact_stratum", self.contact_stratum),
            ("club_stratum", self.club_stratum),
            ("schema_version", self.schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.channel_availability, Mapping):
            raise ValueError("channel_availability must be a mapping")
        if not isinstance(self.ancestry, tuple):
            raise ValueError("ancestry must be a tuple of strings")

    def _validate_basis_and_units(self) -> None:
        if self.units not in ALLOWED_UNITS:
            raise ValueError(
                f"units must be one of {sorted(ALLOWED_UNITS)}, got {self.units!r}"
            )
        if self.control_basis not in ALLOWED_CONTROL_BASES:
            raise ValueError(
                f"control basis must be one of {sorted(ALLOWED_CONTROL_BASES)}, "
                f"got {self.control_basis!r}"
            )

    def _validate_layout(self) -> None:
        if len(self.joint_names) != N_JOINTS:
            raise ValueError(
                f"joint_names must have length {N_JOINTS}, got {len(self.joint_names)}"
            )
        if tuple(self.joint_names) != CANONICAL_JOINTS:
            raise ValueError(
                "joint_names must match CANONICAL_JOINTS order "
                "(compact-1.0 27-coordinate contract)"
            )
        if tuple(self.coefficient_letters) != COEFFICIENT_LETTERS:
            raise ValueError("coefficient_letters must match COEFFICIENT_LETTERS")

    def _is_available(self, channel: str) -> bool:
        status = str(self.channel_availability.get(channel, "unavailable"))
        return status == "available"

    def _require_channel(
        self,
        name: str,
        values: np.ndarray | None,
        n_t: int,
        *,
        required: bool,
    ) -> np.ndarray | None:
        available = self._is_available(name)
        if not available:
            if values is not None and name != "q":
                # Allow construction helpers to pass arrays then drop them.
                return None
            if name == "q" and not available:
                raise ValueError("q channel must be available")
            return None
        arr = require_finite_or_none(name, values, required=True)
        assert arr is not None
        if arr.ndim != 2 or arr.shape[0] != n_t or arr.shape[1] != N_JOINTS:
            raise ValueError(
                f"{name} must have shape ({n_t}, {N_JOINTS}), got {arr.shape}"
            )
        if required and arr is None:
            raise ValueError(f"{name} is required")
        return arr

    def clock_is_uniform(self) -> bool:
        """Return whether the sample clock has constant dt."""
        return _is_uniform_clock(self.sample_times_s)

    def identity_channels(self) -> tuple[str, ...]:
        """Same-time identity channels (not predictive targets)."""
        return _IDENTITY

    def predictive_targets(self) -> tuple[str, ...]:
        """Next-state / predictive targets separated from identity."""
        return _PREDICTIVE

    def content_payload_digest(self) -> str:
        """Stable digest of episode payload (independent of episode_id field)."""
        return self._payload_digest or self._compute_payload_digest()

    def _compute_payload_digest(self) -> str:
        payload: dict[str, Any] = {
            "trial_id": self.trial_id,
            "family_id": self.family_id,
            "model_id": self.model_id,
            "control_basis": self.control_basis,
            "units": self.units,
            "joint_names": list(self.joint_names),
            "coefficient_letters": list(self.coefficient_letters),
            "schema_version": self.schema_version,
            "source_schema": self.source_schema,
            "ancestry": list(self.ancestry),
            "geometry_stratum": self.geometry_stratum,
            "contact_stratum": self.contact_stratum,
            "club_stratum": self.club_stratum,
            "channel_availability": dict(self.channel_availability),
            "sample_times_s": self._array_digest(self.sample_times_s),
            "q": self._array_digest(self.q),
            "v": self._array_digest(self.v),
            "u": self._array_digest(self.u),
            "a_native": self._array_digest(self.a_native),
            "q_next": self._array_digest(self.q_next),
            "coefficients": self._array_digest(self.coefficients),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _array_digest(values: np.ndarray | None) -> str | None:
        if values is None:
            return None
        arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def as_meta_dict(self) -> dict[str, Any]:
        """JSON-serialisable metadata (no large arrays)."""
        return {
            "episode_id": self.episode_id,
            "content_sha256": self.content_sha256,
            "trial_id": self.trial_id,
            "family_id": self.family_id,
            "model_id": self.model_id,
            "control_basis": self.control_basis,
            "units": self.units,
            "schema_version": self.schema_version,
            "source_schema": self.source_schema,
            "joint_names": list(self.joint_names),
            "coefficient_letters": list(self.coefficient_letters),
            "channel_availability": dict(self.channel_availability),
            "ancestry": list(self.ancestry),
            "geometry_stratum": self.geometry_stratum,
            "contact_stratum": self.contact_stratum,
            "club_stratum": self.club_stratum,
            "n_samples": int(self.sample_times_s.shape[0]),
            "clock_uniform": self.clock_is_uniform(),
        }
