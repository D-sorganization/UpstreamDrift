"""Versioned, content-addressed input identity for native-equivalent replay."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np

from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile
from src.shared.python.motion_matching.prefix_fit import normalized_to_simscape

_FIELDS = frozenset(
    {
        "schema_version",
        "coordinate_names",
        "model_sha256",
        "capture_sha256",
        "source_sha256",
        "coefficient_order",
        "time_basis",
        "force_frame",
        "duration_s",
        "q0",
        "qd0",
        "coefficients",
        "marker_labels",
        "marker_bodies",
        "marker_offsets_m",
    }
)


@dataclass(frozen=True)
class NativeReplayCandidate:
    """Validated JSON snapshot; identity includes states, inputs and attachments.

    Build through from_document. Geometry and passive physics belong to the
    hashed model artifact. The adapter must verify that artifact and physical
    initial closure before execution; this object alone is not qualification.
    """

    _canonical_json: str

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        coordinate_order: Sequence[str],
        model_sha256: str,
    ) -> "NativeReplayCandidate":
        if set(document) != _FIELDS:
            raise ValueError("Candidate fields are missing or unsupported")
        try:
            canonical = json.dumps(
                dict(document), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Candidate must be finite JSON data") from error
        data = json.loads(canonical)
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported candidate schema")
        for name, expected in {
            "coordinate_names": list(coordinate_order),
            "model_sha256": model_sha256,
            "coefficient_order": "highest-power-first",
            "time_basis": "absolute-seconds",
            "force_frame": "world",
        }.items():
            if data[name] != expected:
                raise ValueError(f"Candidate {name} differs from the native contract")
        for name in ("model_sha256", "capture_sha256", "source_sha256"):
            if not isinstance(data[name], str) or not re.fullmatch(
                r"[0-9a-f]{64}", data[name]
            ):
                raise ValueError(f"Invalid {name}")
        duration = data["duration_s"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            raise ValueError("Candidate duration must be positive seconds")
        names = data["coordinate_names"]
        NativeEffortProfile(names, data["coefficients"], np.eye(3))
        for name in ("q0", "qd0"):
            vector = np.asarray(data[name], dtype=float)
            if vector.shape != (len(names),) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid initial {name}")
        labels, bodies = data["marker_labels"], data["marker_bodies"]
        if (
            not isinstance(labels, list)
            or not isinstance(bodies, list)
            or not labels
            or len(labels) != len(bodies)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in labels + bodies
            )
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("Invalid marker identity or body inventory")
        offsets = np.asarray(data["marker_offsets_m"], dtype=float)
        if offsets.shape != (len(labels), 3) or not np.isfinite(offsets).all():
            raise ValueError("Invalid marker offsets")
        return cls(canonical)

    @property
    def document(self) -> dict[str, Any]:
        """Return a detached copy suitable for serialization and adapter input."""
        return json.loads(self._canonical_json)

    @property
    def sha256(self) -> str:
        """Hash all candidate content, independent of caller dictionary ordering."""
        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()


def increment_native_candidate(
    candidate: NativeReplayCandidate,
    normalized_increment: np.ndarray,
    *,
    basis_duration_s: float,
) -> NativeReplayCandidate:
    """Add an ascending normalized polynomial without retiming existing inputs.

    The basis duration is explicit and independent of integration coverage.
    Return a new candidate identity; preserve the original state and attachments.
    """
    data = candidate.document
    delta = normalized_to_simscape(normalized_increment, duration_s=basis_duration_s)
    original = np.asarray(data["coefficients"], dtype=float)
    if delta.shape != original.shape:
        raise ValueError("Increment must preserve native coordinate count")
    data["coefficients"] = (original + delta).tolist()
    return NativeReplayCandidate.from_document(
        data, data["coordinate_names"], data["model_sha256"]
    )
