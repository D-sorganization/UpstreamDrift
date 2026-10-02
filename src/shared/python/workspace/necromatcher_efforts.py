"""Unit-preserving authored generalized efforts, never measured dynamics.

Coordinate units bind to declared research-fit units. Native compilation must
independently check that declaration before simulation. Operator-authored seconds
do not qualify the source footage's physical clock.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.piecewise_polynomial import (
    PiecewisePolynomialTorque,
    PolynomialSegment,
)

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary

EFFORT_SCHEMA = "necromatcher/effort-profile/2"
_FIELDS = {
    "schema_version",
    "model_id",
    "model_hash",
    "fit_id",
    "fit_hash",
    "dofs",
    "coordinate_units",
    "effort_units",
    "timebase",
    "provenance",
    "segments",
}
_EFFORT_UNITS = {"m": "N", "rad": "N*m"}


@dataclass(frozen=True)
class AuthoredEffortProfile:
    """Checked ordered units with an explicitly bounded authored time interval."""

    model_id: str
    fit_id: str
    model_hash: str
    fit_hash: str
    dofs: tuple[str, ...]
    coordinate_units: tuple[str, ...]
    effort_units: tuple[str, ...]
    _curve: PiecewisePolynomialTorque

    def evaluate(self, time_s: float) -> NDArray[np.float64]:
        """Reject extrapolation instead of extending controls by endpoint clamping."""
        if (
            type(time_s) not in (float, int)
            or not np.isfinite(time_s)
            or not self._curve.start_s <= time_s <= self._curve.end_s
        ):
            raise ValueError(
                "Effort time must be finite and within its authored interval"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            result = self._curve.evaluate(time_s)
        if not np.isfinite(result).all():
            raise ValueError("Evaluated efforts must be finite")
        return result


def read_segments(payload: dict[str, Any]) -> PiecewisePolynomialTorque:
    """Validate JSON scalar types before canonical polynomial construction."""
    records = payload.get("segments")
    if not isinstance(records, list) or not records:
        raise ValueError("Profile requires polynomial segments")
    segments = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "start_s",
            "end_s",
            "coefficients",
            "is_bernstein",
        }:
            raise ValueError("Profile segment fields must match the schema")
        if type(record["is_bernstein"]) is not bool or any(
            type(record[key]) not in (int, float) for key in ("start_s", "end_s")
        ):
            raise ValueError(
                "Segment timing requires numbers and basis requires boolean"
            )
        coefficients = record["coefficients"]
        if (
            not isinstance(coefficients, list)
            or not coefficients
            or any(
                not isinstance(row, list)
                or not row
                or any(type(value) not in (int, float) for value in row)
                for row in coefficients
            )
        ):
            raise ValueError("Segment coefficients require a finite numeric matrix")
        try:
            values = np.array(coefficients, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Segment coefficients require a finite numeric matrix"
            ) from exc
        # Immutable backing bytes prevent callers re-enabling writes to this array.
        values = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
        segments.append(
            PolynomialSegment(
                record["start_s"], record["end_s"], values, record["is_bernstein"]
            )
        )
    return PiecewisePolynomialTorque(tuple(segments))


def validate_authored_provenance(payload: dict[str, Any]) -> None:
    """Require explicit operator authorship without inferring source timing."""
    provenance = payload.get("provenance")
    if (
        payload.get("timebase") != "physical_seconds"
        or not isinstance(provenance, dict)
        or provenance.get("kind") != "authored"
        or not isinstance(provenance.get("description"), str)
        or not provenance["description"].strip()
    ):
        raise ValueError("Profiles require authored provenance and physical_seconds")
    json.dumps(payload, allow_nan=False)


def read_effort_profile(
    source: Path, library: NecromatcherLibrary, swing_id: str
) -> tuple[dict[str, Any], AuthoredEffortProfile]:
    """Verify exact model/fit versions, ordered declared units and polynomial data."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise ValueError("Effort profile must contain exactly its schema fields")
    if payload["schema_version"] != EFFORT_SCHEMA:
        raise ValueError("Unsupported effort-profile schema")
    for name, kind in (("model", "native_model"), ("fit", "kinematic_fit")):
        if not isinstance(payload[f"{name}_id"], str):
            raise ValueError("Profile parent identity must be a string")
        asset = library.load_asset(payload[f"{name}_id"])
        if asset.kind != kind or asset.session_id != swing_id:
            raise ValueError("Profile parents must belong to the same swing session")
        if payload[f"{name}_hash"] != asset.metadata["hash"]:
            raise ValueError("Profile parent hash mismatch")
    fit = library.load_fit(payload["fit_id"])
    if (
        fit["model_id"] != payload["model_id"]
        or fit["model_hash"] != payload["model_hash"]
        or payload["dofs"] != fit["coordinate_order"]
        or payload["coordinate_units"] != fit["coordinate_units"]
    ):
        raise ValueError("Profile model, coordinate order and units must match its fit")
    expected = [_EFFORT_UNITS[unit] for unit in fit["coordinate_units"]]
    if payload["effort_units"] != expected:
        raise ValueError("Effort units require N for m coordinates and N*m for rad")
    validate_authored_provenance(payload)
    curve = read_segments(payload)
    if curve.n_channels != len(payload["dofs"]):
        raise ValueError("Profile channel count must match model DOFs")
    return payload, AuthoredEffortProfile(
        payload["model_id"],
        payload["fit_id"],
        payload["model_hash"],
        payload["fit_hash"],
        tuple(payload["dofs"]),
        tuple(payload["coordinate_units"]),
        tuple(expected),
        curve,
    )
