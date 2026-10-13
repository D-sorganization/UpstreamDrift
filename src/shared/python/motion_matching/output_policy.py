"""Fail-closed output policy for golf swing outputs (#10382, MS-108).

Owner ruling (2026-10-11): advertise clubhead position/speed and sequencing
with uncertainty; withhold face angle, joint loads and impact/ball-flight as
"not validated" unless the receipt supplies both calibration and evidence.

Receipt contract (optional ``outputs`` mapping, absent means no claims):

    outputs: {
        "<output name>": {
            "validated": bool,        # claim only for withheld outputs
            "value": float,           # optional; finite when present
            "uncertainty": float,     # required for advertised values
            "unit": str,
            "calibration": str,       # required to validate a withheld output
            "evidence": str,          # required to validate a withheld output
        },
    }

Postconditions: a withheld output without both ``calibration`` and ``evidence``
never reports status ``validated``; withheld values are never returned to the
caller for display. When a face observability audit
(:func:`club_observability.audit_face_observability`) finds the club face
unobservable, ``face_angle`` is ``unavailable`` and cannot be validated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.python.motion_matching.club_observability import (
        FaceObservability,
    )

__all__ = [
    "ADVERTISED_OUTPUTS",
    "WITHHELD_OUTPUTS",
    "OutputPolicyError",
    "OutputStatus",
    "evaluate_output_policy",
    "face_observability_of_receipt",
    "output_policy_payload",
    "render_face_observability_markdown",
    "render_output_policy_markdown",
]

ADVERTISED_OUTPUTS: tuple[str, ...] = (
    "clubhead_position",
    "clubhead_speed",
    "sequencing",
)
WITHHELD_OUTPUTS: tuple[str, ...] = (
    "face_angle",
    "joint_loads",
    "impact",
    "ball_flight",
)

_STATUS_ADVERTISED = "advertised"
_STATUS_WITHHELD = "withheld"
_STATUS_VALIDATED = "validated"
_STATUS_UNAVAILABLE = "unavailable"
_FACE_ANGLE = "face_angle"


class OutputPolicyError(ValueError):
    """Raised when a receipt makes an output claim the policy does not allow."""


@dataclass(frozen=True)
class OutputStatus:
    """Display status for one golf output. Withheld values are never carried."""

    name: str
    status: str
    reason: str


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OutputPolicyError(
            f"{label} must be a mapping, got {type(value).__name__}"
        )
    return value


def _check_finite_value(name: str, entry: Mapping[str, Any]) -> None:
    if "value" not in entry:
        return
    value = entry["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OutputPolicyError(f"output '{name}' value must be a finite number")
    if not math.isfinite(float(value)):
        raise OutputPolicyError(f"output '{name}' value must be finite")


def _has_text(entry: Mapping[str, Any], key: str) -> bool:
    raw = entry.get(key)
    return isinstance(raw, str) and bool(raw.strip())


def _evaluate_advertised(name: str, entry: Mapping[str, Any]) -> OutputStatus:
    _check_finite_value(name, entry)
    if "value" in entry:
        uncertainty = entry.get("uncertainty")
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
            raise OutputPolicyError(
                f"advertised output '{name}' requires a finite uncertainty"
            )
        if not math.isfinite(float(uncertainty)) or float(uncertainty) < 0.0:
            raise OutputPolicyError(
                f"advertised output '{name}' requires a finite uncertainty"
            )
    return OutputStatus(name, _STATUS_ADVERTISED, "advertised with uncertainty")


def _evaluate_withheld(name: str, entry: Mapping[str, Any]) -> OutputStatus:
    if not entry.get("validated", False):
        return OutputStatus(name, _STATUS_WITHHELD, "not validated")
    missing = [key for key in ("calibration", "evidence") if not _has_text(entry, key)]
    if missing:
        raise OutputPolicyError(
            f"withheld output '{name}' claims validated without {' and '.join(missing)}"
        )
    _check_finite_value(name, entry)
    return OutputStatus(name, _STATUS_VALIDATED, "calibration and evidence supplied")


def _evaluate_face_angle(
    entry: Mapping[str, Any], observability: FaceObservability | None
) -> OutputStatus:
    if observability is None or observability.observable:
        return _evaluate_withheld(_FACE_ANGLE, entry)
    reason = f"not observable: {observability.reason}"
    if entry.get("validated", False):
        raise OutputPolicyError(
            f"withheld output '{_FACE_ANGLE}' claims validated but is {reason}"
        )
    return OutputStatus(_FACE_ANGLE, _STATUS_UNAVAILABLE, reason)


def evaluate_output_policy(
    receipt: Mapping[str, Any],
    *,
    face_observability: FaceObservability | None = None,
) -> dict[str, OutputStatus]:
    """Classify every golf output in the receipt under the fail-closed policy.

    Args:
        receipt: Parsed execution receipt. Its optional ``outputs`` key holds the
            claims described in the module docstring.
        face_observability: Optional club face audit of the capture. When it is
            unobservable, ``face_angle`` is ``unavailable``.

    Returns:
        Status per known output name. Advertised and withheld outputs always
        appear, even without a claim, so reports state what is withheld.

    Raises:
        OutputPolicyError: On unknown output names, non-mapping claims, a
            withheld output validated without calibration and evidence, an
            advertised value lacking uncertainty, a non-finite value, or a
            validated ``face_angle`` claim the face audit finds unobservable.
    """
    claims = receipt.get("outputs", {})
    claims = _require_mapping(claims, "outputs")

    known = set(ADVERTISED_OUTPUTS) | set(WITHHELD_OUTPUTS)
    unknown = sorted(set(claims) - known)
    if unknown:
        raise OutputPolicyError(f"unknown output '{unknown[0]}' in receipt outputs")

    statuses: dict[str, OutputStatus] = {}
    for name in ADVERTISED_OUTPUTS:
        entry = _require_mapping(claims.get(name, {}), f"output '{name}'")
        statuses[name] = _evaluate_advertised(name, entry)
    for name in WITHHELD_OUTPUTS:
        entry = _require_mapping(claims.get(name, {}), f"output '{name}'")
        if name == _FACE_ANGLE:
            statuses[name] = _evaluate_face_angle(entry, face_observability)
        else:
            statuses[name] = _evaluate_withheld(name, entry)
    return statuses


_DISPLAY_LABELS = {
    _STATUS_WITHHELD: "NOT VALIDATED",
    _STATUS_UNAVAILABLE: "UNAVAILABLE",
}


def _display_label(status: OutputStatus) -> str:
    return _DISPLAY_LABELS.get(status.status, status.status)


def render_output_policy_markdown(statuses: Mapping[str, OutputStatus]) -> str:
    """Render the policy table. Values are intentionally never included."""
    lines = [
        "| Output | Status | Basis |",
        "|---|---|---|",
    ]
    for name, status in statuses.items():
        label = _display_label(status)
        lines.append(f"| `{name}` | **{label}** | {status.reason} |")
    return "\n".join(lines) + "\n"


def face_observability_of_receipt(
    receipt: Mapping[str, Any],
) -> FaceObservability | None:
    """The receipt's ``ik.face_observability`` audit, or None when absent.

    Raises:
        ValueError: When the block is present but malformed (fail closed).
    """
    from src.shared.python.motion_matching.club_observability import (
        face_observability_from_receipt,
    )

    ik = receipt.get("ik")
    block = ik.get("face_observability") if isinstance(ik, Mapping) else None
    return None if block is None else face_observability_from_receipt(block)


def render_face_observability_markdown(
    observability: FaceObservability | None,
) -> str:
    """Render the face-angle audit: coverage and gap runs, never an angle."""
    if observability is None:
        return "Face-angle observability: not audited for this receipt.\n"
    gaps = ", ".join(f"[{a}, {b})" for a, b in observability.gaps) or "none"
    verdict = "observable" if observability.observable else "unobservable"
    return (
        f"Face-angle observability: {verdict} ({observability.reason}).\n\n"
        f"- Observed frames: {observability.observed_frames}/"
        f"{observability.total_frames} ({observability.coverage:.1%})\n"
        f"- Unobserved frame runs: {gaps}\n"
    )


def _face_payload(observability: FaceObservability | None) -> dict[str, Any] | None:
    if observability is None:
        return None
    return {
        "observable": observability.observable,
        "reason": observability.reason,
        "observed_frames": observability.observed_frames,
        "total_frames": observability.total_frames,
        "coverage": observability.coverage,
        "gaps": [[start, stop] for start, stop in observability.gaps],
    }


def _row_payload(status: OutputStatus) -> dict[str, str]:
    return {
        "name": status.name,
        "status": status.status,
        "label": _display_label(status),
        "reason": status.reason,
    }


def output_policy_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Structured, JSON-safe policy view of a receipt for reports and viewers.

    One row per known output (status, display label, reason) plus the face
    audit (coverage, gap runs) when the receipt carries one. Values are never
    included. Fails closed: a receipt the policy rejects yields every output
    ``unavailable`` with the rejection as its reason, never a bare number.
    """
    try:
        audit = face_observability_of_receipt(receipt)
        statuses = evaluate_output_policy(receipt, face_observability=audit)
    except ValueError as exc:  # OutputPolicyError is a ValueError
        failed = OutputStatus("", _STATUS_UNAVAILABLE, f"policy error: {exc}")
        statuses = {
            name: OutputStatus(name, failed.status, failed.reason)
            for name in (*ADVERTISED_OUTPUTS, *WITHHELD_OUTPUTS)
        }
        audit = None
    return {
        "schema_version": "output-policy/1",
        "outputs": [_row_payload(status) for status in statuses.values()],
        "face_observability": _face_payload(audit),
    }
