"""Pure view-model for golf-output policy status in the viewer (#10382, MS-108).

Consumes :func:`output_policy_payload` and returns display rows (name, badge,
tooltip). Contains no Qt so it is testable without a display. A row never
carries a numeric value: withheld or unavailable outputs show their status.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.shared.python.motion_matching.output_policy import (
    ADVERTISED_OUTPUTS,
    WITHHELD_OUTPUTS,
)

__all__ = ["OutputStatusRow", "build_output_status_rows", "face_audit_summary"]

_FACE_ANGLE = "face_angle"
_ADVERTISED_BADGE = "ADVERTISED"
_UNAVAILABLE_BADGE = "UNAVAILABLE"
_NO_PAYLOAD_REASON = "policy status unavailable for this run"


@dataclass(frozen=True)
class OutputStatusRow:
    """One displayed output: status badge plus reason/coverage tooltip."""

    name: str
    badge: str
    tooltip: str


def face_audit_summary(payload: Mapping[str, Any] | None) -> str:
    """Coverage and gap runs of the face audit, or a not-audited notice."""
    face = (payload or {}).get("face_observability")
    if not face:
        return "Face-angle observability: not audited for this receipt."
    gaps = ", ".join(f"[{a}, {b})" for a, b in face["gaps"]) or "none"
    verdict = "observable" if face["observable"] else "unobservable"
    return (
        f"Face-angle observability: {verdict} ({face['reason']}); observed "
        f"{face['observed_frames']}/{face['total_frames']} frames "
        f"({face['coverage']:.1%}); unobserved runs: {gaps}"
    )


def _row(item: Mapping[str, Any], payload: Mapping[str, Any]) -> OutputStatusRow:
    badge = _ADVERTISED_BADGE if item["status"] == "advertised" else item["label"]
    tooltip = str(item["reason"])
    if item["name"] == _FACE_ANGLE:
        tooltip = f"{tooltip}\n{face_audit_summary(payload)}"
    return OutputStatusRow(item["name"], str(badge).upper(), tooltip)


def build_output_status_rows(
    payload: Mapping[str, Any] | None,
) -> list[OutputStatusRow]:
    """Display rows for every known output; fail closed without a payload."""
    if not payload or not payload.get("outputs"):
        return [
            OutputStatusRow(name, _UNAVAILABLE_BADGE, _NO_PAYLOAD_REASON)
            for name in (*ADVERTISED_OUTPUTS, *WITHHELD_OUTPUTS)
        ]
    return [_row(item, payload) for item in payload["outputs"]]
