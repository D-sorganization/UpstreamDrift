"""Pure view-model for output policy status in the matched-swing viewer (#10382)."""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.motion_matching.output_policy import output_policy_payload
from src.tools.matched_swing_browser.output_status import (
    build_output_status_rows,
    face_audit_summary,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_FACE_BLOCK = {
    "observable": False,
    "reason": "club head markers are collinear or coincident",
    "observed_frames": 1,
    "total_frames": 4,
    "gaps": [[1, 4]],
}


def _payload(face: dict[str, Any] | None = None) -> dict[str, Any]:
    receipt: dict[str, Any] = {"engine": "mujoco"}
    if face is not None:
        receipt["ik"] = {"face_observability": face}
    return output_policy_payload(receipt)


def test_every_row_carries_label_and_reason_tooltip() -> None:
    rows = build_output_status_rows(_payload())
    by_name = {row.name: row for row in rows}
    assert by_name["clubhead_speed"].badge == "ADVERTISED"
    assert "uncertainty" in by_name["clubhead_speed"].tooltip
    assert by_name["joint_loads"].badge == "NOT VALIDATED"
    assert by_name["joint_loads"].tooltip == "not validated"


def test_face_angle_row_states_coverage_and_gaps() -> None:
    rows = {r.name: r for r in build_output_status_rows(_payload(_FACE_BLOCK))}
    face = rows["face_angle"]
    assert face.badge == "UNAVAILABLE"
    assert "25.0%" in face.tooltip and "[1, 4)" in face.tooltip


def test_no_row_text_is_a_bare_number() -> None:
    for row in build_output_status_rows(_payload(_FACE_BLOCK)):
        assert row.badge and not row.badge.replace(".", "").isdigit()


def test_missing_payload_is_unavailable_never_blank() -> None:
    rows = build_output_status_rows(None)
    assert rows and all(r.badge == "UNAVAILABLE" for r in rows)


def test_face_audit_summary_unaudited_and_audited() -> None:
    assert "not audited" in face_audit_summary(_payload())
    assert "1/4" in face_audit_summary(_payload(_FACE_BLOCK))
