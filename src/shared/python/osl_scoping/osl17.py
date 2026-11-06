"""OSL-17 (#13131): back-rack bar placement and window start at rest.

Provenance-only checks on committed LIFT-5 bundles: cited high-bar rack in IK
``back_bar`` and a **lead-in** segment before the recorded motion window so the
first bundled state can sit on its contact. Full Hicks-gate acceptance across
the whole window stays **TODO** until OpenSim receipts exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from src.shared.python.motion_matching.same_input import InputBundle

from ._json_schema import validate_payload

ISSUE_ID: Final = 13131
OSL_SLUG: Final = "OSL-17"

FIXTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "issue", "rack_contract", "lead_in_contract"],
    "additionalProperties": True,
    "properties": {
        "schema_version": {"const": "ud.osl_scoping.osl17.v1"},
        "issue": {
            "type": "object",
            "required": ["id", "slug"],
            "properties": {
                "id": {"const": ISSUE_ID},
                "slug": {"const": OSL_SLUG},
            },
        },
        "rack_contract": {
            "type": "object",
            "required": ["rack_name", "landmark"],
            "properties": {
                "rack_name": {"type": "string", "const": "high_bar"},
                "landmark": {"type": "string", "const": "BackTop"},
            },
        },
        "lead_in_contract": {
            "type": "object",
            "required": ["min_duration_s"],
            "properties": {
                "min_duration_s": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Lead-in duration target; below fails until extended",
                },
            },
        },
    },
}

ISSUE_TODOS_WITHOUT_ASSETS: Final[tuple[str, ...]] = (
    "Regenerate the back-squat bundle with bar on cited low/high rack if needed; "
    "keep Tools changes out-of-repo (report Tools issue per #13131).",
    "Extend lead-in or add pre-window states until OpenSim Hicks checks pass at "
    "the first window sample with no ad hoc state exclusion.",
    "Re-run barbell hold, re-track, parity and scripts/measure_opensim_lift_ik_id.py.",
)


def validate_osl17_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_payload(record, FIXTURE_SCHEMA)
    return dict(record)


def issue_todo_block() -> str:
    lines = [f"{OSL_SLUG} (#{ISSUE_ID}) — blocked until regeneration and receipts:", ""]
    lines.extend(f"- TODO: {t}" for t in ISSUE_TODOS_WITHOUT_ASSETS)
    return "\n".join(lines)


def assert_high_bar_rack_in_ik(
    bundle_path: Path, *, fixture: Mapping[str, Any] | None = None
) -> None:
    """Committed back-squat IK records Glassbrook/Wretenberg high bar on ``BackTop``."""
    if fixture is not None:
        validate_payload(fixture, FIXTURE_SCHEMA)
    bundle = InputBundle.load(bundle_path)
    ik = bundle.provenance.get("ik")
    assert isinstance(ik, dict), ik
    bb = ik.get("back_bar")
    assert isinstance(bb, dict), bb
    geom = bb.get("geometry")
    assert isinstance(geom, dict), geom
    rack = geom.get("rack")
    assert isinstance(rack, dict), rack
    if fixture is not None:
        rc = fixture["rack_contract"]
        assert rack.get("name") == rc["rack_name"]
        assert rack.get("landmark") == rc["landmark"]
    else:
        assert rack.get("name") == "high_bar"
        assert rack.get("landmark") == "BackTop"


def assert_lead_in_record_present(bundle_path: Path) -> None:
    """``provenance['lead_in']`` exists (settling run before the window)."""
    bundle = InputBundle.load(bundle_path)
    lead = bundle.provenance.get("lead_in")
    assert isinstance(lead, dict), lead
    assert float(lead["duration_s"]) > 0.0
    reason = str(lead.get("reason", ""))
    assert "13131" in reason or "OSL-17" in reason


def assert_lead_in_duration(bundle_path: Path, *, fixture: Mapping[str, Any]) -> None:
    """``lead_in.duration_s`` meets the fixture minimum (stricter than \"present\")."""
    validate_payload(fixture, FIXTURE_SCHEMA)
    bundle = InputBundle.load(bundle_path)
    lead = bundle.provenance.get("lead_in")
    assert isinstance(lead, dict), lead
    need = float(fixture["lead_in_contract"]["min_duration_s"])
    assert float(lead["duration_s"]) >= need
