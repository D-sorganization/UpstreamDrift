"""OSL-15 (#13093): bar mass in IK so combined CoM stays over mid-foot.

Scoping contracts for **LoadBalance** records embedded in lift bundle provenance
(:mod:`src.shared.python.lifting.loaded_balance`). Full acceptance still needs
regenerated bundles for every lift, Tools/optimizer changes where required,
OpenSim receipts and flipped xfails — those stay **TODO(#13093)** on the issue.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from src.shared.python.motion_matching.same_input import InputBundle

from ._json_schema import validate_payload

ISSUE_ID: Final = 13093
OSL_SLUG: Final = "OSL-15"

FIXTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "issue", "load_balance_contract"],
    "additionalProperties": True,
    "properties": {
        "schema_version": {"const": "ud.osl_scoping.osl15.v1"},
        "issue": {
            "type": "object",
            "required": ["id", "slug"],
            "properties": {
                "id": {"const": ISSUE_ID},
                "slug": {"const": OSL_SLUG},
            },
        },
        "load_balance_contract": {
            "type": "object",
            "required": ["method_keyword", "barbell_load_kg_min"],
            "properties": {
                "method_keyword": {
                    "type": "string",
                    "minLength": 3,
                    "description": "Substring of loaded_balance.METHOD",
                },
                "barbell_load_kg_min": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Minimum IWF bar + plates load in IK (:class:`Barbell.load_kg`)",
                },
            },
        },
    },
}

ASSET_GATED_FOLLOWUPS: Final[tuple[str, ...]] = (
    "Regenerate each bar-loaded lift bundle with bar mass in the full-body IK "
    "(not only the planar optimizer) and combined CoM over mid-foot per frame.",
    "Redefine bar-position bounds against the real bar landmark and loaded contact; "
    "record reasoning in lift evidence docs.",
    "Re-run barbell re-track, hold, parity and scripts/measure_opensim_lift_ik_id.py; "
    "flip strict xfails tied to #13093.",
)


def validate_osl15_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a copy of a scoping fixture dict."""
    validate_payload(record, FIXTURE_SCHEMA)
    return dict(record)


def asset_gated_followups_block() -> str:
    """Work that remains asset- and OpenSim-gated."""
    lines = [f"{OSL_SLUG} (#{ISSUE_ID}) — blocked until regeneration and receipts:", ""]
    prefix = "- Tracked follow-up (#"
    lines.extend(f"{prefix}{ISSUE_ID}): {t}" for t in ASSET_GATED_FOLLOWUPS)
    return "\n".join(lines)


def ik_balance_record_from_bundle(bundle_path: Path) -> dict[str, Any] | None:
    """``provenance['ik']['balance']`` if present."""
    bundle = InputBundle.load(bundle_path)
    ik = bundle.provenance.get("ik")
    if not isinstance(ik, dict):
        return None
    bal = ik.get("balance")
    return dict(bal) if isinstance(bal, dict) else None


def assert_ik_records_load_balance(
    bundle_path: Path, *, fixture: Mapping[str, Any]
) -> None:
    """Assert bundle IK provenance carries a LoadBalance-shaped balance record.

    Raises:
        AssertionError: when the record is missing or violates *fixture* gates.
    """
    validate_payload(fixture, FIXTURE_SCHEMA)
    bal = ik_balance_record_from_bundle(bundle_path)
    assert bal is not None, f"missing ik.balance in {bundle_path}"
    contract = fixture["load_balance_contract"]
    method = str(bal.get("method", ""))
    assert contract["method_keyword"] in method, (method, contract["method_keyword"])
    load = bal.get("load")
    assert isinstance(load, dict), load
    barbell = load.get("barbell")
    assert isinstance(barbell, dict), barbell
    load_kg = float(barbell["load_kg"])
    assert load_kg >= float(contract["barbell_load_kg_min"]), (load_kg, contract)
    assert "max_offset_m" in bal and float(bal["max_offset_m"]) >= 0.0
    assert "correction_rad" in bal and isinstance(bal["correction_rad"], list)


def deadlift_pull_bundle_paths(repo_root: Path | None = None) -> tuple[Path, Path]:
    """Barbell deadlift and clean_and_jerk bundle paths (expected to gain balance)."""
    root = repo_root or Path(__file__).resolve().parents[4]
    ev = root / "docs/development/lifting/evidence"
    return (
        ev / "deadlift" / "deadlift_barbell_bundle.npz",
        ev / "clean_and_jerk" / "clean_and_jerk_barbell_bundle.npz",
    )
