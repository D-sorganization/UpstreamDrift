"""OSL-19 (#13158): Rajagopal folded-arm reach vs bar grips.

Scoping checks on **MuJoCo IK closure** embedded in lift bundle provenance
(``ik.max_closure_m``). That quantity is already sub-millimetre on the
committed squat barbell bundle; the OpenSim bar-reach xfails called out in the
issue are **not** represented here — a separate provenance receipt is reserved
for that pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from src.shared.python.motion_matching.same_input import InputBundle

from ._json_schema import validate_payload

ISSUE_ID: Final = 13158
OSL_SLUG: Final = "OSL-19"
# LIFT / bushings grip weld scale (metres); bundle already beats this today.
GRIP_CLOSURE_PASS_M: Final = 1.0e-3

FIXTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "issue", "reach"],
    "additionalProperties": True,
    "properties": {
        "schema_version": {"const": "ud.osl_scoping.osl19.v1"},
        "issue": {
            "type": "object",
            "required": ["id", "slug"],
            "properties": {
                "id": {"const": ISSUE_ID},
                "slug": {"const": OSL_SLUG},
            },
        },
        "reach": {
            "type": "object",
            "required": ["max_grip_closure_pass_m"],
            "properties": {
                "max_grip_closure_pass_m": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                },
            },
        },
    },
}

ISSUE_TODOS_WITHOUT_ASSETS: Final[tuple[str, ...]] = (
    "Choose cited high-bar grip width and placement inside Rajagopal arm ROM, "
    "or cite an arm variant that reaches the grip — issue table in #13158.",
    "Record reachable workspace (minimum shoulder-to-hand distance by direction) "
    "and align OpenSim IK with MuJoCo bundle grips.",
    "Remove OpenSim strict xfails listed on #13158 when hands land within 1 mm "
    "with no joint-limit overrun.",
)


def validate_osl19_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_payload(record, FIXTURE_SCHEMA)
    return dict(record)


def issue_todo_block() -> str:
    lines = [f"{OSL_SLUG} (#{ISSUE_ID}) — blocked until OpenSim reach matches:", ""]
    lines.extend(f"- TODO: {t}" for t in ISSUE_TODOS_WITHOUT_ASSETS)
    return "\n".join(lines)


def ik_max_grip_closure_m(bundle_path: Path) -> float:
    bundle = InputBundle.load(bundle_path)
    ik = bundle.provenance.get("ik")
    assert isinstance(ik, dict), ik
    return float(ik["max_closure_m"])


def assert_mujoco_grip_closure_passes(
    bundle_path: Path, *, fixture: Mapping[str, Any] | None = None
) -> None:
    """MuJoCo marker IK weld residual stays under the fixture pass line."""
    if fixture is not None:
        validate_payload(fixture, FIXTURE_SCHEMA)
        limit = float(fixture["reach"]["max_grip_closure_pass_m"])
    else:
        limit = GRIP_CLOSURE_PASS_M
    assert ik_max_grip_closure_m(bundle_path) <= limit


def assert_bundle_carries_opensim_reach_receipt(bundle_path: Path) -> None:
    """**Red anchor:** bundle must embed OpenSim reach parity (not produced yet)."""
    bundle = InputBundle.load(bundle_path)
    receipt = bundle.provenance.get("opensim_bar_reach_receipt_v1")
    assert isinstance(receipt, dict), receipt
    miss = float(receipt["max_shoulder_grip_miss_m"])
    assert miss <= GRIP_CLOSURE_PASS_M
