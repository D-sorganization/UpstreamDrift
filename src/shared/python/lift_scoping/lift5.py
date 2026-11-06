"""LIFT-5 (#11745) — same-input lift bundles and cross-engine bar-path parity.

This module only checks **committed evidence** (paths + JSON). It does not run
engines. Full epic acceptance also requires optional stack / parity CI lanes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ._json_validate import validate_payload

ISSUE_ID: Final = 11745

# Mirrors :data:`~src.shared.python.lifting.lift_recipes.RECIPES` bundle paths
# without importing the ``lifting`` package (``lifting/__init__.py`` is heavy).
_REPO: Final[Path] = Path(__file__).resolve().parents[4]
_LIFT_EVIDENCE: Final[Path] = _REPO / "docs/development/lifting/evidence"
COMMITTED_BUNDLE_PATHS: Final[dict[str, Path]] = {
    "squat": _LIFT_EVIDENCE / "back_squat" / "back_squat_bundle.npz",
    "deadlift": _LIFT_EVIDENCE / "deadlift" / "deadlift_bundle.npz",
    "clean_and_jerk": _LIFT_EVIDENCE / "clean_and_jerk" / "clean_and_jerk_bundle.npz",
    "snatch": _LIFT_EVIDENCE / "snatch" / "snatch_bundle.npz",
    "bench_press": _LIFT_EVIDENCE / "bench_press" / "bench_press_bundle.npz",
}
BACK_SQUAT_BAR_PATH_PARITY: Final[Path] = (
    _LIFT_EVIDENCE / "back_squat" / "bar_path_parity.json"
)

# See :data:`~src.shared.python.lifting.bar_path_parity.REQUIRED_ENGINES`.
REQUIRED_PARITY_ENGINES: Final[tuple[str, ...]] = ("mujoco", "drake", "pinocchio")

FIXTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "issue", "parity"],
    "properties": {
        "schema_version": {"const": "ud.lift_scoping.lift5.v1"},
        "issue": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"const": ISSUE_ID}},
        },
        "parity": {
            "type": "object",
            "required": ["record_relative_path", "required_engines"],
            "properties": {
                "record_relative_path": {"type": "string", "minLength": 8},
                "required_engines": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
            },
        },
    },
}

HOST_GATED_FOLLOWUPS: Final[tuple[str, ...]] = (
    "Run optional-stack / parity CI lanes that replay all five bundles in every "
    "parity engine and extend bar_path_parity.json beyond the back squat where "
    "not yet recorded.",
)


def validate_lift5_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_payload(record, FIXTURE_SCHEMA)
    return dict(record)


def host_gated_followups_block() -> str:
    lines = [f"LIFT-5 (#{ISSUE_ID}) — CI / optional-stack gates:", ""]
    prefix = "- Tracked follow-up (#"
    lines.extend(f"{prefix}{ISSUE_ID}): {t}" for t in HOST_GATED_FOLLOWUPS)
    return "\n".join(lines)


def assert_committed_lift_bundles_exist() -> None:
    """Every standard lift has a small committed ``*_bundle.npz``."""
    for lift, path in COMMITTED_BUNDLE_PATHS.items():
        assert path.is_file(), f"missing bundle for {lift}: {path}"


def load_back_squat_bar_path_parity_record() -> dict[str, Any]:
    return json.loads(BACK_SQUAT_BAR_PATH_PARITY.read_bytes())


def assert_back_squat_parity_record(
    record: Mapping[str, Any], *, fixture: Mapping[str, Any] | None = None
) -> None:
    """The committed squat parity JSON lists required engines with clean replays."""
    if fixture is not None:
        validate_payload(fixture, FIXTURE_SCHEMA)
        expected = tuple(fixture["parity"]["required_engines"])
        parity_path = _REPO / fixture["parity"]["record_relative_path"]
        assert parity_path.resolve() == BACK_SQUAT_BAR_PATH_PARITY.resolve()
    else:
        expected = REQUIRED_PARITY_ENGINES

    engines = record.get("engines")
    assert isinstance(engines, dict), engines
    for name in expected:
        assert name in engines, (name, sorted(engines))
        block = engines[name]
        assert isinstance(block, dict), block
        assert block.get("failure") is None, (name, block.get("failure"))
        assert float(block["max_bar_deviation_m"]) <= 1e-6
