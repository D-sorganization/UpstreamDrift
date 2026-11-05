"""Golden parity: puttingSetup.golden.json vs the shared putting core (#14472).

The React scenarios panel reads its setup mapping, schema version and scenario
cap from the same golden file (ui/src/utils/puttingSetup.test.ts).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.putting_analytics import service_setup

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (
        Path(__file__).resolve().parents[3] / "ui/src/utils/puttingSetup.golden.json"
    ).read_text(encoding="utf-8")
)


def test_schema_and_cap_are_the_core_values() -> None:
    assert GOLDEN["schema_version"] == service_setup.SCHEMA_VERSION
    assert GOLDEN["suite"] == service_setup.SUITE
    assert GOLDEN["max_scenarios"] == service_setup.MAX_SCENARIOS


@pytest.mark.parametrize("case", GOLDEN["cases"])
def test_case_setups_are_core_setups(case: dict) -> None:
    setup = case["setup"]
    assert tuple(setup) == service_setup.SETUP_KEYS
    assert service_setup.setup_from_args(setup) == setup


def test_studio_import_is_the_core_import() -> None:
    studio = GOLDEN["studio_import"]
    imported = service_setup.import_setup({"document": studio["document"]})
    assert imported["setup"] == pytest.approx(studio["setup"], rel=1e-12)
