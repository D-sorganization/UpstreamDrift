"""Shared loaders for the swing report contract tests (#12324)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = REPO_ROOT / "schemas" / "swing-report-v1.json"
EXAMPLES_PATH = REPO_ROOT / "schemas" / "swing-report-examples.json"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "swing_report"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def schema() -> dict[str, Any]:
    return load_json(SCHEMA_PATH)


@pytest.fixture
def valid_full() -> dict[str, Any]:
    return load_json(FIXTURE_DIR / "valid_full.json")


@pytest.fixture
def valid_custom() -> dict[str, Any]:
    return load_json(FIXTURE_DIR / "valid_custom.json")
