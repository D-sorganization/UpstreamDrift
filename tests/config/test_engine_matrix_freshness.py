"""CI freshness test for src/config/engine_capability_matrix.json (MS-71, #10351)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.generate_engine_matrix import generate_matrix_data, MATRIX_PATH, REPO_ROOT

pytestmark = pytest.mark.unit


def test_engine_capability_matrix_file_exists() -> None:
    """The engine capability matrix must exist in src/config/."""
    assert MATRIX_PATH.is_file(), (
        f"Missing {MATRIX_PATH}; run python scripts/generate_engine_matrix.py --write"
    )


def test_engine_capability_matrix_matches_generated() -> None:
    """The committed matrix JSON must be completely fresh with generate_matrix_data."""
    assert MATRIX_PATH.is_file(), f"Missing {MATRIX_PATH}"
    committed_raw = MATRIX_PATH.read_text(encoding="utf-8")
    committed_data = json.loads(committed_raw)

    fresh_data = generate_matrix_data(repo_root=REPO_ROOT)

    # Compare key structures
    assert committed_data.get("schema_version") == fresh_data.get("schema_version")
    assert committed_data.get("advertised_engines") == fresh_data.get(
        "advertised_engines"
    )
    assert committed_data.get("experimental_engines") == fresh_data.get(
        "experimental_engines"
    )

    # Deep equality
    assert committed_data == fresh_data, (
        "src/config/engine_capability_matrix.json is stale; run 'python scripts/generate_engine_matrix.py --write'"
    )
