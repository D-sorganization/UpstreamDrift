"""NSX-95 (#12724): committed JSON mirrors the frozen Python catalogue."""

from __future__ import annotations

import pytest

from src.shared.python.loop_nullspace.strategy_catalogue_data import (
    catalogue_matches_config,
    export_catalogue,
    load_catalogue_json,
)

pytestmark = pytest.mark.unit


def test_json_catalogue_matches_python_registry() -> None:
    payload = load_catalogue_json()
    assert catalogue_matches_config(payload)
    assert payload == export_catalogue()
