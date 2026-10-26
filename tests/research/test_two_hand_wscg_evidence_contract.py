"""Characterization tests for committed WSCG audit JSON (hash-masked)."""

from __future__ import annotations

import json

import pytest

from scripts.research.proximal_distal_energy.run_two_hand_wscg_analysis import (
    build_outputs,
)
from tests.research.wscg_analysis_test_support import (
    JSON_PATH,
    committed_case_metric_keys,
    mask_source_hash,
)

pytestmark = pytest.mark.scientific


def test_two_hand_wscg_json_matches_committed_except_source_hash() -> None:
    record, _ = build_outputs()
    committed = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert mask_source_hash(record) == mask_source_hash(committed)
    assert tuple(record["cases"]["base"].keys()) == committed_case_metric_keys()
