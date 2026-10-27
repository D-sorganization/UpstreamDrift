"""Shared helpers for WSCG audit evidence contract tests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SCRIPT = (
    REPO_ROOT / "scripts/research/proximal_distal_energy/run_two_hand_wscg_analysis.py"
)
CASE_SCRIPT = (
    REPO_ROOT / "scripts/research/proximal_distal_energy/run_two_hand_wscg_case.py"
)
JSON_PATH = (
    REPO_ROOT
    / "docs/research/proximal_distal_energy_transfer/data/two_hand_wscg_analysis.json"
)
SCRIPT_HASH_KEY = (
    "scripts/research/proximal_distal_energy/run_two_hand_wscg_analysis.py"
)


def mask_source_hash(payload: dict) -> dict:
    """Return a copy with the audit runner hash masked for byte comparison."""
    masked = json.loads(json.dumps(payload))
    masked["source_sha256"][SCRIPT_HASH_KEY] = "masked"
    return masked


@lru_cache(maxsize=1)
def committed_case_metric_keys() -> tuple[str, ...]:
    record = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return tuple(record["cases"]["base"].keys())
