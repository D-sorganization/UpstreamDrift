"""Architecture-budget wave-0 renewals must cite issue #15029 (acceptance #15029)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARCH_BUDGET = _REPO_ROOT / "scripts/config/architecture_budget.json"
_WAVE0_ISSUE = 15029


def test_architecture_wave0_renewals_reference_issue_15029() -> None:
    data = json.loads(_ARCH_BUDGET.read_text(encoding="utf-8"))
    wave0 = [
        entry
        for entry in data.get("exceptions", [])
        if "Part of #15028" in str(entry.get("reason", ""))
        and f"#{_WAVE0_ISSUE}" in str(entry.get("reason", ""))
    ]
    assert wave0, "expected architecture_budget wave-0 cohort from #15028"
    for entry in wave0:
        assert entry.get("issue") == _WAVE0_ISSUE, entry.get("symbol") or entry.get(
            "path"
        )
