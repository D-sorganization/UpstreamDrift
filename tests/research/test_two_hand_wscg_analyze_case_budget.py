"""Architecture budget for WSCG per-case analysis helpers."""

from __future__ import annotations

import pytest

from scripts.ci.check_architecture_budget import analyze_python_file
from tests.research.wscg_analysis_test_support import CASE_SCRIPT

pytestmark = pytest.mark.scientific


def test_two_hand_wscg_analyze_case_within_architecture_budget() -> None:
    findings = [
        finding
        for finding in analyze_python_file(
            CASE_SCRIPT, max_function_lines=100, max_parameters=8
        )
        if finding.symbol == "_analyze_case"
    ]
    assert findings == []
