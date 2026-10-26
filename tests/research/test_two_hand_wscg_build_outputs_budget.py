"""Architecture budget for WSCG build_outputs orchestration."""

from __future__ import annotations

import pytest

from scripts.ci.check_architecture_budget import analyze_python_file
from tests.research.wscg_analysis_test_support import ANALYSIS_SCRIPT

pytestmark = pytest.mark.scientific


def test_two_hand_wscg_build_outputs_within_architecture_budget() -> None:
    findings = [
        finding
        for finding in analyze_python_file(
            ANALYSIS_SCRIPT, max_function_lines=100, max_parameters=8
        )
        if finding.symbol == "build_outputs"
    ]
    assert findings == []
