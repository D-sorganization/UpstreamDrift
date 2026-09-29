"""Repository logging hygiene checks."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit


def test_audited_production_modules_use_canonical_logger() -> None:
    """Issue #7282: audited production modules must not bypass log config."""
    audited_paths = [
        "src/api/routes/analysis_tools.py",
        "src/api/routes/simulation_ws.py",
        "src/launchers/base.py",
        "src/launchers/cross_engine_dashboard.py",
    ]

    for relative_path in audited_paths:
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "logging.getLogger(" not in content
        assert "logging_pkg.logging_config import get_logger" in content
