"""Guards for docs/motion_pipeline/formats.md accuracy."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_ROOT = Path(__file__).resolve().parents[2]
_FORMATS = _ROOT / "docs" / "motion_pipeline" / "formats.md"


def test_qualisys_row_documents_c3d_trc_export_not_native_qtm_parser() -> None:
    """Qualisys `.qtm` has no native adapter; readers must export C3D/TRC instead."""
    text = _FORMATS.read_text(encoding="utf-8")
    qualisys_rows = [
        line for line in text.splitlines() if re.search(r"\|\s*\*\*Qualisys\*\*", line)
    ]
    assert len(qualisys_rows) == 1, qualisys_rows
    row = qualisys_rows[0]

    assert "_planned_" not in row
    assert "C3DAdapter" in row
    assert "TRCAdapter" in row
