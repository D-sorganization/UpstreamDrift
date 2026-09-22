"""NM-00 (#10615): surrogate discovery pointer re-exports neural_motion audit."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_surrogate_nm00_audit_reexports_neural_motion_surface() -> None:
    from src.shared.python.motion_matching.surrogate import nm00_audit
    from src.shared.python import neural_motion

    assert nm00_audit.AUDIT_SCHEMA == neural_motion.AUDIT_SCHEMA
    assert nm00_audit.audit_neural_artifacts is neural_motion.audit_neural_artifacts
    assert nm00_audit.inspect_parquet_bounded is neural_motion.inspect_parquet_bounded
