"""Part of #14944: published scope for the engine-agnostic ZTCF force helper."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCOPE = (
    Path(__file__).resolve().parents[3] / "docs/architecture/ztcf_force_helper_scope.md"
)


def test_ztcf_scope_doc_states_single_pendulum_restriction() -> None:
    """Capability consumers must not treat compute_ztcf_forces as full Newton–Euler."""
    text = SCOPE.read_text(encoding="utf-8")
    assert "single-pendulum" in text
    assert "allow_simplified" in text
    assert "14944" in text
