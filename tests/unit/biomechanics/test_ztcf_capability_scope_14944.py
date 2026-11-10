"""Part of #14944: published scope for the engine-agnostic ZTCF force helper."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCOPE = (
    Path(__file__).resolve().parents[3] / "docs/architecture/ztcf_force_helper_scope.md"
)
ATLAS = Path(__file__).resolve().parents[3] / "docs/architecture/CAPABILITY_ATLAS.md"


def test_ztcf_scope_doc_states_single_pendulum_restriction() -> None:
    """Capability consumers must not treat compute_ztcf_forces as full Newton–Euler."""
    text = SCOPE.read_text(encoding="utf-8")
    assert "single-pendulum" in text
    assert "allow_simplified" in text
    assert "14944" in text


def test_scope_and_docstring_honesty_14944() -> None:
    """Document that 2D helper is not a true reaction force and atlas does not claim multi-body RNEA."""
    from src.shared.python.biomechanics.ztcf import compute_ztcf_forces

    text, doc = SCOPE.read_text(encoding="utf-8"), compute_ztcf_forces.__doc__ or ""
    phrase = "NOT a physical joint reaction force"
    assert "90" in text and phrase in text and "compute_ztcf_joint_reactions" in text
    assert "m·L·q̈" in doc and "90" in doc and phrase in doc
    atlas = ATLAS.read_text(encoding="utf-8")
    for line in atlas.splitlines():
        if "src/shared/python/biomechanics/ztcf.py" in line:
            assert "Newton" not in line
