"""Contract test for issue #14733: OpenSim net-moment gap proposal write-up.

Refs #13560, #14711. Verifies that the proposal document and changelog fragment
accurately capture the conditioning math, Finding A (recorder inconsistency),
recorder fix options, and governance constraints.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_PATH = ROOT / "docs" / "proposals" / "14733-opensim-net-moment-gap-proposal.md"
FRAGMENT_PATH = ROOT / "changes" / "14733-opensim-net-moment-gap-proposal.md"

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_proposal_document_exists_and_contains_governance_guards() -> None:
    """Proposal must exist and state write-up/governance constraints explicitly."""
    assert PROPOSAL_PATH.is_file(), f"Missing proposal document at {PROPOSAL_PATH}"
    text = PROPOSAL_PATH.read_text(encoding="utf-8")

    assert "#13560" in text
    assert "#14711" in text
    assert "Proposal / write-up only" in text
    assert "net-wrench gate stays in force exactly as it is" in text
    assert "per-hand values remain diagnostics only" in text


def test_proposal_contains_conditioning_math_and_cancellation_analysis() -> None:
    """Section 1 must detail the conditioning math and cancellation ratios."""
    text = PROPOSAL_PATH.read_text(encoding="utf-8")

    assert "1. Conditioning math" in text
    assert "cancellation ratio" in text.lower()
    # Overlap swing cancellation ratio of ~57 and error threshold ~0.049%
    assert "0.049%" in text or "0.05%" in text
    assert "overlap-swing" in text
    assert "statically indeterminate" in text


def test_proposal_contains_finding_a_and_self_consistency_residuals() -> None:
    """Section 2 must document Finding A with reference and OpenSim residuals."""
    text = PROPOSAL_PATH.read_text(encoding="utf-8")

    assert "Finding A" in text
    assert "0.0044" in text or "0.0046" in text
    assert "0.019" in text or "0.037" in text
    assert "central difference" in text
    assert "ClubDynamics.from_spec" in text


def test_proposal_specifies_options_and_risks() -> None:
    """Proposal must detail Option B (recorder fix) and risks/limitations."""
    text = PROPOSAL_PATH.read_text(encoding="utf-8")

    assert "Option B" in text
    assert "recorder fix" in text.lower()
    assert "PairTraceRecorder" in text or "ContactGripSimulator" in text
    assert "Dynamic-consistency check" in text
    assert "additional diagnostic only" in text
    assert "Dieter" in text


def test_change_fragment_exists_and_validates() -> None:
    """Changelog fragment must exist and reference #14733."""
    assert FRAGMENT_PATH.is_file(), f"Missing changelog fragment at {FRAGMENT_PATH}"
    content = FRAGMENT_PATH.read_text(encoding="utf-8")
    assert "14733" in content
