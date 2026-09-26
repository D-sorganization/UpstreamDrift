"""Tests for Historical Closed-Issue and Tools Revision Reconciliation (TB-00 #10585).

TDD test-first suite verifying:
1. Reconciliation of closed #9914, #9921, #10003 against current receipts.
2. Tools revision tracking (gitlink SHA, vendor pin, and ownership boundary).
3. Linking of full-body governing issues #10363, #10378, #10430.
"""

from __future__ import annotations

import pytest

from src.shared.python.tour_baselines.reconciliation import (
    get_governing_epics,
    get_historical_reconciliation,
    get_tools_revision_status,
)

pytestmark = pytest.mark.unit


def test_reconciliation_of_closed_issues():
    """Closed issues must be classified with their actual status, not false qualification."""
    records = get_historical_reconciliation()
    issues = {r.issue_number: r for r in records}

    assert 9914 in issues
    assert 9921 in issues
    assert 10003 in issues

    # #9914: reference overlay, not full-swing dynamic match
    r_9914 = issues[9914]
    assert "reference overlay" in r_9914.scope.lower()
    assert r_9914.is_full_swing_qualified is False
    assert r_9914.receipt_path is not None

    # #9921: Simscape historical authority, terminal error exceeded threshold
    r_9921 = issues[9921]
    assert "simscape" in r_9921.scope.lower()
    assert r_9921.is_full_swing_qualified is False
    assert "r2025b" in r_9921.notes.lower()

    # #10003: OpenSim tour matching, replay divergence
    r_10003 = issues[10003]
    assert "opensim" in r_10003.scope.lower()
    assert r_10003.is_full_swing_qualified is False


def test_tools_revision_tracking():
    """Tools revision status records verified gitlink, historical review divergence, and ownership."""
    status = get_tools_revision_status()

    assert status.vendor_submodule_path == "vendor/ud-tools"
    # Pinned commit in HEAD is 855a10cdaf09ca280d9b2e70b140c0790c0d66cf
    assert status.pinned_commit_sha == "855a10cdaf09ca280d9b2e70b140c0790c0d66cf"
    assert (
        status.historical_review_gitlink == "62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1"
    )
    assert "double_pendulum_golf" in status.owned_package_name
    assert status.source_owner == "Tools"


def test_governing_epics_linked():
    """Governing issues #10363, #10378, #10430 must be linked."""
    epics = get_governing_epics()
    epic_numbers = {e.issue_number for e in epics}

    assert 10363 in epic_numbers
    assert 10378 in epic_numbers
    assert 10430 in epic_numbers
