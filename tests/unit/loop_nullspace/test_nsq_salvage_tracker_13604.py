"""Salvage-tracker receipts for NSX wave rows marked merged (#13604).

The tracker epic lists slices that landed on main while their GitHub issues
stayed open. These tests pin the public seams so a stale row is visible in CI.
They do not close issues (#12583 remains MATLAB-deferred).
"""

from __future__ import annotations

import inspect

import pytest

from src.shared.python.loop_nullspace import envelopes
from src.shared.python.motion_matching import contact_force_allocator as cfa

pytestmark = pytest.mark.unit


def test_nsq22_static_envelopes_seam_on_main() -> None:
    """#12565 / NSX-22: envelope LP API is the merged product view."""
    assert inspect.isfunction(envelopes.static_envelopes)


def test_nsq24_allocate_contact_frame_seam_on_main() -> None:
    """#12567 / NSX-24: legacy allocators delegate through one frame entry."""
    assert inspect.isfunction(cfa.allocate_contact_frame)
    assert inspect.isfunction(cfa.feasible_set_from_frame)


def test_nsq24_no_slsqp_guard_module_present() -> None:
    """Acceptance grep-test for convex allocation paths ships on main."""
    from tests.unit.repo_hygiene import test_no_slsqp_on_convex_allocation

    assert hasattr(test_no_slsqp_on_convex_allocation, "test_allocation_paths_exist")
