"""NSX-16: net bare-club wrench from the live Pinocchio adapter (#12561)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.grip_split import net_club_wrench
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter
from tests.unit.loop_nullspace.test_snapshot_from_adapter import (  # noqa: F401
    PROVENANCE,
    adapter,
    pin_module,  # noqa: F401
    state,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]


def test_net_club_wrench_equals_adapter_bare_club_wrench_in_grip_axes(
    adapter,  # noqa: F811
    state,  # noqa: F811
) -> None:
    q, v = state
    a = np.random.default_rng(12561).normal(size=adapter.nv)
    snap = snapshot_from_adapter(adapter, q, v, a, input_provenance=PROVENANCE)
    ne = adapter.club_newton_euler(q, v, a)
    rec = net_club_wrench(snap, ne)

    force, torque_com = adapter.bare_club_net_wrench(q, v, a)
    origin = snap.grip_frame.origin
    # bare_club_net_wrench torque is about the CoM: transport to the grip origin.
    moment = torque_com + np.cross(np.asarray(ne["com_m"]) - origin, force)
    rot = snap.grip_frame.rotation
    expected = np.concatenate([rot.T @ force, rot.T @ moment])
    assert np.max(np.abs(rec.values - expected)) <= 1e-10 * max(
        1.0, float(np.max(np.abs(expected)))
    )
    assert np.linalg.norm(expected) > 1e-6
