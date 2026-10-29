"""NSX-24 (#12567): both legacy allocator APIs route through one frame seam.

Each frame's constraints must come from ``feasible_set_from_frame`` and reach the
conic core unchanged.  Numerics are pinned by the existing allocator tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching import contact_force_allocator as cfa
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocator,
    FrameFeasibleSet,
    FrameInputs,
    feasible_set_from_frame,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    DrakeSyntheticForceAdapter,
    MultiEngineTorqueAllocator,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

NV = 12
ACT = np.arange(6, NV)
N_SPHERES = 2


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    j_ground = rng.normal(size=(3 * N_SPHERES, NV))
    j_grip = np.zeros((6, NV))
    j_grip[:, 6:12] = rng.normal(size=(6, 6))
    tau = np.zeros(NV)
    tau[ACT] = rng.normal(size=ACT.size)
    return tau + j_ground.T @ np.tile([0.5, -0.3, 8.0], N_SPHERES), j_ground, j_grip


def _allocator() -> ContactForceAllocator:
    return ContactForceAllocator(NV, ACT, n_contact_spheres=N_SPHERES)


class _SeamSpy:
    """Record what flows from ``feasible_set_from_frame`` into ``solve_qp``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.frames: list[FrameFeasibleSet] = []
        self.solved: list[Any] = []
        real_seam, real_solve = cfa.feasible_set_from_frame, cfa.solve_qp

        def seam(*args: Any, **kwargs: Any) -> FrameFeasibleSet:
            self.frames.append(real_seam(*args, **kwargs))
            return self.frames[-1]

        def solve(p: Any, c: Any, constraints: Any, **kw: Any) -> Any:
            self.solved.append(constraints)
            return real_solve(p, c, constraints, **kw)

        monkeypatch.setattr(cfa, "feasible_set_from_frame", seam)
        monkeypatch.setattr(cfa, "solve_qp", solve)


def test_contact_force_allocator_routes_through_feasible_set_from_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SeamSpy(monkeypatch)
    assert _allocator().allocate(*_scene()).success
    assert len(spy.frames) == len(spy.solved) == 1
    np.testing.assert_array_equal(np.asarray(spy.solved[0].A_eq), spy.frames[0].a_eq)
    np.testing.assert_array_equal(spy.solved[0].lo, spy.frames[0].lo)
    np.testing.assert_array_equal(spy.solved[0].hi, spy.frames[0].hi)


def test_multi_engine_allocate_trajectory_routes_every_frame_through_the_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_frames, nv = 4, 44
    q = np.zeros((n_frames, nv))
    q[:, 2] = 0.85
    zeros = np.zeros((n_frames, nv))
    spy = _SeamSpy(monkeypatch)
    allocator = MultiEngineTorqueAllocator(DrakeSyntheticForceAdapter(nv, n_spheres=6))
    out = allocator.allocate_trajectory(
        np.linspace(0.0, 0.03, n_frames), q, zeros, zeros, stride=2
    )
    assert out.n_frames == len(spy.frames) == len(spy.solved) == 2


def _seam(alloc: ContactForceAllocator, **kwargs: Any) -> FrameFeasibleSet:
    tau, j_ground, j_grip = _scene()
    frame = FrameInputs(tau_rnea=tau, j_ground=j_ground, j_grip=j_grip, **kwargs)
    return feasible_set_from_frame(alloc, frame)


def test_seam_hard_zero_trail_pins_trail_coordinates() -> None:
    alloc = _allocator()
    fs = _seam(
        alloc,
        objective=AllocationObjective.HARD_ZERO_TRAIL,
        trail_arm_indices=(6, 7),
    )
    assert fs.has_actuator_slack
    np.testing.assert_array_equal(fs.lo[:2], 0.0)
    np.testing.assert_array_equal(fs.hi[:2], 0.0)


def test_seam_masked_contact_is_zero_with_no_friction_rows() -> None:
    alloc = _allocator()
    masked = _seam(alloc, contact_mask=(True, False))
    start = alloc.n_actuated + 3
    np.testing.assert_array_equal(masked.lo[start : start + 3], 0.0)
    np.testing.assert_array_equal(masked.hi[start : start + 3], 0.0)
    assert _seam(alloc).a_ub.shape[0] == 2 * 9
    assert masked.a_ub.shape[0] == 9
    with pytest.raises(ValueError, match="contact_mask length"):
        _seam(alloc, contact_mask=(True,))
