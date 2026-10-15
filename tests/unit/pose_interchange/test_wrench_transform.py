"""NSX-50: closure twist/wrench duals reuse the NSX-02 converter (#12578)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.loop_nullspace.closure_convention import (
    closure_conversion_matrix,
)
from src.shared.python.pose_interchange import wrench_transform as wt

pytestmark = pytest.mark.unit

ENGINES = ("pinocchio", "mujoco", "drake", "opensim")


def _frames(engine: str, seed: int) -> wt.ClosureFrames:
    rng = np.random.default_rng(seed)
    native = wt.engine_closure_convention(engine)
    r_f = (
        np.eye(3)
        if native.frame == "WORLD"
        else Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    )
    return wt.ClosureFrames(
        native=native,
        native_frame_rotation=r_f,
        grip_frame_rotation=Rotation.from_rotvec(rng.normal(size=3)).as_matrix(),
        native_reference_point=rng.normal(size=3),
        grip_reference_point=rng.normal(size=3),
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_power_invariance_and_round_trips(engine: str, seed: int) -> None:
    frames = _frames(engine, seed)
    rng = np.random.default_rng(seed + 50)
    twist_n, wrench_c = rng.normal(size=6), rng.normal(size=6)
    twist_c = wt.twist_to_canonical(twist_n, frames)
    wrench_n = wt.wrench_from_canonical(wrench_c, frames)
    assert twist_c @ wrench_c == pytest.approx(twist_n @ wrench_n, rel=1e-12)
    wt.check_pairing_preserved(twist_c, wrench_c, twist_n, wrench_n)
    np.testing.assert_allclose(
        wt.twist_from_canonical(twist_c, frames), twist_n, atol=1e-10
    )
    np.testing.assert_allclose(
        wt.wrench_to_canonical(wrench_n, frames), wrench_c, atol=1e-10
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_uses_the_nsx02_matrix(engine: str) -> None:
    frames = _frames(engine, 7)
    expected = closure_conversion_matrix(
        native=frames.native,
        native_frame_rotation=frames.native_frame_rotation,
        grip_frame_rotation=frames.grip_frame_rotation,
        native_reference_point=frames.native_reference_point,
        grip_reference_point=frames.grip_reference_point,
    )
    np.testing.assert_array_equal(frames.matrix(), expected)
    wrench = np.arange(1.0, 7.0)
    np.testing.assert_allclose(
        wt.wrench_from_canonical(wrench, frames), expected.T @ wrench
    )


def test_pairing_check_rejects_a_non_dual_wrench() -> None:
    frames = _frames("drake", 3)
    twist_n, wrench_c = np.arange(1.0, 7.0), np.arange(6.0, 0.0, -1.0)
    twist_c = wt.twist_to_canonical(twist_n, frames)
    wrong = frames.matrix() @ wrench_c  # C, not C^T
    with pytest.raises(ValueError, match="power not preserved"):
        wt.check_pairing_preserved(twist_c, wrench_c, twist_n, wrong)


def test_preconditions() -> None:
    frames = _frames("mujoco", 0)
    with pytest.raises(ValueError, match="length 6"):
        wt.twist_to_canonical(np.zeros(5), frames)
    with pytest.raises(ValueError, match="finite"):
        wt.wrench_from_canonical(np.full(6, np.nan), frames)
    with pytest.raises(ValueError, match="unknown engine"):
        wt.engine_closure_convention("simscape")
