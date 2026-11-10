"""ADR-0026 slice 1/2: canonical state I/O and cross-engine checkpoint guard (#14936)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
    MuJoCoPhysicsEngine,
)
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)
from src.shared.python.pose_interchange import se3
from src.shared.python.engine_core.engine_availability import (
    PINOCCHIO_AVAILABLE,
)
from src.shared.python.pose_interchange.canonical_state import CONVENTION_TAG_V2

_skip_no_pinocchio = pytest.mark.skipif(
    not PINOCCHIO_AVAILABLE,
    reason="pinocchio not installed",
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FLOATING_MJCF = _REPO_ROOT / "tests/fixtures/feedback_controls/native_floating_muscle_11958.xml"

_MINIMAL_FLOATING_URDF = """<?xml version="1.0"?>
<robot name="float_hinge">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="arm">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="floating_base" type="floating">
    <parent link="world"/>
    <child link="base_link"/>
  </joint>
  <joint name="hinge" type="revolute">
    <parent link="base_link"/>
    <child link="arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0"/>
  </joint>
</robot>
"""


def _random_canonical_on_engine(engine) -> None:
    q, v = engine.get_state()
    rng = np.random.default_rng(14936)
    q_arr = q.astype(np.float64).copy()
    v_arr = v.astype(np.float64).copy()
    q_arr[0:3] = rng.normal(scale=0.2, size=3)
    quat = se3.quat_exp(rng.normal(scale=0.15, size=3))
    q_arr[3:7] = quat
    if q_arr.shape[0] > 7:
        q_arr[7:] = rng.uniform(-0.3, 0.3, size=q_arr.shape[0] - 7)
    v_arr[:] = rng.normal(scale=0.5, size=v_arr.shape[0])
    engine.set_state(q_arr, v_arr)
    if hasattr(engine, "a"):
        engine.a = rng.normal(scale=0.2, size=v_arr.shape[0])


@pytest.mark.requires_mujoco
def test_mujoco_canonical_roundtrip_identity() -> None:
    pytest.importorskip("mujoco")
    if not _FLOATING_MJCF.is_file():
        pytest.skip(f"fixture missing: {_FLOATING_MJCF}")

    engine = MuJoCoPhysicsEngine()
    engine.load_from_path(str(_FLOATING_MJCF))
    _random_canonical_on_engine(engine)

    before = engine.get_canonical_state()
    assert before.convention == CONVENTION_TAG_V2
    engine.set_canonical_state(before)
    after = engine.get_canonical_state()

    np.testing.assert_allclose(after.q, before.q, atol=1.0e-12)
    np.testing.assert_allclose(after.v, before.v, atol=1.0e-12)
    np.testing.assert_allclose(after.a, before.a, atol=1.0e-12)
    assert after.t == pytest.approx(before.t)


@pytest.mark.requires_pinocchio
@_skip_no_pinocchio
def test_pinocchio_canonical_roundtrip_identity() -> None:
    engine = PinocchioPhysicsEngine()
    engine.load_from_string(_MINIMAL_FLOATING_URDF, extension="urdf")
    _random_canonical_on_engine(engine)

    before = engine.get_canonical_state()
    engine.set_canonical_state(before)
    after = engine.get_canonical_state()

    np.testing.assert_allclose(after.q, before.q, atol=1.0e-12)
    np.testing.assert_allclose(after.v, before.v, atol=1.0e-12)
    np.testing.assert_allclose(after.a, before.a, atol=1.0e-12)


@pytest.mark.requires_mujoco
@pytest.mark.requires_pinocchio
@_skip_no_pinocchio
def test_mujoco_checkpoint_restore_on_pinocchio_raises() -> None:
    pytest.importorskip("mujoco")
    if not _FLOATING_MJCF.is_file():
        pytest.skip(f"fixture missing: {_FLOATING_MJCF}")

    mujoco_engine = MuJoCoPhysicsEngine()
    mujoco_engine.load_from_path(str(_FLOATING_MJCF))
    checkpoint = mujoco_engine.save_checkpoint()
    assert checkpoint.engine_type == "mujoco"

    pin_engine = PinocchioPhysicsEngine()
    pin_engine.load_from_string(_MINIMAL_FLOATING_URDF, extension="urdf")

    with pytest.raises(ValueError, match="engine_type"):
        pin_engine.restore_checkpoint(checkpoint)


def test_checkpoint_manager_raises_on_checksum_failure() -> None:
    from src.shared.python.engine_core.checkpoint import CheckpointManager, StateCheckpoint

    class _StubEngine:
        engine_type = "stub"

        def save_checkpoint(self) -> StateCheckpoint:
            return StateCheckpoint.create(
                engine_type="stub",
                engine_state={"q": np.zeros(2), "v": np.zeros(2), "t": 0.0},
                q=np.zeros(2),
                v=np.zeros(2),
                timestamp=0.0,
            )

        def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
            pass

    manager = CheckpointManager()
    engine = _StubEngine()
    cp_id = manager.save(engine)
    cp = manager.get(cp_id)
    assert cp is not None
    tampered = StateCheckpoint(
        id=cp.id,
        timestamp=cp.timestamp,
        wall_time=cp.wall_time,
        engine_type=cp.engine_type,
        engine_state=cp.engine_state,
        q=cp.q,
        v=cp.v,
        step_count=cp.step_count,
        metadata=cp.metadata,
        checksum="0000000000000000",
    )
    manager._checkpoints[-1] = tampered
    manager._checkpoint_index[cp.id] = len(manager._checkpoints) - 1

    with pytest.raises(ValueError, match="integrity check"):
        manager.restore(engine, cp.id)
