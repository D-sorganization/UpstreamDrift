"""Multi-sample representation consumers; no alternate-joint dynamics claim.

The real native specification is used with manufactured, nonsingular states.
These samples need not satisfy the closed weld and are never integrated.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import NonCallableMock
from typing import Any

import numpy as np
import pytest

from src.shared.python.pose_interchange.native_joint_state import (
    NativeJointStateAdapter,
)
from src.shared.python.pose_interchange.native_motion_sequence import (
    export_native_motion,
    restore_native_motion,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def native_batch() -> tuple[Any, ...]:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    )
    raw = path.read_bytes()
    adapter = NativeJointStateAdapter(json.loads(raw))
    count = len(adapter.coordinate_order)
    q = np.arange(3 * count, dtype=float).reshape(3, count) / 200
    for group in adapter.groups:
        q[:, adapter.coordinate_order.index(group.coordinates[0])] = [6.4, 6.5, 6.6]
        q[:, adapter.coordinate_order.index(group.coordinates[1])] = [-2, -2.1, -2.2]
    fields = (q, 0.3 + q / 10, -0.2 + q / 20, 1 + q / 30)
    motion = export_native_motion(adapter, [0, 0.17, 0.41], *fields)
    return raw, adapter, motion, fields


def test_named_batch_retains_all_fields_clock_order_and_branch(
    native_batch: tuple[Any, ...],
) -> None:
    _, adapter, motion, fields = native_batch
    restored = restore_native_motion(adapter, motion)
    assert motion.times_s == (0, 0.17, 0.41)
    assert len(motion.coordinate_order) == 27
    for actual, expected in zip(restored, fields, strict=True):
        for row, original in zip(actual, expected, strict=True):
            named = dict(zip(motion.coordinate_order, row, strict=True))
            assert tuple(named) == adapter.coordinate_order
            np.testing.assert_allclose(list(named.values()), original, atol=1e-12)
    # The nonprincipal middle branch and >2pi winding must survive by value.
    for group in adapter.groups:
        first, middle = [
            adapter.coordinate_order.index(n) for n in group.coordinates[:2]
        ]
        assert all(
            row[first] > 2 * np.pi and row[middle] < -np.pi / 2 for row in restored[0]
        )
    with pytest.raises(TypeError):
        restored[0][0][0] = 99


def test_batch_rejects_other_model_before_consumption(
    native_batch: tuple[Any, ...],
) -> None:
    raw, _, motion, _ = native_batch
    changed = json.loads(raw)
    changed["gravity_m_s2"][2] += 0.1
    with pytest.raises(ValueError, match="identity/inventory"):
        restore_native_motion(NativeJointStateAdapter(changed), motion)


def test_batch_rejects_different_native_order(native_batch: tuple[Any, ...]) -> None:
    raw, _, motion, _ = native_batch
    changed = json.loads(raw)
    changed["coordinate_order"].reverse()
    with pytest.raises(ValueError, match="identity/inventory"):
        restore_native_motion(NativeJointStateAdapter(changed), motion)


def test_later_invalid_reference_rejects_whole_batch(
    native_batch: tuple[Any, ...],
) -> None:
    _, adapter, motion, _ = native_batch
    reference = np.array(motion.native_reference)
    middle = adapter.coordinate_order.index(adapter.groups[0].coordinates[1])
    reference[2, middle] = np.pi / 2
    with pytest.raises(ValueError, match=r"sample 2.*0.41"):
        restore_native_motion(adapter, replace(motion, native_reference=reference))


@pytest.mark.live_simulation
@pytest.mark.parametrize("engine_name", ["mujoco", "drake", "pinocchio"])
def test_native_engine_consumes_every_restored_sample(
    native_batch: tuple[Any, ...], engine_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, adapter, motion, fields = native_batch
    package = "pydrake" if engine_name == "drake" else engine_name
    # Unit conftest installs mocks even for missing packages. Never count those
    # as a native runtime; temporarily remove only mocked package entries.
    for name, module in tuple(sys.modules.items()):
        if (name == package or name.startswith(package + ".")) and isinstance(
            module, NonCallableMock
        ):
            monkeypatch.delitem(sys.modules, name)
    pytest.importorskip(package)
    if engine_name == "mujoco":
        from src.engines.physics_engines.mujoco.python.native_model import (
            NativeMujocoModel,
        )

        engine = NativeMujocoModel(raw)
    elif engine_name == "drake":
        from src.engines.physics_engines.drake.python.native_model import (
            NativeDrakeModel,
        )
        from src.shared.python.motion_matching.native_urdf import export_native_urdf

        xml, sidecar = export_native_urdf(raw)
        engine = NativeDrakeModel(xml.encode(), json.dumps(sidecar).encode(), raw)
    else:
        from src.engines.physics_engines.pinocchio.python.native_model import (
            NativePinocchioModel,
        )

        engine = NativePinocchioModel(json.loads(raw))
    restored = restore_native_motion(adapter, motion)
    all_frames = []
    for original, row in zip(fields[0], restored[0], strict=True):
        expected = engine.frame_poses(
            dict(zip(motion.coordinate_order, original, strict=True))
        )
        # Engines must consume names, not the insertion order of a mapping.
        named = dict(reversed(list(zip(motion.coordinate_order, row, strict=True))))
        actual = engine.frame_poses(named)
        assert actual.keys() == expected.keys()
        assert len(actual) == 16
        for name in actual:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=0)
        all_frames.append(np.stack(list(actual.values())))
    assert not np.allclose(all_frames[0], all_frames[-1])
