"""Pure-data unit tests for the Pose Studio controllers and core layout.

These tests never touch Qt — they exercise only the pure-data state
machine and the two controllers, which is the load-bearing math for
the Pose Studio tool (the rest is layout + signal wiring).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.canonical import (
    CanonicalPose,
    canonical_from_reference_setup,
    canonical_zero_pose,
)
from src.tools.pose_studio.controllers import (
    EngineController,
    HistoryController,
)
from src.tools.pose_studio.core import (
    JOINT_REGION_LAYOUT,
    SUPPORTED_ENGINES,
    EngineStatus,
    joint_region_partitions_reference_fields,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Layout / core
# ---------------------------------------------------------------------------


def test_supported_engines_non_empty() -> None:
    """At least one engine must be available; otherwise the picker is dead."""
    assert SUPPORTED_ENGINES, "SUPPORTED_ENGINES must be non-empty"


def test_joint_region_layout_partitions_reference_fields() -> None:
    """Every canonical field must belong to exactly one body region."""
    assert joint_region_partitions_reference_fields()


def test_joint_region_layout_groups_have_unique_fields() -> None:
    """A field never appears in two regions."""
    seen: set[str] = set()
    for region, fields in JOINT_REGION_LAYOUT.items():
        for f in fields:
            assert f not in seen, f"{f} appears in two regions (incl. {region!r})"
            seen.add(f)


# ---------------------------------------------------------------------------
# EngineController
# ---------------------------------------------------------------------------


def test_engine_controller_constructs_with_default() -> None:
    ctrl = EngineController(SUPPORTED_ENGINES[0])
    assert ctrl.engine_name == SUPPORTED_ENGINES[0]
    assert ctrl.status in {EngineStatus.MOCK, EngineStatus.LIVE}
    assert ctrl.adapter is not None
    assert ctrl.service is not None
    assert isinstance(ctrl.pose, CanonicalPose)


def test_engine_controller_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        EngineController("not-a-real-engine")
    with pytest.raises(TypeError):
        EngineController(123)  # type: ignore[arg-type]


def test_engine_controller_switches_engines() -> None:
    if len(SUPPORTED_ENGINES) < 2:
        pytest.skip("need >= 2 engines to exercise switch")
    a, b = SUPPORTED_ENGINES[0], SUPPORTED_ENGINES[1]
    ctrl = EngineController(a)
    status = ctrl.switch_engine(b)
    assert ctrl.engine_name == b
    assert status == ctrl.status
    assert ctrl.status in {EngineStatus.MOCK, EngineStatus.LIVE}


def test_engine_controller_switch_engine_rejects_unknown() -> None:
    ctrl = EngineController(SUPPORTED_ENGINES[0])
    with pytest.raises(ValueError):
        ctrl.switch_engine("not-a-real-engine")


def test_engine_controller_set_pose_updates_state() -> None:
    ctrl = EngineController(SUPPORTED_ENGINES[0])
    ref = canonical_from_reference_setup()
    ctrl.set_pose(ref)
    assert ctrl.pose is ref


def test_engine_controller_set_pose_type_validation() -> None:
    ctrl = EngineController(SUPPORTED_ENGINES[0])
    with pytest.raises(TypeError):
        ctrl.set_pose("not a pose")  # type: ignore[arg-type]


def test_engine_controller_replays_pose_across_switch() -> None:
    if len(SUPPORTED_ENGINES) < 2:
        pytest.skip("need >= 2 engines to exercise switch")
    a, b = SUPPORTED_ENGINES[0], SUPPORTED_ENGINES[1]
    ctrl = EngineController(a)
    ref = canonical_from_reference_setup()
    ctrl.set_pose(ref)
    ctrl.switch_engine(b)
    # Pose must survive the swap so the 3D view stays in sync.
    assert np.allclose(
        ctrl.pose.pelvis_translation_m,
        ref.pelvis_translation_m,
    )
    for key in ref.angles_full_dict_deg():
        assert ctrl.pose.angle_deg(key) == ref.angle_deg(key)


# ---------------------------------------------------------------------------
# HistoryController
# ---------------------------------------------------------------------------


def test_history_controller_initial_state() -> None:
    h = HistoryController(canonical_zero_pose())
    assert h.depth == 1
    assert not h.can_undo
    assert not h.can_redo
    assert h.undo() is None
    assert h.redo() is None


def test_history_controller_push_undo_redo_cycle() -> None:
    p0 = canonical_zero_pose()
    p1 = canonical_from_reference_setup()
    h = HistoryController(p0)
    h.push(p1)

    assert h.can_undo
    assert not h.can_redo
    assert h.depth == 2

    undone = h.undo()
    assert undone is p0
    assert h.can_redo
    assert h.current is p0

    redone = h.redo()
    assert redone is p1
    assert h.current is p1


def test_history_controller_push_after_undo_drops_redo_branch() -> None:
    p0 = canonical_zero_pose()
    p1 = canonical_from_reference_setup()
    p2 = CanonicalPose(
        pelvis_translation_m=np.zeros(3),
        pelvis_rotation_xyz_deg=np.zeros(3),
        joint_angles_deg={"HipStartPositionX": 1.0},
    )
    h = HistoryController(p0)
    h.push(p1)
    h.undo()
    assert h.can_redo
    h.push(p2)
    # The redo branch (p1) must be discarded.
    assert not h.can_redo
    assert h.current is p2


def test_history_controller_max_depth_trim() -> None:
    p0 = canonical_zero_pose()
    h = HistoryController(p0, max_depth=3)
    poses = [
        CanonicalPose(
            pelvis_translation_m=np.zeros(3),
            pelvis_rotation_xyz_deg=np.zeros(3),
            joint_angles_deg={"HipStartPositionX": float(i + 1)},
        )
        for i in range(5)
    ]
    for p in poses:
        h.push(p)
    # Stack should be trimmed to the cap.
    assert h.depth == 3
    # The current cursor still points at the most recently pushed pose.
    assert h.current is poses[-1]


def test_history_controller_rejects_bad_inputs() -> None:
    with pytest.raises(TypeError):
        HistoryController("not a pose")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HistoryController(canonical_zero_pose(), max_depth=1)
    with pytest.raises(TypeError):
        HistoryController(canonical_zero_pose(), max_depth="big")  # type: ignore[arg-type]
    h = HistoryController(canonical_zero_pose())
    with pytest.raises(TypeError):
        h.push("not a pose")  # type: ignore[arg-type]
