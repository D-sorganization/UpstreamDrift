"""Pure-numpy retarget round-trip tests for MyoSuite coordinate map (MS-52, #10345)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.myosuite.python.retarget import (
    RetargetMap,
    interpolate_unmapped,
    load_retarget_map,
    retarget_frame,
    retarget_trajectory,
)

pytestmark = pytest.mark.unit

FIXTURE_MAP = Path(__file__).resolve().parents[4] / (
    "src/engines/physics_engines/myosuite/python/coordinate_map_anthro.json"
)


def test_identity_round_trip_equals_input() -> None:
    """Retarget round-trip on an identity map must reproduce the input."""
    names = ("a", "b", "c")
    identity = RetargetMap(
        source_names=names,
        target_names=names,
        source_to_target={name: (idx, 1.0) for idx, name in enumerate(names)},
        chain_order=names,
        neutral_target=np.zeros(len(names)),
        omitted_target=(),
    )
    q = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    mapped = retarget_frame(q, identity)
    round_trip = identity.project_to_source(mapped)
    np.testing.assert_allclose(round_trip, q, rtol=0, atol=1e-12)


def test_interpolate_unmapped_fills_from_chain_parent() -> None:
    """Unmapped coordinates inherit the nearest mapped ancestor on the chain."""
    names = ("root", "mid", "tip")
    partial = RetargetMap(
        source_names=("root",),
        target_names=names,
        source_to_target={"root": (0, 1.0)},
        chain_order=names,
        neutral_target=np.array([0.0, 0.5, 1.0]),
        omitted_target=("mid", "tip"),
    )
    q = np.array([0.25])
    out = retarget_frame(q, partial)
    np.testing.assert_allclose(out[0], 0.25)
    assert out[1] == pytest.approx(0.25)
    assert out[2] == pytest.approx(0.25)


def test_retarget_trajectory_preserves_time_axis() -> None:
    names = ("x", "y")
    identity = RetargetMap(
        source_names=names,
        target_names=names,
        source_to_target={name: (idx, 1.0) for idx, name in enumerate(names)},
        chain_order=names,
        neutral_target=np.zeros(2),
        omitted_target=(),
    )
    q_traj = np.array([[0.0, 0.0], [0.1, -0.1], [0.2, 0.05]], dtype=np.float64)
    out = retarget_trajectory(q_traj, identity)
    assert out.shape == q_traj.shape
    np.testing.assert_allclose(out, q_traj)


def test_fixture_map_loads_and_maps_subset() -> None:
    if not FIXTURE_MAP.is_file():
        pytest.skip("coordinate_map_anthro.json not present")
    doc = json.loads(FIXTURE_MAP.read_text())
    rmap = load_retarget_map(doc)
    assert rmap.source_names
    assert rmap.target_names
    q_src = np.zeros(len(rmap.source_names))
    q_tgt = retarget_frame(q_src, rmap)
    assert q_tgt.shape == (len(rmap.target_names),)
    assert np.all(np.isfinite(q_tgt))


def test_signed_mapping_applies_multiplier() -> None:
    rmap = RetargetMap(
        source_names=("joint",),
        target_names=("joint",),
        source_to_target={"joint": (0, -1.0)},
        chain_order=("joint",),
        neutral_target=np.zeros(1),
        omitted_target=(),
    )
    out = retarget_frame(np.array([0.3]), rmap)
    assert out[0] == pytest.approx(-0.3)


def test_interpolate_unmapped_respects_neutral_baseline() -> None:
    names = ("a", "b")
    rmap = RetargetMap(
        source_names=(),
        target_names=names,
        source_to_target={},
        chain_order=names,
        neutral_target=np.array([0.2, 0.4]),
        omitted_target=names,
    )
    out = interpolate_unmapped(np.array([0.0, 0.0]), rmap)
    np.testing.assert_allclose(out, [0.2, 0.4])


def _fixture_map() -> tuple[dict, RetargetMap]:
    if not FIXTURE_MAP.is_file():
        pytest.skip("coordinate_map_anthro.json not present")
    doc = json.loads(FIXTURE_MAP.read_text())
    return doc, load_retarget_map(doc)


def _retarget_single(rmap: RetargetMap, source: str, value: float) -> np.ndarray:
    q = np.zeros(rmap.n_source)
    q[rmap.source_names.index(source)] = value
    return retarget_frame(q, rmap)


def test_neck_input_y_drives_neck_flexion_with_fk_sign() -> None:
    """NeckInputY (Ry, head pitch) -> neck_flexion, sign -1 by FK (#11729).

    +NeckInputY pitches the native head forward axis down; +neck_flexion pitches
    the MyoSuite head up (evidence/myosuite_neck/neck_flexion_sign.json).
    """
    _, rmap = _fixture_map()
    assert rmap.source_to_target["NeckInputY"] == (
        rmap.target_names.index("neck_flexion"),
        -1.0,
    )
    out = _retarget_single(rmap, "NeckInputY", 0.3)
    assert out[rmap.target_names.index("neck_flexion")] == pytest.approx(-0.3)
    assert out[rmap.target_names.index("neck_rotation")] == pytest.approx(0.0)


def test_neck_input_x_is_unmapped_and_documented() -> None:
    """NeckInputX is lateral bending; myo_sim has no matching head DOF."""
    doc, rmap = _fixture_map()
    assert "NeckInputX" not in rmap.source_to_target
    assert "NeckInputX" in doc["omitted_source"]
    assert "lateral bending" in doc["omitted_source_notes"]["NeckInputX"]
    out = _retarget_single(rmap, "NeckInputX", 0.3)
    for joint in ("neck_flexion", "neck_rotation"):
        assert out[rmap.target_names.index(joint)] == pytest.approx(0.0)


def test_neck_input_z_alone_drives_neck_rotation() -> None:
    _, rmap = _fixture_map()
    out = _retarget_single(rmap, "NeckInputZ", 0.3)
    assert out[rmap.target_names.index("neck_rotation")] == pytest.approx(0.3)
    assert out[rmap.target_names.index("neck_flexion")] == pytest.approx(0.0)


def test_fixture_map_partitions_source_coordinates() -> None:
    """Every source coordinate is either mapped or listed as omitted, never both."""
    doc, rmap = _fixture_map()
    mapped = set(rmap.source_to_target)
    omitted = set(doc["omitted_source"])
    assert not mapped & omitted
    assert mapped | omitted == set(rmap.source_names)
    assert set(doc["omitted_source_notes"]) == omitted
