"""Segment-group catalog coverage on shipped golf MJCF models (#15398)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
    FULL_BODY_GOLF_SWING_XML,
)
from src.shared.python.body_part_viz.mujoco_segment_angular_momentum import (
    MujocoSegmentAngularMomentumProvider,
)
from src.shared.python.body_part_viz.mujoco_segment_group_catalog import (
    GROUP_EXCLUSIONS,
    catalog_member_for_body,
    infer_segment_group_catalog,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_GOLFER_XML = (
    Path(__file__).resolve().parents[3]
    / "src/engines/physics_engines/mujoco/models/generated/golfer.xml"
)


def _body_names(model: mujoco.MjModel) -> list[str]:
    out: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name:
            out.append(name)
    return out


def _massive_bodies(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for body_id in range(1, model.nbody):
        if model.body_mass[body_id] > 0.0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name:
                names.append(name)
    return names


def _is_excluded_body(name: str) -> bool:
    lower = name.lower()
    return lower in GROUP_EXCLUSIONS or any(tok in lower for tok in GROUP_EXCLUSIONS)


@pytest.mark.parametrize(
    ("xml", "label"),
    (
        (FULL_BODY_GOLF_SWING_XML, "full_body"),
        pytest.param(
            _GOLFER_XML.read_text(encoding="utf-8"),
            "golfer",
            marks=pytest.mark.skipif(
                not _GOLFER_XML.is_file(),
                reason="generated golfer.xml missing",
            ),
        ),
    ),
)
def test_golf_catalog_covers_massive_bodies_and_groups_sum_to_whole_body(
    xml: str, label: str
) -> None:
    model = mujoco.MjModel.from_xml_string(xml)
    catalog = infer_segment_group_catalog(_body_names(model))
    assigned: set[str] = set()
    for members in catalog.as_mapping().values():
        overlap = assigned.intersection(members)
        assert not overlap, f"{label}: body in multiple groups: {overlap}"
        assigned.update(members)

    for name in _massive_bodies(model):
        group = catalog_member_for_body(catalog, name)
        if group is None:
            assert _is_excluded_body(name), (
                f"{label}: massive body {name!r} is neither grouped nor excluded"
            )
        else:
            assert name in assigned

    data = mujoco.MjData(model)
    rng = np.random.default_rng(15398)
    data.qpos[:] = rng.uniform(-0.2, 0.2, model.nq)
    data.qvel[:] = rng.uniform(-1.0, 1.0, model.nv)
    mujoco.mj_forward(model, data)
    frame = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog, time_s=float(data.time)
    ).get_segment_angular_momentum()
    assert frame is not None and frame.whole_body_nm_s is not None
    parts = [
        np.asarray(vec, dtype=float)
        for vec in frame.groups_nm_s.values()
        if vec is not None
    ]
    assert parts, f"{label}: expected grouped contributions"
    excluded = [
        np.asarray(vec, dtype=float)
        for name, vec in frame.contribution_nm_s.items()
        if vec is not None and _is_excluded_body(name)
    ]
    partition_sum = sum(parts + excluded, start=np.zeros(3))
    np.testing.assert_allclose(
        partition_sum,
        frame.whole_body_nm_s,
        rtol=0.0,
        atol=1e-9,
        err_msg=f"{label}: five groups plus excluded bodies must sum to whole-body H",
    )
