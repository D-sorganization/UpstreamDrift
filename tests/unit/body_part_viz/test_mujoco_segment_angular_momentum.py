"""MuJoCo segment angular-momentum provider benchmarks (#15397)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from src.shared.python.body_part_viz.mujoco_segment_angular_momentum import (
    MujocoSegmentAngularMomentumProvider,
    mujoco_subtree_angular_momentum,
)
from src.shared.python.body_part_viz.segment_angular_momentum import SegmentGroupCatalog

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

# Measured 2026-11-21 on MULTILINK_BENCHMARK_XML across 80 random states
# (‖Σ H_i − subtree_angmom‖_∞). Not a theoretical bound.
MULTILINK_ANGMOM_RTOL = 2e-9
MULTILINK_ANGMOM_ATOL = 1e-11

MULTILINK_BENCHMARK_XML = """
<mujoco>
  <option gravity="0 0 -9.80665" timestep="0.002"/>
  <worldbody>
    <body name="pelvis" pos="0 0 1.0">
      <freejoint name="root"/>
      <geom type="sphere" size="0.08" mass="8"/>
      <body name="torso" pos="0 0 0.15">
        <joint name="spine" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.07" mass="12"/>
        <body name="arm" pos="0.15 0 0.3">
          <joint name="shoulder" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.04" mass="3"/>
          <body name="club" pos="0.35 0 0">
            <joint name="wrist" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.02" mass="0.4"/>
          </body>
        </body>
      </body>
      <body name="thigh" pos="0.08 0 -0.05">
        <joint name="hip" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_SPINNING_ROD = """
<mujoco>
  <option gravity="0 0 0"/>
  <compiler eulerseq="xyz"/>
  <worldbody>
    <body name="rod" pos="0 0 1.5" euler="90 0 0">
      <joint name="spin" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 -0.6 0 0 0.6" size="0.02" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""

_FREE_FLIGHT = """
<mujoco>
  <option gravity="0 0 0" timestep="0.005" integrator="RK4"/>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.15 0.1 0.08" mass="6"/>
    </body>
  </worldbody>
</mujoco>
"""


def _catalog_all(*names: str) -> SegmentGroupCatalog:
    if len(names) == 1:
        only = frozenset(names)
        empty: frozenset[str] = frozenset()
        return SegmentGroupCatalog(
            legs=empty, pelvis=only, torso=empty, arms=empty, club=empty
        )
    catalog = {
        "pelvis": frozenset({"pelvis"}),
        "torso": frozenset({"torso"}),
        "arms": frozenset({"arm"}),
        "club": frozenset({"club"}),
        "legs": frozenset({"thigh"}),
    }
    return SegmentGroupCatalog(**catalog)


def _max_sum_error(model: mujoco.MjModel, trials: int = 80) -> float:
    rng = np.random.default_rng(15397)
    peak = 0.0
    catalog = _catalog_all()
    for _ in range(trials):
        data = mujoco.MjData(model)
        data.qpos[:] = rng.uniform(-0.8, 0.8, model.nq)
        data.qvel[:] = rng.uniform(-4.0, 4.0, model.nv)
        mujoco.mj_forward(model, data)
        provider = MujocoSegmentAngularMomentumProvider(
            model, data, catalog=catalog, time_s=0.0
        )
        frame = provider.get_segment_angular_momentum()
        assert frame is not None and frame.whole_body_nm_s is not None
        reference = mujoco_subtree_angular_momentum(model, data, root_body_id=1)
        err = float(np.max(np.abs(np.asarray(frame.whole_body_nm_s) - reference)))
        peak = max(peak, err)
    return peak


@pytest.fixture(scope="module")
def multilink_benchmark_tolerance() -> tuple[float, float]:
    model = mujoco.MjModel.from_xml_string(MULTILINK_BENCHMARK_XML)
    peak = _max_sum_error(model)
    assert peak < 1e-9
    return MULTILINK_ANGMOM_RTOL, max(MULTILINK_ANGMOM_ATOL, peak * 4.0)


def test_segment_contributions_match_subtree_reference(
    multilink_benchmark_tolerance: tuple[float, float],
) -> None:
    rtol, atol = multilink_benchmark_tolerance
    model = mujoco.MjModel.from_xml_string(MULTILINK_BENCHMARK_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = np.linspace(-0.3, 0.3, model.nq)
    data.qvel[:] = np.linspace(-2.0, 2.5, model.nv)
    mujoco.mj_forward(model, data)
    catalog = _catalog_all()
    frame = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog, time_s=0.25
    ).get_segment_angular_momentum()
    assert frame is not None
    assert frame.time_s == 0.25
    reference = mujoco_subtree_angular_momentum(model, data, root_body_id=1)
    assert frame.whole_body_nm_s is not None
    np.testing.assert_allclose(frame.whole_body_nm_s, reference, rtol=rtol, atol=atol)
    group_sum = sum(
        np.asarray(frame.groups_nm_s[name])
        for name in frame.groups_nm_s
        if frame.groups_nm_s[name] is not None
    )
    np.testing.assert_allclose(group_sum, frame.whole_body_nm_s, rtol=0, atol=1e-12)


def test_mujoco_spinning_rod_provider_matches_i_omega() -> None:
    model = mujoco.MjModel.from_xml_string(_SPINNING_ROD)
    data = mujoco.MjData(model)
    data.qpos[:] = [0.0]
    data.qvel[:] = [10.0]
    mujoco.mj_forward(model, data)
    catalog = _catalog_all("rod")
    frame = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog, time_s=0.0
    ).get_segment_angular_momentum()
    assert frame is not None
    contrib = frame.contribution_nm_s["rod"]
    spin = frame.spin_only_nm_s["rod"]
    assert contrib is not None and spin is not None
    np.testing.assert_allclose(contrib, spin, rtol=1e-9, atol=1e-10)


def test_angular_momentum_conserved_in_free_flight() -> None:
    model = mujoco.MjModel.from_xml_string(_FREE_FLIGHT)
    data = mujoco.MjData(model)
    data.qpos[2] = 1.2
    data.qvel[: model.nv] = np.linspace(-1.5, 1.5, model.nv)
    mujoco.mj_forward(model, data)
    catalog = _catalog_all("pelvis")
    provider = MujocoSegmentAngularMomentumProvider(model, data, catalog=catalog)
    initial = provider.get_segment_angular_momentum()
    assert initial is not None and initial.whole_body_nm_s is not None
    for _ in range(200):
        mujoco.mj_step(model, data)
    final = provider.get_segment_angular_momentum()
    assert final is not None and final.whole_body_nm_s is not None
    # RK4 free-flight drift measured ~1e-7 N·m·s over 1 s (gravity off).
    np.testing.assert_allclose(
        np.asarray(initial.whole_body_nm_s),
        np.asarray(final.whole_body_nm_s),
        rtol=0.0,
        atol=5e-7,
    )


def test_world_body_is_unqualified_not_zero() -> None:
    model = mujoco.MjModel.from_xml_string(_SPINNING_ROD)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    catalog = _catalog_all("rod")
    frame = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog
    ).get_segment_angular_momentum()
    assert frame is not None
    assert "world" not in frame.contribution_nm_s
