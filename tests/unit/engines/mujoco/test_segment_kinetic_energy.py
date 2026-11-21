"""Segment kinetic-energy API and physics parity (#15389)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_humanoid_golf.power_flow import PowerFlowAnalyzer
from mujoco_humanoid_golf.segment_kinetic_energy_provider import (
    MujocoSegmentKineticEnergyProvider,
)
from src.shared.python.segment_kinetic_energy import (
    MUJOCO_SYSTEM_KE_ATOL,
    MUJOCO_SYSTEM_KE_RTOL,
    SegmentKineticEnergySeries,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def _arm_model() -> mujoco.MjModel:
    xml = """
    <mujoco>
      <option gravity="0 0 -9.80665"/>
      <worldbody>
        <body name="upper" pos="0 0 2">
          <joint name="j1" type="hinge" axis="0 1 0"/>
          <geom type="box" pos="0.3 0.1 -0.2" size="0.3 0.05 0.05" mass="3"/>
          <body name="lower" pos="0.6 0 0">
            <joint name="j2" type="hinge" axis="0 1 0"/>
            <geom type="box" pos="0.25 -0.15 -0.3" size="0.25 0.05 0.05" mass="2"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="j1"/><motor joint="j2"/></actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _mujoco_ke_truth(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray
) -> float:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    mujoco.mj_energyVel(model, data)
    return float(data.energy[1])


class TestSegmentKeSplit:
    def test_translational_plus_rotational_equals_total_per_body(self) -> None:
        model = _arm_model()
        analyzer = PowerFlowAnalyzer(model)
        qpos = np.array([0.3, -0.7])
        qvel = np.array([1.7, -2.3])
        result = analyzer.compute_power_flow(
            qpos, qvel, np.zeros(model.nv), np.zeros(model.nv)
        )
        assert np.allclose(
            result.segment_kinetic_energy,
            result.segment_ke_translational + result.segment_ke_rotational,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_summed_body_ke_matches_mjdata_energy_vel(self) -> None:
        model = _arm_model()
        analyzer = PowerFlowAnalyzer(model)
        qpos = np.array([0.3, -0.7])
        qvel = np.array([1.7, -2.3])
        result = analyzer.compute_power_flow(
            qpos, qvel, np.zeros(model.nv), np.zeros(model.nv)
        )
        truth = _mujoco_ke_truth(model, qpos, qvel)
        total = float(np.sum(result.segment_kinetic_energy))
        assert total == pytest.approx(
            truth, rel=MUJOCO_SYSTEM_KE_RTOL, abs=MUJOCO_SYSTEM_KE_ATOL
        )


class TestMujocoSegmentKeProvider:
    def test_provider_frame_matches_analyzer_split(self) -> None:
        model = _arm_model()
        provider = MujocoSegmentKineticEnergyProvider(model)
        qpos = np.array([0.2, 0.5])
        qvel = np.array([2.0, -1.0])
        frame = provider.frame_at(qpos, qvel, time_s=0.05)
        assert frame.engine == "mujoco"
        assert frame.system_kinetic_j is not None
        summed = sum(seg.total_j for seg in frame.segments)
        assert summed == pytest.approx(
            frame.system_kinetic_j, rel=MUJOCO_SYSTEM_KE_RTOL, abs=MUJOCO_SYSTEM_KE_ATOL
        )

    def test_time_series_peaks_track_maxima(self) -> None:
        model = _arm_model()
        provider = MujocoSegmentKineticEnergyProvider(model)
        times = np.array([0.0, 0.01, 0.02])
        qpos = np.tile(np.array([0.1, 0.0]), (3, 1))
        qvel = np.array([[0.5, 0.0], [2.0, 0.0], [1.0, 0.0]])
        series = SegmentKineticEnergySeries.from_trajectory(provider, times, qpos, qvel)
        peaks = series.peak_totals()
        assert peaks
        by_body = {p.body: p for p in peaks}
        mid_frame = series.frames[1]
        upper = next(s for s in mid_frame.segments if s.body == "upper")
        assert by_body["upper"].peak_total_j >= upper.total_j - 1e-12
