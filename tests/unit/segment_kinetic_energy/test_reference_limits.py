"""Reference swing segment KE envelope used for fixed shading limits (#15396)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.segment_kinetic_energy import SegmentKineticEnergySeries
from src.shared.python.segment_kinetic_energy.shading import (
    REFERENCE_SWING_PEAK_SEGMENT_KE_J,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def _load_provider_class():
    mujoco = pytest.importorskip("mujoco")
    root = (
        Path(__file__).resolve().parents[3]
        / "src/engines/physics_engines/mujoco/python"
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mujoco_humanoid_golf.segment_kinetic_energy_provider import (
        MujocoSegmentKineticEnergyProvider,
    )

    return mujoco, MujocoSegmentKineticEnergyProvider


def _arm_model(mujoco):

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
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def test_reference_peak_matches_recorded_arm_swing_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "disable")
    mujoco, provider_cls = _load_provider_class()
    model = _arm_model(mujoco)
    provider = provider_cls(model)
    times = np.linspace(0.0, 0.04, 40)
    qpos = np.tile(np.array([0.1, 0.2]), (len(times), 1))
    qvel = np.column_stack(
        [
            2.5 * np.sin(np.linspace(0, 2.0 * np.pi, len(times))),
            1.5 * np.cos(np.linspace(0, 2.0 * np.pi, len(times))),
        ]
    )
    series = SegmentKineticEnergySeries.from_trajectory(provider, times, qpos, qvel)
    peak = max(max(seg.total_j for seg in frame.segments) for frame in series.frames)
    assert pytest.approx(peak, rel=1e-3) == REFERENCE_SWING_PEAK_SEGMENT_KE_J
