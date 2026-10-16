"""Every standard plot must render from a real recording (#12887, #12878).

The shared plot data manager pre-loads channels (``ground_forces`` among
them) that ``BiomechanicalData`` does not record. A channel that was not
recorded is an empty series, not an ``AttributeError`` that kills the app.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="MuJoCo plotter needs mujoco")
pytest.importorskip("matplotlib", reason="plots need matplotlib")

from matplotlib.figure import Figure  # noqa: E402

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.biomechanics import (  # noqa: E402
    SwingRecorder,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.plotting import (  # noqa: E402
    GolfSwingPlotter,
)
from src.shared.python.biomechanics.biomechanics_data import (  # noqa: E402
    BiomechanicalData,
)

pytestmark = [pytest.mark.unit]

STANDARD_PLOTS = (
    "plot_summary_dashboard",
    "plot_joint_angles",
    "plot_joint_velocities",
    "plot_joint_torques",
    "plot_actuator_powers",
    "plot_energy_analysis",
    "plot_club_head_speed",
    "plot_club_head_trajectory",
    "plot_swing_plane",
    "plot_torque_comparison",
)
FRAMES, DOFS = 40, 4


@pytest.fixture
def recorder() -> SwingRecorder:
    rec = SwingRecorder()
    rec.start_recording()
    for i in range(FRAMES):
        t = i * 0.01
        rec.record_frame(
            BiomechanicalData(
                time=t,
                joint_positions=np.full(DOFS, np.sin(t)),
                joint_velocities=np.full(DOFS, np.cos(t)),
                joint_torques=np.full(DOFS, 1.0 + t),
                actuator_powers=np.full(DOFS, 0.5),
                club_head_position=np.array([t, 0.0, 1.0]),
                club_head_speed=10.0 * t,
                kinetic_energy=1.0 + t,
                potential_energy=2.0,
                total_energy=3.0 + t,
            )
        )
    rec.stop_recording()
    return rec


def test_unrecorded_channel_is_an_empty_series_not_an_error(
    recorder: SwingRecorder,
) -> None:
    times, values = recorder.get_time_series("ground_forces")

    assert len(values) == 0
    assert len(times) == FRAMES


def test_plotter_builds_from_a_recording(recorder: SwingRecorder) -> None:
    assert GolfSwingPlotter(recorder) is not None


@pytest.mark.parametrize("plot_name", STANDARD_PLOTS)
def test_standard_plot_renders_from_a_recording(
    recorder: SwingRecorder, plot_name: str
) -> None:
    plotter = GolfSwingPlotter(recorder)
    figure = Figure()

    getattr(plotter, plot_name)(figure)

    assert len(figure.axes) > 0, f"{plot_name} produced an empty figure"
