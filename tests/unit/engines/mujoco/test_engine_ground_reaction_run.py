"""A real MuJoCo run through the recorder to the ground-reaction plot (GCV-5, #11711).

``MuJoCoPhysicsEngine.get_ground_reaction_breakdown`` is sampled each step by
``GenericPhysicsRecorder``; ``compute_ground_reaction_plot`` then reads the
recorder's series for a run whose engine and recorded data carry none.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.api.services.ground_reaction_service import (  # noqa: E402
    compute_ground_reaction_plot,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: E402
    MuJoCoPhysicsEngine,
)
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder  # noqa: E402
from tests.unit.engines.mujoco.test_ground_reaction_wiring import (  # noqa: E402
    SETTLE_S,
    STANCE,
    WEIGHT_N,
    WEIGHT_RTOL,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _loaded_engine() -> MuJoCoPhysicsEngine:
    engine = MuJoCoPhysicsEngine()
    engine.load_from_string(STANCE, "xml")
    return engine


def test_unloaded_engine_reports_no_breakdown() -> None:
    assert MuJoCoPhysicsEngine().get_ground_reaction_breakdown() is None


def test_settled_stance_breakdown_carries_the_body_weight() -> None:
    engine = _loaded_engine()
    while engine.get_time() < SETTLE_S:
        engine.step()
    breakdown = engine.get_ground_reaction_breakdown()
    assert breakdown is not None
    net = np.asarray(breakdown.net.force_n)
    assert abs(net[2] - WEIGHT_N) / WEIGHT_N < WEIGHT_RTOL


def test_recorded_run_reaches_the_plot_payload() -> None:
    engine = _loaded_engine()
    recorder = GenericPhysicsRecorder(engine)
    recorder.start()
    while engine.get_time() < SETTLE_S:
        engine.step()
        recorder.record_step()
    recorder.stop()
    run = SimpleNamespace(
        simulation_data={"body_weight_n": WEIGHT_N},
        engine=SimpleNamespace(),
        recorder=recorder,
    )

    payload = compute_ground_reaction_plot(run)

    assert payload["available"] is True
    assert len(payload["time_s"]) == recorder.current_idx
    net_bw = payload["traces"]["net_force_bw"]["z"]
    assert net_bw[-1] == pytest.approx(1.0, rel=WEIGHT_RTOL)
