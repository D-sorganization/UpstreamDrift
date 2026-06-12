"""Golden desktop/web parity test for counterfactual analyses (issue #7450).

Scientific-integrity guard: the ZTCF/ZVCF definitions had documented
errors in the article layer (2026-06-11 accuracy audit).  The web API
exposes the ORCHESTRATOR's computation as-is; this test pins the web
path (``AnalysisOrchestrator.compute_counterfactual``) to the PyQt6
dashboard's compute path (``GenericPhysicsRecorder.compute_analysis_post_hoc``
+ ``get_counterfactual_series`` — see ``dashboard.window.compute_analysis``)
for the same pendulum fixture, so desktop and web can never disagree.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
    PendulumPhysicsEngine,
)
from src.shared.python.analysis.orchestrator import (
    COUNTERFACTUAL_KIND_REQUIREMENTS,
    AnalysisOrchestrator,
    supported_counterfactual_kinds,
)
from src.shared.python.dashboard.recorder import GenericPhysicsRecorder

pytestmark = [pytest.mark.unit, pytest.mark.scientific, pytest.mark.headless_safe]

N_STEPS = 12
DT = 0.01
TORQUE = np.array([0.8, -0.3])


def _record_pendulum_session() -> GenericPhysicsRecorder:
    """Run a short deterministic double-pendulum sim and record it."""
    engine = PendulumPhysicsEngine()
    recorder = GenericPhysicsRecorder(engine)
    recorder.start()
    recorder.record_step(TORQUE.copy())
    for _ in range(N_STEPS):
        engine.set_control(TORQUE.copy())
        engine.step(DT)
        recorder.record_step(TORQUE.copy())
    recorder.stop()
    return recorder


@pytest.fixture(scope="module")
def desktop_recorder() -> GenericPhysicsRecorder:
    """Desktop path: explicit post-hoc compute, as the PyQt6 button does."""
    recorder = _record_pendulum_session()
    recorder.compute_analysis_post_hoc()
    return recorder


@pytest.fixture(scope="module")
def web_orchestrator() -> AnalysisOrchestrator:
    """Web path: orchestrator on an identically recorded session."""
    return AnalysisOrchestrator(_record_pendulum_session())


@pytest.mark.parametrize("kind", ["ztcf", "zvcf"])
def test_counterfactual_matches_desktop_dashboard(
    kind: str,
    desktop_recorder: GenericPhysicsRecorder,
    web_orchestrator: AnalysisOrchestrator,
) -> None:
    """ZTCF/ZVCF served to the web equal the desktop dashboard's values."""
    times_desktop, values_desktop = desktop_recorder.get_counterfactual_series(kind)
    assert times_desktop.size > 0, "desktop fixture produced no data"

    result = web_orchestrator.compute_counterfactual(kind)

    np.testing.assert_array_equal(np.asarray(result.times), times_desktop)
    np.testing.assert_array_equal(np.asarray(result.values), values_desktop)
    assert result.units == "rad/s^2"
    assert result.metadata["n_frames"] == times_desktop.size
    json.dumps(result.to_dict())  # API payload must be JSON-serializable


@pytest.mark.parametrize("kind", ["gravity", "drift", "control", "total"])
def test_induced_acceleration_matches_desktop_dashboard(
    kind: str,
    desktop_recorder: GenericPhysicsRecorder,
    web_orchestrator: AnalysisOrchestrator,
) -> None:
    """Induced accelerations match the desktop post-hoc decomposition."""
    times_desktop, values_desktop = desktop_recorder.get_induced_acceleration_series(
        kind
    )
    assert times_desktop.size > 0, "desktop fixture produced no data"

    result = web_orchestrator.compute_counterfactual(kind)

    np.testing.assert_array_equal(np.asarray(result.times), times_desktop)
    np.testing.assert_array_equal(np.asarray(result.values), values_desktop)


def test_total_is_drift_plus_control(
    desktop_recorder: GenericPhysicsRecorder,
) -> None:
    """Sanity: superposition identity used by the post-hoc decomposition."""
    _, drift = desktop_recorder.get_induced_acceleration_series("drift")
    _, control = desktop_recorder.get_induced_acceleration_series("control")
    _, total = desktop_recorder.get_induced_acceleration_series("total")
    np.testing.assert_allclose(total, drift + control, rtol=1e-12, atol=1e-12)


def test_pendulum_engine_reports_all_kinds_supported() -> None:
    """Capability probe offers every kind for the full-surface pendulum."""
    engine = PendulumPhysicsEngine()
    assert supported_counterfactual_kinds(engine) == sorted(
        COUNTERFACTUAL_KIND_REQUIREMENTS
    )


def test_capability_probe_is_conservative() -> None:
    """Engines missing any post-hoc method are gated out entirely."""

    class _Partial:
        def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
            pass

        def set_control(self, u: np.ndarray) -> None:
            pass

        def forward(self) -> None:
            pass

        def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
            return np.zeros(2)

        # compute_zvcf / drift / control intentionally missing

    assert supported_counterfactual_kinds(_Partial()) == []
    assert supported_counterfactual_kinds(None) == []
