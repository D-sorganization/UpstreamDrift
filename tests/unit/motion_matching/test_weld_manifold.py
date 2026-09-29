"""Forward dynamics must start on the dual-grip weld's constraint manifold (#11043).

The KKT solve enforces only ``J a = -J̇ q̇``, so the weld conserves whatever
relative velocity it starts with. A replay seeded with a finite-difference
velocity that violates the weld opened the grip at 187 mm/s on the fitted
closure run and drove the tracked replay to an 861 mm receipt.
"""

from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching import full_body_forward_dynamics as fs
from src.shared.python.motion_matching.pipeline import dynamics as pipeline_dynamics
from src.shared.python.motion_matching.weld_manifold import project_onto_weld

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
CANDIDATE = (
    ROOT
    / "docs/development/full_body_models/evidence/native_candidates"
    / "returned81_candidate.json"
)


def _toy_system(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(10, 10))
    mass = basis @ basis.T + 10 * np.eye(10)
    return mass, rng.normal(size=(6, 10)), rng.normal(size=10)


def test_projection_satisfies_the_weld_and_is_idempotent() -> None:
    mass, jac, rates = _toy_system()
    projected = project_onto_weld(mass, jac, rates)
    assert np.abs(jac @ rates).max() > 0.1
    np.testing.assert_allclose(jac @ projected, 0.0, atol=1e-10)
    np.testing.assert_allclose(project_onto_weld(mass, jac, projected), projected)


def test_projection_is_the_minimum_kinetic_energy_impulse() -> None:
    """The removed velocity is mass-orthogonal to every weld-admissible motion,
    which is what an inelastic impulse through the weld does."""
    mass, jac, rates = _toy_system(1)
    removed = rates - project_onto_weld(mass, jac, rates)
    _, _, vt = np.linalg.svd(jac)
    admissible = vt[jac.shape[0] :].T
    np.testing.assert_allclose(removed @ mass @ admissible, 0.0, atol=1e-9)


def test_projection_rejects_mismatched_shapes() -> None:
    mass, jac, rates = _toy_system()
    with pytest.raises(ValueError, match="rates"):
        project_onto_weld(mass, jac, rates[:-1])
    with pytest.raises(ValueError, match="mass"):
        project_onto_weld(mass[:-1], jac, rates)
    with pytest.raises(ValueError, match="finite"):
        project_onto_weld(mass, jac, np.full(10, np.nan))


@pytest.fixture(scope="module")
def simulator() -> fs.FullBodySimulator:
    return fs.FullBodySimulator(NativeMujocoFullBodyModel(SPEC.read_bytes()))


def _candidate_pose(simulator: fs.FullBodySimulator) -> np.ndarray:
    candidate = json.loads(CANDIDATE.read_text())
    q = np.zeros(simulator.nv)
    for name, value in zip(candidate["coordinate_names"], candidate["q0"], strict=True):
        q[simulator.names.index(name)] = value
    return fs.preload_feet(simulator, q)


def _weld_rate_residual(
    simulator: fs.FullBodySimulator, q: np.ndarray, v: np.ndarray
) -> float:
    simulator.acceleration(q, v, np.zeros(simulator.nv))
    return float(np.linalg.norm(simulator.adapter.closure_errors()[1]))


def _violating_rates(simulator: fs.FullBodySimulator) -> np.ndarray:
    return np.random.default_rng(3).normal(scale=0.5, size=simulator.nv)


def test_simulator_consistent_velocity_satisfies_the_weld(
    simulator: fs.FullBodySimulator,
) -> None:
    q0 = _candidate_pose(simulator)
    raw = _violating_rates(simulator)
    consistent = simulator.consistent_velocity(q0, raw)
    assert _weld_rate_residual(simulator, q0, raw) > 0.05
    assert _weld_rate_residual(simulator, q0, consistent) < 1e-9


def test_consistent_start_keeps_the_grip_closed(
    simulator: fs.FullBodySimulator,
) -> None:
    """A violating start opens the weld linearly; a consistent one does not."""
    q0 = _candidate_pose(simulator)
    raw = _violating_rates(simulator)

    def unactuated(t: float, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(simulator.nv)

    def grip_gap_m(v0: np.ndarray) -> float:
        record = simulator.run(q0, v0, unactuated, duration_s=0.03, dt_s=1e-3)
        simulator.acceleration(record.q[-1], record.v[-1], np.zeros(simulator.nv))
        return float(np.linalg.norm(simulator.adapter.closure_errors()[0][:3]))

    assert grip_gap_m(raw) > 1e-3
    assert grip_gap_m(simulator.consistent_velocity(q0, raw)) < 2e-4


def test_replay_starts_on_the_weld(
    simulator: fs.FullBodySimulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finite-difference start velocity of a reference is projected before
    the tracked replay integrates it."""
    q0 = _candidate_pose(simulator)
    times = np.linspace(0.0, 0.1, 11)
    q_track = q0 + np.outer(times, _violating_rates(simulator))
    captured: dict[str, np.ndarray] = {}

    class Captured(Exception):
        pass

    def fake_run(q: np.ndarray, v: np.ndarray, *args: object, **kwargs: object):
        captured["q"], captured["v"] = q, v
        raise Captured

    monkeypatch.setattr(simulator, "run", fake_run)
    with pytest.raises(Captured):
        pipeline_dynamics.replay(simulator, types.SimpleNamespace(times=times), q_track)
    assert _weld_rate_residual(simulator, captured["q"], captured["v"]) < 1e-9
