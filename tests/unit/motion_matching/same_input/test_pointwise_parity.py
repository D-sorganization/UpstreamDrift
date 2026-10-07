"""Pointwise same-input parity of the spec-built engines (#11606, epic #11605).

The fixture is the committed ground-support MuJoCo run: its spec and
``dynamics_record.npz``.  Recorded states violate the dual-grip weld by tens of
micrometres, so they are first projected onto the closure manifold with each
engine's own residuals; the gate is then that MuJoCo (exact KKT), Drake and
Pinocchio return the same constrained accelerations for the same (q, v, tau).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import (
    PARITY_ENGINES,
    VectorPlant,
    project_to_closure,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_mujoco,
    pytest.mark.requires_drake,
    pytest.mark.requires_pinocchio,
]

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)
FRAMES = (0, 400, 420)  # 420 is a contact spike, max|a| ~ 1.3e5
# Acceptance level L1 of epic #11605: |da| <= ATOL + RTOL * max|a|; RTOL is
# well inside eps * cond(M) ~ 1.4e-7 at the frame-420 contact spike.
ACCELERATION_ATOL = 1e-6
ACCELERATION_RTOL = 1e-8
STATE_TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def plants() -> dict[str, VectorPlant]:
    for module in ("mujoco", "pydrake", "pinocchio"):
        pytest.importorskip(module)
    spec = (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()
    return {engine: VectorPlant(engine, spec) for engine in PARITY_ENGINES}


@pytest.fixture(scope="module")
def record() -> dict[str, np.ndarray]:
    with np.load(EVIDENCE / "dynamics_record.npz") as data:
        return {key: data[key] for key in ("q", "v", "tau")}


def test_plants_share_the_spec_coordinate_order(plants) -> None:
    orders = {plant.coordinate_order for plant in plants.values()}
    assert len(orders) == 1


@pytest.mark.parametrize("frame", FRAMES)
def test_projection_gives_the_same_state_in_every_engine(plants, record, frame) -> None:
    q, v = record["q"][frame], record["v"][frame]
    results = {e: project_to_closure(p, q, v) for e, p in plants.items()}
    reference = results["pinocchio"]
    for engine, result in results.items():
        assert result.pose_residual_after <= 1e-13, engine
        assert result.rate_residual_after <= 1e-12, engine
        np.testing.assert_allclose(result.q, reference.q, rtol=0, atol=STATE_TOLERANCE)
        np.testing.assert_allclose(result.v, reference.v, rtol=0, atol=STATE_TOLERANCE)


@pytest.mark.parametrize("frame", FRAMES)
def test_accelerations_agree_on_the_closure_manifold(plants, record, frame) -> None:
    state = project_to_closure(
        plants["pinocchio"], record["q"][frame], record["v"][frame]
    )
    tau = record["tau"][frame]
    accelerations = {
        e: p.acceleration(state.q, state.v, tau) for e, p in plants.items()
    }
    reference = accelerations["pinocchio"]
    bound = ACCELERATION_ATOL + ACCELERATION_RTOL * np.abs(reference).max()
    for engine in ("mujoco", "drake"):
        np.testing.assert_allclose(
            accelerations[engine], reference, rtol=0, atol=bound, err_msg=engine
        )


def test_off_manifold_states_are_formulation_dependent(plants, record) -> None:
    """Why projection is part of the bundle: a violated weld is linearised at
    different points by each engine, so the raw recorded state does not define
    one acceleration."""
    q, v, tau = record["q"][400], record["v"][400], record["tau"][400]
    mujoco = plants["mujoco"].acceleration(q, v, tau)
    pinocchio = plants["pinocchio"].acceleration(q, v, tau)
    assert np.abs(mujoco - pinocchio).max() > 1e3 * ACCELERATION_ATOL


def test_mujoco_pipeline_default_keeps_its_regularisation() -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

    spec = (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()
    assert NativeMujocoFullBodyModel(spec).kkt_regularization == 1e-6
