"""Same-input parity of the spec model in the MyoSuite runtime (#11612).

The spec MJCF is loaded through MyoSuite's own ``MujocoEnv`` layer and driven
with the shared contact law and the exact weld KKT solve.  Gates L0 (mass and
M(q)), L1 (pointwise constrained accelerations) and L2 (open-loop replay of a
30 ms reference) of epic #11605 compare it with the MuJoCo adapter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import (
    InputBundle,
    VectorPlant,
    generate_reference_bundle,
    open_loop,
    project_to_closure,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.requires_mujoco,
]

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)
FRAMES = (0, 400, 420)
ACCELERATION_ATOL = 1e-6
ACCELERATION_RTOL = 1e-8
MASS_TOLERANCE = 1e-12
POSITION_TOLERANCE = 1e-10


@pytest.fixture(scope="module")
def spec() -> bytes:
    pytest.importorskip("mujoco")
    pytest.importorskip("myosuite")
    return (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()


@pytest.fixture(scope="module")
def plants(spec) -> dict[str, VectorPlant]:
    return {e: VectorPlant(e, spec) for e in ("mujoco", "myosuite")}


@pytest.fixture(scope="module")
def record() -> dict[str, np.ndarray]:
    with np.load(EVIDENCE / "dynamics_record.npz") as data:
        return {key: data[key] for key in ("time_s", "q", "v", "tau")}


@pytest.fixture(scope="module")
def bundle(spec, record) -> InputBundle:
    return generate_reference_bundle(
        spec, record["time_s"], record["q"], duration_s=0.03
    )


def test_model_is_held_by_the_myosuite_runtime(plants) -> None:
    adapter = plants["myosuite"]._adapter
    env = adapter.myosuite_env
    assert env.mj_model is adapter.model
    assert env.mj_data is adapter.data
    assert env.mj_spec is not None
    assert adapter.kkt_regularization == 0.0
    assert plants["myosuite"].coordinate_order == plants["mujoco"].coordinate_order


@pytest.mark.parametrize("frame", FRAMES)
def test_l0_mass_and_mass_matrix_are_identical(plants, record, frame) -> None:
    from src.shared.python.engine_core.mujoco_compat import full_mass_matrix

    mats, masses = {}, {}
    for engine, plant in plants.items():
        adapter = plant._adapter
        adapter.generalized_forces(
            plant._named(record["q"][frame]), plant._named(record["v"][frame])
        )
        mats[engine] = full_mass_matrix(adapter._mj, adapter.model, adapter.data)
        masses[engine] = float(np.sum(adapter.model.body_mass))
    assert abs(masses["myosuite"] - masses["mujoco"]) <= MASS_TOLERANCE
    np.testing.assert_allclose(
        mats["myosuite"], mats["mujoco"], rtol=0, atol=MASS_TOLERANCE
    )


@pytest.mark.parametrize("frame", FRAMES)
def test_l1_accelerations_match_mujoco(plants, record, frame) -> None:
    state = project_to_closure(plants["mujoco"], record["q"][frame], record["v"][frame])
    tau = record["tau"][frame]
    reference = plants["mujoco"].acceleration(state.q, state.v, tau)
    result = plants["myosuite"].acceleration(state.q, state.v, tau)
    bound = ACCELERATION_ATOL + ACCELERATION_RTOL * np.abs(reference).max()
    np.testing.assert_allclose(result, reference, rtol=0, atol=bound)


def test_l2_open_loop_replay_matches_the_mujoco_reference(spec, bundle) -> None:
    rollout = open_loop(
        VectorPlant("myosuite", spec),
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
    )
    np.testing.assert_allclose(
        rollout.q, bundle.reference_q, rtol=0, atol=POSITION_TOLERANCE
    )
