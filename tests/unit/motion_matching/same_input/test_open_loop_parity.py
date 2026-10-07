"""Open-loop same-input replay across MuJoCo, Drake and Pinocchio (#11607).

A 30 ms reference is generated with the pipeline tracking controller held per
step on the committed ground-support spec; replaying its efforts open loop
must reproduce MuJoCo bit for bit and agree in Drake and Pinocchio to well
inside the L2 bounds of epic #11605.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import (
    InputBundle,
    VectorPlant,
    closed_loop,
    generate_reference_bundle,
    open_loop,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.requires_mujoco,
    pytest.mark.requires_drake,
    pytest.mark.requires_pinocchio,
]

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    for module in ("mujoco", "pydrake", "pinocchio"):
        pytest.importorskip(module)
    spec = (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()
    with np.load(EVIDENCE / "dynamics_record.npz") as record:
        return generate_reference_bundle(
            spec, record["time_s"], record["q"], duration_s=0.03
        )


def test_reference_stays_on_the_closure_manifold(bundle) -> None:
    assert bundle.provenance["max_pose_drift_m"] < 1e-12


def test_mujoco_open_loop_replay_is_bit_exact(bundle) -> None:
    rollout = open_loop(
        VectorPlant("mujoco", bundle.spec_bytes),
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
    )
    np.testing.assert_array_equal(rollout.q, bundle.reference_q)
    np.testing.assert_array_equal(rollout.v, bundle.reference_v)


@pytest.mark.parametrize("engine", ["drake", "pinocchio"])
def test_engine_replays_the_mujoco_motion(bundle, engine) -> None:
    rollout = open_loop(
        VectorPlant(engine, bundle.spec_bytes),
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
    )
    np.testing.assert_allclose(rollout.q, bundle.reference_q, rtol=0, atol=1e-10)
    np.testing.assert_allclose(rollout.v, bundle.reference_v, rtol=0, atol=1e-8)


@pytest.mark.parametrize("engine", ["drake", "pinocchio"])
def test_closed_loop_with_the_shared_controller_matches(bundle, engine) -> None:
    spec = (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()
    with np.load(EVIDENCE / "dynamics_record.npz") as record:
        rollout = closed_loop(
            engine, spec, record["time_s"], record["q"], duration_s=0.03
        )
    np.testing.assert_allclose(rollout.q, bundle.reference_q, rtol=0, atol=1e-10)
    np.testing.assert_allclose(rollout.efforts, bundle.efforts, rtol=0, atol=1e-6)
