"""Optional native MyoSuite/MuJoCo runtime smoke for the T01 consumer."""

from __future__ import annotations

import importlib.metadata

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.requires_myosuite]


def test_installed_myo_suite_runtime_replays_exact_frozen_excitation() -> None:
    pytest.importorskip("myosuite")
    pytest.importorskip("mujoco")
    if importlib.metadata.version("MyoSuite") != "3.0.0":
        pytest.skip("native evidence is pinned to the MyoSuite 3.0.0 lane")
    if importlib.metadata.version("mujoco") != "3.6.0":
        pytest.skip("native evidence is pinned to the MuJoCo 3.6.0 lane")

    from src.engines.physics_engines.myosuite.python.native_excitation_replay import (
        build_native_myo_suite_excitation_bundle,
        replay_native_myo_suite_excitation_bundle,
    )

    environment_id = "myoElbowPose1D6MFixed-v0"
    times = np.array([0.0, 0.02, 0.04])
    excitations = np.vstack((np.full(6, 0.25), np.full(6, 0.65), np.full(6, 0.65)))
    bundle = build_native_myo_suite_excitation_bundle(
        environment_id, times, excitations
    )

    replay = replay_native_myo_suite_excitation_bundle(bundle, environment_id)

    assert np.array_equal(replay.time_seconds, times)
    assert np.array_equal(replay.applied_muscle_excitations, excitations[:-1])
    assert replay.integration_states.shape[0] == len(times)
    assert replay.muscle_activations.shape[0] == len(times)
    assert np.all(replay.wrapper_states == replay.wrapper_states[0])
    assert np.isfinite(replay.qpos).all() and np.isfinite(replay.qvel).all()
    assert replay.input_sha256 == bundle.applied_input_sha256
    assert replay.policy_sha256 == bundle.policy_sha256
