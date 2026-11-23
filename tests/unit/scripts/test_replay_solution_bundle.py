"""Bundle replay bridge validates independently produced boundary state evidence."""

from dataclasses import replace

import numpy as np
import pytest

from scripts.replay_solution_bundle import playback_bundle, verify_bundle_replay
from src.shared.python.optimization.solution_bundle import (
    SolutionBundle,
    save_solution_bundle,
)
from src.shared.python.simulation_backends.protocol import BackendCapabilities, SimState
from src.shared.python.simulation_backends.validation import ReplayTolerance

pytestmark = pytest.mark.unit


class Plant:
    capabilities = BackendCapabilities(name="unit-mass")

    def reset(self, state):
        self.state = state.copy()

    def get_state(self):
        return self.state.copy()

    def set_control(self, command):
        self.command = command.copy()

    def step(self, dt):
        self.state.q += dt * self.state.v
        self.state.v += dt * self.command
        self.state.time += dt


def make_bundle():
    arrays = {
        "time": np.array([0.0, 0.1, 0.2]),
        "q_ref": np.array([[1.0], [1.0], [1.0]]),
        "qd_ref": np.zeros((3, 1)),
        "tau_ff_ref": np.zeros((3, 1)),
        "q_rollout": np.array([[1.0], [0.9975], [0.98525625]]),
        "qd_rollout": np.array([[0.0], [-0.1], [-0.18975]]),
        "tau_cmd": np.array([[-1.0], [-0.8975], [0.0]]),
        "tau_applied": np.array([[-1.0], [-0.8975], [0.0]]),
        "knots": np.array([0.0, 0.0, 0.0, 0.2, 0.2, 0.2]),
        "delta_tau_coeffs": np.zeros((3, 1)),
        "q0": np.array([1.0]),
        "qd0": np.array([0.0]),
        "controller_state0": np.zeros(0),
        "kp": np.ones(1),
        "kd": np.ones(1),
    }
    return SolutionBundle(
        solution_id="synthetic",
        parent_id=None,
        git_sha="a" * 40,
        model_hash="b" * 64,
        asset_hashes={},
        frame="world",
        joint_order=("joint",),
        control={
            "hold": "zoh",
            "terminal_command": "unused_zero",
            "saturation": "none",
        },
        integrator={
            "kind": "fixed_zoh",
            "dt": 0.1,
            "substeps": 2,
            "runtime_sha256": "c" * 64,
        },
        environment={},
        optimizer_seed=1,
        impact_event={"t": 0.3, "q": [0.0], "qd": [0.0]},
        flight_metrics={},
        feasibility_margins={},
        assumptions=(),
        arrays=arrays,
    )


def test_playback_returns_copies_and_does_not_run_simulation(tmp_path):
    path = save_solution_bundle(make_bundle(), tmp_path / "bundle")
    trace = playback_bundle(path, expected_model_hash="b" * 64)
    trace.q[0, 0] = -8
    again = playback_bundle(path, expected_model_hash="b" * 64)
    assert again.q[0, 0] == 1
    assert again.meta["replay_mode"] == "recorded_playback"


def test_fresh_simulation_compares_only_observed_quantities(tmp_path):
    path = save_solution_bundle(make_bundle(), tmp_path / "bundle")
    result = verify_bundle_replay(
        path,
        expected_model_hash="b" * 64,
        runtime_sha256="c" * 64,
        backend_factory=Plant,
        controller_factory=lambda inputs: lambda s: -s.q - s.v,
        tolerances={
            key: ReplayTolerance(atol=1e-12, rtol=0.0)
            for key in ("time", "q", "qd", "tau_cmd")
        },
    )
    assert all(report.passed for report in result.reports.values())
    assert result.applied_torque_verified is False
    assert result.impact_verified is False
    assert result.independently_verified is False


def test_changed_recorded_states_fail_comparison(tmp_path):
    bundle = make_bundle()
    bundle.arrays["q_rollout"][1, 0] += 0.01
    path = save_solution_bundle(bundle, tmp_path / "bundle")
    result = verify_bundle_replay(
        path,
        expected_model_hash="b" * 64,
        runtime_sha256="c" * 64,
        backend_factory=Plant,
        controller_factory=lambda inputs: lambda s: -s.q - s.v,
        tolerances={
            key: ReplayTolerance(atol=1e-12, rtol=0.0)
            for key in ("time", "q", "qd", "tau_cmd")
        },
    )
    assert not result.reports["q"].passed


@pytest.mark.parametrize(
    "change", ["model", "runtime", "impact", "assets", "state", "schedule", "tiny_grid"]
)
def test_unsupported_or_mismatched_policy_fails_before_factory(tmp_path, change):
    bundle = make_bundle()
    expected, runtime = "b" * 64, "c" * 64
    if change == "model":
        expected = "d" * 64
    elif change == "runtime":
        runtime = "d" * 64
    elif change == "impact":
        bundle = replace(bundle, impact_event={"t": 0.1, "q": [0.0], "qd": [0.0]})
    elif change == "assets":
        bundle = replace(bundle, asset_hashes={"external": "d" * 64})
    elif change == "state":
        bundle.arrays["controller_state0"] = np.ones(1)
    elif change == "tiny_grid":
        bundle.arrays["time"] = np.array([0.0, 1e-18, 2e-18])
        bundle = replace(bundle, integrator={**bundle.integrator, "dt": 1e-20})
    else:
        bundle.arrays["time"][1] += 0.001
    path = save_solution_bundle(bundle, tmp_path / "bundle")

    def forbidden():
        pytest.fail("unsupported policy must fail before plant construction")

    with pytest.raises(ValueError):
        verify_bundle_replay(
            path,
            expected_model_hash=expected,
            runtime_sha256=runtime,
            backend_factory=forbidden,
            controller_factory=forbidden,
            tolerances={
                key: ReplayTolerance(atol=0.0, rtol=0.0)
                for key in ("time", "q", "qd", "tau_cmd")
            },
        )


def test_comparison_rejects_nonfinite_allowance_arithmetic(tmp_path):
    path = save_solution_bundle(make_bundle(), tmp_path / "bundle")
    with pytest.raises(ValueError, match="comparison arithmetic"):
        verify_bundle_replay(
            path,
            expected_model_hash="b" * 64,
            runtime_sha256="c" * 64,
            backend_factory=Plant,
            controller_factory=lambda inputs: lambda s: -s.q - s.v,
            tolerances={
                key: ReplayTolerance(atol=1e308, rtol=1e308)
                for key in ("time", "q", "qd", "tau_cmd")
            },
        )


def test_controller_inputs_exclude_recorded_rollout_and_are_detached(tmp_path):
    path = save_solution_bundle(make_bundle(), tmp_path / "bundle")

    def build(inputs):
        assert not hasattr(inputs, "q_rollout")
        assert not hasattr(inputs, "tau_applied")
        inputs.q_ref[:] = 123.0
        return lambda s: -s.q - s.v

    verify_bundle_replay(
        path,
        expected_model_hash="b" * 64,
        runtime_sha256="c" * 64,
        backend_factory=Plant,
        controller_factory=build,
        tolerances={
            key: ReplayTolerance(atol=1e-12, rtol=0.0)
            for key in ("time", "q", "qd", "tau_cmd")
        },
    )
    assert playback_bundle(path, expected_model_hash="b" * 64).q[0, 0] == 1.0


def test_fresh_real_ode_replay_of_serialized_bundle(tmp_path):
    from scripts.replay_optimized_candidate import replay_candidate
    from src.shared.python.simulation_backends import GolfModelParams
    from src.shared.python.simulation_backends.ode_backend import ODEBackend

    def plant():
        return ODEBackend(GolfModelParams.default())

    def control(inputs):
        return lambda state: (
            -inputs.kp * (state.q - inputs.q_ref[0]) - inputs.kd * state.v
        )

    template = make_bundle()
    a = template.arrays
    n, j = 3, 2
    for key in ("q_ref", "qd_ref", "tau_ff_ref"):
        a[key] = np.zeros((n, j))
    for key in ("q0", "qd0"):
        a[key] = np.zeros(j)
    a["q0"][:] = [0.2, -0.1]
    a["kp"] = np.full(j, 1.0)
    a["kd"] = np.full(j, 0.2)
    a["delta_tau_coeffs"] = np.zeros((3, j))
    original = replay_candidate(
        backend_factory=plant,
        controller_factory=lambda: lambda state: -state.q - 0.2 * state.v,
        initial=SimState(a["q0"], a["qd0"]),
        steps=2,
        dt=0.1,
        substeps=2,
    )
    a["q_rollout"], a["qd_rollout"], a["tau_cmd"] = original.q, original.v, original.u
    a["tau_applied"] = np.zeros((n, j))  # Not observed; never used as replay evidence.
    bundle = replace(
        template,
        joint_order=("shoulder", "wrist"),
        impact_event={"t": 0.3, "q": [0.0, 0.0], "qd": [0.0, 0.0]},
    )
    path = save_solution_bundle(bundle, tmp_path / "bundle")
    result = verify_bundle_replay(
        path,
        expected_model_hash="b" * 64,
        runtime_sha256="c" * 64,
        backend_factory=plant,
        controller_factory=control,
        tolerances={
            key: ReplayTolerance(atol=1e-12, rtol=0.0)
            for key in ("time", "q", "qd", "tau_cmd")
        },
    )
    assert all(report.passed for report in result.reports.values())
    assert result.applied_torque_verified is False
    assert result.independently_verified is False


def test_measured_local_replay_rejects_model_and_runtime_mismatch_before_factory(
    tmp_path,
):
    import hashlib
    from scripts.replay_solution_bundle import (
        verify_measured_local_replay,
        measured_replay_runtime,
    )

    model = tmp_path / "model.json"
    model.write_bytes(b"model identity")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    runtime = measured_replay_runtime()
    bundle = replace(
        make_bundle(),
        model_hash=digest,
        integrator={**make_bundle().integrator, "runtime_sha256": runtime},
    )
    path = save_solution_bundle(bundle, tmp_path / "bundle")

    def forbidden():
        pytest.fail("mismatch must fail before factory")

    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model"):
        verify_measured_local_replay(
            path,
            model_path=model,
            backend_factory=forbidden,
            controller_factory=forbidden,
            tolerances={
                k: ReplayTolerance(atol=0.0, rtol=0.0)
                for k in ("time", "q", "qd", "tau_cmd")
            },
        )
    model.write_bytes(b"model identity")
    wrong = replace(
        bundle, integrator={**bundle.integrator, "runtime_sha256": "d" * 64}
    )
    path2 = save_solution_bundle(wrong, tmp_path / "wrong")
    with pytest.raises(ValueError, match="runtime"):
        verify_measured_local_replay(
            path2,
            model_path=model,
            backend_factory=forbidden,
            controller_factory=forbidden,
            tolerances={
                k: ReplayTolerance(atol=0.0, rtol=0.0)
                for k in ("time", "q", "qd", "tau_cmd")
            },
        )


def test_measured_local_replay_reports_limited_identity_not_independent(tmp_path):
    import hashlib
    from scripts.replay_solution_bundle import (
        verify_measured_local_replay,
        measured_replay_runtime,
    )

    model = tmp_path / "model.json"
    model.write_bytes(b"model identity")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    bundle = replace(
        make_bundle(),
        model_hash=digest,
        integrator={
            **make_bundle().integrator,
            "runtime_sha256": measured_replay_runtime(),
        },
    )
    path = save_solution_bundle(bundle, tmp_path / "bundle")
    result = verify_measured_local_replay(
        path,
        model_path=model,
        backend_factory=Plant,
        controller_factory=lambda inputs: lambda s: -s.q - s.v,
        tolerances={
            k: ReplayTolerance(atol=1e-12, rtol=0.0)
            for k in ("time", "q", "qd", "tau_cmd")
        },
    )
    assert all(x.passed for x in result.replay.reports.values())
    assert result.model_file_hash_verified and result.partial_runtime_verified
    assert (
        not result.factory_model_verified and not result.replay.independently_verified
    )
    assert not result.replay.applied_torque_verified
