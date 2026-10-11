"""Contract tests for fixed-input full-body replay through VectorPlant."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.native_replay_contracts import native_replay_contract_types
from src.shared.python.motion_matching.same_input import ROOT_COORDINATES

pytestmark = pytest.mark.unit


class _Plant:
    coordinate_order = (*ROOT_COORDINATES, "Joint")

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def acceleration(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        self.calls.append((q.copy(), v.copy(), tau.copy()))
        return tau.copy()

    def closure_pose_residual(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[-1]])

    def closure_residuals(
        self, q: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.full(6, q[-1]), np.full(6, v[-1])

    def closure_rate_matrix(self, q: np.ndarray) -> np.ndarray:
        return np.ones((1, len(q)))


def test_validate_generalized_history_rejects_nonzero_root_before_integration() -> None:
    from src.engines.physics_engines.mujoco.python.native_full_body_replay import (
        validate_generalized_effort_history,
    )

    values = np.zeros((2, 7))
    values[-1, 0] = 1.0
    with pytest.raises(ValueError, match="root generalized efforts must be zero"):
        validate_generalized_effort_history(
            _Plant.coordinate_order, np.array([0.0, 0.01]), values
        )


def test_replay_uses_fixed_zoh_rk4_without_projection_or_state_feedback() -> None:
    from src.engines.physics_engines.mujoco.python.native_full_body_replay import (
        integrate_full_body_history,
    )

    plant = _Plant()
    q0, v0 = np.zeros(7), np.zeros(7)
    times = np.array([0.0, 0.01, 0.02])
    efforts = np.zeros((3, 7))
    efforts[:, -1] = 1.0

    output = integrate_full_body_history(plant, q0, v0, times, efforts, substeps=2)

    assert output.time_seconds.tolist() == times.tolist()
    assert output.qpos.shape == (3, 7)
    assert output.qvel.shape == (3, 7)
    assert output.generalized_efforts.shape == (2, 7)
    assert np.all(output.generalized_efforts[:, :6] == 0.0)
    assert np.all(output.generalized_efforts[:, -1] == 1.0)
    # Four RK stages per substep; only the torque table is supplied to dynamics.
    assert len(plant.calls) == 16
    assert all(call[2][-1] == 1.0 for call in plant.calls)
    np.testing.assert_array_equal(output.qpos[0], q0)
    np.testing.assert_array_equal(output.qvel[0], v0)
    assert output.initial_pose_closure_residual.shape == (6,)
    assert output.initial_rate_closure_residual.shape == (6,)


@pytest.mark.parametrize("pose_size, rate_size", [(5, 6), (7, 6), (6, 5), (6, 7)])
def test_replay_rejects_incomplete_dual_grip_weld_observations(
    pose_size: int, rate_size: int
) -> None:
    from src.engines.physics_engines.mujoco.python.native_full_body_replay import (
        integrate_full_body_history,
    )

    class IncompleteClosurePlant(_Plant):
        def closure_residuals(
            self, q: np.ndarray, v: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros(pose_size), np.zeros(rate_size)

    with pytest.raises(ValueError, match="dual-grip weld closure observations"):
        integrate_full_body_history(
            IncompleteClosurePlant(),
            np.zeros(7),
            np.zeros(7),
            np.array([0.0, 0.01]),
            np.zeros((2, 7)),
        )


def test_bundle_builder_uses_t01_generalized_effort_and_complete_qv(
    tmp_path, monkeypatch
) -> None:
    from src.engines.physics_engines.mujoco.python import native_full_body_replay

    contracts = native_replay_contract_types()
    coordinate_order = (*ROOT_COORDINATES, "Joint")
    primitives = ("Px", "Py", "Pz", "Rx", "Ry", "Rz", "Rz")
    spec_path = tmp_path / "synthetic-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "coordinate_order": coordinate_order,
                "joints": [
                    {
                        "primitives": [
                            {"coordinate": name, "primitive": primitive}
                            for name, primitive in zip(
                                coordinate_order, primitives, strict=True
                            )
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_schema = contracts.InitialStateSchema(
        "mujoco-full-body-qv",
        "1.0.0",
        (
            contracts.StateComponentSpec(
                "qpos", contracts.StateComponentRole.POSITION, 7, "SI", "q"
            ),
            contracts.StateComponentSpec(
                "qvel", contracts.StateComponentRole.VELOCITY, 7, "SI", "v"
            ),
        ),
    )
    source_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    model = contracts.ModelIdentity(
        "mujoco",
        "synthetic-full-body",
        "test",
        "1.0.0",
        source_sha,
        native_full_body_replay.PROVIDER_ID,
        "1.0.0",
        "a" * 64,
        state_schema,
        coordinate_order,
        "b" * 64,
    )
    policy = contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id="full-body-weld-kkt",
        solver_version="1.0.0",
        integration_method="python-classical-rk4-8-substeps-over-engine-accelerations",
        step_policy="fixed",
        step_size_seconds=0.01,
        initialization_policy_id="complete-qv-no-assembly-or-reset",
        initialization_policy_version="1.0.0",
        input_player_id="coordinate-generalized-effort-zoh",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="spec-owned-ground-contact-and-dual-grip-weld",
        contact_policy_version="1.0.0",
        contact_policy_sha256="c" * 64,
    )
    monkeypatch.setattr(
        native_full_body_replay,
        "_identity",
        lambda *_args: (model, policy),
    )
    efforts = np.zeros((3, 7))
    efforts[:, -1] = 0.5

    bundle = native_full_body_replay.build_native_full_body_bundle(
        spec_path,
        np.zeros(7),
        np.zeros(7),
        np.array([0.0, 0.01, 0.02]),
        efforts,
        model_id=model.model_id,
        variant_id=model.variant_id,
        experiment_id="opaque:test-full-body",
    )

    assert tuple(value.component_id for value in bundle.initial_state) == (
        "qpos",
        "qvel",
    )
    assert (
        bundle.input_history.input_kind
        is contracts.ActuationInputKind.GENERALIZED_EFFORT
    )
    assert (
        tuple(channel.channel_id for channel in bundle.input_history.channels)
        == coordinate_order
    )
    assert tuple(channel.unit for channel in bundle.input_history.channels) == (
        "N",
        "N",
        "N",
        "N*m",
        "N*m",
        "N*m",
        "N*m",
    )
    assert bundle.policy.integration_method == (
        "python-classical-rk4-8-substeps-over-engine-accelerations"
    )
    assert bundle.policy.state_reset_allowed is False
    assert np.all(np.asarray(bundle.input_history.values)[:, :6] == 0.0)
    native_full_body_replay._validate_full_body_input_contract(
        bundle, spec_path.read_bytes(), contracts
    )

    invalid_channels = (
        contracts.InputChannel(
            "TranslationInputX", "TranslationInputX", "N*m", "TranslationInputX"
        ),
        *bundle.input_history.channels[1:],
    )
    invalid_unit_bundle = SimpleNamespace(
        input_history=SimpleNamespace(
            input_kind=bundle.input_history.input_kind,
            interpolation=bundle.input_history.interpolation,
            timebase_id=bundle.input_history.timebase_id,
            channels=invalid_channels,
        )
    )
    with pytest.raises(ValueError, match="exact source primitive mapping"):
        native_full_body_replay._validate_full_body_input_contract(
            invalid_unit_bundle,
            spec_path.read_bytes(),
            contracts,
        )
    wrong_kind_bundle = SimpleNamespace(
        input_history=SimpleNamespace(
            input_kind=contracts.ActuationInputKind.ACTUATOR_TORQUE,
            interpolation=bundle.input_history.interpolation,
            timebase_id=bundle.input_history.timebase_id,
            channels=bundle.input_history.channels,
        )
    )
    with pytest.raises(ValueError, match="generalized-effort input"):
        native_full_body_replay._validate_full_body_input_contract(
            wrong_kind_bundle,
            spec_path.read_bytes(),
            contracts,
        )

    def rebuild_bundle(*, channels, input_kind, interpolation, timebase_id):
        return contracts.build_experiment_replay_bundle(
            bundle.experiment_id,
            bundle.model,
            bundle.capabilities,
            tuple((item.component_id, item.values) for item in bundle.initial_state),
            channels,
            input_kind,
            interpolation,
            bundle.input_history.time_seconds,
            bundle.input_history.values,
            bundle.policy,
            timebase_id=timebase_id,
        )

    monkeypatch.setattr(
        native_full_body_replay,
        "VectorPlant",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid replay must fail before native model construction"
        ),
    )
    wrong_target = (
        contracts.InputChannel(
            "TranslationInputX", "other-target", "N", "TranslationInputX"
        ),
        *bundle.input_history.channels[1:],
    )
    wrong_unit = (
        contracts.InputChannel(
            "TranslationInputX", "TranslationInputX", "N*m", "TranslationInputX"
        ),
        *bundle.input_history.channels[1:],
    )
    invalid_public_bundles = (
        (
            rebuild_bundle(
                channels=wrong_unit,
                input_kind=contracts.ActuationInputKind.GENERALIZED_EFFORT,
                interpolation=contracts.InputInterpolation.ZERO_ORDER_HOLD,
                timebase_id="simulation_relative",
            ),
            "exact source primitive mapping",
        ),
        (
            rebuild_bundle(
                channels=wrong_target,
                input_kind=contracts.ActuationInputKind.GENERALIZED_EFFORT,
                interpolation=contracts.InputInterpolation.ZERO_ORDER_HOLD,
                timebase_id="simulation_relative",
            ),
            "exact source primitive mapping",
        ),
        (
            rebuild_bundle(
                channels=tuple(
                    contracts.InputChannel(name, name, "N*m", name)
                    for name in coordinate_order
                ),
                input_kind=contracts.ActuationInputKind.ACTUATOR_TORQUE,
                interpolation=contracts.InputInterpolation.ZERO_ORDER_HOLD,
                timebase_id="simulation_relative",
            ),
            "generalized-effort input",
        ),
    )
    for invalid_bundle, message in invalid_public_bundles:
        with pytest.raises(ValueError, match=message):
            native_full_body_replay.replay_native_full_body_bundle(
                invalid_bundle, spec_path
            )

    with pytest.raises(ValueError, match="torque and effort histories require"):
        rebuild_bundle(
            channels=bundle.input_history.channels,
            input_kind=contracts.ActuationInputKind.GENERALIZED_EFFORT,
            interpolation=contracts.InputInterpolation.LINEAR,
            timebase_id="simulation_relative",
        )
    with pytest.raises(ValueError, match="timebase_id must be"):
        rebuild_bundle(
            channels=bundle.input_history.channels,
            input_kind=contracts.ActuationInputKind.GENERALIZED_EFFORT,
            interpolation=contracts.InputInterpolation.ZERO_ORDER_HOLD,
            timebase_id="wall_clock",
        )

    invalid_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    invalid_spec["joints"][0]["primitives"][-1]["primitive"] = "Punknown"
    spec_path.write_text(json.dumps(invalid_spec), encoding="utf-8")
    with pytest.raises(ValueError, match="translational or rotational axis"):
        native_full_body_replay.build_native_full_body_bundle(
            spec_path,
            np.zeros(7),
            np.zeros(7),
            np.array([0.0, 0.01, 0.02]),
            efforts,
            model_id=model.model_id,
            variant_id=model.variant_id,
            experiment_id="opaque:test-invalid-coordinate-units",
        )

    for malformed_primitives in (
        invalid_spec["joints"][0]["primitives"][:-1],
        [
            *invalid_spec["joints"][0]["primitives"][:-1],
            invalid_spec["joints"][0]["primitives"][0],
        ],
    ):
        malformed_spec = {
            "coordinate_order": coordinate_order,
            "joints": [{"primitives": malformed_primitives}],
        }
        spec_path.write_text(json.dumps(malformed_spec), encoding="utf-8")
        with pytest.raises(ValueError, match="one primitive mapping"):
            native_full_body_replay.build_native_full_body_bundle(
                spec_path,
                np.zeros(7),
                np.zeros(7),
                np.array([0.0, 0.01, 0.02]),
                efforts,
                model_id=model.model_id,
                variant_id=model.variant_id,
                experiment_id="opaque:test-invalid-primitive-mapping",
            )
