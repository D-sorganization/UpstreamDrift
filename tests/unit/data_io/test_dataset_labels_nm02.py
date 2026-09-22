"""NM-02 (#10617): native dataset label completeness and semantic correctness.

Negative and behavioral contracts for the existing DatasetGenerator evidence
surface. Synthetic fixtures validate software contracts only; native residual
cases exercise the ODE double-pendulum adapter on its source clock.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.core.error_utils import SimulationError
from src.shared.python.data_io.dataset_generator import (
    ControlProfile,
    DatasetGenerator,
    GeneratorConfig,
)
from src.shared.python.data_io.dataset_generator.labels import (
    AccelerationKind,
    ActuationKind,
    ChannelAvailability,
    ModelDoFLayout,
    dynamics_residual,
    require_finite_array,
)
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine

pytestmark = pytest.mark.unit


def _config(**kwargs: object) -> GeneratorConfig:
    defaults: dict[str, object] = {
        "num_samples": 1,
        "duration": 0.04,
        "timestep": 0.01,
        "seed": 7,
        "vary_initial_positions": False,
        "vary_initial_velocities": False,
        "control_profiles": [
            ControlProfile(
                name="const", profile_type="constant", parameters={"magnitude": 1.5}
            )
        ],
        "record_mass_matrix": True,
        "record_bias_forces": True,
        "record_gravity": True,
        "record_contact_forces": True,
        "record_drift_control": False,
    }
    defaults.update(kwargs)
    return GeneratorConfig(**defaults)  # type: ignore[arg-type]


class TestUnavailableOptionalNotZero:
    def test_engine_raise_on_mass_matrix_marks_unavailable(self) -> None:
        engine = MockPhysicsEngine(num_joints=3)
        engine.load_from_string("<mock/>")

        def _boom() -> np.ndarray:
            raise RuntimeError("mass matrix unsupported")

        engine.compute_mass_matrix = _boom  # type: ignore[method-assign]
        gen = DatasetGenerator(engine)
        sample = gen.generate(_config()).samples[0]

        evidence = sample.channel_evidence["mass_matrices"]
        assert evidence.availability is ChannelAvailability.UNAVAILABLE
        assert sample.mass_matrices is None
        assert evidence.notes


class TestAccelerationSemantics:
    def test_interval_fd_not_labeled_instantaneous(self) -> None:
        engine = MockPhysicsEngine(num_joints=2)
        engine.load_from_string("<mock/>")
        gen = DatasetGenerator(engine)
        sample = gen.generate(
            _config(
                record_mass_matrix=False,
                record_bias_forces=False,
                record_gravity=False,
                record_contact_forces=False,
            )
        ).samples[0]

        assert sample.acceleration_kind is AccelerationKind.INTERVAL_FINITE_DIFFERENCE
        assert sample.interval_accelerations is not None
        np.testing.assert_allclose(sample.accelerations, sample.interval_accelerations)
        ev = sample.channel_evidence["interval_accelerations"]
        assert "interval" in ev.semantic
        assert "instantaneous" not in ev.semantic

    def test_native_acceleration_recorded_separately_when_available(self) -> None:
        engine = MockPhysicsEngine(num_joints=2)
        engine.load_from_string("<mock/>")
        gen = DatasetGenerator(engine)
        sample = gen.generate(
            _config(
                record_mass_matrix=False,
                record_bias_forces=False,
                record_gravity=False,
                record_contact_forces=False,
            )
        ).samples[0]

        assert sample.native_accelerations is not None
        assert (
            sample.channel_evidence["native_accelerations"].availability
            is ChannelAvailability.AVAILABLE
        )
        assert sample.channel_evidence["native_accelerations"].semantic.startswith(
            "instantaneous"
        )


class TestDimensionContracts:
    def test_quaternion_layout_allows_nq_ne_nv(self) -> None:
        engine = MockPhysicsEngine(num_joints=6, num_q=7, num_v=6, num_u=6)
        engine.load_from_string("<mock/>")
        gen = DatasetGenerator(engine)
        sample = gen.generate(
            _config(
                record_mass_matrix=False,
                record_bias_forces=False,
                record_gravity=False,
                record_contact_forces=False,
            )
        ).samples[0]

        assert sample.dimensions.n_q == 7
        assert sample.dimensions.n_v == 6
        assert sample.dimensions.n_u == 6
        assert sample.positions.shape[1] == 7
        assert sample.velocities.shape[1] == 6
        assert sample.torques.shape[1] == 6

    def test_control_dimension_may_differ_from_nv(self) -> None:
        engine = MockPhysicsEngine(num_joints=4, num_q=4, num_v=4, num_u=2)
        engine.load_from_string("<mock/>")
        gen = DatasetGenerator(engine)
        sample = gen.generate(
            _config(
                record_mass_matrix=False,
                record_bias_forces=False,
                record_gravity=False,
                record_contact_forces=False,
            )
        ).samples[0]

        assert sample.dimensions.n_u == 2
        assert sample.dimensions.n_v == 4
        assert sample.requested_controls is not None
        assert sample.requested_controls.shape[1] == 2
        assert sample.applied_controls is not None
        assert sample.applied_controls.shape[1] == 2


class TestSaturationAppliedVsRequested:
    def test_records_applied_control_under_saturation(self) -> None:
        engine = MockPhysicsEngine(
            num_joints=2,
            control_limits=(-1.0, 1.0),
        )
        engine.load_from_string("<mock/>")
        gen = DatasetGenerator(engine)
        sample = gen.generate(
            _config(
                control_profiles=[
                    ControlProfile(
                        name="big",
                        profile_type="constant",
                        parameters={"magnitude": 5.0},
                    )
                ],
                record_mass_matrix=False,
                record_bias_forces=False,
                record_gravity=False,
                record_contact_forces=False,
            )
        ).samples[0]

        assert sample.actuation_kind is ActuationKind.APPLIED_SATURATED
        assert sample.requested_controls is not None
        assert sample.applied_controls is not None
        assert np.all(np.abs(sample.requested_controls) >= 4.9)
        assert np.all(np.abs(sample.applied_controls) <= 1.0 + 1e-12)
        np.testing.assert_allclose(sample.torques, sample.applied_controls)


class TestContactLabelContract:
    def test_missing_contact_label_marks_unavailable(self) -> None:
        engine = MockPhysicsEngine(num_joints=2)
        engine.load_from_string("<mock/>")

        def _boom() -> np.ndarray:
            raise AttributeError("no contact")

        engine.compute_contact_forces = _boom  # type: ignore[method-assign]
        gen = DatasetGenerator(engine)
        sample = gen.generate(
            _config(
                record_mass_matrix=False,
                record_bias_forces=False,
                record_gravity=False,
                record_contact_forces=True,
            )
        ).samples[0]

        assert sample.contact_forces is None
        assert sample.contact_labels is None
        ev = sample.channel_evidence["contact_forces"]
        assert ev.availability is ChannelAvailability.UNAVAILABLE


class TestStateRestore:
    def test_failure_exposes_when_restore_unsupported(self) -> None:
        engine = MagicMock()
        initial = (np.ones(2), np.zeros(2))
        engine.get_state.return_value = initial
        engine.get_time.return_value = 0.0
        engine.model_name = "mock"
        engine.set_control.side_effect = RuntimeError("boom")
        engine.set_state.side_effect = RuntimeError("cannot restore")

        gen = DatasetGenerator(engine)
        with pytest.raises(StateError, match="(?i)restore"):
            gen.generate(_config(num_samples=1, duration=0.02, timestep=0.01))

    def test_successful_generation_restores_state_and_time(self) -> None:
        engine = MockPhysicsEngine(num_joints=2)
        engine.load_from_string("<mock/>")
        engine.set_state(np.array([0.3, -0.2]), np.array([0.1, 0.0]))
        t0 = engine.get_time()
        q0, v0 = engine.get_state()

        gen = DatasetGenerator(engine)
        gen.generate(
            _config(
                record_mass_matrix=False,
                record_bias_forces=False,
                record_gravity=False,
                record_contact_forces=False,
            )
        )
        q1, v1 = engine.get_state()
        np.testing.assert_allclose(q1, q0)
        np.testing.assert_allclose(v1, v0)
        assert engine.get_time() == pytest.approx(t0)


class TestDynamicsResidual:
    def test_mock_native_residual_near_zero(self) -> None:
        engine = MockPhysicsEngine(num_joints=3)
        engine.load_from_string("<mock/>")
        gen = DatasetGenerator(engine)
        sample = gen.generate(
            _config(
                record_mass_matrix=True,
                record_bias_forces=True,
                record_gravity=False,
                record_contact_forces=False,
            )
        ).samples[0]

        assert sample.mass_matrices is not None
        assert sample.bias_forces is not None
        assert sample.native_accelerations is not None
        residual = dynamics_residual(
            mass=sample.mass_matrices[0],
            acceleration=sample.native_accelerations[0],
            bias=sample.bias_forces[0],
            applied=sample.applied_controls[0]
            if sample.applied_controls is not None
            else sample.torques[0],
        )
        assert float(np.linalg.norm(residual)) < 1e-8


class TestRootForceGate:
    def test_unqualified_root_forces_rejected_for_supervised_data(self) -> None:
        layout = ModelDoFLayout(n_q=7, n_v=6, n_u=6, n_force=6)
        with pytest.raises(ValueError, match="(?i)root.?force"):
            layout.ensure_physically_supervised(
                allow_unqualified_root_forces=False, has_root_force_channels=True
            )


class TestDbCOptimize:
    def test_require_finite_rejects_under_optimize(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        snippet = (
            "from src.shared.python.data_io.dataset_generator.labels import "
            "require_finite_array\n"
            "import numpy as np\n"
            "try:\n"
            "    require_finite_array('x', np.array([1.0, float('nan')]))\n"
            "except ValueError as exc:\n"
            "    assert 'finite' in str(exc).lower()\n"
            "else:\n"
            "    raise SystemExit('expected ValueError')\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-O", "-c", snippet],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout


class TestAllSamplesFailStillAttemptsRestore:
    def test_all_samples_fail_still_raises_simulation_error_after_restore_ok(
        self,
    ) -> None:
        engine = MockPhysicsEngine(num_joints=2)
        engine.load_from_string("<mock/>")
        engine.set_state(np.array([1.0, 2.0]), np.zeros(2))

        def _fail_step(dt: float) -> None:
            raise RuntimeError("step failed")

        engine.step = _fail_step  # type: ignore[method-assign]
        gen = DatasetGenerator(engine)
        with pytest.raises(SimulationError):
            gen.generate(
                _config(
                    num_samples=2,
                    record_mass_matrix=False,
                    record_bias_forces=False,
                    record_gravity=False,
                    record_contact_forces=False,
                )
            )
        q, _ = engine.get_state()
        np.testing.assert_allclose(q, [1.0, 2.0])
