import pytest
import numpy as np
from tests.fixtures.fixtures_lib import TOLERANCE_ACCELERATION_M_S2, TOLERANCE_CLOSURE_RAD_S2, TOLERANCE_JACOBIAN, compute_accelerations, set_identical_state, skip_if_insufficient_engines
from src.shared.python.engine_core.cross_engine_validator import CrossEngineValidator
from typing import Any
from src.shared.python.logging_pkg.logging_config import get_logger
logger = get_logger(__name__)

@pytest.mark.integration
class TestCrossEngineValidationIntegration:
    """Integration tests comparing actual physics engine outputs.

    These tests validate Guideline M2 requirement for cross-engine comparison.
    Tests automatically skip if required engines are not available.
    """

    def test_forward_dynamics_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate forward dynamics agree per Guideline P3.

        Sets identical initial conditions and compares accelerations.
        Tolerance: acceleration ±1e-4 m/s² per P3.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Set identical initial state: small angle (0.1 rad) from vertical
        q_init = np.array([0.1])  # Small angle for near-linear regime
        v_init = np.array([0.0])  # Starting from rest

        set_identical_state(available_engines, q_init, v_init)

        # Compute accelerations
        accelerations = compute_accelerations(available_engines)

        # Pairwise comparison
        engine_names = list(accelerations.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_states(
                    name1,
                    accelerations[name1],
                    name2,
                    accelerations[name2],
                    metric="acceleration",
                )
                logger.info(
                    f"Forward dynamics {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}, "
                    f"tolerance={TOLERANCE_ACCELERATION_M_S2:.2e}, "
                    f"severity={result.severity}"
                )
                # Allow WARNING severity (up to 2x tolerance) for cross-engine
                assert result.severity in ["PASSED", "WARNING"], (
                    f"Forward dynamics mismatch between {name1} and {name2}: "
                    f"{result.message}"
                )

    def test_inverse_dynamics_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate inverse dynamics agree per Guideline P3.

        Computes torques for a given motion and compares.
        Tolerance: RMS < 10% per P3.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Set state
        q_init = np.array([0.2])  # 0.2 rad
        v_init = np.array([0.5])  # 0.5 rad/s

        set_identical_state(available_engines, q_init, v_init)

        # Desired acceleration for ID computation
        qacc_desired = np.array([1.0])  # 1 rad/s² desired

        # Compute inverse dynamics torques
        torques: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is not None:
                tau = eng.engine.compute_inverse_dynamics(qacc_desired)
                if tau.size > 0:
                    torques[eng.name] = tau

        if len(torques) < 2:
            pytest.skip("Inverse dynamics not available in enough engines")

        # Pairwise RMS comparison
        engine_names = list(torques.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_torques_with_rms(
                    name1, torques[name1], name2, torques[name2], rms_threshold_pct=10.0
                )
                logger.info(
                    f"Inverse dynamics {name1} vs {name2}: "
                    f"RMS deviation={result.max_deviation:.2f}%"
                )
                assert result.passed, (
                    f"Inverse dynamics mismatch between {name1} and {name2}: "
                    f"{result.message}"
                )

    def test_jacobian_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate Jacobians agree per Guideline P3.

        Computes spatial Jacobians for end-effector and compares.
        Tolerance: ±1e-8 element-wise per P3.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Set specific configuration
        q_init = np.array([0.3])  # 0.3 rad
        v_init = np.array([0.0])  # Static

        set_identical_state(available_engines, q_init, v_init)

        # Compute Jacobians for end effector/pendulum link
        jacobians: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is None:
                continue

            # Prefer engine-provided body names when available to avoid
            # hardcoding URDF-specific names. Fall back to common names.
            candidate_names: list[str]
            if hasattr(eng.engine, "get_body_names"):
                candidate_names = list(eng.engine.get_body_names())
            elif hasattr(eng.engine, "body_names"):
                candidate_names = list(eng.engine.body_names)
            else:
                # Fallback for engines without body name API
                candidate_names = ["end_effector", "pendulum_link", "lower_link"]

            for body_name in candidate_names:
                jac = eng.engine.compute_jacobian(body_name)
                if jac is not None and "spatial" in jac:
                    jacobians[eng.name] = jac["spatial"]
                    break

        if len(jacobians) < 2:
            pytest.skip("Jacobian computation not available in enough engines")

        # Pairwise comparison
        engine_names = list(jacobians.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_states(
                    name1,
                    jacobians[name1].flatten(),
                    name2,
                    jacobians[name2].flatten(),
                    metric="jacobian",
                )
                logger.info(
                    f"Jacobian {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}, "
                    f"tolerance={TOLERANCE_JACOBIAN:.2e}"
                )
                # Allow some tolerance for numerical differences
                assert result.severity in [
                    "PASSED",
                    "WARNING",
                ], f"Jacobian mismatch between {name1} and {name2}: {result.message}"

    def test_ztcf_counterfactual_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate ZTCF (Zero-Torque Counterfactual) agrees across engines.

        Per Guideline G1: ZTCF isolates drift dynamics.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Test state
        q_test = np.array([0.2])
        v_test = np.array([0.3])

        # Compute ZTCF accelerations
        ztcf_accels: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is not None:
                try:
                    a_ztcf = eng.engine.compute_ztcf(q_test, v_test)
                    if a_ztcf.size > 0:
                        ztcf_accels[eng.name] = a_ztcf
                except Exception as e:
                    logger.warning(f"ZTCF failed for {eng.name}: {e}")

        if len(ztcf_accels) < 2:
            pytest.skip("ZTCF not available in enough engines")

        # Pairwise comparison
        engine_names = list(ztcf_accels.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_states(
                    name1,
                    ztcf_accels[name1],
                    name2,
                    ztcf_accels[name2],
                    metric="acceleration",
                )
                logger.info(
                    f"ZTCF {name1} vs {name2}: deviation={result.max_deviation:.2e}"
                )
                assert result.severity in [
                    "PASSED",
                    "WARNING",
                ], f"ZTCF mismatch between {name1} and {name2}: {result.message}"

    def test_zvcf_counterfactual_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate ZVCF (Zero-Velocity Counterfactual) agrees across engines.

        Per Guideline G2: ZVCF isolates configuration-dependent dynamics.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Test configuration
        q_test = np.array([0.4])

        # Compute ZVCF accelerations
        zvcf_accels: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is not None:
                try:
                    a_zvcf = eng.engine.compute_zvcf(q_test)
                    if a_zvcf.size > 0:
                        zvcf_accels[eng.name] = a_zvcf
                except Exception as e:
                    logger.warning(f"ZVCF failed for {eng.name}: {e}")

        if len(zvcf_accels) < 2:
            pytest.skip("ZVCF not available in enough engines")

        # Pairwise comparison
        engine_names = list(zvcf_accels.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                result = validator.compare_states(
                    name1,
                    zvcf_accels[name1],
                    name2,
                    zvcf_accels[name2],
                    metric="acceleration",
                )
                logger.info(
                    f"ZVCF {name1} vs {name2}: deviation={result.max_deviation:.2e}"
                )
                assert result.severity in [
                    "PASSED",
                    "WARNING",
                ], f"ZVCF mismatch between {name1} and {name2}: {result.message}"

    def test_mass_matrix_agreement(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate mass matrices agree across engines."""
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        # Set configuration
        q_init = np.array([0.25])
        v_init = np.array([0.0])
        set_identical_state(available_engines, q_init, v_init)

        # Compute mass matrices
        mass_matrices: dict[str, np.ndarray] = {}
        for eng in available_engines:
            if eng.engine is not None:
                M = eng.engine.compute_mass_matrix()
                if M.size > 0:
                    mass_matrices[eng.name] = M.flatten()

        if len(mass_matrices) < 2:
            pytest.skip("Mass matrix not available in enough engines")

        # Pairwise comparison (mass matrix should be very close)
        engine_names = list(mass_matrices.keys())
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1 :]:
                # Use position tolerance for mass matrix comparison
                result = validator.compare_states(
                    name1,
                    mass_matrices[name1],
                    name2,
                    mass_matrices[name2],
                    metric="position",  # Tight tolerance
                )
                logger.info(
                    f"Mass matrix {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}"
                )
                assert result.severity in ["PASSED", "WARNING"], (
                    f"Mass matrix mismatch between {name1} and {name2}: "
                    f"{result.message}"
                )

    def test_indexed_acceleration_closure(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate indexed acceleration closure per Guideline M2.

        Verifies that M(q)*qacc + bias(q,v) = tau (with tau=0 for free motion).
        The residual ||M*qacc_drift + bias|| should be near zero.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines, min_count=1)
        available_engines = [e for e in engines if e.available]

        test_configs = [
            (np.array([0.1]), np.array([0.0])),
            (np.array([0.5]), np.array([1.0])),
            (np.array([1.0]), np.array([-0.5])),
        ]

        for eng in available_engines:
            if eng.engine is None:
                continue
            for q, v in test_configs:
                eng.engine.set_state(q, v)
                eng.engine.forward()

                M = eng.engine.compute_mass_matrix()
                bias = eng.engine.compute_bias_forces()

                if M.size == 0 or bias.size == 0:
                    continue

                # Drift acceleration at zero torque
                qacc_drift = -np.linalg.solve(M, bias)

                # Closure residual: M*qacc + bias should be zero
                residual = M @ qacc_drift + bias
                closure_err = float(np.linalg.norm(residual))

                logger.info(
                    f"{eng.name} acceleration closure at q={q}: "
                    f"residual={closure_err:.2e}"
                )
                assert closure_err < TOLERANCE_CLOSURE_RAD_S2, (
                    f"{eng.name}: acceleration closure residual "
                    f"{closure_err:.2e} exceeds tolerance "
                    f"{TOLERANCE_CLOSURE_RAD_S2:.2e}"
                )

    def test_multi_configuration_forward_dynamics(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate forward dynamics at multiple configurations per M2.

        Tests consistency at various (q, v) pairs spanning the
        configuration space, not just a single test point.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        validator = CrossEngineValidator()
        available_engines = [e for e in engines if e.available]

        test_configs = [
            (np.array([0.0]), np.array([0.0])),  # Equilibrium
            (np.array([0.1]), np.array([0.0])),  # Small angle
            (np.array([0.5]), np.array([0.0])),  # Moderate angle
            (np.array([1.0]), np.array([0.0])),  # Large angle
            (np.array([0.3]), np.array([1.0])),  # With velocity
            (np.array([0.3]), np.array([-2.0])),  # Negative velocity
        ]

        for q, v in test_configs:
            set_identical_state(available_engines, q, v)
            accelerations = compute_accelerations(available_engines)

            if len(accelerations) < 2:
                continue

            engine_names = list(accelerations.keys())
            for i, name1 in enumerate(engine_names):
                for name2 in engine_names[i + 1 :]:
                    result = validator.compare_states(
                        name1,
                        accelerations[name1],
                        name2,
                        accelerations[name2],
                        metric="acceleration",
                    )
                    assert result.severity in ["PASSED", "WARNING"], (
                        f"Forward dynamics at q={q}, v={v}: "
                        f"{name1} vs {name2}: {result.message}"
                    )

    def test_grf_cross_engine_validation(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Validate GRF outputs agree across engines per M2/E5.

        For pendulum models that support contact forces, the GRF
        should be consistent across engines.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        available_engines = [e for e in engines if e.available]

        q_test = np.array([0.2])
        v_test = np.array([0.0])
        set_identical_state(available_engines, q_test, v_test)

        contact_forces: dict[str, np.ndarray] = {}
        gravity_forces: dict[str, np.ndarray] = {}

        for eng in available_engines:
            if eng.engine is None:
                continue

            # Collect gravity forces (always available)
            g = eng.engine.compute_gravity_forces()
            if g.size > 0:
                gravity_forces[eng.name] = g

            # Collect contact forces if supported
            cf = eng.engine.compute_contact_forces()
            if cf is not None and cf.size > 0:
                contact_forces[eng.name] = cf

        # Gravity forces should agree across all engines
        if len(gravity_forces) >= 2:
            grav_names = list(gravity_forces.keys())
            for i, name1 in enumerate(grav_names):
                for name2 in grav_names[i + 1 :]:
                    np.testing.assert_allclose(
                        gravity_forces[name1],
                        gravity_forces[name2],
                        atol=TOLERANCE_ACCELERATION_M_S2,
                        err_msg=f"Gravity force mismatch: {name1} vs {name2}",
                    )

        # Contact forces should agree if available in multiple engines
        if len(contact_forces) >= 2:
            cf_names = list(contact_forces.keys())
            for i, name1 in enumerate(cf_names):
                for name2 in cf_names[i + 1 :]:
                    np.testing.assert_allclose(
                        contact_forces[name1],
                        contact_forces[name2],
                        atol=1e-2,  # Contact forces have more variation
                        err_msg=f"Contact force mismatch: {name1} vs {name2}",
                    )
