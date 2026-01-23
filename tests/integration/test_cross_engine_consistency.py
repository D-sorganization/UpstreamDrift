"""Integration test for verifying consistency across physics engines.

Uses shared fixtures from tests/fixtures/conftest.py to load
gold-standard models (simple pendulum, double pendulum) into
available physics engines and compare results.

Per Guideline M2/P3: Cross-engine validation with explicit tolerances.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from src.shared.python.cross_engine_validator import CrossEngineValidator
from tests.fixtures.fixtures_lib import (
    TOLERANCE_ACCELERATION_M_S2,
    _check_drake_available,
    _check_mujoco_available,
    _check_pinocchio_available,
    compute_accelerations,
    set_identical_state,
    skip_if_insufficient_engines,
)

LOGGER = logging.getLogger(__name__)

# Tolerance multiplier for triangulation outlier detection
# A relaxed 10x threshold is used to identify engines with systematic deviations
TRIANGULATION_TOLERANCE_MULTIPLIER = 10.0


def _get_available_engine_count() -> int:
    """Count available physics engines."""
    count = 0
    if _check_mujoco_available():
        count += 1
    if _check_drake_available():
        count += 1
    if _check_pinocchio_available():
        count += 1
    return count


@pytest.mark.integration
class TestCrossEngineConsistency:
    """Compare physics quantities across engines using shared fixtures.

    Per Guideline M2: Cross-engine comparison with gold-standard models.
    Per Guideline P3: Tolerance-based validation with severity classification.
    """

    def test_mass_matrix_consistency(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Check if Mass Matrix is consistent between engines.

        Sets identical configuration and compares M(q) matrices.
        Tolerance: Position-level (tight) per P3.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        available = [e for e in engines if e.available]
        validator = CrossEngineValidator()

        # Set fixed state
        q = np.array([0.5])  # 0.5 rad
        v = np.array([0.0])  # Static
        set_identical_state(available, q, v)

        # Compute mass matrices
        results: dict[str, np.ndarray] = {}
        for eng in available:
            if eng.engine is not None:
                M = eng.engine.compute_mass_matrix()
                if M.size > 0:
                    results[eng.name] = M.flatten()

        if len(results) < 2:
            pytest.skip("Mass matrix not available in enough engines")

        # Pairwise comparison
        names = list(results.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                result = validator.compare_states(
                    name1, results[name1], name2, results[name2], metric="position"
                )
                LOGGER.info(
                    f"Mass matrix {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}"
                )
                assert result.severity in ["PASSED", "WARNING"], (
                    f"Mass matrix mismatch between {name1} and {name2}: "
                    f"{result.message}"
                )

    def test_gravity_forces_consistency(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Check gravity vector G(q) consistency.

        Tolerance: Torque-level per P3 (±1e-3 N·m).
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        available = [e for e in engines if e.available]
        validator = CrossEngineValidator()

        # Set fixed configuration
        q = np.array([0.3])
        v = np.array([0.0])
        set_identical_state(available, q, v)

        # Compute gravity forces
        results: dict[str, np.ndarray] = {}
        for eng in available:
            if eng.engine is not None:
                try:
                    g = eng.engine.compute_gravity_forces()
                    if g.size > 0:
                        results[eng.name] = g
                except Exception as e:
                    LOGGER.warning(f"Gravity forces failed for {eng.name}: {e}")

        if len(results) < 2:
            pytest.skip("Gravity forces not available in enough engines")

        # Pairwise comparison
        names = list(results.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                result = validator.compare_states(
                    name1, results[name1], name2, results[name2], metric="torque"
                )
                LOGGER.info(
                    f"Gravity forces {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}"
                )
                assert result.severity in ["PASSED", "WARNING"], (
                    f"Gravity force mismatch between {name1} and {name2}: "
                    f"{result.message}"
                )

    def test_bias_forces_consistency(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Check bias forces C(q,v) + G(q) consistency.

        With non-zero velocity, includes Coriolis/centrifugal terms.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        available = [e for e in engines if e.available]
        validator = CrossEngineValidator()

        # Set state with velocity
        q = np.array([0.4])
        v = np.array([0.5])
        set_identical_state(available, q, v)

        # Compute bias forces
        results: dict[str, np.ndarray] = {}
        for eng in available:
            if eng.engine is not None:
                try:
                    bias = eng.engine.compute_bias_forces()
                    if bias.size > 0:
                        results[eng.name] = bias
                except Exception as e:
                    LOGGER.warning(f"Bias forces failed for {eng.name}: {e}")

        if len(results) < 2:
            pytest.skip("Bias forces not available in enough engines")

        # Pairwise comparison
        names = list(results.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                result = validator.compare_states(
                    name1, results[name1], name2, results[name2], metric="torque"
                )
                LOGGER.info(
                    f"Bias forces {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}"
                )
                assert result.severity in [
                    "PASSED",
                    "WARNING",
                ], f"Bias force mismatch between {name1} and {name2}: {result.message}"

    def test_forward_dynamics_trajectory_consistency(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Test trajectory consistency over multiple simulation steps.

        Simulates for a short duration and compares final states.
        Per Guideline M2: Gold-standard test motions.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        skip_if_insufficient_engines(engines)

        available = [e for e in engines if e.available]
        validator = CrossEngineValidator()

        # Set identical initial conditions
        q0 = np.array([0.2])
        v0 = np.array([0.0])
        set_identical_state(available, q0, v0)

        # Simulate for 0.1 seconds with small timestep
        dt = 0.001
        n_steps = 100

        final_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for eng in available:
            if eng.engine is not None:
                # Reset to initial
                eng.engine.set_state(q0, v0)
                eng.engine.forward()

                # Simulate
                for _ in range(n_steps):
                    eng.engine.set_control(np.zeros(1))  # Zero torque
                    eng.engine.step(dt)

                # Record final state
                q_final, v_final = eng.engine.get_state()
                final_states[eng.name] = (q_final, v_final)

        if len(final_states) < 2:
            pytest.skip("Not enough engines completed simulation")

        # Compare final positions and velocities
        names = list(final_states.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                q1, v1 = final_states[name1]
                q2, v2 = final_states[name2]

                # Position comparison
                pos_result = validator.compare_states(
                    name1, q1, name2, q2, metric="position"
                )
                LOGGER.info(
                    f"Trajectory position {name1} vs {name2}: "
                    f"deviation={pos_result.max_deviation:.2e}"
                )

                # Velocity comparison
                vel_result = validator.compare_states(
                    name1, v1, name2, v2, metric="velocity"
                )
                LOGGER.info(
                    f"Trajectory velocity {name1} vs {name2}: "
                    f"deviation={vel_result.max_deviation:.2e}"
                )

                # Allow WARNING for trajectory tests (integration differences)
                # ERROR (>5x tolerance) indicates unacceptable systematic deviation
                assert pos_result.severity in [
                    "PASSED",
                    "WARNING",
                ], f"Trajectory position mismatch: {pos_result.message}"
                assert vel_result.severity in [
                    "PASSED",
                    "WARNING",
                ], f"Trajectory velocity mismatch: {vel_result.message}"


@pytest.mark.integration
class TestThreeWayTriangulation:
    """Three-way engine comparisons for tiebreaking.

    When two engines disagree, the third serves as tiebreaker to
    identify which implementation is deviating.

    Per Guideline M2: Triangulation protocol.
    """

    def test_acceleration_triangulation(
        self,
        mujoco_pendulum: Any,
        drake_pendulum: Any,
        pinocchio_pendulum: Any,
    ) -> None:
        """Use three-way comparison for acceleration validation.

        If MuJoCo and Drake disagree, Pinocchio decides which is correct.
        """
        engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
        available = [e for e in engines if e.available]

        if len(available) < 3:
            pytest.skip(f"Need all 3 engines for triangulation, got {len(available)}")

        validator = CrossEngineValidator()

        # Set identical state
        q = np.array([0.25])
        v = np.array([0.3])
        set_identical_state(available, q, v)

        # Compute accelerations
        accelerations = compute_accelerations(available)

        if len(accelerations) < 3:
            pytest.skip("Not all engines computed accelerations")

        # Compute all pairwise deviations
        pairs = [
            ("MuJoCo", "Drake"),
            ("MuJoCo", "Pinocchio"),
            ("Drake", "Pinocchio"),
        ]

        deviations: dict[tuple[str, str], float] = {}
        for name1, name2 in pairs:
            if name1 in accelerations and name2 in accelerations:
                result = validator.compare_states(
                    name1,
                    accelerations[name1],
                    name2,
                    accelerations[name2],
                    metric="acceleration",
                )
                deviations[(name1, name2)] = result.max_deviation
                LOGGER.info(
                    f"Triangulation {name1} vs {name2}: "
                    f"deviation={result.max_deviation:.2e}, "
                    f"severity={result.severity}"
                )

        # Triangulation logic: if one engine disagrees with both others,
        # it's likely the outlier
        agreement_threshold = TOLERANCE_ACCELERATION_M_S2 * 10  # 10x tolerance

        # Check if any engine is the clear outlier
        for engine_name in ["MuJoCo", "Drake", "Pinocchio"]:
            other_engines = [
                n for n in ["MuJoCo", "Drake", "Pinocchio"] if n != engine_name
            ]

            disagrees_with_first = False
            disagrees_with_second = False

            sorted_pair1 = sorted([engine_name, other_engines[0]])
            pair1: tuple[str, str] = (sorted_pair1[0], sorted_pair1[1])
            sorted_pair2 = sorted([engine_name, other_engines[1]])
            pair2: tuple[str, str] = (sorted_pair2[0], sorted_pair2[1])

            if pair1 in deviations and deviations[pair1] > agreement_threshold:
                disagrees_with_first = True
            if pair2 in deviations and deviations[pair2] > agreement_threshold:
                disagrees_with_second = True

            if disagrees_with_first and disagrees_with_second:
                # Check if the other two agree
                sorted_other = sorted(other_engines)
                other_pair: tuple[str, str] = (sorted_other[0], sorted_other[1])
                if other_pair in deviations:
                    if deviations[other_pair] < agreement_threshold:
                        LOGGER.warning(
                            f"Triangulation identified {engine_name} as outlier: "
                            f"disagrees with both {other_engines[0]} and {other_engines[1]}"
                        )

        # For now, just verify that at least two engines agree closely
        min_deviation = min(deviations.values()) if deviations else float("inf")
        assert (
            min_deviation < agreement_threshold
        ), f"No engine pair agrees within threshold: min deviation={min_deviation:.2e}"
