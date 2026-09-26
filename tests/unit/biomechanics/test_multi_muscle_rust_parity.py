"""Numerical parity tests between Rust and Python multi-muscle backends.

Verifies that MuscleGroup and AntagonistPair yield identical net torque
(within rtol=1e-9, atol=1e-12) between the Rust backend (upstream_muscle)
and the pure-Python reference implementation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

upstream_muscle = pytest.importorskip("upstream_muscle", exc_type=ImportError)

from src.shared.python.biomechanics.hill_muscle import (
    HillMuscleModel,
    MuscleParameters,
)
from src.shared.python.biomechanics.multi_muscle import (
    AntagonistPair,
    MuscleGroup,
    create_elbow_muscle_system,
)
from src.shared.python.core.contracts import PreconditionError

pytestmark = [pytest.mark.unit]

RTOL = 1e-9
ATOL = 1e-12


def _make_flexor_muscles() -> tuple[HillMuscleModel, HillMuscleModel]:
    biceps = HillMuscleModel(
        MuscleParameters(
            F_max=1000.0,
            l_opt=0.15,
            l_slack=0.20,
            v_max=10.0,
            pennation_angle=0.05,
            damping=0.05,
        )
    )
    brachialis = HillMuscleModel(
        MuscleParameters(
            F_max=800.0,
            l_opt=0.12,
            l_slack=0.10,
            v_max=10.0,
            pennation_angle=0.10,
            damping=0.05,
        )
    )
    return biceps, brachialis


def _make_extensor_muscles() -> HillMuscleModel:
    return HillMuscleModel(
        MuscleParameters(
            F_max=1200.0,
            l_opt=0.18,
            l_slack=0.22,
            v_max=8.0,
            pennation_angle=0.0,
            damping=0.08,
        )
    )


class TestMuscleGroupRustParity:
    """Parity tests for MuscleGroup compute_net_torque."""

    @pytest.mark.parametrize("act_val", [0.0, 0.1, 0.35, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize(
        ("l_ce", "v_ce"),
        [
            (0.15, 0.0),  # optimal isometric
            (0.12, -0.05),  # shortened concentric
            (0.18, 0.05),  # lengthened eccentric
            (0.10, 0.0),  # extreme shortened
            (0.20, 0.0),  # extreme lengthened
        ],
    )
    def test_single_muscle_torque_parity(
        self, act_val: float, l_ce: float, v_ce: float
    ) -> None:
        """Verify net torque matches between Rust and Python for a single muscle."""
        p = MuscleParameters(
            F_max=1000.0,
            l_opt=0.15,
            l_slack=0.20,
            v_max=10.0,
            pennation_angle=0.05,
            damping=0.05,
        )
        m_rust = HillMuscleModel(p)
        m_py = HillMuscleModel(p)

        group_rust = MuscleGroup("Flexors", enable_rust=True)
        group_rust.add_muscle("biceps", m_rust, moment_arm=0.04)

        group_py = MuscleGroup("Flexors", enable_rust=False)
        group_py.add_muscle("biceps", m_py, moment_arm=0.04)

        activations = {"biceps": act_val}
        states = {"biceps": (l_ce, v_ce)}

        tau_rust = group_rust.compute_net_torque(activations, states)
        tau_py = group_py.compute_net_torque(activations, states)

        assert group_rust._rust_backend is not None
        assert group_py._rust_backend is None
        assert math.isclose(tau_rust, tau_py, rel_tol=RTOL, abs_tol=ATOL)

    @pytest.mark.parametrize(
        "activations",
        [
            {"biceps": 0.0, "brachialis": 0.0},
            {"biceps": 1.0, "brachialis": 1.0},
            {"biceps": 0.7, "brachialis": 0.3},
            {"biceps": 0.2, "brachialis": 0.9},
            {"biceps": 0.5},  # brachialis omitted from activations
        ],
    )
    def test_synergist_muscles_torque_parity(
        self, activations: dict[str, float]
    ) -> None:
        """Verify net torque matches for multiple synergist muscles."""
        bic_r, brach_r = _make_flexor_muscles()
        group_rust = MuscleGroup("Flexors", enable_rust=True)
        group_rust.add_muscle("biceps", bic_r, moment_arm=0.04)
        group_rust.add_muscle("brachialis", brach_r, moment_arm=0.03)

        bic_py, brach_py = _make_flexor_muscles()
        group_py = MuscleGroup("Flexors", enable_rust=False)
        group_py.add_muscle("biceps", bic_py, moment_arm=0.04)
        group_py.add_muscle("brachialis", brach_py, moment_arm=0.03)

        states = {
            "biceps": (0.14, -0.02),
            "brachialis": (0.13, 0.01),
        }

        tau_rust = group_rust.compute_net_torque(activations, states)
        tau_py = group_py.compute_net_torque(activations, states)

        assert math.isclose(tau_rust, tau_py, rel_tol=RTOL, abs_tol=ATOL)


class TestAntagonistPairRustParity:
    """Parity tests for AntagonistPair compute_net_torque."""

    @pytest.mark.parametrize(
        ("flex_act", "ext_act"),
        [
            ({"biceps": 0.5, "brachialis": 0.5}, {"triceps": 0.0}),  # Pure flexion
            ({"biceps": 0.0, "brachialis": 0.0}, {"triceps": 0.5}),  # Pure extension
            (
                {"biceps": 0.8, "brachialis": 0.8},
                {"triceps": 0.1},
            ),  # Low co-contraction
            (
                {"biceps": 0.8, "brachialis": 0.8},
                {"triceps": 0.8},
            ),  # High co-contraction
            (
                {"biceps": 0.2, "brachialis": 0.4},
                {"triceps": 0.7},
            ),  # Extension-dominant
            ({"biceps": 0.0, "brachialis": 0.0}, {"triceps": 0.0}),  # Fully passive
            ({"biceps": 1.0, "brachialis": 1.0}, {"triceps": 1.0}),  # Full activation
        ],
    )
    @pytest.mark.parametrize(
        "states",
        [
            {
                "biceps": (0.15, 0.0),
                "brachialis": (0.12, 0.0),
                "triceps": (0.18, 0.0),
            },
            {
                "biceps": (0.13, -0.04),
                "brachialis": (0.10, -0.03),
                "triceps": (0.20, 0.04),
            },
            {
                "biceps": (0.17, 0.06),
                "brachialis": (0.14, 0.05),
                "triceps": (0.16, -0.06),
            },
        ],
    )
    def test_elbow_system_parity(
        self,
        flex_act: dict[str, float],
        ext_act: dict[str, float],
        states: dict[str, tuple[float, float]],
    ) -> None:
        """Verify AntagonistPair net torque matches across co-contraction levels and states."""
        pair_rust = create_elbow_muscle_system(enable_rust=True)
        pair_py = create_elbow_muscle_system(enable_rust=False)

        assert pair_rust._rust_backend is not None
        assert pair_py._rust_backend is None

        tau_rust = pair_rust.compute_net_torque(flex_act, ext_act, states)
        tau_py = pair_py.compute_net_torque(flex_act, ext_act, states)

        assert math.isclose(tau_rust, tau_py, rel_tol=RTOL, abs_tol=ATOL)


class TestPreconditionParity:
    """Verify that both Rust and Python paths enforce DbC preconditions identically."""

    def test_out_of_bounds_agonist_activation_raises_on_both(self) -> None:
        pair_rust = create_elbow_muscle_system(enable_rust=True)
        pair_py = create_elbow_muscle_system(enable_rust=False)

        states = {
            "biceps": (0.15, 0.0),
            "brachialis": (0.12, 0.0),
            "triceps": (0.18, 0.0),
        }

        for pair in (pair_rust, pair_py):
            with pytest.raises((ValueError, PreconditionError)):
                pair.compute_net_torque(
                    {"biceps": 1.5, "brachialis": 0.5}, {"triceps": 0.2}, states
                )

            with pytest.raises((ValueError, PreconditionError)):
                pair.compute_net_torque(
                    {"biceps": -0.1, "brachialis": 0.5}, {"triceps": 0.2}, states
                )

    def test_out_of_bounds_antagonist_activation_raises_on_both(self) -> None:
        pair_rust = create_elbow_muscle_system(enable_rust=True)
        pair_py = create_elbow_muscle_system(enable_rust=False)

        states = {
            "biceps": (0.15, 0.0),
            "brachialis": (0.12, 0.0),
            "triceps": (0.18, 0.0),
        }

        for pair in (pair_rust, pair_py):
            with pytest.raises((ValueError, PreconditionError)):
                pair.compute_net_torque(
                    {"biceps": 0.5, "brachialis": 0.5}, {"triceps": 2.0}, states
                )

            with pytest.raises((ValueError, PreconditionError)):
                pair.compute_net_torque(
                    {"biceps": 0.5, "brachialis": 0.5}, {"triceps": -0.5}, states
                )

    def test_missing_activations_raises_on_both(self) -> None:
        pair_rust = create_elbow_muscle_system(enable_rust=True)
        pair_py = create_elbow_muscle_system(enable_rust=False)

        states = {
            "biceps": (0.15, 0.0),
            "brachialis": (0.12, 0.0),
            "triceps": (0.18, 0.0),
        }

        for pair in (pair_rust, pair_py):
            with pytest.raises(ValueError):
                pair.compute_net_torque(None, {"triceps": 0.2}, states)  # type: ignore[arg-type]

            with pytest.raises(ValueError):
                pair.compute_net_torque({"biceps": 0.5}, None, states)  # type: ignore[arg-type]
