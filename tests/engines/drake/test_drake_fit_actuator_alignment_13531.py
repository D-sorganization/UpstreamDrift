"""Issue #13531: autodiff fit theta must match the canonical URDF actuator count."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.motion_matching.fit_swing_autodiff import (
    COEFFS_PER_JOINT,
    FitOptions,
    fit_swing_drake_autodiff,
)
from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
    CANONICAL_URDF,
    count_actuators_for_urdf,
    resolve_plant_n_actuators,
)
from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    _resolve_n_actuators,
)

pytestmark = pytest.mark.unit


def test_resolve_plant_n_actuators_prefers_num_actuators() -> None:
    plant = MagicMock()
    plant.num_actuators.return_value = 43
    plant.num_velocities.return_value = 100
    assert resolve_plant_n_actuators(plant) == 43
    plant.num_velocities.assert_not_called()


def test_resolve_plant_n_actuators_falls_back_to_nv_minus_six() -> None:
    plant = MagicMock()
    plant.num_actuators.return_value = 0
    plant.num_velocities.return_value = 25
    assert resolve_plant_n_actuators(plant) == 19


def test_simulate_resolve_n_actuators_delegates_to_humanoid_helper() -> None:
    plant = MagicMock()
    plant.num_actuators.return_value = 12
    assert _resolve_n_actuators(plant) == resolve_plant_n_actuators(plant)


def test_n_joints_hint_must_match_plant_not_override() -> None:
    """DbC: hints cannot resize theta away from the loaded plant (#13531)."""
    target = MagicMock()
    target.time = np.linspace(0.0, 0.2, 5)
    target.grip = np.zeros((5, 3))

    fake_pydrake = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "pydrake": fake_pydrake,
            "pydrake.autodiffutils": fake_pydrake,
            "pydrake.solvers": fake_pydrake,
            "pydrake.multibody.plant": fake_pydrake,
            "pydrake.systems.framework": fake_pydrake,
        },
    ):
        with (
            patch(
                "src.engines.physics_engines.drake.python.motion_matching."
                "fit_swing_autodiff._build_autodiff_plant",
                return_value=(MagicMock(), MagicMock(), 1, 1, 43),
            ),
            patch(
                "src.engines.physics_engines.drake.python.motion_matching."
                "fit_swing_autodiff._grip_log_from_target",
                return_value=(target.time, target.grip),
            ),
        ):
            wrong_len = 19 * COEFFS_PER_JOINT
            theta = np.zeros(wrong_len)
            opts = FitOptions(n_joints_hint=19)
            with pytest.raises(ValueError, match="n_joints_hint"):
                fit_swing_drake_autodiff(target, opts, initial_theta=theta)


@pytest.mark.requires_drake
def test_canonical_urdf_actuator_count_matches_fit_options_default() -> None:
    """Live Drake: default fit plant exposes a stable positive actuator count."""
    pytest.importorskip("pydrake")
    n = count_actuators_for_urdf(CANONICAL_URDF)
    assert n > 0
    assert n != 19, (
        "CANONICAL_URDF no longer matches legacy EXPECTED_NUM_REVOLUTE_DOF=19; "
        "tests must size theta from count_actuators_for_urdf(), not a hardcoded joint count"
    )
