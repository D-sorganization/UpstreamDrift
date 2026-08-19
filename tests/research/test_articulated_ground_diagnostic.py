from __future__ import annotations

import pytest
from scripts.research.proximal_distal_energy.articulated_ground_diagnostic import (
    BRANCHES,
    ENGINES,
    STEPS_S,
)

pytestmark = pytest.mark.scientific


def test_ground_diagnostic_design_is_preregistered_and_complete() -> None:
    assert BRANCHES == (
        "fixed_zero",
        "translation_perturbed",
        "free_moment_perturbed",
        "coupled_perturbed",
        "coupled_natural_zero",
        "coupled_gravity_only",
        "coupled_conditional",
    )
    assert ENGINES == ("mujoco", "pinocchio")
    assert STEPS_S == (0.00025, 0.000125, 0.0000625)
