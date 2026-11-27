"""Backspin against spin loft for every impact model (#15443, F2).

Reference: closed-form oblique impact of a ball on a face moving at speed ``v``
with face loft ``L`` (centred strike, zero attack angle, so spin loft = ``L``):
normal approach ``v cos L``, slip ``v sin L``, ball tangential impulse
``min(mu J_n, v_t / (1/m + 1/M + R^2/I))`` and spin ``R J_t / I``.
Sources: Cross, R., "Grip-slip behavior of a bouncing ball", Am. J. Phys. 70(11),
1093-1101 (2002); Penner, A. R., "The physics of golf", Rep. Prog. Phys. 66,
131-171 (2003). Full citation and data: ``fixtures/spin_vs_spin_loft_15443.json``.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.impact_parameters.club_impact import club_impact_for
from src.shared.python.physics.impact_model.models import create_impact_model
from src.shared.python.physics.impact_model.types import (
    ImpactModelType,
    ImpactParameters,
    PreImpactState,
)

pytestmark = pytest.mark.unit

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "spin_vs_spin_loft_15443.json").read_text()
)
# Regression values are integer rpm; the spring-damper normal impulse differs from
# the rigid one by < 0.5 %, which the friction-capped spin inherits at most.
REL_TOL = 5e-3
RPM = 60.0 / (2.0 * math.pi)


def _solve_rpm(model: ImpactModelType, club: dict, loft_deg: float) -> float:
    lam = math.radians(loft_deg)
    pre = PreImpactState(
        np.array([club["club_speed_mps"], 0.0, 0.0]),
        np.zeros(3),
        np.array([math.cos(lam), 0.0, math.sin(lam)]),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        clubhead_mass=club["head_mass_kg"],
        clubhead_moi=club["head_moi_kg_m2"],
    )
    post = create_impact_model(model).solve(pre, ImpactParameters(cor=club["cor"]))
    return float(np.linalg.norm(post.ball_angular_velocity)) * RPM


def _closed_form_rpm(club: dict, loft_deg: float) -> float:
    m, big_m = float(GOLF_BALL_MASS_KG), club["head_mass_kg"]
    radius = float(GOLF_BALL_RADIUS_M)
    inertia = float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)
    mu = ImpactParameters().friction_coefficient
    lam, v = math.radians(loft_deg), club["club_speed_mps"]
    j_n = (1.0 + club["cor"]) * v * math.cos(lam) / (1.0 / m + 1.0 / big_m)
    j_roll = v * math.sin(lam) / (1.0 / m + 1.0 / big_m + radius**2 / inertia)
    return radius * min(mu * j_n, j_roll) / inertia * RPM


CASES = [
    (c, i) for c in FIXTURE["clubs"] for i in range(len(FIXTURE["spin_lofts_deg"]))
]


@pytest.mark.parametrize("model", list(ImpactModelType))
@pytest.mark.parametrize(("name", "idx"), CASES)
def test_backspin_matches_closed_form_and_fixture(model, name, idx):
    club, loft = FIXTURE["clubs"][name], FIXTURE["spin_lofts_deg"][idx]
    rpm = _solve_rpm(model, club, loft)
    assert rpm == pytest.approx(_closed_form_rpm(club, loft), rel=REL_TOL)
    assert rpm == pytest.approx(club["model_backspin_rpm"][idx], rel=REL_TOL)


@pytest.mark.parametrize("name", list(FIXTURE["clubs"]))
def test_backspin_rises_with_spin_loft(name):
    rpm = [
        _solve_rpm(ImpactModelType.RIGID_BODY, FIXTURE["clubs"][name], lo)
        for lo in FIXTURE["spin_lofts_deg"]
    ]
    assert all(b > a for a, b in zip(rpm, rpm[1:], strict=False))


def test_iron_fixture_uses_the_catalogue_head():
    iron = club_impact_for("iron7")
    assert iron is not None
    club = FIXTURE["clubs"]["iron7"]
    assert club["head_mass_kg"] == pytest.approx(iron.head_mass_kg)
    assert club["head_moi_kg_m2"] == pytest.approx(iron.head_moi_kg_m2, rel=1e-4)


def test_centred_spin_ignores_head_inertia():
    club = dict(FIXTURE["clubs"]["iron7"])
    base = _solve_rpm(ImpactModelType.RIGID_BODY, club, 30.0)
    club["head_moi_kg_m2"] *= 3.0
    assert _solve_rpm(ImpactModelType.RIGID_BODY, club, 30.0) == pytest.approx(base)
