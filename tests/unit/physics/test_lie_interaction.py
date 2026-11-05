"""SHOW-B5 (#14569): parametric lie/rough modifier on impact conditions."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics._friction_laws import check_friction_cone
from src.shared.python.physics.lie_interaction import (
    LIE_PARAMETERS,
    ImpactPerturbation,
    LieType,
    effective_face_friction,
    lie_from_code,
    lie_perturbation,
    perturb_impact_rows,
)
from src.shared.python.sg_optimizer.course.conditions import RoughModel
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES
from src.shared.python.sg_optimizer.shot_model.skill_family import IMPACT_FIELDS

pytestmark = pytest.mark.unit


def test_fairway_is_the_identity() -> None:
    p = lie_perturbation(LieType.FAIRWAY)
    assert p.is_identity()
    assert p.apply(speed_mph=100.0, backspin_rpm=2500.0, face_deg=1.5) == (
        100.0,
        2500.0,
        1.5,
    )
    samples = p.sample(np.random.default_rng(0), 50)
    assert np.all(samples.speed_factor == 1.0)
    assert np.all(samples.spin_factor == 1.0)
    assert np.all(samples.face_twist_deg == 0.0)


def test_rough_lowers_spin_and_speed_and_reuses_the_course_rough_model() -> None:
    rough = RoughModel(0.5)
    p = lie_perturbation(LieType.ROUGH, severity=0.5)
    assert p.spin_factor == pytest.approx(1.0 - rough.spin_reduction())
    assert p.speed_factor == pytest.approx(rough.distance_multiplier())
    assert p.flyer_probability == pytest.approx(rough.flyer_probability())
    speed, spin, _ = p.apply(100.0, 2500.0, 0.0)
    assert speed < 100.0 and spin < 2500.0


def test_rough_increases_distance_variance_and_severity_is_monotone() -> None:
    rng = np.random.default_rng(1)
    fairway = lie_perturbation(LieType.FAIRWAY).sample(rng, 4000)
    light = lie_perturbation(LieType.ROUGH, severity=0.2).sample(rng, 4000)
    heavy = lie_perturbation(LieType.ROUGH, severity=0.8).sample(rng, 4000)
    # Distance varies with ball speed and with spin (flyers): compare the
    # spread of the speed factor.
    assert np.std(heavy.speed_factor) > np.std(light.speed_factor)
    assert np.std(light.speed_factor) > np.std(fairway.speed_factor) == 0.0
    assert np.mean(heavy.spin_factor) < np.mean(light.spin_factor) < 1.0


def test_flyers_launch_with_less_spin_and_more_speed() -> None:
    p = lie_perturbation(LieType.ROUGH, severity=0.5)
    s = p.sample(np.random.default_rng(2), 20000)
    assert s.is_flyer.any() and not s.is_flyer.all()
    assert s.is_flyer.mean() == pytest.approx(p.flyer_probability, abs=0.02)
    assert s.spin_factor[s.is_flyer].mean() < s.spin_factor[~s.is_flyer].mean()
    assert s.speed_factor[s.is_flyer].mean() > s.speed_factor[~s.is_flyer].mean()


def test_rough_twists_the_face_closed_and_other_lies_are_ordered() -> None:
    assert lie_perturbation(LieType.ROUGH, severity=0.7).face_twist_deg < 0.0
    bunker = lie_perturbation(LieType.BUNKER)
    hardpan = lie_perturbation(LieType.HARDPAN)
    first_cut = lie_perturbation(LieType.FIRST_CUT)
    assert bunker.speed_factor < first_cut.speed_factor <= 1.0
    assert hardpan.speed_factor <= 1.0


def test_every_parameter_is_cited_or_flagged_as_assumption() -> None:
    assert LIE_PARAMETERS
    for record in LIE_PARAMETERS.values():
        assert record.basis in {"cited", "assumption"}
        assert len(record.source) > 15
        if record.basis == "assumption":
            assert record.rationale
    assert any(r.basis == "assumption" for r in LIE_PARAMETERS.values())


def test_contracts_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        lie_perturbation(LieType.ROUGH, severity=1.5)
    with pytest.raises(ValueError):
        lie_perturbation(LieType.ROUGH, severity=-0.1)
    with pytest.raises(ValueError):
        lie_perturbation(LieType.FAIRWAY).apply(-1.0, 2500.0, 0.0)
    with pytest.raises(ValueError):
        lie_perturbation(LieType.FAIRWAY).sample(np.random.default_rng(0), 0)
    assert isinstance(lie_perturbation(LieType.FAIRWAY), ImpactPerturbation)


def test_face_friction_term_uses_the_shared_friction_cone() -> None:
    mu_dry = 0.45
    assert effective_face_friction(mu_dry, lie_perturbation(LieType.FAIRWAY)) == mu_dry
    mu_rough = effective_face_friction(mu_dry, lie_perturbation(LieType.ROUGH, 0.8))
    assert 0.0 < mu_rough < mu_dry
    tangent = np.array([0.0, mu_dry * 100.0 * 0.99, 0.0])
    assert check_friction_cone(100.0, tangent, mu_dry)
    assert not check_friction_cone(100.0, tangent, mu_rough)


def test_lie_codes_map_to_lie_types_and_reject_unmodelled_codes() -> None:
    assert lie_from_code(LIE_CODES["fairway"]) is LieType.FAIRWAY
    assert lie_from_code(LIE_CODES["tee"]) is LieType.FAIRWAY
    assert lie_from_code(LIE_CODES["rough"]) is LieType.ROUGH
    assert lie_from_code(LIE_CODES["sand"]) is LieType.BUNKER
    with pytest.raises(ValueError):
        lie_from_code(LIE_CODES["water"])


def test_impact_rows_pass_through_on_fairway_and_shift_in_rough() -> None:
    rows = np.tile(np.array([100.0, 1.0, 2.0, -3.0, 4.0, 5.0]), (4000, 1))
    rng = np.random.default_rng(1)
    out, spin = perturb_impact_rows(rows, lie_perturbation(LieType.FAIRWAY), rng)
    assert np.array_equal(out, rows) and np.all(spin == 1.0)
    rough = lie_perturbation(LieType.ROUGH, 0.8)
    out, spin = perturb_impact_rows(rows, rough, np.random.default_rng(1))
    speed, face = (
        IMPACT_FIELDS.index("clubhead_speed_mph"),
        IMPACT_FIELDS.index("face_angle_deg"),
    )
    assert out[:, speed].mean() < 100.0 and out[:, speed].std() > 0.0
    assert out[:, face].mean() < 1.0  # rough closes the face
    assert spin.mean() < 1.0
    other = [i for i in range(len(IMPACT_FIELDS)) if i not in (speed, face)]
    assert np.array_equal(out[:, other], rows[:, other])
    assert np.array_equal(rows[0], [100.0, 1.0, 2.0, -3.0, 4.0, 5.0])  # input intact
