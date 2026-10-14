"""#12199: the character builder's ``mass_kg`` sets the compiled body mass.

The compiled body mass excludes the club, whose mass is the club model's own
(``document["club"]["total_mass_kg"]``) and does not depend on the subject.
The lower limbs carry the part of ``mass_kg`` the upper body does not, in the
Rajagopal reference proportions. Reference capture builds keep the unscaled
Rajagopal legs (no ``match_subject_mass``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.humanoid_character_builder.presets.loader import (
    list_character_presets,
)
from src.shared.python.humanoid_character_builder.spec_export import (
    compile_character,
)
from src.shared.python.motion_matching.anthropometric_geometry import (
    build_upper_body,
)
from src.shared.python.motion_matching.execution import assets
from src.shared.python.motion_matching.execution.spec_builder import (
    LEG_BODIES,
    compose_anthropometric_document,
    read_osim,
    scale_leg_mass,
)
from src.shared.python.motion_matching.execution.spec_builder import (
    _body_mass_without_club as _builder_body_mass_without_club,
)

pytestmark = pytest.mark.unit

#: Body mass equals the request up to float summation error.
MASS_TOLERANCE_KG = 1e-6
LEG_SUFFIXES = ("_r", "_l")


def _masses(document: dict) -> dict[str, float]:
    return {
        b["name"]: sum(s["mass_kg"] for s in b["solids"]) for b in document["bodies"]
    }


def _body_mass_without_club(document: dict) -> float:
    return sum(_masses(document).values()) - document["club"]["total_mass_kg"]


def _legs(document: dict) -> dict[str, float]:
    return {k: v for k, v in _masses(document).items() if k.endswith(LEG_SUFFIXES)}


def _hand_solids(document: dict) -> dict[str, dict]:
    body = next(b for b in document["bodies"] if b["name"].endswith("Clubface Vector"))
    return {
        s["name"].rsplit("/", 1)[-1]: s
        for s in body["solids"]
        if s.get("name", "").rsplit("/", 1)[-1] in ("LHand", "RHand")
    }


@pytest.mark.parametrize("preset", list_character_presets())
@pytest.mark.parametrize("mass_kg", [30.0, 84.0, 200.0])
def test_body_mass_equals_requested_mass(preset: str, mass_kg: float) -> None:
    document = compile_character(preset, {"mass_kg": mass_kg}).document
    assert _body_mass_without_club(document) == pytest.approx(
        mass_kg, abs=MASS_TOLERANCE_KG
    )


def test_leg_masses_scale_with_mass_in_rajagopal_proportions() -> None:
    light = _legs(compile_character("tour_average_male", {"mass_kg": 60.0}).document)
    heavy = _legs(compile_character("tour_average_male", {"mass_kg": 120.0}).document)
    assert sum(heavy.values()) > 1.9 * sum(light.values())
    ratios = {name: heavy[name] / light[name] for name in light}
    assert max(ratios.values()) == pytest.approx(min(ratios.values()), rel=1e-12)


def test_leg_inertia_scales_with_leg_mass() -> None:
    def femur(mass_kg: float) -> dict:
        document = compile_character("tour_average_male", {"mass_kg": mass_kg})
        body = next(b for b in document.document["bodies"] if b["name"] == "femur_r")
        return body["solids"][0]

    light, heavy = femur(60.0), femur(120.0)
    factor = heavy["mass_kg"] / light["mass_kg"]
    np.testing.assert_allclose(
        heavy["inertia_com_kg_m2"],
        np.asarray(light["inertia_com_kg_m2"]) * factor,
        rtol=1e-12,
    )
    assert heavy["com_m"] == pytest.approx(light["com_m"], abs=1e-15)


def _compose(mass_kg: float, **kwargs: object) -> dict:
    return compose_anthropometric_document(
        native_path=assets.get_native_geometry_spec(None),
        osim_path=assets.get_opensim_model(None),
        native_candidate_path=assets.get_candidate_geometry_spec(None),
        stature_m=1.80,
        mass_kg=mass_kg,
        **kwargs,
    ).document


def test_reference_build_keeps_rajagopal_leg_masses() -> None:
    bodies, _joints = read_osim(assets.get_opensim_model(None))
    legs = _legs(_compose(84.0))
    assert legs == pytest.approx({name: bodies[name]["mass"] for name in legs})


def test_mass_below_upper_body_mass_is_rejected() -> None:
    # The upper body alone outweighs 0.5 kg.
    with pytest.raises(ValueError, match="leg mass"):
        _compose(0.5, match_subject_mass=True)


def test_build_upper_body_scales_hand_mass_and_inertia() -> None:
    doc_60 = build_upper_body(1.80, 60.0, "driver", scale_hand_mass=True)
    hands_60 = _hand_solids(doc_60)
    assert set(hands_60) == {"LHand", "RHand"}
    for side in ("LHand", "RHand"):
        assert hands_60[side]["mass_kg"] == pytest.approx(0.0061 * 60.0, abs=1e-12)

    doc_100 = build_upper_body(1.80, 100.0, "driver", scale_hand_mass=True)
    hands_100 = _hand_solids(doc_100)
    for side in ("LHand", "RHand"):
        assert hands_100[side]["mass_kg"] == pytest.approx(0.0061 * 100.0, abs=1e-12)
        factor = 100.0 / 60.0
        np.testing.assert_allclose(
            hands_100[side]["inertia_com_kg_m2"],
            np.asarray(hands_60[side]["inertia_com_kg_m2"]) * factor,
            rtol=1e-12,
        )


def test_build_upper_body_preserves_native_hand_mass_by_default() -> None:
    doc_default = build_upper_body(1.80, 80.0, "driver")
    hands = _hand_solids(doc_default)
    for side in ("LHand", "RHand"):
        assert hands[side]["mass_kg"] == pytest.approx(0.5491357853472616, abs=1e-9)

    doc_explicit_false = build_upper_body(1.80, 80.0, "driver", scale_hand_mass=False)
    hands_false = _hand_solids(doc_explicit_false)
    for side in ("LHand", "RHand"):
        assert hands_false[side]["mass_kg"] == pytest.approx(
            0.5491357853472616, abs=1e-9
        )


def test_compose_anthropometric_document_scales_hands_and_total_body_mass() -> None:
    composed = compose_anthropometric_document(
        1.80, 95.0, "driver", match_subject_mass=True
    ).document
    hands = _hand_solids(composed)
    for side in ("LHand", "RHand"):
        assert hands[side]["mass_kg"] == pytest.approx(0.0061 * 95.0, abs=1e-12)
    assert _body_mass_without_club(composed) == pytest.approx(
        95.0, abs=MASS_TOLERANCE_KG
    )


def test_mass_sums_do_not_depend_on_python_version() -> None:
    """Sums are correctly rounded, so the spec hash matches on 3.11 and 3.12+.

    Python 3.12 made the builtin ``sum`` compensated; on 3.11 ten masses of
    0.1 kg sum to 0.9999999999999999. CI's 3.11 then compiled another spec
    SHA than 3.13 did (#12199).
    """
    legs = {f"{n}{s}" for n in LEG_BODIES for s in LEG_SUFFIXES}
    bodies = {name: {"mass": 0.1, "inertia": np.ones(3)} for name in legs}
    scaled = scale_leg_mass(bodies, 1.0)
    assert {body["mass"] for body in scaled.values()} == {0.1}
    document = {
        "bodies": [{"solids": [{"mass_kg": 0.1}]} for _ in range(10)],
        "club": {"total_mass_kg": 0.0},
    }
    assert _builder_body_mass_without_club(document) == 1.0


@pytest.mark.parametrize("leg_mass_kg", [0.0, -1.0, float("nan")])
def test_scale_leg_mass_rejects_non_positive_target(leg_mass_kg: float) -> None:
    with pytest.raises(ValueError, match="leg mass"):
        scale_leg_mass({}, leg_mass_kg)
