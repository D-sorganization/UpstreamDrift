"""Cross-engine hand mass and length consistency (HRG-HX7, #13764).

The flagship specification welds both hand solids to the club body, so no
engine has a hand body to read. The opt-in separated-hand ``full-body-v2``
document (#13309) gives each hand its own rigid body, and every flagship
exporter (MuJoCo MJCF, Drake and Pinocchio from the URDF bundle, OpenSim XML)
builds it. Each engine's own hand body mass must equal the specification's.

Hand length is visual only (#13995): of the flagship exports only OpenSim
draws grip hands, so only OpenSim carries a hand length. The other engines
report it as unrepresentable with a reason; no quantity passes silently.
Simscape and the MyoSuite golfer are reported, never gating.

The hand-frame assertion (0.5 deg) is the issue's second slice. Related:
#13176 (cross-engine tests) and #13560.
"""

from __future__ import annotations

import pytest

from tests.helpers import hand_consistency as hc

pytestmark = pytest.mark.integration

MASS_TOL_KG = 1e-9
LENGTH_TOL_M = 1e-12


@pytest.fixture(scope="module")
def doc() -> dict:
    return hc.separated_flagship_doc()


@pytest.mark.parametrize("engine", hc.GATING_ENGINES)
def test_hand_mass_and_length_agree_across_flagship_exports(doc, engine) -> None:
    reading = hc.read_hand(engine, doc)
    expected_mass = hc.spec_hand_masses(doc)
    assert set(reading.mass_kg) == set(expected_mass) == set(hc.HANDS)
    for hand, mass in expected_mass.items():
        assert reading.mass_kg[hand] == pytest.approx(mass, abs=MASS_TOL_KG), (
            f"{engine} {hand} hand mass {reading.mass_kg[hand]!r} kg, spec {mass!r} kg"
        )
    if isinstance(reading.length_m, hc.Unrepresentable):
        assert reading.length_m.reason
        return
    expected_length = hc.spec_hand_length_m(doc)
    assert set(reading.length_m) == set(hc.HANDS)
    for hand, length in reading.length_m.items():
        assert length == pytest.approx(expected_length, abs=LENGTH_TOL_M), (
            f"{engine} {hand} hand length {length!r} m, spec {expected_length!r} m"
        )


def test_unrepresentable_capability_reported_not_passed() -> None:
    report = hc.capability_report()
    assert set(report) == set(hc.GATING_ENGINES) | set(hc.REPORTED_ENGINES)
    for engine in hc.REPORTED_ENGINES:
        entry = report[engine]
        assert isinstance(entry["mass_kg"], hc.Unrepresentable), engine
        assert entry["mass_kg"].reason and not entry["gating"]
    for engine in hc.GATING_ENGINES:
        assert report[engine]["gating"]
    missing = [
        engine
        for engine, entry in report.items()
        for quantity in ("mass_kg", "length_m")
        if entry[quantity] is None
    ]
    assert not missing, f"capability neither measured nor reported: {missing}"
    assert any(
        isinstance(entry["length_m"], hc.Unrepresentable)
        for engine, entry in report.items()
        if engine in hc.GATING_ENGINES
    ), "visual-only hand length must be reported where an export lacks hands"
