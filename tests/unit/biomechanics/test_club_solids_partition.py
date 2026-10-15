"""Bare-club versus welded-hand partition of the club body solids (NSX-01, #12550).

The expected mass, centre of mass and inertia are hand calculations on a small
synthetic body, not a re-run of the implementation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.club_inertia_partition import (
    INTERFACE_BARE_CLUB,
    INTERFACE_LEGACY_COMPOSITE,
    RigidInertia,
    club_solids_partition,
    newton_euler_keywords,
)

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
CLUB_BODY = "solid_reference:Test/Club/Clubface Vector"


def _placement(rot: np.ndarray, xyz: tuple[float, float, float]) -> list[list[float]]:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t.tolist()


RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _solid(name, mass, diag, rot, xyz) -> dict:
    return {
        "name": name,
        "mass_kg": mass,
        "com_m": [0.0, 0.0, 0.0],
        "inertia_com_kg_m2": np.diag(diag).tolist(),
        "placement": _placement(rot, xyz),
    }


def _spec(total_mass: float | None = 3.0) -> dict:
    spec: dict = {
        "closure": {"body_b": CLUB_BODY},
        "bodies": [
            {
                "name": CLUB_BODY,
                "solids": [
                    _solid(
                        f"{CLUB_BODY}/Clubhead",
                        1.0,
                        (0.1, 0.2, 0.3),
                        np.eye(3),
                        (0.0, -1.0, 0.0),
                    ),
                    _solid(
                        f"{CLUB_BODY}/Rigid Shaft",
                        2.0,
                        (0.01, 0.02, 0.03),
                        RZ90,
                        (0.0, 1.0, 0.0),
                    ),
                    _solid(
                        "Test/Grip/LHand",
                        0.5,
                        (1e-3, 1e-3, 1e-3),
                        np.eye(3),
                        (0.0, 2.0, 0.0),
                    ),
                    _solid(
                        "Test/Grip/LHandStandoff",
                        0.01,
                        (1e-6,) * 3,
                        np.eye(3),
                        (0.0, 2.0, 0.1),
                    ),
                    _solid(
                        "Test/Grip/RHand",
                        0.6,
                        (2e-3, 2e-3, 2e-3),
                        np.eye(3),
                        (0.0, 1.5, 0.0),
                    ),
                ],
            }
        ],
    }
    if total_mass is not None:
        spec["club"] = {"total_mass_kg": total_mass}
    return spec


def test_bare_club_matches_hand_calculation() -> None:
    bare, _ = club_solids_partition(_spec())
    # mass 1 + 2; CoM y = (1*(-1) + 2*1) / 3 = 1/3
    assert bare.mass_kg == pytest.approx(3.0, abs=1e-12)
    np.testing.assert_allclose(bare.com_m, (0.0, 1.0 / 3.0, 0.0), atol=1e-12)
    # head offset -4/3, shaft offset +2/3 along y: parallel axis adds m d^2 to
    # I_xx and I_zz.  RZ90 swaps the shaft's x and y principal moments.
    shift = 1.0 * (4.0 / 3.0) ** 2 + 2.0 * (2.0 / 3.0) ** 2  # 24 / 9
    expected = np.diag([0.1 + 0.02 + shift, 0.2 + 0.01, 0.3 + 0.03 + shift])
    np.testing.assert_allclose(bare.inertia_com_kg_m2, expected, atol=1e-12)


def test_hand_solids_are_partitioned_out_and_kept_individually() -> None:
    bare, hands = club_solids_partition(_spec())
    assert set(hands) == {"LHand", "LHandStandoff", "RHand"}
    assert hands["LHand"].mass_kg == pytest.approx(0.5)
    assert hands["RHand"].mass_kg == pytest.approx(0.6)
    np.testing.assert_allclose(hands["RHand"].com_m, (0.0, 1.5, 0.0), atol=1e-12)
    assert bare.mass_kg == pytest.approx(3.0)  # hands excluded


def test_inertia_about_com_is_symmetric_and_positive_definite() -> None:
    bare, _ = club_solids_partition(_spec())
    inertia = bare.inertia_com_kg_m2
    np.testing.assert_allclose(inertia, inertia.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(inertia) > 0.0)


def test_mass_mismatch_with_club_total_raises_descriptive_error() -> None:
    with pytest.raises(ValueError, match=r"3\.0.*total_mass_kg.*3\.1"):
        club_solids_partition(_spec(total_mass=3.1))


def test_mass_check_is_tight_not_loose() -> None:
    club_solids_partition(_spec(total_mass=3.0 + 1e-10))
    with pytest.raises(ValueError, match="total_mass_kg"):
        club_solids_partition(_spec(total_mass=3.0 + 1e-8))


def test_spec_without_club_field_skips_the_mass_check() -> None:
    bare, _ = club_solids_partition(_spec(total_mass=None))
    assert bare.mass_kg == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.pop("closure"), "closure"),
        (lambda s: s["closure"].update(body_b="nope"), "nope"),
        (
            lambda s: s["bodies"][0]["solids"].__setitem__(slice(0, 2), []),
            "club solids",
        ),
    ],
)
def test_malformed_spec_is_rejected(mutate, message) -> None:
    spec = copy.deepcopy(_spec(total_mass=None))
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        club_solids_partition(spec)


@pytest.mark.parametrize(
    ("name", "total"),
    [("driver", 0.313), ("iron7", 0.428)],
)
def test_real_specs_bare_mass_equals_club_total(name: str, total: float) -> None:
    spec = json.loads((MODELS / f"full_body_spec_anthro_{name}.json").read_text())
    bare, hands = club_solids_partition(spec)
    assert bare.mass_kg == pytest.approx(total, abs=1e-9)
    assert sum(h.mass_kg for h in hands.values()) == pytest.approx(
        2 * 0.5491357853472616 + 0.01, abs=1e-9
    )


@pytest.mark.parametrize("name", ["v1", "v2"])
def test_legacy_specs_without_club_field_still_partition(name: str) -> None:
    spec = json.loads((MODELS / f"full_body_spec_{name}.json").read_text())
    bare, hands = club_solids_partition(spec)
    assert 0.3 < bare.mass_kg < 0.5
    assert {"LHand", "RHand"} <= set(hands)


def test_expressed_in_translates_and_rotates_consistently() -> None:
    inertia = RigidInertia(2.0, (0.0, 1.0, 0.0), np.diag([1.0, 2.0, 3.0]))
    transform = np.eye(4)
    transform[:3, :3] = RZ90
    transform[:3, 3] = (1.0, 0.0, 0.0)
    moved = inertia.expressed_in(transform)
    np.testing.assert_allclose(moved.com_m, (0.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(moved.inertia_com_kg_m2, np.diag([2.0, 1.0, 3.0]))
    assert moved.mass_kg == 2.0


def test_newton_euler_keywords_hand_calculation() -> None:
    inertia = RigidInertia(2.0, (0.0, 0.5, 0.0), np.diag([1.0, 2.0, 3.0]))
    kw = newton_euler_keywords(
        inertia,
        rot_world_from_body=RZ90,
        origin_world=(1.0, 0.0, 0.0),
        omega=(0.0, 0.0, 2.0),
        alpha=(0.0, 0.0, 1.0),
        accel_origin=(0.0, 0.0, 0.0),
        gravity=(0.0, 0.0, -9.81),
    )
    # r = R c = (-0.5, 0, 0); com = (0.5, 0, 0)
    np.testing.assert_allclose(kw["com_m"], (0.5, 0.0, 0.0), atol=1e-12)
    # a = alpha x r + omega x (omega x r) = (0,0,1)x(-.5,0,0) + 4*(0.5,0,0)
    np.testing.assert_allclose(
        kw["com_acceleration_m_s2"], (2.0, -0.5, 0.0), atol=1e-12
    )
    np.testing.assert_allclose(
        kw["inertia_world_kg_m2"], np.diag([2.0, 1.0, 3.0]), atol=1e-12
    )
    assert kw["mass_kg"] == 2.0


def test_interface_labels_are_the_documented_strings() -> None:
    assert INTERFACE_BARE_CLUB == "hand_on_bare_club"
    assert INTERFACE_LEGACY_COMPOSITE == "wrist_on_hands_plus_club"
