"""R6 (#12686): engine-neutral flexible-shaft chain against the FEM reference.

Acceptance criteria covered here (no physics engine needed):

1. Static tip deflection of the lumped chain equals the FEM cantilever
   (``FiniteElementShaftModel``) within 1 %, and the N -> 2N change is < 1 %.
3. The chain partitions the rigid club exactly: mass, centre of mass and
   inertia at zero deflection equal the rigid club's to 1e-9.
- The default profile is a generic STIFF flex flagged ``default_not_measured``.
- DbC: a chain spec is refused for an engine without chain support.

The modal (2), bit-identical default (4) and rigid-limit (5) criteria need
MuJoCo and live in ``test_full_body_mujoco_flex_shaft.py``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.biomechanics.club_inertia_partition import _compose
from src.shared.python.motion_matching.club_models import (
    CLUB_BODY_SUFFIX,
    CLUBS,
    DRIVER,
    IRON_7,
    ClubSpec,
)
from src.shared.python.motion_matching.club_shaft_chain import (
    CPM_CLAMP_M,
    DEFAULT_FLEX,
    DEFAULT_LINKS,
    PROFILE_STATUS,
    ShaftChainUnsupportedError,
    apply_shaft_chain,
    build_shaft_chain,
    clamped_head_frequencies,
    flex_frequency_hz,
    passive_coordinates,
    require_engine_support,
    shaft_model,
    span_properties,
    station_profile,
    tube_properties,
    fem_tip_deflection,
    calibrated_modulus_pa,
)

pytestmark = pytest.mark.unit
CLUB_NAMES = ("driver", "iron7")
TIP_LOAD_N = 1.0


def lumped_tip_deflection(club: ClubSpec, n_links: int) -> float:
    """Small-angle tip deflection of the chain under a unit transverse tip load.

    Joint ``i`` carries the moment ``F (L - s_i)`` and turns by it over
    ``k_i``; the tip moves by that angle times the lever ``L - s_i``.
    """
    chain = build_shaft_chain(club, n_links)
    span = chain.span_m
    return sum(
        TIP_LOAD_N * (span - s) ** 2 / k
        for s, k in zip(chain.joint_s_m, chain.bend_stiffness, strict=True)
    )


@pytest.fixture(scope="module")
def fem_deflection() -> dict[str, float]:
    return {
        name: fem_tip_deflection(span_properties(CLUBS[name]), TIP_LOAD_N)
        for name in CLUB_NAMES
    }


def test_default_profile_is_generic_stiff_and_flagged() -> None:
    chain = build_shaft_chain(DRIVER)
    record = chain.record
    assert DEFAULT_FLEX == "Stiff" and chain.n_links == DEFAULT_LINKS
    assert record["flex"] == "Stiff"
    assert record["flex_cpm"] == pytest.approx(280.0)
    assert record["profile_status"] == PROFILE_STATUS == "default_not_measured"
    assert not chain.torsion and chain.stiffness_scale == 1.0
    assert any("Milne" in s for s in record["sources"])
    assert any("MacKenzie" in s for s in record["sources"])
    profile = station_profile(span_properties(DRIVER), DRIVER)
    assert PROFILE_STATUS in profile.provenance.source_name


@pytest.mark.parametrize("name", CLUB_NAMES)
def test_modulus_reproduces_flex_frequency_in_the_butt_clamp(name: str) -> None:
    club = CLUBS[name]
    clamp = tube_properties(club, CPM_CLAMP_M, calibrated_modulus_pa(club))
    f1 = float(clamped_head_frequencies(clamp, club)[0])
    # Frequency is linear in sqrt(E); the generalized eigensolver limits it.
    assert f1 == pytest.approx(flex_frequency_hz(DEFAULT_FLEX), rel=1e-6)


@pytest.mark.parametrize("name", CLUB_NAMES)
def test_profile_keeps_the_club_shaft_mass(name: str) -> None:
    club = CLUBS[name]
    profile = station_profile(span_properties(club), club)
    density = [st.linear_density_kg_m for st in profile.stations]
    expected = club.shaft_mass_kg / (club.length_m - club.grip_length_m)
    np.testing.assert_allclose(density, expected, rtol=1e-12)


@pytest.mark.parametrize("name", CLUB_NAMES)
def test_static_tip_deflection_matches_fem(
    name: str, fem_deflection: dict[str, float]
) -> None:
    """Criterion 1: the default chain is within 1 % of the FEM cantilever."""
    chain = lumped_tip_deflection(CLUBS[name], DEFAULT_LINKS)
    assert chain == pytest.approx(fem_deflection[name], rel=0.01)


@pytest.mark.parametrize("name", CLUB_NAMES)
def test_static_deflection_converges_with_links(
    name: str, fem_deflection: dict[str, float]
) -> None:
    """Criterion 1: N -> 2N changes < 1 % at the default N; error shrinks ~1/N^2."""
    club = CLUBS[name]
    values = {n: lumped_tip_deflection(club, n) for n in (2, 4, 8, 16)}
    change = abs(values[2 * DEFAULT_LINKS] - values[DEFAULT_LINKS])
    assert change / values[DEFAULT_LINKS] < 0.01
    errors = [abs(values[n] / fem_deflection[name] - 1.0) for n in (2, 4, 8, 16)]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-3  # second order: N = 16 within 0.1 %


def test_fem_without_head_agrees_with_the_tools_modal_solver() -> None:
    """DRY cross-check: the physics FEM and the vendored Tools modal solver
    give the same clamped-free first frequency for the ``ShaftStation``
    profile (no head)."""
    from scipy.linalg import eigh

    from shared.python.golf_club.shaft_dynamics import solve_shaft_bending_modes
    from src.shared.python.physics.flexible_shaft import FiniteElementShaftModel

    properties = span_properties(DRIVER)
    fem = FiniteElementShaftModel(64)
    fem.initialize(properties)
    f_fem = math.sqrt(eigh(fem.K, fem.M, eigvals_only=True)[0]) / (2 * math.pi)
    tools = solve_shaft_bending_modes(station_profile(properties, DRIVER))
    assert tools.frequencies_x_hz[0] == pytest.approx(f_fem, rel=5e-3)
    assert tools.frequencies_y_hz[0] == pytest.approx(f_fem, rel=5e-3)


def test_station_profile_uses_the_canonical_tools_golf_club_module() -> None:
    """The profile must come from the one ``shared.python.golf_club`` tree.

    Importing the ``src.shared.python.golf_club`` spelling builds a second
    module tree whose import re-runs Tools' turf-variable registration and
    trips the registry's duplicate-key precondition (#12686 CI).
    """
    import shared.python.golf_club.shaft_profile as canonical

    profile = station_profile(span_properties(DRIVER), DRIVER)
    assert type(profile) is canonical.ShaftProfile


def test_chain_contracts() -> None:
    chain = build_shaft_chain(DRIVER, 3, torsion=True)
    assert chain.coordinates() == (
        "shaft_bend_x_0",
        "shaft_bend_z_0",
        "shaft_twist_0",
        "shaft_bend_x_1",
        "shaft_bend_z_1",
        "shaft_twist_1",
        "shaft_bend_x_2",
        "shaft_bend_z_2",
        "shaft_twist_2",
    )
    assert chain.link_bounds_m[-1][1] == pytest.approx(chain.span_m)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            build_shaft_chain(DRIVER, bad)  # type: ignore[arg-type]
    for scale in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            build_shaft_chain(DRIVER, stiffness_scale=scale)
    with pytest.raises(ValueError, match="unknown flex"):
        build_shaft_chain(DRIVER, flex="Noodle")
    stiff = build_shaft_chain(DRIVER, stiffness_scale=1e6)
    base = build_shaft_chain(DRIVER)
    np.testing.assert_allclose(
        stiff.bend_stiffness, np.asarray(base.bend_stiffness) * 1e6, rtol=1e-12
    )
    # Damping scales with sqrt(scale): modal damping ratios are kept.
    assert stiff.damping_beta_s == pytest.approx(base.damping_beta_s / 1e3)


# --- specification level ------------------------------------------------------


def _club_document(club: ClubSpec) -> dict[str, Any]:
    """Minimal rigid-club spec: the club body of ``club_solids`` plus a frame."""
    from src.shared.python.motion_matching.club_models import club_solids

    prefix = f"Test{CLUB_BODY_SUFFIX}"
    return {
        "coordinate_order": ["root_x"],
        "bodies": [
            {"name": "world", "solids": []},
            {"name": prefix, "solids": club_solids(prefix, club)},
        ],
        "joints": [
            {
                "name": "root",
                "parent": "world",
                "child": prefix,
                "parent_to_base": np.eye(4).tolist(),
                "child_to_follower": np.eye(4).tolist(),
                "primitives": [{"primitive": "Px", "coordinate": "root_x"}],
            }
        ],
        "frames": [
            {"name": "Clubhead", "body": prefix, "placement": np.eye(4).tolist()},
            {"name": "Grip", "body": prefix, "placement": np.eye(4).tolist()},
        ],
        "club": {"name": club.name},
    }


def _club_solids_in_club_frame(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """All club solids; at zero deflection every link frame is the club frame."""
    return [
        solid
        for body in doc["bodies"]
        if CLUB_BODY_SUFFIX in body["name"]
        for solid in body["solids"]
    ]


@pytest.mark.parametrize("club", [DRIVER, IRON_7], ids=lambda c: c.name)
@pytest.mark.parametrize("n_links", [1, 4, 7])
def test_chain_preserves_club_mass_com_and_inertia(
    club: ClubSpec, n_links: int
) -> None:
    """Criterion 3: the exact rod partition keeps the rigid inertia to 1e-9."""
    rigid_doc = _club_document(club)
    chain_doc = apply_shaft_chain(rigid_doc, build_shaft_chain(club, n_links))
    rigid = _compose(_club_solids_in_club_frame(rigid_doc))
    chain = _compose(_club_solids_in_club_frame(chain_doc))
    assert chain.mass_kg == pytest.approx(rigid.mass_kg, abs=1e-9)
    np.testing.assert_allclose(chain.com_m, rigid.com_m, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        chain.inertia_com_kg_m2, rigid.inertia_com_kg_m2, rtol=0, atol=1e-9
    )


def test_apply_chain_appends_coordinates_and_moves_the_head() -> None:
    rigid_doc = _club_document(DRIVER)
    snapshot = json.dumps(rigid_doc, sort_keys=True)
    chain = build_shaft_chain(DRIVER)
    doc = apply_shaft_chain(rigid_doc, chain)
    assert json.dumps(rigid_doc, sort_keys=True) == snapshot  # input untouched
    assert doc["coordinate_order"] == ["root_x", *chain.coordinates()]
    shaft = doc["club"]["shaft"]
    assert shaft_model(doc) == "chain" and shaft["profile_status"] == PROFILE_STATUS
    assert passive_coordinates(doc) == chain.coordinates()
    frames = {f["name"]: f["body"] for f in doc["frames"]}
    assert frames["Clubhead"] == shaft["tip_link"] != shaft["club_body"]
    assert frames["Grip"] == shaft["club_body"]
    tip = next(b for b in doc["bodies"] if b["name"] == shaft["tip_link"])
    assert any(s["name"].endswith("/Clubhead") for s in tip["solids"])
    springs = [
        p for j in doc["joints"] for p in j["primitives"] if "stiffness_n_m_rad" in p
    ]
    assert [p["coordinate"] for p in springs] == list(chain.coordinates())
    with pytest.raises(ValueError, match="already"):
        apply_shaft_chain(doc, chain)
    with pytest.raises(ValueError, match="rigid driver solids"):
        apply_shaft_chain(_club_document(IRON_7), chain)


def test_rigid_default_has_no_passive_coordinates() -> None:
    doc = _club_document(DRIVER)
    assert shaft_model(doc) == "rigid" and passive_coordinates(doc) == ()
    doc["club"]["shaft"] = {"model": "spaghetti"}
    with pytest.raises(ValueError, match="unknown club.shaft.model"):
        shaft_model(doc)


@pytest.mark.parametrize("engine", ["drake", "pinocchio", "simscape"])
def test_chain_spec_is_refused_for_engines_without_chain_support(engine: str) -> None:
    """DbC: never a silent rigid replay of a flexible-shaft bundle."""
    doc = apply_shaft_chain(_club_document(DRIVER), build_shaft_chain(DRIVER))
    with pytest.raises(ShaftChainUnsupportedError, match="silent rigid replay"):
        require_engine_support(json.dumps(doc).encode(), engine)
    for supported in ("mujoco", "opensim", "myosuite"):  # parity gates, #12691
        require_engine_support(doc, supported)
    require_engine_support(_club_document(DRIVER), engine)  # rigid: any engine
