"""Club hand wrench and external-load assembly (SPN-9c, #12758).

Every number is INVENTED.  The wrench maths is checked against closed-form
rigid-body results; the ExternalLoads round trip uses the synthetic fixture
spine model (bodies ``pelvis`` and ``thorax_proxy``), not a licensed asset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
    ContactPoint,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_loads import (
    ClubInertia,
    LoadSet,
    club_net_wrench,
    combine_load_sets,
    describe_load_set,
    hand_reaction_loads,
    split_between_hands,
    write_load_set,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    prepare_model_for_id,
)
from src.shared.python.core.constants import GRAVITY_FLOAT
from tests.helpers.opensim_support import require_opensim

INERTIA = ClubInertia(
    mass_kg=0.4,
    com_local=(0.0, 0.5, 0.0),
    inertia_local=((0.04, 0.0, 0.01), (0.0, 0.02, 0.0), (0.01, 0.0, 0.05)),
)
T = np.linspace(0.0, 0.3, 61)


@pytest.mark.unit
def test_inertia_preconditions() -> None:
    with pytest.raises(ValueError, match="mass_kg"):
        ClubInertia(0.0, (0, 0, 0), np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        ClubInertia(1.0, (0, 0, 0), ((1, 0.5, 0), (0, 1, 0), (0, 0, 1)))
    with pytest.raises(ValueError, match="positive definite"):
        ClubInertia(1.0, (0, 0, 0), np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="com_local"):
        ClubInertia(1.0, (0, 0), np.eye(3))


@pytest.mark.unit
def test_translation_gives_m_times_a_minus_g() -> None:
    accel = np.array([2.0, 0.5, -1.0])
    com = 0.5 * T[:, None] * T[:, None] * accel
    rot = np.tile(np.eye(3), (T.size, 1, 1))
    wrench = club_net_wrench(T, com, rot, INERTIA)
    expected = INERTIA.mass_kg * (accel - np.array([0.0, -GRAVITY_FLOAT, 0.0]))
    np.testing.assert_allclose(
        wrench.force_n, np.tile(expected, (T.size, 1)), atol=1e-8
    )
    np.testing.assert_allclose(wrench.moment_com_nm, 0.0, atol=1e-8)


@pytest.mark.unit
def test_steady_spin_gives_the_gyroscopic_moment() -> None:
    omega = 6.0
    rot = Rotation.from_rotvec(np.outer(omega * T, [1.0, 0.0, 0.0])).as_matrix()
    com = np.zeros((T.size, 3))
    wrench = club_net_wrench(T, com, rot, INERTIA)
    inertia = np.array(INERTIA.inertia_local)
    w = np.array([omega, 0.0, 0.0])
    expected = np.array([np.cross(w, r @ inertia @ r.T @ w) for r in rot])
    np.testing.assert_allclose(wrench.moment_com_nm, expected, atol=2e-3)
    assert np.abs(expected).max() > 0.05  # the check is not vacuous


@pytest.mark.unit
def test_constant_angular_acceleration_gives_inertia_times_alpha() -> None:
    alpha = 12.0
    angle = 0.5 * alpha * T * T
    rot = Rotation.from_rotvec(np.outer(angle, [0.0, 1.0, 0.0])).as_matrix()
    wrench = club_net_wrench(T, np.zeros((T.size, 3)), rot, INERTIA)
    inertia = np.array(INERTIA.inertia_local)
    w = np.outer(alpha * T, [0.0, 1.0, 0.0])
    expected = np.array(
        [
            r @ inertia @ r.T @ np.array([0.0, alpha, 0.0])
            + np.cross(wi, r @ inertia @ r.T @ wi)
            for r, wi in zip(rot, w, strict=True)
        ]
    )
    np.testing.assert_allclose(wrench.moment_com_nm[5:-5], expected[5:-5], atol=5e-3)


@pytest.mark.unit
def test_wrench_preconditions() -> None:
    rot = np.tile(np.eye(3), (T.size, 1, 1))
    com = np.zeros((T.size, 3))
    with pytest.raises(ValueError, match="shape"):
        club_net_wrench(T, com[:-1], rot, INERTIA)
    with pytest.raises(ValueError, match="strictly increasing"):
        club_net_wrench(T[::-1], com, rot, INERTIA)
    with pytest.raises(ValueError, match="finite"):
        club_net_wrench(T, com * np.nan, rot, INERTIA)
    with pytest.raises(ValueError, match="proper rotation"):
        club_net_wrench(T, com, rot * 2.0, INERTIA)
    with pytest.raises(ValueError, match="at least 4"):
        club_net_wrench(T[:3], com[:3], rot[:3], INERTIA)


def _random_net(rng: np.random.Generator, n: int = 7):
    from src.engines.physics_engines.opensim.python.spine.fullbody_loads import (
        ClubWrench,
    )

    return ClubWrench(
        force_n=rng.normal(size=(n, 3)), moment_com_nm=rng.normal(size=(n, 3))
    )


@pytest.mark.unit
def test_split_conserves_force_and_moment_about_the_com() -> None:
    rng = np.random.default_rng(1)
    net = _random_net(rng)
    com = rng.normal(size=(7, 3))
    grips = com[:, None, :] + rng.normal(size=(7, 2, 3)) * 0.1
    hands = split_between_hands(net, com, grips, weights=(0.7, 0.3))
    np.testing.assert_allclose(hands.force_n.sum(axis=1), net.force_n, atol=1e-12)
    moment = (hands.torque_nm + np.cross(grips - com[:, None, :], hands.force_n)).sum(
        axis=1
    )
    np.testing.assert_allclose(moment, net.moment_com_nm, atol=1e-12)
    np.testing.assert_allclose(hands.force_n[:, 0], 0.7 * net.force_n, atol=1e-12)


@pytest.mark.unit
def test_split_preconditions() -> None:
    rng = np.random.default_rng(2)
    net = _random_net(rng)
    com = rng.normal(size=(7, 3))
    grips = rng.normal(size=(7, 2, 3))
    with pytest.raises(ValueError, match="sum to 1"):
        split_between_hands(net, com, grips, weights=(0.7, 0.7))
    with pytest.raises(ValueError, match="non-negative"):
        split_between_hands(net, com, grips, weights=(1.5, -0.5))
    with pytest.raises(ValueError, match="one weight per hand"):
        split_between_hands(net, com, grips, weights=(1.0,))
    with pytest.raises(ValueError, match="shape"):
        split_between_hands(net, com[:-1], grips, weights=(0.5, 0.5))


@pytest.mark.unit
def test_reaction_on_the_hand_is_the_opposite_wrench_at_the_hand_origin() -> None:
    rng = np.random.default_rng(3)
    net = _random_net(rng)
    com = rng.normal(size=(7, 3))
    grips = com[:, None, :] + rng.normal(size=(7, 2, 3)) * 0.1
    origins = grips + rng.normal(size=(7, 2, 3)) * 0.05
    hands = split_between_hands(net, com, grips, weights=(0.5, 0.5))
    force, torque = hand_reaction_loads(hands, grips, origins)
    np.testing.assert_allclose(force, -hands.force_n)
    # equal wrench about any common point: hand reaction + club wrench = 0
    point = rng.normal(size=(7, 3))
    on_club = (hands.torque_nm + np.cross(grips - point[:, None], hands.force_n)).sum(
        axis=1
    )
    on_hand = (torque + np.cross(origins - point[:, None], force)).sum(axis=1)
    np.testing.assert_allclose(on_hand, -on_club, atol=1e-12)


def _points() -> tuple[ContactPoint, ...]:
    return (
        ContactPoint("hand_lead", "thorax_proxy", (0.0, 0.0, 0.0), ""),
        ContactPoint("hand_trail", "thorax_proxy", (0.0, 0.0, 0.0), ""),
    )


@pytest.mark.unit
def test_load_set_validation_and_combination() -> None:
    pts = _points()
    ones = np.ones((5, 2, 3))
    kinds = ("club_hand_wrench", "club_hand_wrench")
    a = LoadSet(pts, ones, 2 * ones, kinds)
    foot = LoadSet(
        (ContactPoint("heel_a_r", "pelvis", (0.0, 0.0, 0.0), "r"),),
        np.ones((5, 1, 3)),
        np.zeros((5, 1, 3)),
        ("ground_reaction",),
    )
    both = combine_load_sets(a, foot)
    assert [p.name for p in both.points] == ["hand_lead", "hand_trail", "heel_a_r"]
    assert both.forces.shape == (5, 3, 3)
    assert both.kinds[-1] == "ground_reaction"
    with pytest.raises(ValueError, match="duplicate"):
        combine_load_sets(a, a)
    with pytest.raises(ValueError, match="frame count"):
        combine_load_sets(
            a, LoadSet(foot.points, np.ones((4, 1, 3)), np.zeros((4, 1, 3)), foot.kinds)
        )
    with pytest.raises(ValueError, match="shape"):
        LoadSet(pts, np.ones((5, 3, 3)), 2 * ones, kinds)
    with pytest.raises(ValueError, match="kind"):
        LoadSet(pts, ones, ones, ("club_hand_wrench", "mystery"))
    with pytest.raises(ValueError, match="finite"):
        LoadSet(pts, ones * np.nan, ones, kinds)
    with pytest.raises(ValueError, match="at least one"):
        combine_load_sets()


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_written_loads_are_picked_up_by_the_id_preparation(tmp_path: Path) -> None:
    osim = require_opensim()
    model = write_fixture_osim(FixtureSpec(), tmp_path / "fixture.osim")
    time_s = np.linspace(0.0, 0.2, 21)
    force = np.zeros((21, 2, 3))
    force[:, :, 1] = 3.0
    torque = np.zeros((21, 2, 3))
    torque[:, 1, 2] = 0.25
    loads = LoadSet(_points(), force, torque, ("club_hand_wrench",) * 2)
    xml = write_load_set(tmp_path, time_s, loads, stem="unit_loads")
    assert xml.name == "unit_loads.xml" and (tmp_path / "unit_loads.sto").is_file()
    prepared = prepare_model_for_id(model, external_loads=xml)
    assert "2 ExternalForce" in prepared.external_loads
    assert "hand_lead->thorax_proxy" in prepared.external_loads
    table = osim.TimeSeriesTable(str(tmp_path / "unit_loads.sto"))
    labels = list(table.getColumnLabels())
    assert "hand_trail_torque_z" in labels
    column = table.getDependentColumn("hand_trail_torque_z").to_numpy()
    np.testing.assert_allclose(column, 0.25)
    text = describe_load_set(loads)
    assert [item["body"] for item in text] == ["thorax_proxy", "thorax_proxy"]
    assert all(item["kind"] == "club_hand_wrench" for item in text)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_force_free_copy_keeps_external_forces_only_on_request(tmp_path: Path) -> None:
    from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
        _force_free_copy,
    )

    osim = require_opensim()
    model = write_fixture_osim(FixtureSpec(), tmp_path / "fixture.osim")
    time_s = np.linspace(0.0, 0.2, 21)
    loads = LoadSet(
        _points(),
        np.ones((21, 2, 3)),
        np.zeros((21, 2, 3)),
        ("club_hand_wrench",) * 2,
    )
    xml = write_load_set(tmp_path, time_s, loads, stem="keep")
    prepared = prepare_model_for_id(model, external_loads=xml)

    def applying(copy) -> list[bool]:
        forces = copy.getForceSet()
        return [
            bool(forces.get(i).get_appliesForce())
            for i in range(forces.getSize())
            if forces.get(i).getConcreteClassName() == "ExternalForce"
        ]

    assert applying(_force_free_copy(prepared.model)) == [False, False]
    assert applying(_force_free_copy(prepared.model, keep_external=True)) == [
        True,
        True,
    ]
    assert osim is not None
