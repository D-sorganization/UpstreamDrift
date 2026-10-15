"""Hand-on-club wrench uses the bare club, not the club plus both hands (NSX-01).

Issue #12550, epic #12593.  The spec welds ``LHand``, ``LHandStandoff`` and
``RHand`` into the club body (1.4 kg for a 0.3 kg club).  The net wrench the
hands exert on the club must come from head, shaft and grip only.  Same-input
bundle as ``test_full_body_grip_wrench``; legacy composite values stay
available under ``interface="wrist_on_hands_plus_club"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.club_inertia_partition import (
    INTERFACE_BARE_CLUB,
    INTERFACE_LEGACY_COMPOSITE,
    club_solids_partition,
)
from src.shared.python.motion_matching.same_input import (
    VectorPlant,
    project_to_closure,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.requires_mujoco,
    pytest.mark.requires_drake,
    pytest.mark.requires_pinocchio,
]

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
FRAMES = (0, 400, 420)
ATOL = 1e-6
RTOL = 1e-8
KEYS = (
    "mass_kg",
    "com_m",
    "com_acceleration_m_s2",
    "inertia_world_kg_m2",
    "angular_velocity_rad_s",
    "angular_acceleration_rad_s2",
)


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    return SPEC_PATH.read_bytes()


@pytest.fixture(scope="module")
def bare_mass(spec_bytes) -> float:
    return club_solids_partition(json.loads(spec_bytes)).bare_club.mass_kg


@pytest.fixture(scope="module")
def plants(spec_bytes) -> dict[str, VectorPlant]:
    for module in ("mujoco", "pydrake", "pinocchio"):
        pytest.importorskip(module)
    return {e: VectorPlant(e, spec_bytes) for e in ("mujoco", "drake", "pinocchio")}


@pytest.fixture(scope="module")
def record() -> dict[str, np.ndarray]:
    with np.load(EVIDENCE / "dynamics_record.npz") as data:
        return {key: data[key] for key in ("q", "v", "tau")}


def _state(plants, record, frame):
    st = project_to_closure(plants["pinocchio"], record["q"][frame], record["v"][frame])
    return st.q, st.v, record["tau"][frame]


def _named(plant, q, v, tau):
    return plant._named(q), plant._named(v), plant._named(tau)


def _accel(plant, q, v, tau):
    return plant._adapter.accelerations(*_named(plant, q, v, tau))


def _ne(engine: str, plants, q, v, tau, interface: str | None = None) -> dict:
    adapter = plants[engine]._adapter
    acceleration, _ = adapter._constrained_solve(*_named(plants[engine], q, v, tau))
    kwargs = {} if interface is None else {"interface": interface}
    return adapter._club_newton_euler(acceleration, **kwargs)


@pytest.mark.parametrize("engine", ["mujoco", "drake"])
@pytest.mark.parametrize("frame", FRAMES)
def test_default_newton_euler_mass_is_the_bare_club(
    plants, record, bare_mass, engine, frame
) -> None:
    q, v, tau = _state(plants, record, frame)
    assert _ne(engine, plants, q, v, tau)["mass_kg"] == pytest.approx(
        bare_mass, abs=1e-9
    )
    legacy = _ne(engine, plants, q, v, tau, INTERFACE_LEGACY_COMPOSITE)
    assert legacy["mass_kg"] > 1.4  # club plus both hands


@pytest.mark.parametrize("engine", ["mujoco", "drake"])
@pytest.mark.parametrize("frame", FRAMES)
def test_net_force_is_bare_club_newton_euler(
    plants, record, bare_mass, engine, frame
) -> None:
    q, v, tau = _state(plants, record, frame)
    plant = plants[engine]
    grip = plant._adapter.grip_analysis(*_named(plant, q, v, tau))
    ne = _ne(engine, plants, q, v, tau)
    expected = bare_mass * (
        np.asarray(ne["com_acceleration_m_s2"]) - np.asarray(ne["gravity_m_s2"])
    )
    np.testing.assert_allclose(
        grip.net_force_n, expected, rtol=0, atol=ATOL + RTOL * np.abs(expected).max()
    )
    assert grip.metadata["interface"] == INTERFACE_BARE_CLUB


@pytest.mark.parametrize("engine", ["mujoco", "drake"])
def test_legacy_interface_is_preserved_and_labelled(plants, record, engine) -> None:
    q, v, tau = _state(plants, record, 400)
    plant = plants[engine]
    args = _named(plant, q, v, tau)
    legacy = plant._adapter.grip_analysis(*args, interface=INTERFACE_LEGACY_COMPOSITE)
    bare = plant._adapter.grip_analysis(*args)
    assert legacy.metadata["interface"] == INTERFACE_LEGACY_COMPOSITE
    # the hands' own inertial load is a large part of the composite net force
    assert np.linalg.norm(legacy.net_force_n) > 1.5 * np.linalg.norm(bare.net_force_n)


@pytest.mark.parametrize("engine", ["mujoco", "drake"])
def test_unknown_interface_is_rejected(plants, record, engine) -> None:
    q, v, tau = _state(plants, record, 0)
    plant = plants[engine]
    with pytest.raises(ValueError, match="interface"):
        plant._adapter.grip_analysis(*_named(plant, q, v, tau), interface="bogus")


@pytest.mark.parametrize("frame", FRAMES)
def test_mujoco_and_drake_agree_on_bare_club_net_and_split(
    plants, record, frame
) -> None:
    q, v, tau = _state(plants, record, frame)
    mj, dr = (
        plants[e]._adapter.grip_analysis(*_named(plants[e], q, v, tau))
        for e in ("mujoco", "drake")
    )
    scale = np.abs(np.array(dr.net_force_n)).max()
    for field in ("net_force_n", "couple_at_midpoint_nm"):
        a, b = np.array(getattr(mj, field)), np.array(getattr(dr, field))
        bound = ATOL + RTOL * max(np.abs(b).max(), scale)
        np.testing.assert_allclose(a, b, rtol=0, atol=bound, err_msg=field)


@pytest.mark.parametrize("frame", FRAMES)
def test_pinocchio_bare_club_newton_euler_matches_mujoco_and_drake(
    plants, record, spec_bytes, frame
) -> None:
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )

    q, v, tau = _state(plants, record, frame)
    adapter = PinocchioForceAdapter(spec_bytes)
    order = adapter.coordinate_order
    plant = plants["pinocchio"]
    acc = _accel(plant, q, v, tau)
    to_vec = lambda named: np.array([named[n] for n in order])  # noqa: E731
    pin_ne = adapter.club_newton_euler(
        to_vec(plant._named(q)), to_vec(plant._named(v)), to_vec(acc)
    )
    for engine in ("mujoco", "drake"):
        other = _ne(engine, plants, q, v, tau)
        for key in KEYS:
            a, b = np.asarray(pin_ne[key]), np.asarray(other[key])
            bound = ATOL + RTOL * max(np.abs(b).max(), 1.0)
            np.testing.assert_allclose(
                a, b, rtol=0, atol=bound, err_msg=f"{engine}:{key}"
            )
    force, _torque = adapter.bare_club_net_wrench(
        to_vec(plant._named(q)), to_vec(plant._named(v)), to_vec(acc)
    )
    mj = plants["mujoco"]._adapter.grip_analysis(*_named(plants["mujoco"], q, v, tau))
    np.testing.assert_allclose(
        force, mj.net_force_n, rtol=0, atol=ATOL + RTOL * np.abs(force).max()
    )
