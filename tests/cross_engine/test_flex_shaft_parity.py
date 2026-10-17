"""R11 (#12691): the R6 chain shaft in OpenSim and MyoSuite, against MuJoCo.

Every engine builds the chain from the same chain spec
(``apply_shaft_chain``); no engine carries its own shaft parameter.

1. Static and modal: the shaft block of each engine's mass matrix with the
   engine's compiled springs gives the first bending frequency per axis (also
   against R6's FEM reference), and the Clubhead compliance through each
   engine's kinematics equals MuJoCo's.
2. Same-input parity: pointwise constrained accelerations with a deflected
   shaft, and open- and closed-loop 30 ms replays of a MuJoCo chain bundle,
   at the existing same-input tolerances (unchanged).
3. Topology: coordinate order with the appended shaft coordinates, chain
   frames, total mass and the mass matrix agree with MuJoCo; at zero
   deflection the rigid-coordinate block equals the rigid club's.
4. The refusal is lifted only for passing engines: Drake and Pinocchio
   (no chain yet) still raise ``ShaftChainUnsupportedError``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DET
import numpy as np
import pytest

from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    CHAIN_ENGINES,
    ShaftChainUnsupportedError,
    apply_shaft_chain,
    build_shaft_chain,
    clamped_head_frequencies,
    span_properties,
)

pytestmark = [pytest.mark.unit, pytest.mark.cross_engine]

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
FRAMES = (0, 400, 420)
#: Existing same-input tolerances (test_opensim_parity, test_open_loop_parity).
ACCELERATION_ATOL = 1e-6
ACCELERATION_RTOL = 1e-8
REPLAY_Q_ATOL = 1e-10
REPLAY_V_ATOL = 1e-8
REPLAY_EFFORT_ATOL = 1e-6
#: R6 convergence tolerance (#12686) for the FEM comparison.
R6_RTOL = 0.01
#: Engine-to-engine tolerance of the static and modal checks; far inside R6's
#: 1 % because both engines realise the identical lumped chain.
CHAIN_RTOL = 1e-8
FD_STEP_RAD = 1e-6
REPLAY_S = 0.03
OPENSIM_MARKS = (pytest.mark.requires_opensim, pytest.mark.integration)
MYOSUITE_MARKS = (pytest.mark.requires_myosuite, pytest.mark.integration)
ENGINES = [
    pytest.param("opensim", marks=OPENSIM_MARKS),
    pytest.param("myosuite", marks=MYOSUITE_MARKS),
]


def _require(engine: str) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip(engine)


@pytest.fixture(scope="module")
def rigid_bytes() -> bytes:
    return (SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes()


@pytest.fixture(scope="module")
def chain_spec(rigid_bytes: bytes) -> dict[str, Any]:
    return apply_shaft_chain(json.loads(rigid_bytes), build_shaft_chain(DRIVER))


@pytest.fixture(scope="module")
def chain_bytes(chain_spec: dict[str, Any]) -> bytes:
    return json.dumps(chain_spec).encode()


@pytest.fixture(scope="module")
def shaft(chain_spec: dict[str, Any]) -> tuple[str, ...]:
    return tuple(chain_spec["club"]["shaft"]["coordinates"])


@pytest.fixture(scope="module")
def record(shaft: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The R6 seed record, padded with an undeflected shaft."""
    with np.load(SEEDS / "dynamics_record.npz") as data:
        out = {key: data[key] for key in ("time_s", "q", "v", "tau")}
    out["q_chain"] = np.hstack([out["q"], np.zeros((out["q"].shape[0], len(shaft)))])
    return out


@pytest.fixture(scope="module")
def plants(chain_bytes: bytes) -> dict[str, Any]:
    """Lazily built chain plants, one per engine (module cache)."""
    from src.shared.python.motion_matching.same_input import VectorPlant

    cache: dict[str, Any] = {}

    def get(engine: str) -> Any:
        if engine not in cache:
            _require("mujoco" if engine == "mujoco" else engine)
            cache[engine] = VectorPlant(engine, chain_bytes)
        return cache[engine]

    return {"get": get}


# --- criterion 4: refusal lifted only for passing engines ----------------------


def test_chain_support_is_exactly_the_passing_engines() -> None:
    assert {"mujoco", "opensim", "myosuite"} == set(CHAIN_ENGINES)


@pytest.mark.parametrize("engine", ["drake", "pinocchio"])
def test_engine_without_a_passing_gate_still_refuses(
    chain_bytes: bytes, engine: str
) -> None:
    from src.shared.python.motion_matching.same_input import VectorPlant

    with pytest.raises(ShaftChainUnsupportedError, match="silent rigid replay"):
        VectorPlant(engine, chain_bytes)


# --- OpenSim export (pure XML) -------------------------------------------------


def _osim(spec_bytes: bytes) -> tuple[Any, dict[str, Any]]:
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        export_full_body_osim,
    )

    xml, metadata = export_full_body_osim(spec_bytes)
    # OpenSim's ``Class::Nested`` tag names are not namespace-well-formed.
    return DET.fromstring(xml.replace("::", "__")), metadata


def test_osim_export_has_one_spec_spring_per_shaft_coordinate(
    chain_spec: dict[str, Any], chain_bytes: bytes, shaft: tuple[str, ...]
) -> None:
    root, metadata = _osim(chain_bytes)
    assert metadata["coordinate_order"] == chain_spec["coordinate_order"]
    springs = {s.findtext("coordinate"): s for s in root.iter("SpringGeneralizedForce")}
    assert tuple(springs) == shaft
    primitives = {
        p["coordinate"]: p for j in chain_spec["joints"] for p in j["primitives"]
    }
    for name, spring in springs.items():
        spec = primitives[name]
        assert float(spring.findtext("stiffness")) == spec["stiffness_n_m_rad"]
        assert float(spring.findtext("viscosity")) == spec["damping_n_m_s_rad"]
        assert float(spring.findtext("rest_length")) == spec["springref_rad"]
    actuated = {a.findtext("coordinate") for a in root.iter("CoordinateActuator")}
    assert not actuated & set(shaft)
    n_links = len(shaft) // 2  # two bending coordinates per joint, no torsion
    bodies = {b.get("name") for b in root.iter("Body")}
    assert {f"ShaftLink{i}" for i in range(n_links)} <= bodies


def test_rigid_osim_export_has_no_spring(rigid_bytes: bytes) -> None:
    root, _ = _osim(rigid_bytes)
    assert next(root.iter("SpringGeneralizedForce"), None) is None


# --- criterion 3: topology, frames, mass and inertia -----------------------------


def _mujoco_mass_matrix(plant: Any, q: np.ndarray) -> np.ndarray:
    from src.shared.python.engine_core.mujoco_compat import full_mass_matrix

    adapter = plant._adapter
    adapter.generalized_forces(plant._named(q), plant._named(np.zeros(plant.nv)))
    idx = [adapter._indices[name] for name in plant.coordinate_order]
    raw = full_mass_matrix(adapter._mj, adapter.model, adapter.data)
    return raw[np.ix_(idx, idx)]


def _mass_matrix(engine: str, plant: Any, q: np.ndarray) -> np.ndarray:
    if engine == "opensim":
        return plant._adapter.mass_matrix(plant._named(q))
    return _mujoco_mass_matrix(plant, q)


def _springs(engine: str, plant: Any) -> dict[str, tuple[float, float]]:
    """Compiled ``{coordinate: (stiffness, damping)}`` of the shaft springs."""
    adapter = plant._adapter
    if engine == "opensim":
        return {n: (k, c) for n, (k, _, c) in adapter.shaft_springs().items()}
    model = adapter.model
    return {
        name: (
            float(model.jnt_stiffness[model.joint(name).id]),
            float(model.dof_damping[adapter._indices[name]]),
        )
        for name in adapter.passive_coordinates
    }


def _deflected(record: dict[str, np.ndarray], frame: int, n: int) -> np.ndarray:
    q = record["q_chain"][frame].copy()
    q[-n:] = np.random.default_rng(frame).normal(scale=1e-3, size=n)
    return q


@pytest.mark.parametrize("engine", ENGINES)
def test_topology_and_springs_match_mujoco(
    plants: dict[str, Any],
    chain_spec: dict[str, Any],
    shaft: tuple[str, ...],
    engine: str,
) -> None:
    plant, reference = plants["get"](engine), plants["get"]("mujoco")
    assert plant.coordinate_order == tuple(chain_spec["coordinate_order"])
    assert plant.coordinate_order[-len(shaft) :] == shaft
    assert tuple(plant._adapter.passive_coordinates) == shaft
    assert _springs(engine, plant) == _springs("mujoco", reference)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("frame", FRAMES)
def test_mass_matrix_and_chain_frames_match_mujoco(
    plants: dict[str, Any],
    record: dict[str, np.ndarray],
    shaft: tuple[str, ...],
    engine: str,
    frame: int,
) -> None:
    plant, reference = plants["get"](engine), plants["get"]("mujoco")
    q = _deflected(record, frame, len(shaft))
    mass, mass_ref = _mass_matrix(engine, plant, q), _mujoco_mass_matrix(reference, q)
    assert np.abs(mass - mass_ref).max() <= 1e-9 * np.abs(mass_ref).max()
    frames, frames_ref = plant.kinematic_frames(q), reference.kinematic_frames(q)
    assert frames.keys() == frames_ref.keys()
    for name, pose in frames.items():
        np.testing.assert_allclose(pose, frames_ref[name], rtol=0, atol=1e-12)


@pytest.mark.parametrize("engine", [pytest.param("opensim", marks=OPENSIM_MARKS)])
def test_rigid_block_is_the_rigid_club_at_zero_deflection(
    plants: dict[str, Any],
    rigid_bytes: bytes,
    record: dict[str, np.ndarray],
    engine: str,
) -> None:
    from src.shared.python.motion_matching.same_input import VectorPlant

    plant = plants["get"](engine)
    rigid = VectorPlant(engine, rigid_bytes)
    assert plant._adapter.mass_kg == pytest.approx(rigid._adapter.mass_kg, rel=1e-12)
    q = record["q_chain"][400]
    mass = _mass_matrix(engine, plant, q)[: rigid.nv, : rigid.nv]
    mass_rigid = _mass_matrix(engine, rigid, q[: rigid.nv])
    assert np.abs(mass - mass_rigid).max() <= 1e-9 * np.abs(mass_rigid).max()


# --- criterion 1: static and modal ------------------------------------------------


def _first_frequency(
    engine: str, plant: Any, q: np.ndarray, shaft: tuple[str, ...], axis: str
) -> float:
    """Lowest frequency (Hz) of one bending family with the club body held.

    The shaft block of the full-body mass matrix is the grip-clamped chain's
    (the links' inertia about their joints does not depend on the rest of
    the body), so this is R6's clamped modal check in the swing model.
    """
    from scipy.linalg import eigh

    family = [i for i, n in enumerate(shaft) if f"_bend_{axis}_" in n]
    dofs = [plant.coordinate_order.index(shaft[i]) for i in family]
    mass = _mass_matrix(engine, plant, q)[np.ix_(dofs, dofs)]
    springs = _springs(engine, plant)
    stiffness = np.diag([springs[shaft[i]][0] for i in family])
    return float(np.sqrt(eigh(stiffness, mass, eigvals_only=True)[0]) / (2 * np.pi))


def _clubhead_compliance(
    engine: str, plant: Any, q: np.ndarray, shaft: tuple[str, ...]
) -> np.ndarray:
    """``J K^-1 J^T`` (m/N) of the Clubhead origin over the shaft springs."""
    springs = _springs(engine, plant)
    columns = []
    for name in shaft:
        k = plant.coordinate_order.index(name)
        step = np.zeros(plant.nv)
        step[k] = FD_STEP_RAD
        ahead = plant.kinematic_frames(q + step)["Clubhead"][:3, 3]
        behind = plant.kinematic_frames(q - step)["Clubhead"][:3, 3]
        columns.append((ahead - behind) / (2 * FD_STEP_RAD))
    jacobian = np.column_stack(columns)
    inverse = np.diag([1.0 / springs[name][0] for name in shaft])
    return jacobian @ inverse @ jacobian.T


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("axis", ["x", "z"])
def test_first_bending_frequency_matches_mujoco_and_fem(
    plants: dict[str, Any],
    record: dict[str, np.ndarray],
    shaft: tuple[str, ...],
    engine: str,
    axis: str,
) -> None:
    q = record["q_chain"][400]
    value = _first_frequency(engine, plants["get"](engine), q, shaft, axis)
    reference = _first_frequency("mujoco", plants["get"]("mujoco"), q, shaft, axis)
    assert value == pytest.approx(reference, rel=CHAIN_RTOL)
    gyration = DRIVER.head_gyration_m["xyz".index(axis)]
    fem = clamped_head_frequencies(
        span_properties(DRIVER), DRIVER, DRIVER.head_mass_kg * gyration**2
    )[0]
    assert value == pytest.approx(float(fem), rel=R6_RTOL)


@pytest.mark.parametrize("engine", ENGINES)
def test_static_clubhead_compliance_matches_mujoco(
    plants: dict[str, Any],
    record: dict[str, np.ndarray],
    shaft: tuple[str, ...],
    engine: str,
) -> None:
    q = record["q_chain"][400]
    value = _clubhead_compliance(engine, plants["get"](engine), q, shaft)
    reference = _clubhead_compliance("mujoco", plants["get"]("mujoco"), q, shaft)
    assert np.linalg.eigvalsh(reference)[-1] > 1e-3  # the shaft does bend
    assert np.abs(value - reference).max() <= CHAIN_RTOL * np.abs(reference).max()


# --- criterion 2: same-input parity --------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("frame", FRAMES)
def test_pointwise_accelerations_match_mujoco_with_a_deflected_shaft(
    plants: dict[str, Any],
    record: dict[str, np.ndarray],
    shaft: tuple[str, ...],
    engine: str,
    frame: int,
) -> None:
    from src.shared.python.motion_matching.same_input import project_to_closure

    plant, reference = plants["get"](engine), plants["get"]("mujoco")
    rng = np.random.default_rng(frame)
    v = np.r_[record["v"][frame], rng.normal(scale=1e-2, size=len(shaft))]
    state = project_to_closure(reference, _deflected(record, frame, len(shaft)), v)
    tau = np.r_[record["tau"][frame], np.zeros(len(shaft))]
    expected = reference.acceleration(state.q, state.v, tau)
    assert np.abs(expected[-len(shaft) :]).max() > 1e3  # springs are loaded
    bound = ACCELERATION_ATOL + ACCELERATION_RTOL * np.abs(expected).max()
    result = plant.acceleration(state.q, state.v, tau)
    np.testing.assert_allclose(result, expected, rtol=0, atol=bound)
    tau[-len(shaft) :] = 1.0  # the shaft coordinates are unactuated
    np.testing.assert_array_equal(plant.acceleration(state.q, state.v, tau), result)


@pytest.fixture(scope="module")
def bundle(chain_bytes: bytes, record: dict[str, np.ndarray]) -> Any:
    from src.shared.python.motion_matching.same_input import generate_reference_bundle

    pytest.importorskip("mujoco")
    return generate_reference_bundle(
        chain_bytes, record["time_s"], record["q_chain"], duration_s=REPLAY_S
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_open_loop_replay_matches_the_mujoco_chain_bundle(
    bundle: Any, shaft: tuple[str, ...], engine: str
) -> None:
    from src.shared.python.motion_matching.same_input import VectorPlant, open_loop

    _require(engine)
    assert np.abs(bundle.reference_q[:, -len(shaft) :]).max() > 1e-3
    rollout = open_loop(
        VectorPlant(engine, bundle.spec_bytes),
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
    )
    np.testing.assert_allclose(
        rollout.q, bundle.reference_q, rtol=0, atol=REPLAY_Q_ATOL
    )
    np.testing.assert_allclose(
        rollout.v, bundle.reference_v, rtol=0, atol=REPLAY_V_ATOL
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_closed_loop_replay_matches_the_mujoco_chain_bundle(
    bundle: Any, chain_bytes: bytes, record: dict[str, np.ndarray], engine: str
) -> None:
    from src.shared.python.motion_matching.same_input import closed_loop

    _require(engine)
    rollout = closed_loop(
        engine, chain_bytes, record["time_s"], record["q_chain"], duration_s=REPLAY_S
    )
    np.testing.assert_allclose(
        rollout.q, bundle.reference_q, rtol=0, atol=REPLAY_Q_ATOL
    )
    np.testing.assert_allclose(
        rollout.efforts, bundle.efforts, rtol=0, atol=REPLAY_EFFORT_ATOL
    )
