"""Same-input bushing-grip parity: MuJoCo, Drake, Pinocchio vs OpenSim (#11739).

Acceptance (fixed before any comparison was run, issue #11739 phase 2):

* per engine and club over 0 to 1.8 s, per-hand force, net force, internal
  force, squeeze, couple and deflection agree with the OpenSim reference
  within 5 % at the peak and 2 % RMS (normalised by the reference peak),
  see :mod:`src.shared.python.grip_contact.parity`;
* static hold: the two hand forces support the club weight within 1 %;
* ``F = K delta``: a pure deflection of the club frame gives the bushing
  stiffness force through the engine's own force path.

The OpenSim reference series is committed evidence
(``evidence/grip_kinetics/parity/opensim_<club>_series.npz``) produced by
``run_grip_parity.py``; the engine runs are recomputed here.  Every engine
skips cleanly when its package is not importable.
"""

from __future__ import annotations

import json
import types
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    ClubDynamics,
    CoordinateSwing,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.parity import (
    GripKineticsSeries,
    parity_errors,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
REFERENCE = MODELS / "evidence/grip_kinetics/parity"
CLUBS = ("driver", "iron7")
WEIGHT_TOLERANCE = 0.01
STIFFNESS_TOLERANCE = 1e-6
ENGINE_MODULES = {
    "mujoco": ("mujoco", "src.engines.physics_engines.mujoco.python.grip_bushing"),
    "drake": ("pydrake", "src.engines.physics_engines.drake.python.grip_bushing"),
    "pinocchio": (
        "pinocchio",
        "src.engines.physics_engines.pinocchio.python.grip_bushing",
    ),
    "myosuite": (
        "myosuite",
        "src.engines.physics_engines.myosuite.python.grip_bushing",
    ),
}


def _engine(name: str) -> types.ModuleType:
    package, module = ENGINE_MODULES[name]
    try:
        real = import_module(package)
    except ImportError as exc:
        pytest.skip(f"{package} is not installed: {exc}")
    if not hasattr(real, "__file__"):
        pytest.skip(f"{package} in sys.modules is a test double")
    if name == "myosuite":
        # a partial install imports the package but not the scene loader
        pytest.importorskip("myosuite.envs.env_base")
    return import_module(module)


def _spec_bytes(club: str) -> bytes:
    return (MODELS / f"full_body_spec_anthro_{club}.json").read_bytes()


def _swing(club: str) -> CoordinateSwing:
    names = json.loads(_spec_bytes(club))["coordinate_order"]
    return load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz", FIXTURES / "address_poses.json", club, names
    )


def _hold(swing: CoordinateSwing, duration_s: float = 0.2) -> CoordinateSwing:
    n = int(round(duration_s / 0.002)) + 1
    return CoordinateSwing(
        swing.names,
        np.arange(n) * 0.002,
        np.repeat(swing.q[:1], n, axis=0),
        swing.sha256,
    )


ENGINES = sorted(ENGINE_MODULES)


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(e, marks=pytest.mark.slow) if e == "pinocchio" else e
        for e in ENGINES
    ],
)
def test_static_hold_supports_the_club_weight(engine: str) -> None:
    module = _engine(engine)
    spec_bytes = _spec_bytes("driver")
    spec = json.loads(spec_bytes)
    series = module.simulate_grip_bushing(spec_bytes, _hold(_swing("driver")))
    gravity = np.asarray(spec["gravity_m_s2"], float)
    weight = ClubDynamics.from_spec(spec).mass_kg * float(np.linalg.norm(gravity))
    up = -gravity / np.linalg.norm(gravity)
    total = series.force_on_club_n["L"][-1] + series.force_on_club_n["R"][-1]
    assert float(total @ up) == pytest.approx(weight, rel=WEIGHT_TOLERANCE)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_force_equals_stiffness_times_deflection(engine: str, axis: int) -> None:
    """Translate the club by 0.1 mm along a grip axis: F = -K_t delta per hand.

    The total is also read back from the engine's own generalised force on
    the club (MuJoCo ``qfrc_passive``, Drake's bushing force element,
    Pinocchio's free-flyer joint torque), so the engine force path is checked,
    not only the shared law.
    """
    module = _engine(engine)
    spec_bytes = _spec_bytes("driver")
    interface = GripInterface.from_spec(json.loads(spec_bytes))
    delta = 1e-4 * np.eye(3)[axis]
    probe = module.probe_bushing_forces(spec_bytes, _swing("driver"), delta)
    k_t = np.asarray(interface.bushing.translational_stiffness_n_m)
    expected = -probe.hand_rotation @ (k_t * delta)
    for side in ("L", "R"):
        np.testing.assert_allclose(
            probe.force_n[side], expected, rtol=STIFFNESS_TOLERANCE, atol=1e-6
        )
    np.testing.assert_allclose(
        probe.engine_total_force_n, 2.0 * expected, rtol=STIFFNESS_TOLERANCE, atol=1e-6
    )


def _reference(club: str) -> GripKineticsSeries:
    path = REFERENCE / f"opensim_{club}_series.npz"
    if not path.is_file():
        pytest.fail(f"missing committed OpenSim reference {path}")
    return GripKineticsSeries.load_npz(path)


_RUNS: dict[tuple[str, str], GripKineticsSeries] = {}


def _run(engine: str, club: str) -> GripKineticsSeries:
    key = (engine, club)
    if key not in _RUNS:
        module = _engine(engine)
        _RUNS[key] = module.simulate_grip_bushing(_spec_bytes(club), _swing(club))
    return _RUNS[key]


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("club", CLUBS)
@pytest.mark.parametrize("engine", ENGINES)
def test_full_swing_matches_the_opensim_reference(engine: str, club: str) -> None:
    errors = parity_errors(_run(engine, club), _reference(club))
    failing = {k: v.to_dict() for k, v in errors.items() if not v.passes()}
    assert not failing, failing
