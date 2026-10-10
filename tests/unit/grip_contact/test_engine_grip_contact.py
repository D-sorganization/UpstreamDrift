"""Contact-grip static hold, friction and slip in the engines (#11739 phase 3).

Acceptance (fixed before the runs, issue #11739):

* static hold: after settling, the pad forces on the club sum to the club
  weight within 1 %;
* friction keeps the club from sliding under its weight: the hand-to-club
  displacement along the grip axis stays under 0.1 mm over a 0.2 s hold, and a
  nearly frictionless grip (mu = 1e-4, below weight/squeeze) does slide, so the test can fail;
* the squeeze (total pad normal force) stays at its prescribed value.
"""

from __future__ import annotations

import json
import types
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    ClubDynamics,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SQUEEZE = 1100.0
WEIGHT_TOLERANCE = 0.01
SLIP_LIMIT_M = 1.0e-4
ENGINE_MODULES = {
    "mujoco": (
        "mujoco",
        "src.engines.physics_engines.mujoco.python.grip_contact_sim",
    ),
    "drake": (
        "pydrake",
        "src.engines.physics_engines.drake.python.grip_contact_sim",
    ),
    "pinocchio": (
        "pinocchio",
        "src.engines.physics_engines.pinocchio.python.grip_contact_sim",
    ),
}
#: hold duration per engine [s]; the implicit and error-controlled engines are
#: slower per simulated second and the contact ring settles in about 20 ms.
HOLD_DURATION_S = {"mujoco": 0.2, "drake": 0.05, "pinocchio": 0.05}
SLOW = {"drake", "pinocchio"}


def _engine(name: str) -> types.ModuleType:
    package, module = ENGINE_MODULES[name]
    try:
        real = import_module(package)
    except ImportError as exc:
        pytest.skip(f"{package} is not installed: {exc}")
    if not hasattr(real, "__file__"):
        pytest.skip(f"{package} in sys.modules is a test double")
    return import_module(module)


def _setup(friction: float | None = None):
    spec_bytes = (MODELS / "full_body_spec_anthro_driver.json").read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)
    pads = build_pad_model(interface, SQUEEZE)
    if friction is not None:
        pads = replace(
            pads,
            law=replace(pads.law, static_friction=friction, dynamic_friction=friction),
        )
    return spec_bytes, spec, names, swing, interface, pads


def _params():
    return [
        pytest.param(e, marks=pytest.mark.slow) if e in SLOW else e
        for e in sorted(ENGINE_MODULES)
    ]


def _hold(engine: str, friction: float | None = None):
    """Hold the hands still at the address pose; return the contact run."""
    module = _engine(engine)
    spec_bytes, spec, names, swing, interface, pads = _setup(friction)
    q0 = np.asarray(swing.q[0], float)
    duration = HOLD_DURATION_S[engine]
    if engine == "mujoco":
        sim = module.ClubInHands(spec_bytes, names, interface, pads)
        sim.calibrate(q0)
        return spec, module.hold_run(sim, q0, duration), pads
    run = module.simulate_grip_contact(
        spec_bytes, swing, pads, interface, t_end_s=duration, hold=True
    )
    return spec, run, pads


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("engine", _params())
def test_static_hold_supports_the_club_weight(engine: str) -> None:
    spec, run, _ = _hold(engine)
    gravity = np.asarray(spec["gravity_m_s2"], float)
    weight = ClubDynamics.from_spec(spec).mass_kg * float(np.linalg.norm(gravity))
    up = -gravity / np.linalg.norm(gravity)
    total = run.series.force_on_club_n["L"][-1] + run.series.force_on_club_n["R"][-1]
    assert float(total @ up) == pytest.approx(weight, rel=WEIGHT_TOLERANCE)
    assert float(np.linalg.norm(total - (total @ up) * up)) < 0.01 * weight
    for side in ("L", "R"):
        # the engine's squeeze is its own normal-plus-friction force sum
        assert run.squeeze_n[side][-1] == pytest.approx(SQUEEZE, rel=0.02)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("engine", _params())
def test_friction_keeps_the_club_from_sliding(engine: str) -> None:
    slips = {}
    for label, mu in (("grip", None), ("greased", 1e-4)):
        _, run, _ = _hold(engine, mu)
        slips[label] = max(float(np.abs(run.axial_slip_m[s]).max()) for s in "LR")
    assert slips["grip"] < SLIP_LIMIT_M
    assert slips["greased"] > 10.0 * SLIP_LIMIT_M


def test_calibrated_pads_carry_the_shared_law_force() -> None:
    module = _engine("mujoco")
    spec_bytes, _, names, swing, interface, pads = _setup()
    q0 = np.asarray(swing.q[0], float)
    sim = module.ClubInHands(spec_bytes, names, interface, pads)
    assert sim.calibrate(q0) < 1e-3
    sim.hold_pose(q0)
    force, depth = sim.pad_forces_and_penetrations()
    assert len(force) == 2 * pads.layout.pad_count
    for key, f in force.items():
        assert f == pytest.approx(pads.law.stiffness_n_m * depth[key], rel=1e-3)


def test_friction_time_and_hand_mode_reach_the_model() -> None:
    """Issue #11986 diagnostics: the knobs are validated and change the model."""
    module = _engine("mujoco")
    spec_bytes, _, names, swing, interface, pads = _setup()
    with pytest.raises(ValueError, match="hand_mode"):
        module.ClubInHands(spec_bytes, names, interface, pads, hand_mode="both")
    with pytest.raises(ValueError, match="friction_time_s"):
        module.ClubInHands(spec_bytes, names, interface, pads, friction_time_s=0.0)
    q0 = np.asarray(swing.q[0], float)
    sim = module.ClubInHands(spec_bytes, names, interface, pads, friction_time_s=2e-3)
    assert sim.calibrate(q0) < 1e-3
    assert float(sim.model.pair_solreffriction[0][0]) == pytest.approx(2e-3)
    lead = module.ClubInHands(spec_bytes, names, interface, pads, hand_mode="lead_only")
    assert lead.calibrate(q0) < 1e-3
    assert len(lead.model.pair_geom1) == pads.layout.pad_count


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mode", ["trail_follows_club", "lead_only"])
def test_diagnostic_hand_modes_support_the_club_weight(mode: str) -> None:
    module = _engine("mujoco")
    spec_bytes, spec, names, swing, interface, pads = _setup()
    q0 = np.asarray(swing.q[0], float)
    sim = module.ClubInHands(spec_bytes, names, interface, pads, hand_mode=mode)
    sim.calibrate(q0)
    run = module.hold_run(sim, q0, 0.1)
    gravity = np.asarray(spec["gravity_m_s2"], float)
    weight = ClubDynamics.from_spec(spec).mass_kg * float(np.linalg.norm(gravity))
    up = -gravity / np.linalg.norm(gravity)
    total = run.series.force_on_club_n["L"][-1] + run.series.force_on_club_n["R"][-1]
    assert float(total @ up) == pytest.approx(weight, rel=WEIGHT_TOLERANCE)


def test_trail_shift_moves_only_the_trail_hand() -> None:
    """Issue #11986: a constant trail shift is applied in the trail grip frame."""
    module = _engine("mujoco")
    spec_bytes, _, names, swing, interface, pads = _setup()
    q0 = np.asarray(swing.q[0], float)
    sim = module.ClubInHands(spec_bytes, names, interface, pads)
    base = sim.hold_pose(q0)
    sim.trail_shift_m = np.array([1e-3, 0.0, 0.0])
    moved = sim.hold_pose(q0)
    axis = base["R"].rotation[:, 0]
    assert moved["R"].position_m - base["R"].position_m == pytest.approx(1e-3 * axis)
    assert moved["L"].position_m == pytest.approx(base["L"].position_m)
    with pytest.raises(ValueError, match="trail_shift_m"):
        module.simulate_grip_contact(
            spec_bytes, swing, pads, interface, t_end_s=0.0, trail_shift_m=(0.0, 0.0)
        )
