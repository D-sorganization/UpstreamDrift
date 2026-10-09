"""OpenSim contact grip: ElasticFoundationForce on a closed grip mesh (#11739).

Skips when ``opensim`` is not importable.  The quick tests run in the default
unit lane; the full-model holds are slow-marked (they build the 24-pad,
two-mesh model) and run on the simulation host.  Acceptance (fixed before the
runs, issue #11739): a pad on a fixed grip supports the expected load; a
static hold of the club sums to the club weight within 1 % after settling;
friction keeps the club from sliding under its weight (a nearly frictionless
grip does slide, so the test can fail); the per-hand wrenches balance the
gravity moment about the centre of mass.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("opensim")

from src.engines.physics_engines.opensim.python.grip_contact_osim_sim import (  # noqa: E402
    ContactGripSimulator,
    calibrated_foundation,
    pad_normal_force_n,
    single_pad_model,
    write_calibration_mesh,
)
from src.shared.python.grip_contact import (  # noqa: E402
    ClubDynamics,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.elastic_foundation import (  # noqa: E402
    elastic_foundation_parameters,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model  # noqa: E402
from src.shared.python.grip_contact.static_balance import static_balance  # noqa: E402

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SQUEEZE = 1100.0
WEIGHT_TOLERANCE = 0.01
SLIP_LIMIT_M = 1.0e-4
HOLD_S = 0.2
SAMPLE_DT_S = 0.002
G = 9.80665


@pytest.fixture(scope="module")
def pads():
    spec = json.loads((MODELS / "full_body_spec_anthro_driver.json").read_text())
    return build_pad_model(GripInterface.from_spec(spec), SQUEEZE)


@pytest.fixture(scope="module")
def foundation(pads, tmp_path_factory):
    directory = tmp_path_factory.mktemp("calibration")
    return calibrated_foundation(elastic_foundation_parameters(pads), directory)


@pytest.fixture(scope="module")
def mesh(foundation, tmp_path_factory):
    return write_calibration_mesh(foundation, tmp_path_factory.mktemp("mesh"))


def test_single_pad_carries_the_matched_squeeze_at_the_preload(foundation, mesh):
    f = pad_normal_force_n(foundation, mesh, foundation.preload_penetration_m)
    assert f == pytest.approx(foundation.force_per_pad_n, rel=1e-3)


def test_tangent_stiffness_is_close_to_the_shared_pad_stiffness(foundation, mesh):
    """Winkler force is quadratic in depth, so the match is approximate: measure it."""
    d0, h = foundation.preload_penetration_m, 5.0e-5
    tangent = (
        pad_normal_force_n(foundation, mesh, d0 + h)
        - pad_normal_force_n(foundation, mesh, d0 - h)
    ) / (2.0 * h)
    assert tangent == pytest.approx(foundation.pad_stiffness_n_m, rel=0.10)


def test_pad_force_vanishes_without_contact(foundation, mesh):
    assert pad_normal_force_n(foundation, mesh, -1.0e-3) == 0.0


def test_pad_on_a_fixed_grip_supports_the_expected_load(foundation, mesh):
    """Regression of the feasibility probe: a pad at the equilibrium depth stays."""
    import opensim as osim

    weight = foundation.force_per_pad_n
    model, force, coord = single_pad_model(foundation, mesh, weight / G)
    state = model.initSystem()
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-6)
    manager.initialize(state)
    forces, depths = [], []
    for t in np.linspace(0.01, 0.15, 15):
        state = manager.integrate(float(t))
        model.realizeDynamics(state)
        forces.append(force.getRecordValues(state).get(1))
        depths.append(coord.getValue(state))
    assert np.mean(forces) == pytest.approx(weight, rel=WEIGHT_TOLERANCE)
    layout = foundation.layout
    start = (
        layout.grip_radius_m + layout.pad_radius_m - foundation.preload_penetration_m
    )
    assert abs(np.mean(depths) - start) < 0.05 * foundation.preload_penetration_m


# ------------------------------------------------------------- full-model holds
def _hold(friction: float | None = None, pads=None):
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
    model = build_pad_model(interface, SQUEEZE)
    if friction is not None:
        model = replace(
            model,
            law=replace(model.law, static_friction=friction, dynamic_friction=friction),
        )
    n = int(round(HOLD_S / SAMPLE_DT_S)) + 1
    q = np.tile(np.asarray(swing.q[0], float), (n, 1))
    sim = ContactGripSimulator(
        spec_bytes, names, np.arange(n) * SAMPLE_DT_S, q, model, interface
    )
    return spec, interface, sim.run(accuracy=1e-5).to_contact_run()


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_static_hold_supports_the_club_weight():
    spec, interface, run = _hold()
    gravity = np.asarray(spec["gravity_m_s2"], float)
    club = ClubDynamics.from_spec(spec)
    weight = club.mass_kg * float(np.linalg.norm(gravity))
    up = -gravity / np.linalg.norm(gravity)
    total = run.series.force_on_club_n["L"][-1] + run.series.force_on_club_n["R"][-1]
    assert float(total @ up) == pytest.approx(weight, rel=WEIGHT_TOLERANCE)
    assert float(np.linalg.norm(total - (total @ up) * up)) < 0.01 * weight
    bal = static_balance(run.series, interface, club, gravity)
    assert float(np.linalg.norm(bal.moment_residual_nm[-1])) < 0.01 * float(
        bal.gravity_moment_nm[-1]
    )
    for side in "LR":
        assert run.squeeze_n[side][-1] == pytest.approx(SQUEEZE, rel=0.03)


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_friction_keeps_the_club_from_sliding():
    slips = {}
    for label, mu in (("grip", None), ("greased", 1e-4)):
        _, _, run = _hold(mu)
        slips[label] = max(float(np.abs(run.axial_slip_m[s]).max()) for s in "LR")
    assert slips["grip"] < SLIP_LIMIT_M
    assert slips["greased"] > 10.0 * SLIP_LIMIT_M
