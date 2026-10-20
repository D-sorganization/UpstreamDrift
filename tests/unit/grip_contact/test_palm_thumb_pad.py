"""HRG-H2 (#13190): compliant palm-thumb pad pair, diagnostic spec revision.

Pins the H2 acceptance contract of the physical ``trail_palm_lead_thumb`` pair
(trail palm sphere on lead thumb capsule) evaluated with the shared G2
``pair_contact`` law (Hunt-Crossley plus Stribeck) in ``mode="separated_v2"``
(re-scoped by HRG-GS-G3 #13301 and HRG-GS-G11 #13309):

* isolated 1 mm separation gives a force <= 1e-8 N; the normal force is never
  negative;
* friction-cone residual <= 1e-8 N and normalised action/reaction <= 1e-8;
* halving the timestep changes peak normal force and impulse by <= 1 %;
* preload sensitivity is reported;
* hand-drift gate: along and across drift < 2 mm before any force is reported;
* the pad ships as a new spec revision with its own ``spec_sha256``; the parent
  document is untouched.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import BushingState, GripInterface
from src.shared.python.grip_contact.contact_topology import GripStyle, contact_topology
from src.shared.python.grip_contact.hand_drift import HandDrift
from src.shared.python.grip_contact.palm_thumb_pad import (
    ACTION_REACTION_TOL,
    FRICTION_CONE_TOL_N,
    HAND_DRIFT_GATE_MM,
    PALM_THUMB_PAD_REVISION,
    PALM_THUMB_PAIR,
    SEPARATION_FORCE_TOL_N,
    TIMESTEP_SENSITIVITY_TOL,
    hand_drift_gate,
    palm_thumb_impact_response,
    palm_thumb_pad_force,
    palm_thumb_pad_spec_revision,
    palm_thumb_preload_sensitivity,
    timestep_halving_sensitivity,
)
from src.shared.python.grip_contact.topology_contact import build_topology_model
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.full_body_v2 import (
    derive_full_body_v2_spec,
    segment_boundary_locations,
    validate_full_body_v2_spec,
    upper_body_slice,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"

STYLES = [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]


@pytest.fixture(scope="module")
def v1_spec() -> dict:
    return json.loads(DRIVER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def interface(v1_spec: dict) -> GripInterface:
    return GripInterface.from_spec(v1_spec)


def _frame(side, position=None, velocity=None) -> BushingState:
    pos = np.asarray(side.position_m, float) if position is None else position
    vel = np.zeros(3) if velocity is None else velocity
    return BushingState(np.asarray(side.rotation, float), pos, vel, np.zeros(3))


def _model(interface: GripInterface, style: GripStyle = GripStyle.OVERLAP):
    return build_topology_model(
        interface, topology=contact_topology(GripSpec(style=style))
    )


def _no_drift(n: int = 3) -> HandDrift:
    z = np.zeros(n)
    return HandDrift(along_mm=z, across_mm=z, rotation_deg=z)


def _neutral(interface: GripInterface, style: GripStyle = GripStyle.OVERLAP):
    model = _model(interface, style)
    lead, trail = _frame(interface.left), _frame(interface.right)
    gate = hand_drift_gate(_no_drift())
    return model, lead, trail, palm_thumb_pad_force(model, lead, trail, gate=gate)


@pytest.mark.unit
def test_constants_pin_the_h2_contract() -> None:
    assert PALM_THUMB_PAIR == "trail_palm_lead_thumb"
    assert HAND_DRIFT_GATE_MM == 2.0
    assert SEPARATION_FORCE_TOL_N == 1.0e-8
    assert FRICTION_CONE_TOL_N == 1.0e-8
    assert ACTION_REACTION_TOL == 1.0e-8
    assert TIMESTEP_SENSITIVITY_TOL == 0.01


@pytest.mark.unit
@pytest.mark.parametrize("style", STYLES)
def test_neutral_pad_is_preloaded_and_satisfies_contracts(
    interface: GripInterface, style: GripStyle
) -> None:
    _, _, _, pad = _neutral(interface, style)
    assert pad.available, pad.reason
    assert pad.method == "physical"
    assert pad.normal_force_n > 0.0
    assert pad.penetration_m > 0.0
    assert pad.friction_cone_residual_n <= FRICTION_CONE_TOL_N
    assert pad.action_reaction_error <= ACTION_REACTION_TOL
    np.testing.assert_allclose(
        np.add(pad.force_on_lead_n, pad.force_on_trail_n), 0.0, atol=1e-12
    )


@pytest.mark.unit
def test_isolated_one_millimetre_separation_gives_no_force(
    interface: GripInterface,
) -> None:
    model, lead, trail, pad = _neutral(interface)
    n = np.asarray(pad.normal)  # from trail palm (A) toward lead thumb (B)
    # Back the trail hand off so the surfaces are exactly 1 mm apart, still
    # closing at speed: only the palm-thumb pair is evaluated (isolated).
    back = trail.position_m - (pad.penetration_m + 1.0e-3) * n
    sep = palm_thumb_pad_force(
        model,
        lead,
        _frame(interface.right, back, 0.5 * n + np.array([0.0, 0.3, 0.0])),
        gate=hand_drift_gate(_no_drift()),
    )
    assert sep.available, sep.reason
    assert sep.penetration_m == 0.0
    assert np.linalg.norm(sep.force_on_trail_n) <= SEPARATION_FORCE_TOL_N
    assert sep.normal_force_n <= SEPARATION_FORCE_TOL_N
    assert sep.tangential_force_n <= SEPARATION_FORCE_TOL_N


@pytest.mark.unit
@pytest.mark.parametrize("speed", [-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0])
@pytest.mark.parametrize("depth_mm", [0.0, 0.01, 0.1, 0.25, 0.5, 1.0])
def test_normal_force_never_negative_and_cone_holds(
    interface: GripInterface, speed: float, depth_mm: float
) -> None:
    """Fast separation (negative closing speed) never pulls the hands together."""
    model, lead, trail, pad = _neutral(interface)
    n = np.asarray(pad.normal)
    pos = trail.position_m + (depth_mm * 1e-3 - pad.penetration_m) * n
    vel = speed * n + np.array([0.2, -0.4, 0.1])
    out = palm_thumb_pad_force(
        model,
        lead,
        _frame(interface.right, pos, vel),
        gate=hand_drift_gate(_no_drift()),
    )
    assert out.available, out.reason
    assert out.normal_force_n >= 0.0
    # Force on the trail palm never points toward the lead thumb (no adhesion).
    assert float(np.dot(out.force_on_trail_n, n)) <= SEPARATION_FORCE_TOL_N
    assert out.friction_cone_residual_n <= FRICTION_CONE_TOL_N
    assert out.action_reaction_error <= ACTION_REACTION_TOL


@pytest.mark.unit
@pytest.mark.parametrize(
    ("along", "across", "passed"),
    [(0.0, 0.0, True), (1.99, 1.99, True), (2.0, 0.0, False), (0.0, 2.5, False)],
)
def test_hand_drift_gate_blocks_force_report(
    interface: GripInterface, along: float, across: float, passed: bool
) -> None:
    drift = HandDrift(
        along_mm=np.array([0.0, -along]),
        across_mm=np.array([0.0, across]),
        rotation_deg=np.zeros(2),
    )
    gate = hand_drift_gate(drift)
    assert gate.passed is passed
    assert gate.limit_mm == HAND_DRIFT_GATE_MM
    model = _model(interface)
    out = palm_thumb_pad_force(
        model, _frame(interface.left), _frame(interface.right), gate=gate
    )
    if passed:
        assert out.available
        assert gate.reason == ""
    else:
        assert not out.available
        assert "drift" in out.reason and gate.reason in out.reason
        # Unavailable is NaN plus a reason, never zero.
        assert math.isnan(out.normal_force_n)
        assert math.isnan(out.tangential_force_n)
        assert all(math.isnan(v) for v in out.force_on_trail_n)


@pytest.mark.unit
def test_hand_drift_gate_preconditions() -> None:
    with pytest.raises(TypeError):
        hand_drift_gate({"along_mm": 0.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hand_drift_gate(_no_drift(), limit_mm=0.0)
    nan = np.array([np.nan])
    gate = hand_drift_gate(HandDrift(nan, nan, nan))
    assert not gate.passed and "finite" in gate.reason


@pytest.mark.unit
def test_pad_force_type_preconditions(interface: GripInterface) -> None:
    model = _model(interface)
    with pytest.raises(TypeError):
        palm_thumb_pad_force(
            model, _frame(interface.left), _frame(interface.right), gate=None
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        palm_thumb_pad_force(
            object(),  # type: ignore[arg-type]
            _frame(interface.left),
            _frame(interface.right),
            gate=hand_drift_gate(_no_drift()),
        )


@pytest.mark.unit
def test_timestep_halving_changes_peak_and_impulse_by_at_most_one_percent(
    interface: GripInterface,
) -> None:
    model = _model(interface)
    kwargs = {
        "mass_kg": 0.5,
        "approach_speed_m_s": 0.2,
        "initial_gap_m": 2.0e-4,
        "duration_s": 0.01,
    }
    report = timestep_halving_sensitivity(
        model, _frame(interface.left), _frame(interface.right), dt_s=2.0e-5, **kwargs
    )
    coarse = report["coarse"]
    fine = report["fine"]
    assert fine.dt_s == pytest.approx(coarse.dt_s / 2.0)
    assert coarse.peak_normal_force_n > 0.0 and coarse.impulse_n_s > 0.0
    assert report["peak_relative_change"] <= TIMESTEP_SENSITIVITY_TOL
    assert report["impulse_relative_change"] <= TIMESTEP_SENSITIVITY_TOL
    # Momentum bookkeeping: the impulse cannot exceed the full reversal 2 m v0.
    assert fine.impulse_n_s <= 2.0 * 0.5 * 0.2 * (1.0 + 1e-6)


@pytest.mark.unit
def test_impact_response_preconditions(interface: GripInterface) -> None:
    model = _model(interface)
    lead, trail = _frame(interface.left), _frame(interface.right)
    good = {
        "mass_kg": 0.5,
        "approach_speed_m_s": 0.2,
        "initial_gap_m": 1e-4,
        "dt_s": 1e-5,
        "duration_s": 1e-3,
    }
    palm_thumb_impact_response(model, lead, trail, **good)
    for key, bad in [
        ("mass_kg", 0.0),
        ("approach_speed_m_s", -0.1),
        ("initial_gap_m", -1e-3),
        ("dt_s", 0.0),
        ("duration_s", float("nan")),
    ]:
        with pytest.raises(ValueError):
            palm_thumb_impact_response(model, lead, trail, **{**good, key: bad})


@pytest.mark.unit
def test_preload_sensitivity_is_reported(interface: GripInterface) -> None:
    model, lead, trail, pad = _neutral(interface)
    report = palm_thumb_preload_sensitivity(model, lead, trail)
    assert report["pair"] == PALM_THUMB_PAIR
    assert report["preload_penetration_m"] == pytest.approx(pad.penetration_m)
    assert report["normal_force_n"] == pytest.approx(pad.normal_force_n)
    # The shared law is linear in penetration at rest: dF/d(delta) is the
    # law stiffness and the elasticity (delta/F) dF/d(delta) is one.
    k = float(model.law.stiffness_n_m)
    assert report["stiffness_n_m"] == pytest.approx(k, rel=1e-6)
    assert report["elasticity"] == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError):
        palm_thumb_preload_sensitivity(model, lead, trail, relative_step=0.0)


@pytest.fixture(scope="module")
def separated_spec(v1_spec: dict) -> dict:
    return derive_full_body_v2_spec(
        v1_spec,
        segment_boundary_locations(v1_spec),
        provenance="HRG-H2 test separated v2",
        separate_hands=True,
    )


@pytest.mark.unit
def test_spec_revision_has_new_sha_and_leaves_parent_untouched(
    separated_spec: dict,
) -> None:
    before = copy.deepcopy(separated_spec)
    parent_sha = canonical_sha256(separated_spec)
    rev = palm_thumb_pad_spec_revision(separated_spec)
    assert separated_spec == before  # parent document untouched
    block = rev["palm_thumb_pad"]
    assert block["revision"] == PALM_THUMB_PAD_REVISION
    assert block["pair"] == PALM_THUMB_PAIR
    assert block["mode"] == "separated_v2"
    assert block["diagnostic_only"] is True
    assert block["native_collision"] is False
    assert block["hand_drift_gate_mm"] == HAND_DRIFT_GATE_MM
    assert block["parent_spec_sha256"] == parent_sha
    body = {k: v for k, v in rev.items() if k != "spec_sha256"}
    assert rev["spec_sha256"] == canonical_sha256(body)
    assert rev["spec_sha256"] != parent_sha
    # Deterministic, and the physics slice is unchanged by a diagnostic pad.
    assert palm_thumb_pad_spec_revision(separated_spec) == rev
    assert rev["upper_body_sha256"] == separated_spec["upper_body_sha256"]
    validate_full_body_v2_spec(rev, upper_body_slice(rev))
    with pytest.raises(ValueError, match="already"):
        palm_thumb_pad_spec_revision(rev)


@pytest.mark.unit
def test_spec_revision_refuses_welded_documents(v1_spec: dict) -> None:
    welded = derive_full_body_v2_spec(
        v1_spec,
        segment_boundary_locations(v1_spec),
        provenance="HRG-H2 test welded v2",
    )
    with pytest.raises(SchemaFeatureUnavailableError):
        palm_thumb_pad_spec_revision(welded)
    with pytest.raises(TypeError):
        palm_thumb_pad_spec_revision([])  # type: ignore[arg-type]
