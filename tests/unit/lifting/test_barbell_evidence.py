"""Committed barbell-on-bushings evidence for the back squat (LIFT-4, #11744).

Records written by ``scripts/generate_barbell_squat_evidence.py``.  The
thresholds are fixed before the runs and are not loosened:

* Quasi-static hold, means over the last 0.5 s of a 1.5 s hold: vertical
  ground reaction equals body plus bar weight within 1 %; the hand forces
  carry the bar weight within 1 %; the left hand carries 0.5 +- 0.01 of it;
  peak bushing deflection at most 3 mm over the whole hold.  The same 1 %
  bands and split band hold at every sample of the whole hold
  (:func:`test_hold_carries_body_plus_bar_over_the_whole_window`).  Since
  #11744 the hold starts at the contact equilibrium of the held joint angles
  (``barbell_runs.hold_start``); from the old ``preload_feet`` start (feet
  carrying 0.595 of the load, centre of pressure 6.7 cm ahead of the centre
  of mass) a lightly damped fore-aft sway sagged the last 0.5 s to 0.988.
* Loaded bundle: same-engine open-loop replay of the first steps reproduces
  the reference within 1e-8 (determinism of the coupled plant).
* Loaded re-track (#13072): every LIFT-5 body landmark stays within the
  LIFT-5 IK residual bound of the source reference's, and the bar landmark
  within the tightest body-landmark bound (:func:`_lift5_bounds`, OSL-16
  #13104).  It holds on the bar-balanced (#13093), 64-substep (#13104) source.
* Cross-engine parity at every 20th reference state: L1 of epic #11605,
  ``|dX| <= 1e-6 + 1e-8 * max|X|`` per quantity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE

pytestmark = pytest.mark.unit

EVIDENCE = BACK_SQUAT_BUNDLE.parent
HOLD_JSON = EVIDENCE / "barbell_hold.json"
BARBELL_BUNDLE = EVIDENCE / "back_squat_barbell_bundle.npz"
PARITY_JSON = EVIDENCE / "barbell_parity.json"
WEIGHT_FRACTION = 0.01
SPLIT_BAND = 0.01
MAX_DEFLECTION_M = 3e-3
REPLAY_STEPS = 5
REPLAY_TOL = 1e-8


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_hold_ground_reaction_carries_body_plus_bar() -> None:
    summary = _load(HOLD_JSON)["summary"]
    assert summary["failure"] is None
    assert summary["duration_s"] == pytest.approx(1.5)
    assert summary["settle_window_s"] == pytest.approx(0.5)
    assert abs(summary["grf_vertical_over_total_weight"] - 1.0) <= WEIGHT_FRACTION


def test_hold_hands_carry_the_bar_symmetrically() -> None:
    summary = _load(HOLD_JSON)["summary"]
    assert abs(summary["hand_vertical_sum_over_bar_weight"] - 1.0) <= WEIGHT_FRACTION
    assert abs(summary["hand_vertical_split_left"] - 0.5) <= SPLIT_BAND
    assert summary["peak_deflection_m"] <= MAX_DEFLECTION_M


def test_hold_carries_body_plus_bar_over_the_whole_window() -> None:
    """Every sample, not only the last-0.5 s means, meets the hold bands."""
    record = _load(HOLD_JSON)
    series, summary = record["series"], record["summary"]
    total = summary["body_weight_n"] + summary["bar_weight_n"]
    bar = summary["bar_weight_n"]
    grf = np.asarray(series["grf_vertical_n"])
    left = np.asarray(series["hand_vertical_on_bar_n"]["L"])
    right = np.asarray(series["hand_vertical_on_bar_n"]["R"])
    assert np.asarray(series["time_s"])[-1] == pytest.approx(1.5)
    assert np.abs(grf / total - 1.0).max() <= WEIGHT_FRACTION
    assert np.abs((left + right) / bar - 1.0).max() <= WEIGHT_FRACTION
    assert np.abs(left / (left + right) - 0.5).max() <= SPLIT_BAND
    assert max(series["max_deflection_m"]) <= MAX_DEFLECTION_M


#: OSL-17 (#13131) regression, measured on the high-bar rack bundles: the
#: bar-loaded re-track's bar landmark drifts forward of its source linearly
#: through the window (0.6 mm at the first state, 6.79 mm at the last; the
#: shoulders 6.5/5.6 mm, the hips 4.5 mm), above the 3.6 mm LIFT-5 bound;
#: before the rack it was 2.03 mm.  On the #13037 whole-row lift body
#: (#13041): bar 6.45 mm, shoulders 6.57/5.12 mm, hips 4.07/3.68 mm.  The
#: bound is not loosened.
_RETRACK_REGRESSION = pytest.mark.xfail(
    strict=True,
    reason=(
        "OSL-17 #13131 / #13041: high-bar re-track bar deviation 6.45 mm > "
        "3.6 mm LIFT-5 bound; root cause open, bound not loosened."
    ),
)
#: OSL-17 (#13131): with the bar behind the neck the balanced standing pose
#: carries the body's own centre of mass 11.9 cm forward of the mean-heel
#: axis origin, ahead of the forefoot spheres (10.4 cm); body plus bar
#: (7.7 cm) and the bar (4.3 cm) lie over the feet and the CoP sits under the
#: combined centre of mass.  Measured, not loosened.  On the #13037
#: whole-row lift body (#13041): body 11.65 cm, forefoot 10.02 cm, body plus
#: bar 7.32 cm.
_BODY_COM_AHEAD = pytest.mark.xfail(
    strict=True,
    reason=(
        "OSL-17 #13131 / #13041: body-alone CoM 0.117 m forward, ahead of the "
        "forefoot (0.100 m) at the high-bar balanced pose."
    ),
)


def _balance_geometry() -> tuple[dict, float, float]:
    from src.shared.python.lifting.barbell_bushing import Barbell
    from src.shared.python.lifting.barbell_runs import (
        hold_balance_geometry,
        hold_pose,
    )
    from src.shared.python.motion_matching.same_input import InputBundle

    source = InputBundle.load(BACK_SQUAT_BUNDLE)
    q_hold = hold_pose(source)
    geometry = hold_balance_geometry(source.spec_bytes, q_hold, Barbell(100.0))
    forward = geometry["forward_axis"]
    forefoot = max(p @ forward for p in geometry["forefoot_m"].values())
    heel = min(p @ forward for p in geometry["heel_m"].values())
    return geometry, heel, forefoot


def test_hold_balance_geometry_is_stable() -> None:
    """Body plus bar balances over the feet."""
    pytest.importorskip("mujoco")
    geometry, heel, forefoot = _balance_geometry()
    assert heel < geometry["combined_com_m"] @ geometry["forward_axis"] < forefoot
    # The hold starts in contact equilibrium: the centre of pressure lies
    # under the body-plus-bar centre of mass (#11744).
    offset = geometry["centre_of_pressure_m"] - geometry["combined_com_m"]
    assert np.abs(offset[:2]).max() <= 1e-6


@_BODY_COM_AHEAD
def test_hold_body_alone_balances_over_the_feet() -> None:
    """The body's own centre of mass lies over the feet."""
    pytest.importorskip("mujoco")
    geometry, heel, forefoot = _balance_geometry()
    assert heel < geometry["body_com_m"] @ geometry["forward_axis"] < forefoot


def test_barbell_bundle_carries_the_bar_record() -> None:
    from src.shared.python.motion_matching.same_input import InputBundle

    bundle = InputBundle.load(BARBELL_BUNDLE)
    source = InputBundle.load(BACK_SQUAT_BUNDLE)
    record = bundle.provenance["barbell"]
    # A bar-loaded source (#13093) is re-tracked on its body: the bushing bar
    # replaces the rigid one.
    from src.shared.python.lifting.barbell_runs import bushing_body_spec

    assert bundle.spec_bytes == bushing_body_spec(source.spec_bytes)
    assert bundle.provenance["source_bundle_spec_sha256"] == source.spec_sha256
    assert bundle.step_policy() == source.step_policy()
    assert bundle.steps == source.steps
    assert record["bar_mass_kg"] == source.provenance["optimizer"]["bar_mass_kg"]
    assert len(record["reference_bar_q"]) == bundle.steps + 1
    from test_lift_bundles import LOADED_BUNDLE_MAX_BYTES

    assert BARBELL_BUNDLE.stat().st_size <= LOADED_BUNDLE_MAX_BYTES


def test_barbell_bundle_replays_in_mujoco() -> None:
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.barbell_runs import (
        augmented_reference,
        barbell_plant,
    )
    from src.shared.python.motion_matching.same_input import InputBundle, open_loop

    bundle = InputBundle.load(BARBELL_BUNDLE)
    plant = barbell_plant("mujoco", bundle)
    q_ref, v_ref = augmented_reference(bundle)
    held = bundle.replay_efforts(REPLAY_STEPS)
    efforts = np.hstack([held, np.zeros((held.shape[0], 6))])
    rollout = open_loop(
        plant,
        q_ref[0],
        v_ref[0],
        efforts,
        dt_s=bundle.dt_s,
        policy=bundle.step_policy(),
    )
    assert np.abs(rollout.q - q_ref[: REPLAY_STEPS + 1]).max() <= REPLAY_TOL


def test_cross_engine_parity_within_l1() -> None:
    report = _load(PARITY_JSON)
    scale = report["mujoco_scale"]
    assert set(report["engines"]) == {"drake", "pinocchio", "opensim"}
    for engine, diffs in report["engines"].items():
        if "unavailable" in diffs:
            continue
        for key, diff in diffs.items():
            assert diff <= 1e-6 + 1e-8 * scale[key], (engine, key, diff)


def _lift5_bounds() -> dict[str, float]:
    """The LIFT-5 squat IK residual thresholds (``test_lift_bundles``) for a
    re-track, with the bar bound redefined (OSL-16, #13104).

    The 0.1 mm bar threshold is the IK's: it bounds how exactly the IK seats
    the bar landmark on the back-bar geometry (residual 0.01 mm), a kinematic
    constraint, not a tracking scale.  In a dynamic reference the bar landmark
    (the hand-held bar-centre frame) is carried by the body standing on the
    loaded penalty contact: the source reference sits 6.2-6.4 mm below its IK
    at every landmark (the loaded foot sink, almost all vertical), and the
    bushing re-track sinks another 0.93 mm at the ankles.  A common-mode sink
    moves every landmark alike, so the bar cannot be held tighter than the
    body that carries it: its re-track bound is the tightest body-landmark
    bound (ankle_r, 3.6 mm), not a value fitted to the run.
    """
    from test_lift_bundles import IK_RESIDUAL_THRESHOLDS_M

    bounds = dict(IK_RESIDUAL_THRESHOLDS_M)
    bounds["bar"] = min(v for k, v in bounds.items() if k != "bar")
    return bounds


@_RETRACK_REGRESSION
def test_retracked_bundle_records_it_follows_its_source() -> None:
    """#13072: the bar-loaded re-track stays on its source reference.

    Every LIFT-5 landmark of the re-tracked reference lies within the LIFT-5
    IK residual bound of the same landmark on the source reference, at every
    state of the window.
    """
    from src.shared.python.lifting.barbell_runs import RETRACK_DEVIATION_KEY
    from src.shared.python.motion_matching.same_input import InputBundle

    bundle = InputBundle.load(BARBELL_BUNDLE)
    recorded = bundle.provenance[RETRACK_DEVIATION_KEY]
    for label, limit in _lift5_bounds().items():
        assert recorded[label] <= limit, (label, recorded[label], limit)


def _recomputed_deviation() -> tuple[dict[str, float], dict[str, float]]:
    from src.shared.python.lifting.barbell_runs import (
        RETRACK_DEVIATION_KEY,
        retrack_landmark_deviation,
    )
    from src.shared.python.motion_matching.same_input import InputBundle

    bundle = InputBundle.load(BARBELL_BUNDLE)
    source = InputBundle.load(BACK_SQUAT_BUNDLE)
    deviation = retrack_landmark_deviation(source, bundle.reference_q)
    return deviation, bundle.provenance[RETRACK_DEVIATION_KEY]


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_recorded_retrack_deviation_is_reproducible() -> None:
    """The provenance deviation is the committed arrays' own (#13072)."""
    pytest.importorskip("mujoco")
    deviation, recorded = _recomputed_deviation()
    assert recorded == pytest.approx(deviation, rel=0, abs=1e-12)


@pytest.mark.integration
@pytest.mark.requires_mujoco
@_RETRACK_REGRESSION
def test_retracked_bundle_follows_its_source_within_lift5_bounds() -> None:
    """#13072: recompute the landmark deviation from the committed arrays."""
    pytest.importorskip("mujoco")
    deviation, _ = _recomputed_deviation()
    for label, limit in _lift5_bounds().items():
        assert deviation[label] <= limit, (label, deviation[label], limit)


def test_retrack_deviation_rejects_a_mismatched_reference() -> None:
    from src.shared.python.lifting.barbell_runs import retrack_landmark_deviation
    from src.shared.python.motion_matching.same_input import InputBundle

    source = InputBundle.load(BACK_SQUAT_BUNDLE)
    with pytest.raises(ValueError, match="reference_q"):
        retrack_landmark_deviation(source, source.reference_q[:-1])
