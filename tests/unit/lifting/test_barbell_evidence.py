"""Committed barbell-on-bushings evidence for the back squat (LIFT-4, #11744).

Records written by ``scripts/generate_barbell_squat_evidence.py``.  The
thresholds are fixed before the runs and are not loosened:

* Quasi-static hold, means over the last 0.5 s of a 1.5 s hold: vertical
  ground reaction equals body plus bar weight within 1 %; the hand forces
  carry the bar weight within 1 %; the left hand carries 0.5 +- 0.01 of it;
  peak bushing deflection at most 3 mm over the whole hold.  On the
  64-substep loaded policy (OSL-16, #13104) the split and the deflection
  pass, but the 1 % bands still FAIL narrowly (ground 0.988, hands 0.986 of
  their loads): the hold carries body plus bar at 0.75-1.0 s (1.000) and then
  sags.  They are kept as strict expected failures with the thresholds
  unchanged (:data:`HOLD_GAP`).
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


HOLD_GAP = (
    "#13093 gap: at 64 RK4 substeps (#13104) the hold carries body plus bar "
    "(1.000 at 0.75-1.0 s; was 0.455 overall at 8 substeps), then sags to "
    "0.982 by 1.5 s under a steady -102 N net horizontal ground force, so "
    "the last-0.5 s means miss the 1 % bands (ground 0.988, hands 0.986)"
)


@pytest.mark.xfail(strict=True, reason=HOLD_GAP)
def test_hold_ground_reaction_carries_body_plus_bar() -> None:
    summary = _load(HOLD_JSON)["summary"]
    assert summary["failure"] is None
    assert summary["duration_s"] == pytest.approx(1.5)
    assert summary["settle_window_s"] == pytest.approx(0.5)
    assert abs(summary["grf_vertical_over_total_weight"] - 1.0) <= WEIGHT_FRACTION


@pytest.mark.xfail(strict=True, reason=HOLD_GAP)
def test_hold_hands_carry_the_bar_symmetrically() -> None:
    summary = _load(HOLD_JSON)["summary"]
    assert abs(summary["hand_vertical_sum_over_bar_weight"] - 1.0) <= WEIGHT_FRACTION
    assert abs(summary["hand_vertical_split_left"] - 0.5) <= SPLIT_BAND
    assert summary["peak_deflection_m"] <= MAX_DEFLECTION_M


def test_hold_starts_supported_then_fails() -> None:
    """The start carries body plus bar; the last 0.5 s miss the 1 % band."""
    record = _load(HOLD_JSON)
    series, summary = record["series"], record["summary"]
    time = np.asarray(series["time_s"])
    grf = np.asarray(series["grf_vertical_n"])
    total = summary["body_weight_n"] + summary["bar_weight_n"]
    early = (time >= 0.15) & (time <= 0.3)
    assert abs(grf[early].mean() / total - 1.0) <= 0.05
    assert summary["failure"] is None
    assert abs(summary["grf_vertical_over_total_weight"] - 1.0) > WEIGHT_FRACTION


def test_hold_balance_geometry_is_stable() -> None:
    """Body alone and body plus bar both balance over the feet."""
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.barbell_bushing import Barbell
    from src.shared.python.lifting.barbell_runs import hold_balance_geometry
    from src.shared.python.motion_matching.same_input import InputBundle

    source = InputBundle.load(BACK_SQUAT_BUNDLE)
    q_hold = np.asarray(source.provenance["ik"]["q_standing"], dtype=float)
    geometry = hold_balance_geometry(source.spec_bytes, q_hold, Barbell(100.0))
    forward = geometry["forward_axis"]
    forefoot = max(p @ forward for p in geometry["forefoot_m"].values())
    heel = min(p @ forward for p in geometry["heel_m"].values())
    assert heel < geometry["body_com_m"] @ forward < forefoot
    assert heel < geometry["combined_com_m"] @ forward < forefoot


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
