"""GCV-9 (#11715): Simscape per-hand grip wrench against an R2025b fixture.

The fixture was exported by ``scripts/matlab/export_grip_wrench_fixture.m``
(``source="run102"``) on MATLAB R2025b (receipt alongside): the qualified
run-102 replay, an 0.85 s window of the slow early swing.  It is simulation
output, not measured data. The test checks that the shared two-contact
reduction (:func:`analyze_grip`) applied to the per-hand columns reproduces
the model's own logged net hand force and equivalent midpoint couple, and
that the run is physically plausible (the model-workspace coefficients
diverge to ~1e7 N, #11778, and must never back this fixture).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.simscape.force_channels import load_simscape_force_series
from src.shared.python.biomechanics.grip_wrench import HandWrench, analyze_grip

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[4] / "tests/fixtures/simscape"
CSV_PATH = FIXTURES / "simscape_grip_wrench_fixture.csv"
RECEIPT_PATH = FIXTURES / "simscape_grip_wrench_receipt.json"

#: Relative bound on the reconstruction residual, scaled by the largest
#: logged magnitude. The CSV stores ~15 significant digits, so a correct
#: reduction agrees to round-off; a sign or reference-point error is O(1).
REL_TOL = 1e-9


def _columns() -> dict[str, np.ndarray]:
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _vec(cols: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.stack([cols[f"{prefix}{i}"] for i in "123"], axis=1)


def _receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_receipt_is_r2025b_and_matches_fixture() -> None:
    receipt = _receipt()
    assert receipt["matlab_release"] == "2025b"
    assert receipt["issue"] == "#11715"
    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert digest == receipt["fixture_sha256"]


#: Plausibility ceilings for a human swing (#11778): grip loads stay in the
#: low kN range and the clubhead below ~70 m/s even at impact.
MAX_HAND_FORCE_N = 5_000.0
MAX_CLUBHEAD_SPEED_MPS = 100.0


def test_fixture_is_a_plausible_swing_not_the_divergent_default() -> None:
    receipt = _receipt()
    assert receipt["coefficient_source"] == "run102"
    assert receipt["hand_force_max_n"] < MAX_HAND_FORCE_N
    assert receipt["clubhead_speed_max_mps"] < MAX_CLUBHEAD_SPEED_MPS
    c = _columns()
    for prefix in ("LWLogs_LHonClubFGlobal_", "RWLogs_RHonClubFGlobal_"):
        assert np.linalg.norm(_vec(c, prefix), axis=1).max() < MAX_HAND_FORCE_N


def test_analyze_grip_reproduces_logged_net_force_and_couple() -> None:
    c = _columns()
    r_l, r_r = _vec(c, "LWLogs_LHGlobalPosition_"), _vec(c, "RWLogs_RHGlobalPosition_")
    f_l, f_r = _vec(c, "LWLogs_LHonClubFGlobal_"), _vec(c, "RWLogs_RHonClubFGlobal_")
    t_l, t_r = _vec(c, "LWLogs_LHonClubTGlobal_"), _vec(c, "RWLogs_RHonClubTGlobal_")
    logged_net = _vec(c, "CalculatedSignalsLogs_TotalHandForceGlobal_")
    logged_m = _vec(c, "MomentandCoupleLogs_EquivalentMidpointCoupleGlobal_")
    logged_mid = _vec(c, "MidpointCalcsLogs_MPGlobalPosition_")
    assert len(c["time"]) >= 2
    assert np.abs(logged_m).max() > 0.0, "fixture must carry non-zero loads"
    for i in range(len(c["time"])):
        g = analyze_grip(
            HandWrench("L", r_l[i], f_l[i], t_l[i]),
            HandWrench("R", r_r[i], f_r[i], t_r[i]),
            split_method="logged",
        )
        assert g.net_force_n is not None and g.couple_at_midpoint_nm is not None
        assert g.midpoint_m is not None
        # Per-row scale: the largest load magnitude entering this row.
        scale = max(1.0, *(np.abs(a[i]).max() for a in (f_l, f_r, t_l, t_r)))
        np.testing.assert_allclose(g.midpoint_m, logged_mid[i], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            g.net_force_n, logged_net[i], rtol=0, atol=REL_TOL * scale
        )
        np.testing.assert_allclose(
            g.couple_at_midpoint_nm, logged_m[i], rtol=0, atol=REL_TOL * scale
        )


def test_loader_exposes_both_hands_from_fixture() -> None:
    series, missing = load_simscape_force_series(CSV_PATH)
    assert not [m for m in missing if m.startswith("grip:hand_")]
    for frame in series:
        by = {w.label: w for w in frame.wrenches}
        left, right = by["grip:hand_left"], by["grip:hand_right"]
        total = by["grip:total_hand"]
        assert left.force_n is not None and right.force_n is not None
        assert total.force_n is not None
        summed = np.add(left.force_n, right.force_n)
        np.testing.assert_allclose(
            summed, total.force_n, rtol=0, atol=REL_TOL * max(1.0, np.abs(summed).max())
        )
