"""Simscape rigid-body stance vs. the GCV-2 stance-parity criteria (#11709).

``GS3DX_FullBodyContact`` has no affordable held posture: passive it topples
(feet slip up to 1.50 m over 2 s), a joint-space PD hold grows an unbounded
momentum residual, and locking every joint but the pelvis root costs ~140 s
of wall clock per simulated ms (a 2 s stance would take ~78 h) because an
algebraic loop through the leg servo and every upper-body torque input is
paid on each model evaluation regardless of the solver (see
``docs/.../exploratory_gs3dx/docs/GROUND_CONTACT.md``, "Rigid-Body Stance
Check").

Director-approved option (b) (2026-10-10): a rigid body carrying the
golfer's total mass and centre of mass stands on the same six sole-sphere
contacts, same world placement, same Spatial Contact Force law and
parameter values (by workspace variable name), as ``GS3DX_FullBodyContact``.
This module applies the UNCHANGED GCV-2 tolerances
(``tests/integration/cross_engine/test_ground_reaction_parity.py``) to the
settled state of that run:

* net vertical GRF equals body weight (``WEIGHT_RTOL``);
* the two feet carry the load symmetrically (``FOOT_SPLIT_RTOL`` of half
  weight each);
* the net CoP lies on the ground plane, between the two feet
  (``COP_X_ATOL_M`` / ``COP_Y_ATOL_M`` of the foot midpoint).

GCV-2's own absolute-origin version of the CoP check assumes its synthetic
stance is centred at world ``(x, y) = (0, 0)``; GS3DX's world frame has no
such convention, so this module re-centres on the two feet's own midpoint
before applying the SAME tolerance values -- never different numbers, never
a different criterion. GCV-2's fourth check ("left foot on world +Y") is an
engine coordinate-mapping convention (e.g. OpenSim's Y-up axis), not one of
the three physical criteria this issue asks for, so it is not replicated
here.

The fixture is a simulation output (MATLAB R2025b, ``gs3dx_rigid_stance``),
not measured data; it is exported by
``scripts/matlab/export_gs3dx_rigid_stance_fixture.m`` into the same column
schema ``export_gs3dx_grf_fixture.m`` uses, so it is read with the same
loader (:func:`load_simscape_force_series`) as every other GS3DX fixture.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.simscape.force_channels import load_simscape_force_series

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/simscape"
CSV_PATH = FIXTURES / "simscape_gs3dx_rigid_stance_fixture.csv"
RECEIPT_PATH = FIXTURES / "simscape_gs3dx_rigid_stance_receipt.json"


def _gcv2_tolerances():
    """Load the GCV-2 parity module by path: reuse its constants, never redefine them."""
    path = ROOT / "tests/integration/cross_engine/test_ground_reaction_parity.py"
    spec = importlib.util.spec_from_file_location("_gcv2_rigid_stance_tolerances", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GCV2 = _gcv2_tolerances()
WEIGHT_RTOL = _GCV2.WEIGHT_RTOL
FOOT_SPLIT_RTOL = _GCV2.FOOT_SPLIT_RTOL
COP_Y_ATOL_M = _GCV2.COP_Y_ATOL_M
COP_X_ATOL_M = _GCV2.COP_X_ATOL_M


def _receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def _settled_wrenches() -> dict:
    """Wrenches of the last logged frame: the settled state of the 2 s run."""
    series, missing = load_simscape_force_series(CSV_PATH)
    assert not [m for m in missing if m.startswith("contact:")]
    frame = series[-1]
    return {w.label: w for w in frame.wrenches}


def test_receipt_is_r2025b_and_matches_fixture() -> None:
    receipt = _receipt()
    assert receipt["matlab_release"] == "2025b"
    assert receipt["issue"] == "#11709"
    assert receipt["check"] == "rigid_body_stance"
    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert digest == receipt["fixture_sha256"]


def test_net_grf_equals_body_weight() -> None:
    receipt = _receipt()
    w = _settled_wrenches()
    net = np.asarray(w["contact:grf_net"].force_n)
    assert net[2] == pytest.approx(receipt["weight_n"], rel=WEIGHT_RTOL)


def test_feet_share_the_load_symmetrically() -> None:
    receipt = _receipt()
    w = _settled_wrenches()
    for side in ("left", "right"):
        fz = w[f"contact:grf_{side}"].force_n[2]
        assert fz == pytest.approx(receipt["weight_n"] / 2, rel=FOOT_SPLIT_RTOL), side


def test_net_cop_is_on_the_ground_between_the_feet() -> None:
    receipt = _receipt()
    w = _settled_wrenches()
    left_xy = np.asarray(w["contact:grf_left"].point_m[:2])
    right_xy = np.asarray(w["contact:grf_right"].point_m[:2])
    mid_xy = (left_xy + right_xy) / 2
    x, y, z = w["contact:grf_net"].point_m
    assert z == pytest.approx(receipt["ground_height_m"], abs=1e-9)
    assert abs(y - mid_xy[1]) < COP_Y_ATOL_M
    assert abs(x - mid_xy[0]) < COP_X_ATOL_M
