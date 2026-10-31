"""MuJoCo static-hold mu_crit calibration of the pad contact (#13885 PR 1)."""

from __future__ import annotations

import json
import math

import pytest

from src.engines.physics_engines.mujoco.python.grip_hold_calibration import (
    BIAS_BAND,
    HoldRig,
    calibrate,
    declared_bias,
    holds,
)
from src.shared.python.core.physics_constants import GRAVITY_M_S2

pytestmark = pytest.mark.unit

RIG = HoldRig()


def test_mu_crit_is_the_weight_over_both_squeezes() -> None:
    rig = HoldRig(block_mass_kg=2.0, squeeze_n=40.0)
    assert rig.mu_crit == pytest.approx(2.0 * float(GRAVITY_M_S2) / 80.0)


def test_hold_and_slip_bracket_the_critical_friction() -> None:
    assert holds(RIG, 1.2 * RIG.mu_crit, "Euler")
    assert not holds(RIG, 0.8 * RIG.mu_crit, "Euler")


@pytest.mark.parametrize("integrator", ["Euler", "implicitfast"])
def test_transition_is_recorded_per_integrator(integrator: str) -> None:
    record = calibrate(RIG, integrator)
    assert record.integrator == integrator
    assert record.mu_crit == RIG.mu_crit
    assert record.ratio == pytest.approx(record.transition_mu / RIG.mu_crit)
    assert 0.9 < record.ratio < 1.1
    assert record.declared_bias == declared_bias(record.ratio)
    assert holds(RIG, record.transition_mu, integrator)
    assert record == calibrate(RIG, integrator)  # deterministic


def test_bias_outside_the_band_is_declared_either_way() -> None:
    assert declared_bias(1.0 + BIAS_BAND) is None
    assert declared_bias(1.0 - BIAS_BAND) is None
    assert declared_bias(1.07) == 1.07
    assert declared_bias(0.9) == 0.9


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"block_mass_kg": 0.0}, "block_mass_kg"), ({"squeeze_n": -1.0}, "squeeze_n")]
    + [({"timestep_s": math.nan}, "timestep_s"), ({"hold_s": 0.0}, "hold_s")],
)
def test_rig_contract(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HoldRig(**kwargs)


def test_calibration_contract() -> None:
    with pytest.raises(ValueError, match="integrator"):
        holds(RIG, 1.0, "Verlet")
    with pytest.raises(ValueError, match="friction"):
        holds(RIG, 0.0, "Euler")
    with pytest.raises(ValueError, match="bracket"):
        calibrate(RIG, "Euler", lo=1.2, hi=2.0)


def test_cli_writes_a_receipt(tmp_path) -> None:
    from scripts.grip import mujoco_mu_crit_calibration as cli

    out = tmp_path / "calibration.json"
    assert cli.main(["--integrators", "Euler", "--out", str(out)]) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["issue"] == 13885
    assert receipt["bias_band"] == BIAS_BAND
    (row,) = receipt["calibrations"]
    assert row["integrator"] == "Euler"
    assert 0.9 < row["ratio"] < 1.1
