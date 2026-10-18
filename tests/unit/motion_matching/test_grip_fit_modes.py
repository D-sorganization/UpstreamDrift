"""Grip fitting modes, provenance stamping and golf-term reports (HRG-GR3, #13204)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import grip_fit
from src.shared.python.motion_matching.execution.grip_rotation import (
    apply_grip_config,
    grip_spec_of,
)
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    HandAngles,
    hand_to_euler,
)
from src.shared.python.motion_matching.grip_fit_modes import (
    GRIP_FIT_MODES,
    GripFitSettings,
    HandProblem,
    fit_hand,
    fitted_document,
    grip_mode_of,
    multistart_sensitivity,
    profile_interval,
)
from src.shared.python.motion_matching.grip_report import (
    describe_hand,
    describe_rotation_deg,
)
from src.shared.python.motion_matching.pipeline.lane import fitted_grip

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
RANGES = [(-90.0, 90.0), (-40.0, 25.0), (-70.0, 70.0)]
HAND = HandAngles(10.0, 55.0, 20.0)


def _chain(p: float, c: float, f: float) -> np.ndarray:
    rz = Rotation.from_euler("z", p).as_matrix()
    rx = Rotation.from_euler("x", c).as_matrix()
    return np.asarray(rz @ BASE @ rx @ Rotation.from_euler("z", f).as_matrix())


def _relative(seed: int, hidden_deg: tuple[float, float, float] = (0, 0, 0)):
    """Frames whose wrist angles sit inside the ranges once ``hidden`` is undone."""
    rng = np.random.default_rng(seed)
    lo, hi = np.radians(np.asarray(RANGES)).T
    angles = rng.uniform(lo * 0.5, hi * 0.5, size=(60, 3))
    good = np.stack([_chain(*row) for row in angles])
    hidden = grip_fit.grip_rotation(hidden_deg)
    return good @ hidden.T


def _problem(seed: int = 0, hidden=(0, 0, 0), declared=HAND) -> HandProblem:
    # the stack is measured with the declared grip in the document
    return HandProblem(_relative(seed, hidden), BASE, RANGES, declared)


def test_modes_are_the_three_named() -> None:
    assert GRIP_FIT_MODES == ("user_fixed", "bounded_prior", "free_fit")
    with pytest.raises(ValueError):
        GripFitSettings(mode="measured")


def test_user_fixed_has_zero_drift_and_no_fit() -> None:
    result = fit_hand(_problem(1, hidden=(40, -30, 25)), GripFitSettings("user_fixed"))
    assert result.fitted == HAND
    assert result.drift_deg == (0.0, 0.0, 0.0)
    assert result.cost_after == result.cost_before
    assert result.interval is None and result.sensitivity is None
    assert "user_fixed" in result.unavailable_reason


def test_bounded_prior_stays_in_bounds_and_never_worsens() -> None:
    settings = GripFitSettings("bounded_prior", bound_half_deg=(10.0, 10.0, 15.0))
    result = fit_hand(_problem(2, hidden=(45, -35, 30)), settings)
    assert result.cost_before > 1.0, "the hidden rotation must create ROM excess"
    assert result.cost_after <= result.cost_before
    for got, want, half in zip(
        (result.fitted.alpha_deg, result.fitted.psi_deg, result.fitted.gamma_deg),
        (HAND.alpha_deg, HAND.psi_deg, HAND.gamma_deg),
        (10.0, 10.0, 15.0),
        strict=True,
    ):
        assert abs(got - want) <= half + 1e-9
    assert any(result.bound_hits.values()), "an excess this large must hit a bound"


def test_bounded_prior_inside_plateau_does_not_move() -> None:
    result = fit_hand(_problem(3), GripFitSettings("bounded_prior"))
    assert result.cost_before == pytest.approx(0.0, abs=1e-9)
    assert max(abs(v) for v in result.drift_deg) < 1e-2


def test_plateau_has_a_unique_deterministic_answer_under_free_fit() -> None:
    problem = _problem(4)
    settings = GripFitSettings("free_fit")
    first = fit_hand(problem, settings, uncertainty=False)
    again = fit_hand(problem, settings, uncertainty=False)
    assert first.fitted == again.fitted
    assert first.cost_after == pytest.approx(0.0, abs=1e-6)
    sens = multistart_sensitivity(problem, settings, n_starts=10, seed=7)
    assert sens is not None
    assert max(sens["spread_deg"].values()) < 0.5
    other = multistart_sensitivity(problem, settings, n_starts=10, seed=99)
    assert other is not None
    for name, value in sens["best_deg"].items():
        assert other["best_deg"][name] == pytest.approx(value, abs=0.5)


def test_without_the_tie_breaker_the_plateau_is_not_unique() -> None:
    problem = _problem(4)
    flat = GripFitSettings("free_fit", tie_lambda=0.0)
    sens = multistart_sensitivity(problem, flat, n_starts=10, seed=7)
    assert sens is not None
    assert max(sens["spread_deg"].values()) > 2.0


def test_profile_interval_and_sensitivity_are_reported() -> None:
    result = fit_hand(_problem(5, hidden=(20, 10, -15)), GripFitSettings("free_fit"))
    assert result.interval is not None
    for name in ("alpha_deg", "psi_deg", "gamma_deg"):
        lo, hi = result.interval[name]["range_deg"]
        assert lo <= getattr(result.fitted, name) <= hi
        assert result.interval[name]["tolerance_deg"] == 0.1
    assert result.sensitivity is not None
    assert result.sensitivity["n_starts"] >= 2
    document = json.dumps(result.as_document())
    assert "rms_marker" not in document and "confidence" not in document


def test_unavailable_uncertainty_is_null_never_zero() -> None:
    result = fit_hand(_problem(6), GripFitSettings("free_fit"), uncertainty=False)
    assert result.interval is None and result.sensitivity is None
    doc = result.as_document()
    assert doc["interval"] is None and doc["sensitivity"] is None
    assert "not requested" in doc["unavailable_reason"]
    assert profile_interval(_problem(6), GripFitSettings("user_fixed")) is None


def test_existing_fit_is_unchanged() -> None:
    before = grip_fit.fit_grip_rotation(_relative(8, (30, 0, 0)), BASE, RANGES)
    assert before.cost_after <= before.cost_before


def test_grip_mode_defaults_to_user_fixed_and_validates() -> None:
    assert grip_mode_of({"subject": {}}) == "user_fixed"
    assert grip_mode_of({"subject": {"grip_mode": "free_fit"}}) == "free_fit"
    with pytest.raises(ValueError):
        grip_mode_of({"subject": {"grip_mode": "auto"}})


@pytest.fixture(scope="module")
def configured() -> dict[str, Any]:
    document = json.loads(DRIVER.read_text(encoding="utf-8"))
    return apply_grip_config(document, GripSpec(lead=HAND, trail=HAND))


def test_fitted_document_stamps_rom_set_identified(configured: dict[str, Any]) -> None:
    assert not fitted_grip(configured)
    moved = fit_hand(_problem(9, hidden=(30, 20, -20)), GripFitSettings("free_fit"))
    out = fitted_document(configured, {"lead": moved}, mode="free_fit")
    prov = out["subject"]["grip_provenance"]
    assert prov["lead.alpha_deg"] == "rom_set_identified"
    assert prov["trail.alpha_deg"] == "user_declared"
    assert prov["lead_below_butt_m"] == "user_declared"
    assert out["subject"]["grip_mode"] == "free_fit"
    assert fitted_grip(out)
    assert grip_spec_of(out).lead == moved.fitted
    assert "grip_provenance" not in json.dumps(configured["subject"].get("grip_mode"))


def test_fitted_document_with_user_fixed_changes_nothing_physical(
    configured: dict[str, Any],
) -> None:
    fixed = fit_hand(_problem(9, hidden=(30, 20, -20)), GripFitSettings("user_fixed"))
    out = fitted_document(configured, {"lead": fixed}, mode="user_fixed")
    assert out["subject"]["grip_config"] == configured["subject"]["grip_config"]
    assert not fitted_grip(out)


def test_golf_language() -> None:
    assert describe_hand(HandAngles(-15.0, 25.0, 0.0)) == "weak, fingers"
    assert describe_hand(HandAngles(0.0, 40.0, 0.0)) == "neutral, neutral"
    assert describe_hand(HandAngles(14.0, 66.0, 0.0)) == "strong, palm"
    lines = describe_rotation_deg({"L": hand_to_euler(HAND), "R": hand_to_euler(HAND)})
    assert len(lines) == 2
    assert "alpha=10.0" in lines[0] and "psi=55.0" in lines[0]
    assert not any(math.isnan(0.0) for _ in lines)
