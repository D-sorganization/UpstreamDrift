"""The drag-crisis constants are a least-squares fit to Choi et al. 2006 (#15443).

The zero-spin drag law ``cd_dimpled_sphere`` has four fitted parameters: the
crisis centre ``_RE_MIN_CD``, the crisis width ``_TRANSITION_WIDTH``, the
recovery centre ``_DRAG_CRISIS_RECOVERY_RE`` and the floor
``GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT``.  This test re-derives them from the
digitised wind-tunnel points in ``wind_tunnel_drag_15443.json`` by fitting the
production function itself, so the named constants stay traceable to the data
rather than being magic numbers.  The Lyu et al. 2018 abstract values are a
held-out check and are never fitted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from src.shared.python.core.physics_constants import (
    GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT,
)
from src.shared.python.physics import atmosphere

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).with_name("wind_tunnel_drag_15443.json")
_DATA: dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
_CHOI = np.array(_DATA["choi_2006_dimpled_sphere_cd_vs_re"]["points_re_cd"])
_LYU = _DATA["lyu_2018_still_air"]["checks_re_cd"]

# Acceptance: the law must match Choi to within twice the fixture's digitising
# uncertainty (0.005 in Cd, rms), and the rounded Lyu abstract values ("near
# 0.5", "close to 0.2") to the same 0.010.
MAX_RMS_CD = 0.010
LYU_TOLERANCE_CD = 0.010
# The named constants are the fit rounded to three significant figures.
CENTRE_TOLERANCE_DECADES = 0.005
WIDTH_TOLERANCE_DECADES = 0.001
FLOOR_TOLERANCE_CD = 0.001
# Start of the search: the pre-refit law (crisis 7e4, width 0.16, floor 0.20,
# recovery 7e5), so the fit does not start from its own answer.
_START = (math.log10(7.0e4), 0.16, 0.20, math.log10(7.0e5))


def _law(params: np.ndarray, reynolds: np.ndarray) -> np.ndarray:
    """Production ``cd_dimpled_sphere`` evaluated with trial parameters."""
    log_centre, width, floor, log_recovery = (float(p) for p in params)
    with mock.patch.multiple(
        atmosphere,
        _RE_MIN_CD=10.0**log_centre,
        _TRANSITION_WIDTH=width,
        _DRAG_CRISIS_RECOVERY_RE=10.0**log_recovery,
    ):
        return np.array(
            [atmosphere.cd_dimpled_sphere(float(r), floor) for r in reynolds]
        )


def _choi_fit() -> np.ndarray:
    """Least-squares parameters of the drag law over the Choi points."""
    optimize = pytest.importorskip("scipy.optimize")
    result = optimize.least_squares(
        lambda p: _law(p, _CHOI[:, 0]) - _CHOI[:, 1], np.array(_START)
    )
    assert result.success, result.message
    return np.asarray(result.x)


def _rms(cd: np.ndarray) -> float:
    return float(np.sqrt(np.mean((cd - _CHOI[:, 1]) ** 2)))


def test_named_constants_reproduce_the_choi_fit() -> None:
    log_centre, width, floor, log_recovery = _choi_fit()
    assert math.log10(atmosphere._RE_MIN_CD) == pytest.approx(
        log_centre, abs=CENTRE_TOLERANCE_DECADES
    )
    assert width == pytest.approx(
        atmosphere._TRANSITION_WIDTH, abs=WIDTH_TOLERANCE_DECADES
    )
    assert float(GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT) == pytest.approx(
        floor, abs=FLOOR_TOLERANCE_CD
    )
    assert math.log10(atmosphere._DRAG_CRISIS_RECOVERY_RE) == pytest.approx(
        log_recovery, abs=CENTRE_TOLERANCE_DECADES
    )


def test_production_law_matches_choi_within_rms() -> None:
    floor = float(GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT)
    cd = np.array([atmosphere.cd_dimpled_sphere(float(r), floor) for r in _CHOI[:, 0]])
    assert _rms(cd) <= MAX_RMS_CD


def test_pre_refit_law_was_outside_the_rms() -> None:
    """Guards the test itself: the old, broad crisis must fail the criterion."""
    assert _rms(_law(np.array(_START), _CHOI[:, 0])) > MAX_RMS_CD


@pytest.mark.parametrize(("reynolds", "expected"), _LYU)
def test_held_out_lyu_check(reynolds: float, expected: float) -> None:
    floor = float(GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT)
    cd = atmosphere.cd_dimpled_sphere(float(reynolds), floor)
    assert cd == pytest.approx(expected, abs=LYU_TOLERANCE_CD)
