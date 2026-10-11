"""Cross-engine static-hold parity of the per-hand bar wrench (LIFT-4, #11744).

For every engine whose lift pack and Python package are installed, the sum of
the two hand forces equals the bar weight within 1 % and the per-hand split
agrees across engines. Engines whose pack or package is absent are skipped,
never faked; at least two engines must be present for a parity comparison.

Measured on pack commits MuJoCo_Models 284b9b2, OpenSim_Models 16cb5cc,
Pinocchio_Models 5939024, Drake_Models 18118ac (120 kg bar):

* sum of vertical hand forces / bar weight: all engines within 3.3e-7
  (MuJoCo's settled hold), and 0 for the kinematic engines;
* bar weight differs by 0.034 % between Pinocchio (g = 9.81) and the others
  (g = 9.80665) -- inside the 1 % bound, so the comparison is on the
  weight-normalised left share;
* left share of the weight: MuJoCo 0.5 +- 1e-15, OpenSim 0.5 +- 6e-16,
  Pinocchio 0.5 +- 2.8e-7, Drake 0.356 / -0.170 / 0.378 / 0.268
  (deadlift / bench_press / snatch / clean_and_jerk) because the Drake pack
  does not centre the bar between the hands (Drake_Models#402).
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

_ENGINES = ("mujoco", "opensim", "pinocchio", "drake")
_PACKAGE = {
    "mujoco": "mujoco",
    "opensim": "opensim",
    "pinocchio": "pinocchio",
    "drake": "pydrake",
}
_HAND_HELD = ("deadlift", "bench_press", "snatch", "clean_and_jerk")
#: Issue #11744 acceptance: hand forces sum to the bar weight within 1 %.
_SUM_REL_TOL = 0.01
#: Engines that centre the bar between the hands split 0.5; this is the
#: largest measured deviation among them (Pinocchio 2.8e-7) with headroom
#: of one order of magnitude, not a loosened physical tolerance.
_SPLIT_ABS_TOL = 1e-5
#: Engines with a known off-centre bar; strict xfail until the pack is fixed.
_KNOWN_OFF_CENTRE = {
    "drake": "Drake_Models#402: bar not centred between the hands at the start pose"
}


def _real_package(engine: str) -> bool:
    try:
        module = importlib.import_module(_PACKAGE[engine])
    except (ImportError, OSError):
        return False
    return hasattr(module, "__file__")  # not a MagicMock test double


def _result(engine: str, lift: str) -> dict:
    if not _real_package(engine):
        pytest.skip(f"{_PACKAGE[engine]} is not installed (or is a test double)")
    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    pack = locate_pack(engine)
    if pack is None:
        pytest.skip(f"{engine} lift pack checkout not found (set LIFT_PACK_ROOT)")
    return create_adapter(pack, lift, Anthropometry()).bar_hold_wrench()


@pytest.mark.parametrize("lift", _HAND_HELD)
@pytest.mark.parametrize("engine", _ENGINES)
def test_hand_forces_sum_to_bar_weight(engine: str, lift: str) -> None:
    result = _result(engine, lift)

    assert result["available"], result["reason"]
    assert abs(result["sum_vertical_n"] - result["bar_weight_n"]) <= (
        _SUM_REL_TOL * result["bar_weight_n"]
    )


@pytest.mark.parametrize("lift", _HAND_HELD)
@pytest.mark.parametrize("engine", _ENGINES)
def test_per_hand_split_matches_the_symmetric_reference(
    engine: str, lift: str, request: pytest.FixtureRequest
) -> None:
    if engine in _KNOWN_OFF_CENTRE:
        request.applymarker(
            pytest.mark.xfail(strict=True, reason=_KNOWN_OFF_CENTRE[engine])
        )
    result = _result(engine, lift)

    assert result["available"], result["reason"]
    assert abs(result["split_left_fraction"] - 0.5) <= _SPLIT_ABS_TOL, result


@pytest.mark.parametrize("lift", _HAND_HELD)
def test_weight_normalised_split_agrees_across_available_engines(lift: str) -> None:
    shares = {}
    for engine in _ENGINES:
        if engine in _KNOWN_OFF_CENTRE or not _real_package(engine):
            continue
        shares[engine] = _result(engine, lift)["split_left_fraction"]
    if len(shares) < 2:
        pytest.skip(f"need >= 2 centred-bar engines to compare, have {sorted(shares)}")

    spread = max(shares.values()) - min(shares.values())
    assert spread <= _SPLIT_ABS_TOL, shares
