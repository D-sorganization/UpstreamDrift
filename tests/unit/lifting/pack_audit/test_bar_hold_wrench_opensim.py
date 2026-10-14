"""Static-hold per-hand bar wrench, OpenSim slice (LIFT-4, #11744, GCV-7).

A rigid two-hand hold of a rigid bar is statically indeterminate, and the
OpenSim pack welds the bar to the hands with a ``WeldJoint`` (left) and a
``WeldConstraint`` (right), so OpenSim cannot supply the hand/hand split
directly. The adapter instead computes the bar's mass/COM from its own
``barbell``-prefixed bodies and forms the net static-equilibrium wrench the
hands must exert, split by the shared ``allocate_min_norm`` (GCV-7). This is
a kinematic calculation (no simulation): ``bar_linear_accel_mps2`` is always
``None``.

Acceptance from #11744: the hand forces sum to the bar plus plate weight
within 1 %, and a symmetric grip splits symmetrically.

OpenSim is only importable inside the ud-sim container; every test here
skips cleanly (``pytest.importorskip``) when it is absent or is a test
double (no autouse mock is installed for ``opensim`` the way it is for
``pinocchio`` in ``tests/unit/conftest.py``).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit

_UNAVAILABLE_NUMERIC = (
    "bar_mass_kg",
    "bar_weight_n",
    "hand_force_n",
    "sum_vertical_n",
    "relative_error",
    "split_left_fraction",
    "couple_at_midpoint_nm",
    "bar_linear_accel_mps2",
)
#: Symmetric grip -> even split. Measured |share - 0.5| on every hand-held
#: lift of the real OpenSim pack: deadlift 5.6e-16, bench_press 5.0e-16,
#: snatch 1.1e-14, clean_and_jerk 6.7e-16 -- floating-point noise, not a
#: modeling error (both grip points come straight from each hand's own
#: WeldJoint/WeldConstraint frame, with no virtual-link masses to merge,
#: unlike the Pinocchio pack).
_SPLIT_ABS_TOL = 1e-12
#: Measured ``hand_grip_residual_m`` on the same four lifts: L in
#: [5.2e-18, 1.7e-17] (a zero-DOF WeldJoint -- floating-point noise only),
#: R in [6.7e-16, 2.6e-14] (the WeldConstraint's assembled residual, far
#: inside OpenSim_Models' own <5 mm acceptance bound, #394/#426).
_LEFT_RESIDUAL_TOL_M = 1e-9
_RIGHT_RESIDUAL_TOL_M = 1e-9


def _mod():
    require_opensim()
    from src.shared.python.lifting.pack_audit.adapters import opensim_bar_hold

    return opensim_bar_hold


# --------------------------------------------------------------------------
# Helper-function unit tests with duck-typed fakes (no real OpenSim model).
# --------------------------------------------------------------------------


class _FakeSet:
    def __init__(self, items: list[Any]) -> None:  # type: ignore[name-defined]
        self._items = items

    def getSize(self) -> int:
        return len(self._items)

    def get(self, i: int):  # type: ignore[no-untyped-def]
        return self._items[i]


class _FakeFrame:
    def __init__(self, base_name: str) -> None:
        self._base_name = base_name

    def findBaseFrame(self) -> _FakeFrame:
        return self

    def getName(self) -> str:
        return self._base_name


class _FakeJoint:
    def __init__(self, class_name: str, parent: str, child: str) -> None:
        self._class_name = class_name
        self._parent = _FakeFrame(parent)
        self._child = _FakeFrame(child)

    def getConcreteClassName(self) -> str:
        return self._class_name

    def getParentFrame(self) -> _FakeFrame:
        return self._parent

    def getChildFrame(self) -> _FakeFrame:
        return self._child


class _FakeModelNoBarAttachment:
    """A model with no hand-to-bar WeldJoint/WeldConstraint (e.g. squat)."""

    def getJointSet(self) -> _FakeSet:
        return _FakeSet([_FakeJoint("WeldJoint", "torso", "barbell_shaft")])

    def getConstraintSet(self) -> _FakeSet:
        return _FakeSet([])


def test_find_left_weld_joint_returns_none_when_bar_is_on_torso() -> None:
    mod = _mod()
    assert mod._find_left_weld_joint(_FakeModelNoBarAttachment()) is None


def test_find_right_weld_constraint_returns_none_when_absent() -> None:
    mod = _mod()
    assert mod._find_right_weld_constraint(_FakeModelNoBarAttachment()) is None


def test_bar_hold_wrench_unavailable_when_bar_is_on_torso() -> None:
    mod = _mod()
    result = mod.bar_hold_wrench(_FakeModelNoBarAttachment(), object())

    assert result["available"] is False
    assert "torso" in result["reason"]
    for field in _UNAVAILABLE_NUMERIC:
        assert result[field] is None, f"{field} must be None, never 0"


# --------------------------------------------------------------------------
# Real-pack integration tests (require an OpenSim_Models checkout AND the
# real `opensim` package -- only available inside the ud-sim container).
# --------------------------------------------------------------------------


def _opensim_lift_adapter(lift: str):
    osim = require_opensim()
    if not hasattr(osim, "__file__"):
        pytest.skip("opensim in sys.modules is a test double, not the real package")
    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    pack = locate_pack("opensim")
    if pack is None:
        pytest.skip("OpenSim lift pack checkout not found (set LIFT_PACK_ROOT)")
    return create_adapter(pack, lift, Anthropometry())


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize(
    "lift", ("deadlift", "bench_press", "snatch", "clean_and_jerk")
)
def test_pack_hand_held_lift_static_hold(lift: str) -> None:
    adapter = _opensim_lift_adapter(lift)
    result = adapter.bar_hold_wrench()

    assert result["available"], result["reason"]
    assert result["split_method"] == "allocation"
    assert result["bar_mass_kg"] == pytest.approx(
        adapter.anthro.bar_total_mass_kg, rel=1e-6
    )
    assert result["relative_error"] <= 0.01, result
    assert abs(result["split_left_fraction"] - 0.5) <= _SPLIT_ABS_TOL, result
    assert result["bar_linear_accel_mps2"] is None
    assert result["n_welds"] == 2
    # Sign (ADR-0052): the wrench is exerted by the hand ON the bar -> up.
    assert result["hand_force_n"]["L"][2] > 0.0
    assert result["hand_force_n"]["R"][2] > 0.0
    assert set(result["hand_grip_residual_m"]) == {"L", "R"}
    left_residual = result["hand_grip_residual_m"]["L"]
    right_residual = result["hand_grip_residual_m"]["R"]
    # Left is a zero-DOF WeldJoint (kinematically exact); right is a
    # position-level WeldConstraint OpenSim assembles to within ~5 mm
    # (OpenSim_Models#394/#426) -- see the tolerance comments above for the
    # actual measured values, both far inside that 5 mm bound.
    assert math.isfinite(left_residual) and 0.0 <= left_residual <= _LEFT_RESIDUAL_TOL_M
    assert (
        math.isfinite(right_residual) and 0.0 <= right_residual <= _RIGHT_RESIDUAL_TOL_M
    )


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_pack_squat_is_unavailable_bar_on_torso() -> None:
    result = _opensim_lift_adapter("squat").bar_hold_wrench()

    assert result["available"] is False
    assert result["bar_mass_kg"] is None
    assert "torso" in result["reason"]
    for field in _UNAVAILABLE_NUMERIC:
        assert result[field] is None, f"{field} must be None, never 0"
