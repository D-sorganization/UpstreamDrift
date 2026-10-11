"""Static-hold per-hand bar wrench, Drake slice (LIFT-4, #11744, GCV-7).

A rigid two-hand hold of a rigid bar is statically indeterminate: the Drake
pack welds the bar to ``hand_l`` with a fixed SDF joint and to ``hand_r``
with a ``biomech:weld`` loop-closure, so Drake cannot supply the hand/hand
split directly. The adapter instead computes the bar's mass/COM from its own
``barbell``-prefixed bodies and forms the net static-equilibrium wrench the
hands must exert, split by the shared ``allocate_min_norm`` (GCV-7). This is
a kinematic calculation (no simulation): ``bar_linear_accel_mps2`` is always
``None``.

Acceptance from #11744: the hand forces sum to the bar plus plate weight
within 1 %, and a symmetric grip splits symmetrically.

The helper-level tests use duck-typed fakes and need no pydrake; the
real-pack tests skip cleanly when pydrake or the Drake_Models checkout is
absent (``tests/unit/conftest.py`` installs a MagicMock for ``pydrake`` when
it is not installed, so a real install is detected via ``__file__``).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

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
#: Symmetric grip -> even split. The other three packs measure
#: |share - 0.5| <= 2.8e-7; the Drake pack does not (see the xfail below).
_SPLIT_ABS_TOL = 1e-5
#: Measured left-hand share of the bar weight on the Drake pack (pack commit
#: 18118ac), where the bar is not centred between the hands:
#: deadlift 0.356, bench_press -0.170, snatch 0.378, clean_and_jerk 0.268.
_DRAKE_OFF_CENTRE = (
    "Drake_Models#402: the bar is off the hand midline at the start pose "
    "(measured left share 0.356 deadlift, -0.170 bench_press, 0.378 snatch, "
    "0.268 clean_and_jerk vs 0.5 in the MuJoCo/OpenSim/Pinocchio packs)"
)
_RIGHT_RESIDUAL_TOL_M = 5e-3  # Drake_Models weld closure bound (#393)


def _mod():
    from src.shared.python.lifting.pack_audit.adapters import drake_bar_hold

    return drake_bar_hold


# --------------------------------------------------------------------------
# Helper-function unit tests with duck-typed fakes (no pydrake).
# --------------------------------------------------------------------------


class _FakeBody:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _FakeJoint:
    def __init__(self, parent: str, child: str) -> None:
        self._parent = _FakeBody(parent)
        self._child = _FakeBody(child)

    def parent_body(self) -> _FakeBody:
        return self._parent

    def child_body(self) -> _FakeBody:
        return self._child


class _FakePlant:
    def __init__(self, joints: list[_FakeJoint]) -> None:
        self._joints = joints

    def GetJointIndices(self, _instance: Any) -> range:  # noqa: N802
        return range(len(self._joints))

    def get_joint(self, i: int) -> _FakeJoint:
        return self._joints[i]


def _fake_loaded(joints: list[_FakeJoint], welds: list[tuple[str, str]]):
    specs = tuple(
        SimpleNamespace(name=f"w{i}", parent=p, child=c)
        for i, (p, c) in enumerate(welds)
    )
    return SimpleNamespace(
        plant=_FakePlant(joints),
        model_instance=object(),
        weld_specs=specs,
        weld_poses={},
    )


def test_has_left_bar_joint_false_when_bar_is_on_torso() -> None:
    mod = _mod()
    plant = _FakePlant([_FakeJoint("torso", "barbell_shaft")])
    assert mod._has_left_bar_joint(plant, object()) is False


def test_has_left_bar_joint_true_for_hand_l_to_bar() -> None:
    mod = _mod()
    plant = _FakePlant([_FakeJoint("hand_l", "barbell_shaft")])
    assert mod._has_left_bar_joint(plant, object()) is True


def test_find_right_weld_requires_hand_r_to_barbell() -> None:
    mod = _mod()
    none_match = (SimpleNamespace(name="a", parent="torso", child="barbell_shaft"),)
    match = (SimpleNamespace(name="b", parent="hand_r", child="barbell_shaft"),)
    assert mod._find_right_weld(none_match) is None
    assert mod._find_right_weld(match).name == "b"


def test_bar_hold_wrench_unavailable_when_bar_is_on_torso() -> None:
    mod = _mod()
    loaded = _fake_loaded(
        [_FakeJoint("torso", "barbell_shaft")], [("torso", "barbell_shaft")]
    )
    result = mod.bar_hold_wrench(loaded)

    assert result["available"] is False
    assert "torso" in result["reason"]
    for field in _UNAVAILABLE_NUMERIC:
        assert result[field] is None, f"{field} must be None, never 0"


def test_bar_hold_wrench_rejects_object_without_plant() -> None:
    mod = _mod()
    with pytest.raises(TypeError):
        mod.bar_hold_wrench(object())


# --------------------------------------------------------------------------
# Real-pack integration tests (require a Drake_Models checkout AND pydrake).
# --------------------------------------------------------------------------


def _drake_lift_adapter(lift: str):
    pydrake = pytest.importorskip("pydrake")
    if not hasattr(pydrake, "__file__"):
        pytest.skip("pydrake in sys.modules is a test double, not the real package")
    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    pack = locate_pack("drake")
    if pack is None:
        pytest.skip("Drake lift pack checkout not found (set LIFT_PACK_ROOT)")
    return create_adapter(pack, lift, Anthropometry())


_HAND_HELD = ("deadlift", "bench_press", "snatch", "clean_and_jerk")


@pytest.mark.requires_drake
@pytest.mark.integration
@pytest.mark.parametrize("lift", _HAND_HELD)
def test_pack_hand_held_lift_static_hold(lift: str) -> None:
    adapter = _drake_lift_adapter(lift)
    result = adapter.bar_hold_wrench()

    assert result["available"], result["reason"]
    assert result["split_method"] == "allocation"
    assert result["bar_mass_kg"] == pytest.approx(
        adapter.anthro.bar_total_mass_kg, rel=1e-6
    )
    assert result["relative_error"] <= 0.01, result
    assert result["bar_linear_accel_mps2"] is None
    assert result["n_welds"] == 2
    assert set(result["hand_grip_residual_m"]) == {"L", "R"}
    # hand_l IS the left grip point (exact by construction).
    assert result["hand_grip_residual_m"]["L"] == 0.0
    r_residual = result["hand_grip_residual_m"]["R"]
    assert math.isfinite(r_residual) and 0.0 <= r_residual <= _RIGHT_RESIDUAL_TOL_M


@pytest.mark.requires_drake
@pytest.mark.integration
@pytest.mark.xfail(strict=True, reason=_DRAKE_OFF_CENTRE)
@pytest.mark.parametrize("lift", _HAND_HELD)
def test_pack_symmetric_grip_splits_evenly_and_both_hands_lift(lift: str) -> None:
    result = _drake_lift_adapter(lift).bar_hold_wrench()

    assert abs(result["split_left_fraction"] - 0.5) <= _SPLIT_ABS_TOL, result
    # Sign (ADR-0052): the wrench is exerted by the hand ON the bar -> up.
    assert result["hand_force_n"]["L"][2] > 0.0
    assert result["hand_force_n"]["R"][2] > 0.0


@pytest.mark.requires_drake
@pytest.mark.integration
def test_pack_squat_is_unavailable_bar_on_torso() -> None:
    result = _drake_lift_adapter("squat").bar_hold_wrench()

    assert result["available"] is False
    assert result["bar_mass_kg"] is None
    assert "torso" in result["reason"] or "hand" in result["reason"]
    for field in _UNAVAILABLE_NUMERIC:
        assert result[field] is None, f"{field} must be None, never 0"
