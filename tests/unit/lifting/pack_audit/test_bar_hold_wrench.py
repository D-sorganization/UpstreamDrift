"""Static-hold per-hand bar wrench (LIFT-4, #11744, GCV-7/GCV-8).

The MuJoCo adapter reduces a static hold of the pack's start pose to the
per-hand wrench on the bar through the shared GCV-7 grip analysis
(``biomechanics.grip_wrench``) and the GCV-8 weld ``efc_force`` extraction
(``grip_efc``) -- no second wrench-transport routine.  A lift whose bar is
not welded to both hands (back squat: bar welded to the torso) is reported
unavailable with a precise reason, never as zero -- and so is a lift whose
bar has no joint at all: the current MuJoCo_Models pack's barbell bodies are
kinematic fixtures (``body_dofnum == 0``), so their weight never enters the
dynamics and no per-hand split is physically recoverable from the weld
reaction (see the comment above the real-pack tests below for the full
investigation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_G = 9.81
_MASS_KG = 20.0

# Two static (0-DOF) "hand" bodies holding a free barbell shaft at the
# relpose MuJoCo derives from this reference layout (zero initial residual).
_HELD_XML = f"""
<mujoco>
<option gravity="0 0 -{_G}"/>
<worldbody>
 <body name="hand_r" pos="0.3 0 1">
  <geom type="sphere" size="0.01" mass="0.001" contype="0" conaffinity="0"/>
 </body>
 <body name="hand_l" pos="-0.3 0 1">
  <geom type="sphere" size="0.01" mass="0.001" contype="0" conaffinity="0"/>
 </body>
 <body name="barbell_shaft" pos="0 0 1">
  <freejoint/>
  <geom type="box" size="0.3 0.02 0.02" mass="{_MASS_KG}" contype="0" conaffinity="0"/>
 </body>
</worldbody>
<equality>
 <weld name="barbell_to_hand_r" body1="hand_r" body2="barbell_shaft"/>
 <weld name="barbell_to_hand_l" body1="hand_l" body2="barbell_shaft"/>
</equality>
</mujoco>"""

_HELD_WELDS = [
    {"name": "barbell_to_hand_r", "body1": "hand_r", "body2": "barbell_shaft"},
    {"name": "barbell_to_hand_l", "body1": "hand_l", "body2": "barbell_shaft"},
]

# Barbell welded to the torso (the back-squat mechanism): no hand weld exists.
_TORSO_XML = f"""
<mujoco>
<option gravity="0 0 -{_G}"/>
<worldbody>
 <body name="torso" pos="0 0 1.4">
  <geom type="sphere" size="0.05" mass="1.0" contype="0" conaffinity="0"/>
 </body>
 <body name="barbell_shaft" pos="0 0 1.4">
  <freejoint/>
  <geom type="box" size="0.3 0.02 0.02" mass="{_MASS_KG}" contype="0" conaffinity="0"/>
 </body>
</worldbody>
<equality>
 <weld name="barbell_to_torso" body1="torso" body2="barbell_shaft"/>
</equality>
</mujoco>"""

_TORSO_WELDS = [
    {"name": "barbell_to_torso", "body1": "torso", "body2": "barbell_shaft"},
]


def _mod():
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.pack_audit.adapters import mujoco_bar_hold

    return mujoco_bar_hold


@pytest.mark.requires_mujoco
def test_static_hold_balances_weight_within_a_tight_tolerance() -> None:
    mod = _mod()
    result = mod.bar_hold_wrench(_HELD_XML, _HELD_WELDS)
    assert result["available"], result["reason"]
    assert result["bar_mass_kg"] == pytest.approx(_MASS_KG)
    assert result["bar_weight_n"] == pytest.approx(_MASS_KG * _G)
    # Two FULL 6-DOF welds pinning one 6-DOF free body are structurally
    # redundant (12 constraint rows for 6 DOF); MuJoCo's default soft
    # (compliant) equality solver does not fully cancel that redundancy in
    # one static mj_forward solve. Measured 0.0526 (5.26%) for this fixture;
    # asserted with margin, not loosened further.
    assert result["relative_error"] < 0.06, result


@pytest.mark.requires_mujoco
def test_static_hold_sign_is_up_and_split_is_symmetric() -> None:
    mod = _mod()
    result = mod.bar_hold_wrench(_HELD_XML, _HELD_WELDS)
    left = result["hand_force_n"]["L"]
    right = result["hand_force_n"]["R"]
    assert left[2] > 0.0
    assert right[2] > 0.0
    # Symmetric geometry (hands equidistant from the bar's midpoint, same
    # height) measured to split evenly well under 1e-6.
    assert result["split_left_fraction"] == pytest.approx(0.5, abs=1e-6)


@pytest.mark.requires_mujoco
def test_static_hold_bar_acceleration_is_negligible() -> None:
    mod = _mod()
    result = mod.bar_hold_wrench(_HELD_XML, _HELD_WELDS)
    # Residual acceleration from the same redundant-weld compliance as above
    # (measured 0.516 m/s^2); two orders of magnitude below the bar's own
    # free-fall acceleration (g = 9.81 m/s^2), not loosened further.
    assert result["bar_linear_accel_mps2"] < 0.6, result


@pytest.mark.requires_mujoco
def test_non_hand_weld_is_unavailable_not_zero() -> None:
    mod = _mod()
    result = mod.bar_hold_wrench(_TORSO_XML, _TORSO_WELDS)
    assert result["available"] is False
    assert result["bar_mass_kg"] is None
    assert result["hand_force_n"] is None
    assert result["relative_error"] is None
    assert "torso" in result["reason"]


@pytest.mark.requires_mujoco
def test_rejects_empty_xml_and_welds() -> None:
    mod = _mod()
    with pytest.raises(TypeError):
        mod.bar_hold_wrench("", _HELD_WELDS)
    with pytest.raises(ValueError):
        mod.bar_hold_wrench(_HELD_XML, [])


def test_base_adapter_default_is_unavailable_with_none_fields() -> None:
    """A minimal ``EngineAdapter`` subclass needs no engine import at all."""
    from src.shared.python.lifting.pack_audit.model import (
        Anthropometry,
        EngineAdapter,
        PoseEval,
    )
    from src.shared.python.lifting.pack_audit.packs import PackLocation

    class _StubAdapter(EngineAdapter):
        engine = "stub"

        def model_text(self) -> str:
            return ""

        def coordinates(self) -> list:
            return []

        def structure(self) -> dict:
            return {}

        def segment_masses(self) -> dict:
            return {}

        def evaluate(self, q):
            return PoseEval(positions={}, com=np.zeros(3), total_mass=0.0)

        def smoke_step(self) -> dict:
            return {"loaded": True, "stepped": True}

    pack = PackLocation(
        engine="stub",
        repo="x",
        root=Path("."),
        src=Path("."),
        package="x",
        commit=None,
        licence=None,
    )
    adapter = _StubAdapter(pack, "deadlift", Anthropometry())
    result = adapter.bar_hold_wrench()

    assert result["available"] is False
    assert result["reason"] == "not implemented for this engine"
    numeric_fields = (
        "split_method",
        "bar_mass_kg",
        "bar_weight_n",
        "hand_force_n",
        "sum_vertical_n",
        "relative_error",
        "split_left_fraction",
        "couple_at_midpoint_nm",
        "bar_linear_accel_mps2",
        "method",
        "n_welds",
    )
    for field in numeric_fields:
        assert result[field] is None, f"{field} must be None, never 0"


def _mujoco_lift_adapter(lift: str):
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    pack = locate_pack("mujoco")
    if pack is None:
        pytest.skip("MuJoCo lift pack checkout not found (set LIFT_PACK_ROOT)")
    return create_adapter(pack, lift, Anthropometry(80.0, 1.78, 50.0))


_HAND_HELD_LIFTS = ("deadlift", "bench_press", "snatch", "clean_and_jerk")

# INVESTIGATION FINDING (LIFT-4, not a defect in this change): on the current
# MuJoCo_Models lift pack (sibling checkout, `main`, grip-weld fixes #437/
# #441/#443 already applied), the barbell bodies (`barbell_shaft`,
# `barbell_left_sleeve`, `barbell_right_sleeve`) are added via
# `create_barbell_bodies()` with no `<freejoint>` or any other joint --
# confirmed empirically: `model.body_dofnum` is 0 for all three, for every
# hand-held lift (deadlift/bench_press/snatch/clean_and_jerk). A body with
# body_dofnum == 0 never enters MuJoCo's equations of motion (M*qacc = ...);
# its `body_mass` is carried for bookkeeping (it sums correctly to
# `Anthropometry.bar_total_mass_kg`) but gravity never acts on it the way it
# acts on a jointed body. The `barbell_to_hand_{l,r}` weld equalities are
# therefore purely kinematic position constraints (pinning the hand to the
# bar's fixed location), not a load path: no per-hand split of the bar's
# weight is physically recoverable from the weld reaction in this pack,
# regardless of `grip_efc`'s own correctness. `bar_hold_wrench` detects this
# (`body_dofnum` all zero across the matched bar bodies) and reports
# `available=False` with a precise reason instead of a wrong or zero number.
# Making this measurement physically meaningful needs a follow-up in
# MuJoCo_Models: give `barbell_shaft` its own `<freejoint>` so the bar's mass
# is dynamically coupled to the grip.


def _mujoco_lift_adapter(lift: str):
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    pack = locate_pack("mujoco")
    if pack is None:
        pytest.skip("MuJoCo lift pack checkout not found (set LIFT_PACK_ROOT)")
    return create_adapter(pack, lift, Anthropometry(80.0, 1.78, 50.0))


@pytest.mark.requires_mujoco
@pytest.mark.integration
@pytest.mark.parametrize("lift", _HAND_HELD_LIFTS)
def test_pack_hand_held_lifts_report_unavailable_bar_has_no_joint(lift: str) -> None:
    adapter = _mujoco_lift_adapter(lift)
    result = adapter.bar_hold_wrench()
    assert result["available"] is False
    assert "dofnum=0" in result["reason"]
    assert "barbell_shaft" in result["reason"]
    numeric_fields = (
        "bar_mass_kg",
        "bar_weight_n",
        "hand_force_n",
        "sum_vertical_n",
        "relative_error",
        "split_left_fraction",
        "couple_at_midpoint_nm",
        "bar_linear_accel_mps2",
    )
    for field in numeric_fields:
        assert result[field] is None, f"{field} must be None, never 0 ({lift})"


@pytest.mark.requires_mujoco
@pytest.mark.integration
def test_pack_squat_is_unavailable_bar_on_torso() -> None:
    adapter = _mujoco_lift_adapter("squat")
    result = adapter.bar_hold_wrench()
    assert result["available"] is False
    assert result["bar_mass_kg"] is None
    assert "torso" in result["reason"]
