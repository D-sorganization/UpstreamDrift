"""Static-hold per-hand bar wrench (LIFT-4, #11744, GCV-7/GCV-8).

The MuJoCo adapter reduces a settled static hold of the pack's start pose to
the per-hand wrench on the bar through the shared GCV-7 grip analysis
(``biomechanics.grip_wrench``) and the GCV-8 weld ``efc_force`` extraction
(``grip_efc``) -- no second wrench-transport routine.  A lift whose bar is
not welded to both hands (back squat: bar welded to the torso), or whose bar
has no joint, is reported unavailable with a reason, never as zero.

Acceptance from #11744: the hand forces sum to the bar plus plate weight
within 1 %, and a symmetric grip splits symmetrically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_G = 9.81
_MASS_KG = 20.0
#: #11744 acceptance: sum of hand forces equals the bar weight within 1 %.
_WEIGHT_REL_TOL = 0.01
#: Symmetric grip -> even split.  Measured |share - 0.5| <= 3e-16 on every
#: hand-held lift of the MuJoCo pack (284b9b2) and on the fixture below.
_SPLIT_ABS_TOL = 1e-6
#: The settled bar must be at rest.  Measured <= 3.3e-6 m/s^2 on the pack.
_REST_ACCEL_MPS2 = 1e-3

# Two static "hand" bodies holding a free barbell shaft at the relpose MuJoCo
# derives from this layout (zero initial residual).
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

# The same hold with a jointless (world-fixed) bar.
_FIXED_BAR_XML = _HELD_XML.replace("  <freejoint/>\n", "")

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


def _mod():
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.pack_audit.adapters import mujoco_bar_hold

    return mujoco_bar_hold


def _assert_static_hold(result: dict) -> None:
    assert result["available"], result["reason"]
    assert result["split_method"] == "efc_force"
    assert result["relative_error"] <= _WEIGHT_REL_TOL, result
    assert abs(result["split_left_fraction"] - 0.5) <= _SPLIT_ABS_TOL, result
    assert result["bar_linear_accel_mps2"] < _REST_ACCEL_MPS2, result
    # Sign (ADR-0052): the wrench is exerted by the hand ON the bar -> up.
    assert result["hand_force_n"]["L"][2] > 0.0
    assert result["hand_force_n"]["R"][2] > 0.0


@pytest.mark.requires_mujoco
def test_static_hold_balances_weight_and_splits_evenly() -> None:
    result = _mod().bar_hold_wrench(_HELD_XML, _HELD_WELDS)
    assert result["bar_mass_kg"] == pytest.approx(_MASS_KG)
    assert result["bar_weight_n"] == pytest.approx(_MASS_KG * _G)
    _assert_static_hold(result)


@pytest.mark.requires_mujoco
def test_non_hand_weld_is_unavailable_not_zero() -> None:
    result = _mod().bar_hold_wrench(_TORSO_XML, _TORSO_WELDS)
    assert result["available"] is False
    assert "torso" in result["reason"]
    for field in _UNAVAILABLE_NUMERIC:
        assert result[field] is None, f"{field} must be None, never 0"


@pytest.mark.requires_mujoco
def test_world_fixed_bar_is_unavailable_not_zero() -> None:
    result = _mod().bar_hold_wrench(_FIXED_BAR_XML, _HELD_WELDS)
    assert result["available"] is False
    assert "no joint" in result["reason"]
    for field in _UNAVAILABLE_NUMERIC:
        assert result[field] is None, f"{field} must be None, never 0"


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
    result = _StubAdapter(pack, "deadlift", Anthropometry()).bar_hold_wrench()

    assert result["available"] is False
    assert result["reason"] == "not implemented for this engine"
    for field in (*_UNAVAILABLE_NUMERIC, "split_method", "method", "n_welds"):
        assert result[field] is None, f"{field} must be None, never 0"


def _mujoco_lift_adapter(lift: str):
    pytest.importorskip("mujoco")
    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    pack = locate_pack("mujoco")
    if pack is None:
        pytest.skip("MuJoCo lift pack checkout not found (set LIFT_PACK_ROOT)")
    return create_adapter(pack, lift, Anthropometry())


@pytest.mark.requires_mujoco
@pytest.mark.integration
@pytest.mark.parametrize(
    "lift", ("deadlift", "bench_press", "snatch", "clean_and_jerk")
)
def test_pack_hand_held_lift_static_hold(lift: str) -> None:
    adapter = _mujoco_lift_adapter(lift)
    result = adapter.bar_hold_wrench()
    assert result["bar_mass_kg"] == pytest.approx(
        adapter.anthro.bar_total_mass_kg, rel=1e-6
    )
    _assert_static_hold(result)


@pytest.mark.requires_mujoco
@pytest.mark.integration
def test_pack_squat_is_unavailable_bar_on_torso() -> None:
    result = _mujoco_lift_adapter("squat").bar_hold_wrench()
    assert result["available"] is False
    assert result["bar_mass_kg"] is None
    assert "torso" in result["reason"]
