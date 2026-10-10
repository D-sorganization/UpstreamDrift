"""Static-hold per-hand bar wrench, Pinocchio slice (LIFT-4, #11744, GCV-7).

A rigid two-hand hold of a rigid bar is statically indeterminate, and the
Pinocchio pack URDF fuses the bar into the left hand's body (fixed joints),
so Pinocchio cannot supply the hand/hand split directly. The adapter instead
computes the bar's mass/COM from its own URDF ``<inertial>`` elements and
forms the net static-equilibrium wrench the hands must exert, split by the
shared ``allocate_min_norm`` (GCV-7). This is a kinematic calculation (no
simulation): ``bar_linear_accel_mps2`` is always ``None``.

Acceptance from #11744: the hand forces sum to the bar plus plate weight
within 1 %, and a symmetric grip splits symmetrically.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_G = 9.81
#: Symmetric grip -> even split.  Measured |share - 0.5| <= 2.8e-7 on every
#: hand-held lift of the Pinocchio pack (deadlift/bench_press 2.5e-7, snatch
#: 1.4e-7, clean_and_jerk 2.7e-7); the residual comes from the tiny
#: (1e-6 kg) virtual-link masses, not a modeling error.
_SPLIT_ABS_TOL = 1e-5
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
    pytest.importorskip("pinocchio")
    from src.shared.python.lifting.pack_audit.adapters import pinocchio_bar_hold

    return pinocchio_bar_hold


# The pure split is tested engine-free in test_bar_hold_static.py.


def test_link_mass_and_local_com_reads_urdf_inertial() -> None:
    mod = _mod()
    from defusedxml import ElementTree as ET

    link = ET.fromstring(
        '<link name="barbell_shaft">'
        '<inertial><origin xyz="0.01 0.02 0.03"/><mass value="20.0"/>'
        '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
        "</link>"
    )
    mass, com = mod._link_mass_and_local_com(link)
    assert mass == pytest.approx(20.0)
    assert np.allclose(com, (0.01, 0.02, 0.03))


def test_link_mass_and_local_com_handles_no_inertial() -> None:
    mod = _mod()
    from defusedxml import ElementTree as ET

    link = ET.fromstring('<link name="foot_l"/>')
    mass, com = mod._link_mass_and_local_com(link)
    assert mass == 0.0
    assert np.allclose(com, (0.0, 0.0, 0.0))


def test_vec3_rejects_wrong_length() -> None:
    mod = _mod()
    with pytest.raises(ValueError):
        mod._vec3("1.0 2.0")


# --------------------------------------------------------------------------
# bar_hold_wrench() input validation (duck-typed model/data; no real pack).
# --------------------------------------------------------------------------


def test_bar_hold_wrench_rejects_non_element_root() -> None:
    mod = _mod()

    class _FakeModel:
        nq = 1

    with pytest.raises(TypeError):
        mod.bar_hold_wrench(_FakeModel(), object(), object(), np.zeros(1))


def test_bar_hold_wrench_rejects_wrong_length_q() -> None:
    mod = _mod()
    from defusedxml import ElementTree as ET

    class _FakeModel:
        nq = 3

    root = ET.fromstring("<robot/>")
    with pytest.raises(TypeError):
        mod.bar_hold_wrench(_FakeModel(), object(), root, np.zeros(1))


def test_bar_hold_wrench_unavailable_when_right_grip_missing() -> None:
    """No ``barbell_grip_r`` link (e.g. squat welds the bar to the torso)."""
    mod = _mod()
    from defusedxml import ElementTree as ET

    class _FakeModel:
        nq = 1

    root = ET.fromstring(
        '<robot name="back_squat"><link name="torso"/>'
        '<link name="barbell_shaft"/></robot>'
    )
    result = mod.bar_hold_wrench(_FakeModel(), object(), root, np.zeros(1))

    assert result["available"] is False
    assert "torso" in result["reason"]
    for field in _UNAVAILABLE_NUMERIC:
        assert result[field] is None, f"{field} must be None, never 0"


# --------------------------------------------------------------------------
# Real-pack integration tests (require a Pinocchio_Models checkout AND the
# real `pinocchio` package -- only available inside the ud-sim container).
# --------------------------------------------------------------------------


def _pinocchio_lift_adapter(lift: str):
    pin = pytest.importorskip("pinocchio")
    if not hasattr(pin, "__file__"):
        # tests/unit/conftest.py installs a MagicMock when pinocchio is absent.
        pytest.skip("pinocchio in sys.modules is a test double, not the real package")
    from src.shared.python.lifting.pack_audit.adapters import create_adapter
    from src.shared.python.lifting.pack_audit.model import Anthropometry
    from src.shared.python.lifting.pack_audit.packs import locate_pack

    pack = locate_pack("pinocchio")
    if pack is None:
        pytest.skip("Pinocchio lift pack checkout not found (set LIFT_PACK_ROOT)")
    return create_adapter(pack, lift, Anthropometry())


@pytest.mark.requires_pinocchio
@pytest.mark.integration
@pytest.mark.parametrize(
    "lift", ("deadlift", "bench_press", "snatch", "clean_and_jerk")
)
def test_pack_hand_held_lift_static_hold(lift: str) -> None:
    adapter = _pinocchio_lift_adapter(lift)
    result = adapter.bar_hold_wrench()

    assert result["available"], result["reason"]
    assert result["split_method"] == "allocation"
    assert result["bar_mass_kg"] == pytest.approx(
        adapter.anthro.bar_total_mass_kg, rel=1e-6
    )
    assert result["relative_error"] <= 0.01, result
    assert abs(result["split_left_fraction"] - 0.5) <= _SPLIT_ABS_TOL, result
    assert result["bar_linear_accel_mps2"] is None
    # Sign (ADR-0052): the wrench is exerted by the hand ON the bar -> up.
    assert result["hand_force_n"]["L"][2] > 0.0
    assert result["hand_force_n"]["R"][2] > 0.0
    assert set(result["hand_grip_residual_m"]) == {"L", "R"}
    # grip_l IS the hand_l frame (see pinocchio_bar_hold module docstring):
    # the left residual is exactly 0 by construction, not a measurement.
    assert result["hand_grip_residual_m"]["L"] == 0.0
    r_residual = result["hand_grip_residual_m"]["R"]
    assert math.isfinite(r_residual) and 0.0 <= r_residual <= 1e-5, result


@pytest.mark.requires_pinocchio
@pytest.mark.integration
def test_pack_squat_is_unavailable_bar_on_torso() -> None:
    result = _pinocchio_lift_adapter("squat").bar_hold_wrench()

    assert result["available"] is False
    assert result["bar_mass_kg"] is None
    assert "torso" in result["reason"]
    for field in _UNAVAILABLE_NUMERIC:
        assert result[field] is None, f"{field} must be None, never 0"
