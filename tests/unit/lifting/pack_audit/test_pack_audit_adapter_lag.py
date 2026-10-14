"""Pack-audit adapters must track the packs (UpstreamDrift#12504, LIFT-3 #11743).

Three false gaps came from adapters reading the old pack structure:

1. ``grip_attachment``: a right hand welded to the bar as a leaf (Drake's
   ``biomech:weld`` of ``hand_r`` to ``barbell_shaft``) was not recognised.
2. ``grip_width``: spacing was measured at the neutral pose instead of the
   start pose where the packs close their grips.
3. ``no_grf``: fixed "unavailable" strings cited closed pack issues instead
   of asking the pack's contact API.

Everything here uses small fakes; no engine is needed.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.shared.python.lifting.pack_audit import analysis as an
from src.shared.python.lifting.pack_audit.adapters import _common
from src.shared.python.lifting.pack_audit.adapters.drake_adapter import DrakeAdapter
from src.shared.python.lifting.pack_audit.adapters.pinocchio_adapter import (
    PinocchioAdapter,
)

pytestmark = pytest.mark.unit


# --- 1. hand/bar weld recognition (shared helper) ---------------------------


@pytest.mark.parametrize(
    ("a", "b", "side"),
    [
        ("hand_l", "barbell_shaft", "l"),
        ("barbell_shaft", "hand_l", "l"),
        ("hand_r", "barbell_shaft", "r"),
        ("barbell_shaft", "hand_r", "r"),
        ("barbell_grip_r", "barbell_shaft", None),  # dummy frame only
        ("barbell_shaft", "barbell_grip_r", None),
        ("forearm_r", "hand_r", None),
    ],
)
def test_hand_bar_weld_side_is_orientation_agnostic(a, b, side):
    assert _common.hand_bar_weld_side(a, b) == side


def test_hand_bar_weld_side_rejects_non_str():
    with pytest.raises(TypeError):
        _common.hand_bar_weld_side("hand_r", None)  # type: ignore[arg-type]


def test_leaf_weld_hand_r_to_bar_is_attached_as_joint_or_constraint():
    as_joint = _common.hand_attachment(
        [
            ("barbell_to_hand_l", "fixed", "hand_l", "barbell_shaft"),
            ("barbell_to_hand_r", "fixed", "barbell_shaft", "hand_r"),
        ]
    )
    assert as_joint["l"].startswith("fixed joint")
    assert as_joint["r"].startswith("fixed joint")
    as_weld = _common.hand_attachment(
        [("barbell_to_hand_l", "fixed", "hand_l", "barbell_shaft")],
        welds=[("hand_r", "barbell_shaft")],
    )
    assert as_weld["r"] == "weld constraint (loop closure)"


def test_dummy_grip_frame_alone_is_not_attached():
    out = _common.hand_attachment(
        [
            ("barbell_to_hand_l", "fixed", "hand_l", "barbell_shaft"),
            ("barbell_to_hand_r", "fixed", "barbell_shaft", "barbell_grip_r"),
        ]
    )
    assert out["r"] == "none (only dummy frame barbell_grip_r)"


class _FakeJoint:
    def __init__(self, name: str, kind: str, parent: str, child: str) -> None:
        self._v = (name, kind, parent, child)

    def name(self) -> str:
        return self._v[0]

    def type_name(self) -> str:
        return self._v[1]

    def parent_body(self) -> SimpleNamespace:
        return SimpleNamespace(name=lambda: self._v[2])

    def child_body(self) -> SimpleNamespace:
        return SimpleNamespace(name=lambda: self._v[3])


def _drake(joints: list[_FakeJoint], welds: list[tuple[str, str]]) -> DrakeAdapter:
    adapter = object.__new__(DrakeAdapter)
    adapter.plant = SimpleNamespace(
        GetJointIndices=lambda _inst: range(len(joints)),
        get_joint=lambda i: joints[i],
    )
    adapter.loaded = SimpleNamespace(
        model_instance=0,
        weld_specs=tuple(
            SimpleNamespace(name="w", parent=p, child=c) for p, c in welds
        ),
    )
    return adapter


def test_drake_right_hand_biomech_weld_is_attached():
    adapter = _drake(
        [_FakeJoint("barbell_to_left_hand", "fixed", "hand_l", "barbell_shaft")],
        welds=[("hand_r", "barbell_shaft")],
    )
    out = adapter._hand_attachment()
    assert not out["l"].startswith("none")
    assert not out["r"].startswith("none")


def test_drake_weld_with_hand_as_child_is_attached():
    adapter = _drake(
        [_FakeJoint("barbell_to_right_hand", "fixed", "barbell_shaft", "hand_r")],
        welds=[],
    )
    assert not adapter._hand_attachment()["r"].startswith("none")


def test_drake_without_any_right_weld_stays_unattached():
    adapter = _drake(
        [_FakeJoint("barbell_to_left_hand", "fixed", "hand_l", "barbell_shaft")],
        welds=[],
    )
    assert adapter._hand_attachment()["r"] == "none"


def _pinocchio(urdf_joints: str) -> PinocchioAdapter:
    adapter = object.__new__(PinocchioAdapter)
    adapter._root = ET.fromstring(f"<robot>{urdf_joints}</robot>")
    return adapter


def _urdf_joint(name: str, parent: str, child: str) -> str:
    return (
        f'<joint name="{name}" type="fixed"><parent link="{parent}"/>'
        f'<child link="{child}"/></joint>'
    )


def test_pinocchio_dummy_grip_frame_is_still_not_attached():
    adapter = _pinocchio(
        _urdf_joint("barbell_to_hand_l", "hand_l", "barbell_shaft")
        + _urdf_joint("barbell_to_hand_r", "barbell_shaft", "barbell_grip_r")
    )
    out = adapter._hand_attachment()
    assert out["l"].startswith("fixed joint")
    assert out["r"] == "none (only dummy frame barbell_grip_r)"


def test_pinocchio_leaf_hand_r_weld_is_attached_either_orientation():
    for parent, child in (("barbell_shaft", "hand_r"), ("hand_r", "barbell_shaft")):
        adapter = _pinocchio(
            _urdf_joint("barbell_to_hand_l", "hand_l", "barbell_shaft")
            + _urdf_joint("barbell_to_hand_r", parent, child)
        )
        assert adapter._hand_attachment()["r"].startswith("fixed joint")


# --- 2. grip width at the start pose ----------------------------------------


def _receipt(zero_half: float, start_half: float, intended: float = 0.2248) -> dict:
    def summary(half: float) -> dict[str, Any]:
        return {
            "grip_width_m": 2 * half,
            "hand_bar": {"l": {"lateral_m": -intended}},
        }

    res = {
        "same_q": {"zero": {"summary": summary(zero_half)}},
        "start": {
            "summary": summary(start_half),
            "closure_native": 0.0,
            "notes": [],
        },
        "structure": {"bar_hand_attachment": {"l": "weld", "r": "weld"}},
    }
    return {"results": {"deadlift": {"drake": res}}}


def test_grip_width_is_measured_at_the_start_pose_not_neutral():
    row = an.grip_rows(_receipt(zero_half=0.1709, start_half=0.2248))[0]
    assert row["hand_half_spacing_start_m"] == pytest.approx(0.2248)
    assert row["mismatch_m"] == pytest.approx(0.0)


def test_grip_width_still_flags_a_real_start_pose_mismatch():
    row = an.grip_rows(_receipt(zero_half=0.2248, start_half=0.1709))[0]
    assert abs(row["mismatch_m"]) > an.GRIP_TOL_M


def test_grip_tolerance_is_not_loosened():
    assert an.GRIP_TOL_M == 0.01


# --- 3. GRF from the pack's actual API --------------------------------------


@pytest.fixture
def fake_pack(monkeypatch):
    """Register a fake pack package; returns (pack, module-registrar)."""
    name = "fake_lift_pack"
    pkg = types.ModuleType(name)
    pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, pkg)

    def add(module: str, **attrs: Any) -> None:
        mod = types.ModuleType(f"{name}.{module}")
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, f"{name}.{module}", mod)

    return SimpleNamespace(package=name), add


def test_pack_api_detects_present_and_absent(fake_pack):
    pack, add = fake_pack
    assert _common.pack_api(pack.package, "contact_results", "f") is None
    add("contact_results", f=lambda: 1)
    assert _common.pack_api(pack.package, "contact_results", "f") is not None
    assert _common.pack_api(pack.package, "contact_results", "missing") is None


def test_contact_unavailable_demands_a_reason():
    with pytest.raises(ValueError):
        _common.contact_unavailable("")


def _drake_with_diagram(pack) -> DrakeAdapter:
    adapter = object.__new__(DrakeAdapter)
    adapter.pack = pack
    adapter.plant = object()
    diagram = SimpleNamespace(
        CreateDefaultContext=lambda: "root",
        GetMutableSubsystemContext=lambda _plant, _root: "plant_ctx",
    )
    adapter.loaded = SimpleNamespace(builder=SimpleNamespace(Build=lambda: diagram))
    return adapter


def test_drake_grf_comes_from_the_pack_contact_results_api(fake_pack):
    pack, add = fake_pack
    seen: list[Any] = []

    def measure(plant: Any, ctx: Any) -> SimpleNamespace:
        seen.append(ctx)
        feet = {"foot_l": SimpleNamespace(in_contact=True)}
        return SimpleNamespace(vertical_force=784.8, feet=feet)

    add("contact_results", measure_ground_reaction=measure)
    out = _drake_with_diagram(pack).start_contact_force()
    assert out["value_n"] == pytest.approx(784.8)
    assert out["reason"] is None
    assert seen == ["plant_ctx"]


def test_drake_pack_without_grf_api_reports_a_current_reason(fake_pack):
    pack, _add = fake_pack
    out = _drake_with_diagram(pack).start_contact_force()
    assert out["value_n"] is None
    assert "#371" not in out["reason"]
    assert pack.package in out["reason"]


def test_drake_grf_api_failure_is_reported_not_zero(fake_pack):
    pack, add = fake_pack

    def boom(_plant: Any, _ctx: Any) -> None:
        raise ValueError("plant has no body named 'ground_plane'")

    add("contact_results", measure_ground_reaction=boom)
    out = _drake_with_diagram(pack).start_contact_force()
    assert out["value_n"] is None
    assert "ground_plane" in out["reason"]


def _pinocchio_with_model(pack) -> PinocchioAdapter:
    adapter = object.__new__(PinocchioAdapter)
    adapter.pack = pack
    adapter.model = "model"
    adapter._initial_q = lambda: np.zeros(7)  # type: ignore[method-assign]
    return adapter


def test_pinocchio_grf_comes_from_the_pack_contact_dynamics_api(fake_pack):
    pack, add = fake_pack
    calls: list[Any] = []

    def grf(model: Any, q: Any) -> SimpleNamespace:
        calls.append((model, q.shape))
        return SimpleNamespace(
            total_force=np.array([0.0, 0.0, 784.8]),
            per_foot={
                "foot_l": SimpleNamespace(in_contact=True),
                "foot_r": SimpleNamespace(in_contact=False),
            },
        )

    add("contact_dynamics", ground_reaction_forces=grf)
    out = _pinocchio_with_model(pack).start_contact_force()
    assert out["value_n"] == pytest.approx(784.8)
    assert out["n_feet_in_contact"] == 1
    assert calls == [("model", (7,))]


def test_pinocchio_pack_without_grf_api_reports_a_current_reason(fake_pack):
    pack, _add = fake_pack
    out = _pinocchio_with_model(pack).start_contact_force()
    assert out["value_n"] is None
    assert "#432" not in out["reason"]
    assert pack.package in out["reason"]
