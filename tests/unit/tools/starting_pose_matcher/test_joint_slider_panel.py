"""Coverage for the joint-slider + rigid-transform widget (issue #4706).

Mirrors the import-guard pattern in ``test_data_sources_panel.py`` so
the suite runs cleanly on the Windows / Python 3.14 box where PyQt6
DLLs sometimes refuse to coexist with PySide6.
"""

from __future__ import annotations

import os

# Headless Qt platform must be set before any PyQt6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

if "PySide6" in sys.modules:
    pytest.skip(
        "PySide6 already loaded — PyQt6 DLLs unavailable", allow_module_level=True
    )

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    _HAVE_QT = True
except Exception:  # noqa: BLE001
    _HAVE_QT = False

if not _HAVE_QT:  # pragma: no cover - environment-dependent
    pytest.skip("PyQt6.QtWidgets unavailable", allow_module_level=True)


import math  # noqa: E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.starting_pose_matcher.widgets.joint_slider_panel import (  # noqa: E402
    DEFAULT_JOINT_COORDS,
    RIGID_COORDS,
    JointSliderPanel,
    PoseState,
    resolve_coord_names,
)


pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# --------------------------------------------------------------------------- #
# Pure helpers — no Qt instance required                                      #
# --------------------------------------------------------------------------- #


def test_default_coords_includes_canonical_joints() -> None:
    """Sanity check that the fallback list covers Drake's 23-coord set."""
    assert len(DEFAULT_JOINT_COORDS) == 23
    assert "spine_yaw" in DEFAULT_JOINT_COORDS
    assert "club_grip" in DEFAULT_JOINT_COORDS


def test_resolve_coord_names_falls_back_when_provider_is_none() -> None:
    assert resolve_coord_names(None) == DEFAULT_JOINT_COORDS


def test_resolve_coord_names_uses_provider_method() -> None:
    class FakeProvider:
        def coord_names(self) -> tuple[str, ...]:
            return ("a", "b", "c")

    assert resolve_coord_names(FakeProvider()) == ("a", "b", "c")


def test_resolve_coord_names_recovers_from_exception() -> None:
    class BrokenProvider:
        def coord_names(self) -> tuple[str, ...]:
            raise RuntimeError("kaboom")

    assert resolve_coord_names(BrokenProvider()) == DEFAULT_JOINT_COORDS


def test_pose_state_round_trip() -> None:
    state = PoseState(joint_angles={"a": 0.5, "b": -0.25}, tx=0.1, rz=math.pi / 4)
    blob = state.to_dict()
    decoded = PoseState.from_dict(blob, coord_names=("a", "b"))
    assert decoded.joint_angles["a"] == pytest.approx(0.5)
    assert decoded.joint_angles["b"] == pytest.approx(-0.25)
    assert decoded.tx == pytest.approx(0.1)
    assert decoded.rz == pytest.approx(math.pi / 4)


def test_pose_state_from_dict_drops_unknown_joints() -> None:
    blob = {"joint_angles": {"a": 1.0, "stranger": 9.9}}
    decoded = PoseState.from_dict(blob, coord_names=("a", "b"))
    assert "stranger" not in decoded.joint_angles
    assert decoded.joint_angles["b"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Widget-level tests — require a QApplication                                 #
# --------------------------------------------------------------------------- #


def test_widget_instantiates_with_defaults(qapp: QApplication) -> None:
    panel = JointSliderPanel()
    assert panel.coord_names == DEFAULT_JOINT_COORDS
    state = panel.pose_state()
    # Midpoint sliders -> all zeros (within rounding).
    for name in DEFAULT_JOINT_COORDS:
        assert state.joint_angles[name] == pytest.approx(0.0, abs=1e-2)
    for name in RIGID_COORDS:
        assert getattr(state, name) == pytest.approx(0.0, abs=1e-2)


def test_widget_rejects_empty_coord_list(qapp: QApplication) -> None:
    with pytest.raises(ValueError):
        JointSliderPanel(coord_names=())


def test_joint_slider_emits_pose_changed(qapp: QApplication) -> None:
    panel = JointSliderPanel(coord_names=("hip", "knee"))
    received: list[PoseState] = []
    panel.pose_changed.connect(lambda s: received.append(s))

    panel._joint_sliders["hip"].setValue(0)  # min -> -pi
    assert received, "pose_changed never fired"
    assert received[-1].joint_angles["hip"] == pytest.approx(-math.pi, abs=1e-2)
    # Other joint untouched.
    assert received[-1].joint_angles["knee"] == pytest.approx(0.0, abs=1e-2)


def test_rigid_sliders_update_independently(qapp: QApplication) -> None:
    panel = JointSliderPanel(coord_names=("a",))
    received: list[PoseState] = []
    panel.pose_changed.connect(lambda s: received.append(s))

    panel._rigid_sliders["tx"].setValue(1000)  # max -> +2 m
    panel._rigid_sliders["rz"].setValue(0)  # min -> -pi rad

    assert len(received) >= 2
    final = received[-1]
    assert final.tx == pytest.approx(2.0, abs=1e-2)
    assert final.rz == pytest.approx(-math.pi, abs=1e-2)
    # Joints stay at zero.
    assert final.joint_angles["a"] == pytest.approx(0.0, abs=1e-2)
    # ty / tz / rx / ry untouched.
    assert final.ty == pytest.approx(0.0, abs=1e-2)
    assert final.rx == pytest.approx(0.0, abs=1e-2)


def test_reset_button_restores_defaults(qapp: QApplication) -> None:
    panel = JointSliderPanel(coord_names=("hip",))
    panel._joint_sliders["hip"].setValue(0)
    panel._rigid_sliders["tx"].setValue(1000)
    assert panel.pose_state().tx > 1.0  # confirm the move took

    received: list[PoseState] = []
    panel.pose_changed.connect(lambda s: received.append(s))
    panel.btn_reset.click()

    state = panel.pose_state()
    assert state.joint_angles["hip"] == pytest.approx(0.0, abs=1e-2)
    assert state.tx == pytest.approx(0.0, abs=1e-2)
    # Reset emits exactly one pose_changed signal (batched).
    assert len(received) == 1


def test_set_pose_state_round_trips_through_widget(qapp: QApplication) -> None:
    panel = JointSliderPanel(coord_names=("hip", "knee"))
    target = PoseState(joint_angles={"hip": 0.5, "knee": -0.5}, tx=0.25, ry=0.1)
    panel.set_pose_state(target)
    state = panel.pose_state()
    assert state.joint_angles["hip"] == pytest.approx(0.5, abs=1e-2)
    assert state.joint_angles["knee"] == pytest.approx(-0.5, abs=1e-2)
    assert state.tx == pytest.approx(0.25, abs=1e-2)
    assert state.ry == pytest.approx(0.1, abs=1e-2)
