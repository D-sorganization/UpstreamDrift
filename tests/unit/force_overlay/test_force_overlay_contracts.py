"""Unit tests for force_overlay data contracts (WrenchKind, OverlayWrench, ForceTorqueFrame, ForceTorqueProvider)."""

from __future__ import annotations

import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    ForceTorqueProvider,
    OverlayWrench,
    WrenchKind,
    read_force_torque_frame,
)
from src.shared.python.motion_matching.force_torque import SpatialWrench

pytestmark = pytest.mark.unit


def test_wrench_kind_values():
    assert WrenchKind.JOINT_ACTUATOR.value == "joint_actuator"
    assert WrenchKind.JOINT_REACTION.value == "joint_reaction"
    assert WrenchKind.CONTACT.value == "contact"
    assert WrenchKind.GRIP.value == "grip"
    assert WrenchKind.EXTERNAL.value == "external"
    assert WrenchKind.GRAVITY.value == "gravity"
    assert WrenchKind.MUSCLE.value == "muscle"


def test_overlay_wrench_torque_only():
    w = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:elbow",
        body="forearm",
        point_m=(0.0, 0.2, 0.5),
        force_n=None,
        torque_nm=(1.0, 2.0, 3.0),
        source="engine:test",
    )
    assert w.force_n is None
    assert w.torque_nm == (1.0, 2.0, 3.0)
    assert w.direction_convention == "applied_to_body"
    assert w.world_frame == "world_Zup"

    # Round trip via dict
    d = w.to_dict()
    assert d["force_n"] is None
    assert d["torque_nm"] == [1.0, 2.0, 3.0]
    restored = OverlayWrench.from_dict(d)
    assert restored == w
    assert restored.force_n is None


def test_overlay_wrench_force_only():
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot_0",
        body="foot",
        point_m=(0.1, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        torque_nm=None,
        source="engine:test",
    )
    assert w.force_n == (0.0, 0.0, 100.0)
    assert w.torque_nm is None

    d = w.to_dict()
    assert d["force_n"] == [0.0, 0.0, 100.0]
    assert d["torque_nm"] is None
    restored = OverlayWrench.from_dict(d)
    assert restored == w


def test_overlay_wrench_both_halves_none_rejected():
    with pytest.raises(ValueError, match="At least one of force_n or torque_nm"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:test",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=None,
            torque_nm=None,
            source="test",
        )


def test_overlay_wrench_label_validation():
    # Valid label patterns
    w = OverlayWrench(
        kind=WrenchKind.EXTERNAL,
        label="ext:payload_1.a-b",
        body="hand",
        point_m=(0.0, 0.0, 0.0),
        force_n=(1.0, 0.0, 0.0),
        source="test",
    )
    assert w.label == "ext:payload_1.a-b"

    # Invalid labels
    with pytest.raises(ValueError, match="label"):
        OverlayWrench(
            kind=WrenchKind.EXTERNAL,
            label="InvalidLabelNoColon",
            body="hand",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 0.0, 0.0),
            source="test",
        )


def test_overlay_wrench_non_finite_rejected():
    with pytest.raises(ValueError, match="must be finite"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:foot",
            body="foot",
            point_m=(float("nan"), 0.0, 0.0),
            force_n=(0.0, 0.0, 1.0),
            source="test",
        )
    with pytest.raises(ValueError, match="must be finite"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:foot",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=(0.0, float("inf"), 1.0),
            source="test",
        )


def test_overlay_wrench_to_spatial_wrench():
    # Torque-only cannot be converted to SpatialWrench
    w_torque = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:wrist",
        body="hand",
        point_m=(0.0, 0.0, 0.0),
        torque_nm=(0.0, 5.0, 0.0),
        source="test",
    )
    with pytest.raises(ValueError, match="missing force_n"):
        w_torque.to_spatial_wrench()

    # Full wrench can be converted
    w_full = OverlayWrench(
        kind=WrenchKind.GRIP,
        label="grip:club",
        body="handle",
        point_m=(0.1, 0.2, 0.3),
        force_n=(10.0, 20.0, 30.0),
        torque_nm=(1.0, 2.0, 3.0),
        source="test",
    )
    sw = w_full.to_spatial_wrench()
    assert isinstance(sw, SpatialWrench)
    assert sw.application_frame == "world"
    assert sw.direction_convention == "applied_to_body"
    assert sw.point_m == (0.1, 0.2, 0.3)
    assert sw.force_n == (10.0, 20.0, 30.0)
    assert sw.torque_nm == (1.0, 2.0, 3.0)


def test_overlay_wrench_dict_rejects_unknown_keys():
    d = {
        "kind": "contact",
        "label": "contact:foot",
        "body": "foot",
        "point_m": [0.0, 0.0, 0.0],
        "force_n": [0.0, 0.0, 1.0],
        "torque_nm": None,
        "source": "test",
        "unexpected_key": 123,
    }
    with pytest.raises(ValueError, match="Unknown keys"):
        OverlayWrench.from_dict(d)


def test_force_torque_frame_duplicate_labels_rejected():
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 1.0),
        source="test",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.EXTERNAL,
        label="contact:foot",  # Duplicate!
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 2.0),
        source="test",
    )
    with pytest.raises(ValueError, match="Duplicate wrench labels"):
        ForceTorqueFrame(
            time_s=0.1,
            engine="test_engine",
            wrenches=(w1, w2),
        )


def test_force_torque_frame_axial_load_time_mismatch_rejected():
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 1.0),
        source="test",
    )
    al = AxialLoadFrame(time_s=0.5, values_n={"shaft": 10.0}, source="test")
    with pytest.raises(ValueError, match="time_s"):
        ForceTorqueFrame(
            time_s=0.1,  # mismatch with 0.5
            engine="test_engine",
            wrenches=(w,),
            axial_loads=al,
        )


def test_force_torque_frame_round_trip_and_version():
    w = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:wrist",
        body="hand",
        point_m=(0.1, 0.2, 0.3),
        torque_nm=(0.0, 5.0, 0.0),
        source="test",
    )
    al = AxialLoadFrame(time_s=0.1, values_n={"shaft": 15.0}, source="test")
    frame = ForceTorqueFrame(
        time_s=0.1,
        engine="mujoco",
        wrenches=(w,),
        axial_loads=al,
    )
    d = frame.to_dict()
    assert d["schema_version"] == "force-torque-frame-v1"
    assert d["units"] == {"force": "N", "torque": "N*m", "length": "m"}
    assert d["world_frame"] == "world_Zup"

    restored = ForceTorqueFrame.from_dict(d)
    assert restored.time_s == frame.time_s
    assert restored.engine == frame.engine
    assert restored.wrenches == frame.wrenches
    assert restored.axial_loads.values_n == frame.axial_loads.values_n

    # Wrong schema version rejected
    d_bad_version = dict(d)
    d_bad_version["schema_version"] = "wrong-v2"
    with pytest.raises(ValueError, match="schema_version"):
        ForceTorqueFrame.from_dict(d_bad_version)

    # Unknown key rejected
    d_unknown = dict(d)
    d_unknown["bad_key"] = "bad"
    with pytest.raises(ValueError, match="Unknown keys"):
        ForceTorqueFrame.from_dict(d_unknown)


def test_force_torque_frame_by_kind():
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 1.0),
        source="test",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:wrist",
        body="hand",
        point_m=(0.0, 0.0, 0.0),
        torque_nm=(0.0, 1.0, 0.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w1, w2))
    assert frame.by_kind(WrenchKind.CONTACT) == (w1,)
    assert frame.by_kind("joint_actuator") == (w2,)
    assert frame.by_kind(WrenchKind.EXTERNAL) == ()


def test_force_torque_provider_protocol_and_reader():
    class synthetic_Provider:
        def __init__(self, frame: ForceTorqueFrame | None):
            self._frame = frame

        def get_force_torque_frame(self) -> ForceTorqueFrame | None:
            return self._frame

    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.2, engine="test", wrenches=(w,))
    provider = synthetic_Provider(frame)

    assert isinstance(provider, ForceTorqueProvider)

    # Read with exact time match
    read_dict = read_force_torque_frame(provider, time_s=0.2)
    assert read_dict is not None
    assert read_dict["time_s"] == 0.2

    # Read with time mismatch returns None
    assert read_force_torque_frame(provider, time_s=0.5) is None

    # Read from non-provider returns None
    assert read_force_torque_frame("not_a_provider", time_s=0.2) is None

    # Provider returning None
    none_provider = synthetic_Provider(None)
    assert read_force_torque_frame(none_provider) is None

    # Non-finite time_s raises ValueError
    with pytest.raises(ValueError, match="time_s must be finite"):
        read_force_torque_frame(provider, time_s=float("nan"))

    # Provider returning wrong type raises TypeError
    class BadProvider:
        def get_force_torque_frame(self):
            return "not_a_frame"

    with pytest.raises(TypeError, match="must return ForceTorqueFrame or None"):
        read_force_torque_frame(BadProvider())


def test_overlay_wrench_validation_edge_cases():
    # String kind coercion
    w = OverlayWrench(
        kind="joint_actuator",
        label="joint:test",
        body="torso",
        point_m=(0.0, 0.0, 0.0),
        force_n=(1.0, 0.0, 0.0),
        source="test",
    )
    assert w.kind == WrenchKind.JOINT_ACTUATOR

    # Empty body
    with pytest.raises(ValueError, match="body must be a non-empty string"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:test",
            body="",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 0.0, 0.0),
            source="test",
        )

    # Empty source
    with pytest.raises(ValueError, match="source must be a non-empty string"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:test",
            body="hand",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 0.0, 0.0),
            source="",
        )


def test_force_torque_frame_validation_edge_cases():
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:test",
        body="hand",
        point_m=(0.0, 0.0, 0.0),
        force_n=(1.0, 0.0, 0.0),
        source="test",
    )

    # Non-finite time
    with pytest.raises(ValueError, match="time_s must be a finite number"):
        ForceTorqueFrame(time_s=float("nan"), engine="test", wrenches=(w,))

    # Empty engine
    with pytest.raises(ValueError, match="engine must be a non-empty string"):
        ForceTorqueFrame(time_s=0.0, engine="", wrenches=(w,))

    # Invalid axial_loads type
    with pytest.raises(TypeError, match="axial_loads must be an AxialLoadFrame"):
        ForceTorqueFrame(
            time_s=0.0, engine="test", wrenches=(w,), axial_loads={"invalid": 1}
        )
