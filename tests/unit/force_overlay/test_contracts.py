"""Unit tests for force/torque overlay contracts (ADR-0052, #11286)."""

from __future__ import annotations

import math
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


def test_overlay_wrench_construction_and_validation():
    # Both halves None raises ValueError
    with pytest.raises(ValueError, match="At least one of force_n or torque_nm"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:ground_left",
            body="foot_left",
            point_m=(0.0, 0.0, 0.0),
            force_n=None,
            torque_nm=None,
            source="mujoco",
        )

    # Valid torque-only wrench
    w_torque = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:hip_flexion",
        body="femur_r",
        point_m=(0.1, 0.2, 0.3),
        force_n=None,
        torque_nm=(0.0, 10.5, 0.0),
        source="pinocchio",
    )
    assert w_torque.force_n is None
    assert w_torque.torque_nm == (0.0, 10.5, 0.0)

    # Valid force-only wrench
    w_force = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:heel_r",
        body="calcn_r",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 500.0),
        torque_nm=None,
        source="drake",
    )
    assert w_force.force_n == (0.0, 0.0, 500.0)
    assert w_force.torque_nm is None

    # Valid both halves
    w_both = OverlayWrench(
        kind=WrenchKind.GRIP,
        label="grip:lead_hand",
        body="hand_lead",
        point_m=(0.2, 0.1, 1.2),
        force_n=(10.0, -5.0, 20.0),
        torque_nm=(1.0, 2.0, -0.5),
        source="simscape",
    )
    assert w_both.force_n == (10.0, -5.0, 20.0)
    assert w_both.torque_nm == (1.0, 2.0, -0.5)


def test_overlay_wrench_invalid_inputs():
    # Invalid label regex: must match ^[a-z_]+:[A-Za-z0-9_.:-]+$
    with pytest.raises(ValueError, match="label"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="INVALID_LABEL",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 2.0, 3.0),
            torque_nm=None,
            source="test",
        )

    # Empty body
    with pytest.raises(ValueError, match="body"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:heel",
            body="",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 2.0, 3.0),
            torque_nm=None,
            source="test",
        )

    # Empty source
    with pytest.raises(ValueError, match="source"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:heel",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 2.0, 3.0),
            torque_nm=None,
            source="",
        )

    # Non-finite coordinates rejected via validate_vec3
    with pytest.raises(ValueError, match="finite"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:heel",
            body="foot",
            point_m=(0.0, float("nan"), 0.0),
            force_n=(1.0, 2.0, 3.0),
            torque_nm=None,
            source="test",
        )

    with pytest.raises(ValueError, match="finite"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:heel",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=(float("inf"), 2.0, 3.0),
            torque_nm=None,
            source="test",
        )


def test_overlay_wrench_to_spatial_wrench():
    w_both = OverlayWrench(
        kind=WrenchKind.GRIP,
        label="grip:lead_hand",
        body="hand_lead",
        point_m=(0.2, 0.1, 1.2),
        force_n=(10.0, -5.0, 20.0),
        torque_nm=(1.0, 2.0, -0.5),
        source="simscape",
    )
    sw = w_both.to_spatial_wrench()
    assert isinstance(sw, SpatialWrench)
    assert sw.application_frame == "world"
    assert sw.direction_convention == "applied_to_body"
    assert sw.point_m == (0.2, 0.1, 1.2)
    assert sw.force_n == (10.0, -5.0, 20.0)
    assert sw.torque_nm == (1.0, 2.0, -0.5)

    w_torque_only = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:hip",
        body="femur_r",
        point_m=(0.0, 0.0, 0.0),
        force_n=None,
        torque_nm=(0.0, 10.0, 0.0),
        source="test",
    )
    with pytest.raises(ValueError, match="force_n"):
        w_torque_only.to_spatial_wrench()

    w_force_only = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:heel",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        torque_nm=None,
        source="test",
    )
    with pytest.raises(ValueError, match="torque_nm"):
        w_force_only.to_spatial_wrench()


def test_force_torque_frame_validation_and_methods():
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:heel_r",
        body="foot_r",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 400.0),
        torque_nm=None,
        source="engine",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:knee_r",
        body="tibia_r",
        point_m=(0.0, 0.0, 0.4),
        force_n=None,
        torque_nm=(50.0, 0.0, 0.0),
        source="engine",
    )

    frame = ForceTorqueFrame(
        time_s=1.5,
        engine="mujoco",
        wrenches=(w1, w2),
        axial_loads=None,
    )
    assert frame.time_s == 1.5
    assert frame.engine == "mujoco"
    assert frame.world_frame == "world_Zup"
    assert frame.units == {"force": "N", "torque": "N*m", "length": "m"}
    assert frame.by_kind(WrenchKind.CONTACT) == (w1,)
    assert frame.by_kind(WrenchKind.JOINT_ACTUATOR) == (w2,)
    assert frame.by_kind(WrenchKind.MUSCLE) == ()

    # Duplicate labels rejected
    w2_dup = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:heel_r",
        body="foot_l",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 400.0),
        torque_nm=None,
        source="engine",
    )
    with pytest.raises(ValueError, match="Duplicate wrench label.*contact:heel_r"):
        ForceTorqueFrame(
            time_s=1.5,
            engine="mujoco",
            wrenches=(w1, w2_dup),
        )

    # Axial loads time mismatch rejected
    axial = AxialLoadFrame(time_s=1.0, values_n={"pelvis": 200.0}, source="engine")
    with pytest.raises(ValueError, match="axial_loads time_s"):
        ForceTorqueFrame(
            time_s=1.5,
            engine="mujoco",
            wrenches=(w1,),
            axial_loads=axial,
        )

    # Matching axial loads accepted
    axial_matching = AxialLoadFrame(
        time_s=1.5, values_n={"pelvis": 200.0}, source="engine"
    )
    frame_axial = ForceTorqueFrame(
        time_s=1.5,
        engine="mujoco",
        wrenches=(w1,),
        axial_loads=axial_matching,
    )
    assert frame_axial.axial_loads is not None
    assert frame_axial.axial_loads.values_n["pelvis"] == 200.0


def test_force_torque_frame_dict_roundtrip():
    w_torque = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:hip_flexion",
        body="femur_r",
        point_m=(0.1, 0.2, 0.3),
        force_n=None,
        torque_nm=(0.0, 10.5, 0.0),
        source="pinocchio",
    )
    axial = AxialLoadFrame(
        time_s=0.5, values_n={"femur_r": -120.0, "tibia_r": None}, source="sim"
    )
    frame = ForceTorqueFrame(
        time_s=0.5,
        engine="pinocchio",
        wrenches=(w_torque,),
        axial_loads=axial,
    )

    data = frame.to_dict()
    assert data["schema_version"] == "force-torque-frame-v1"
    assert data["time_s"] == 0.5
    assert data["engine"] == "pinocchio"
    assert data["wrenches"][0]["force_n"] is None  # serializes as null
    assert data["wrenches"][0]["torque_nm"] == [0.0, 10.5, 0.0]
    assert data["axial_loads"]["values_n"]["femur_r"] == -120.0
    assert data["axial_loads"]["values_n"]["tibia_r"] is None

    restored = ForceTorqueFrame.from_dict(data)
    assert restored.time_s == frame.time_s
    assert restored.engine == frame.engine
    assert len(restored.wrenches) == 1
    assert restored.wrenches[0].force_n is None
    assert restored.wrenches[0].torque_nm == (0.0, 10.5, 0.0)
    assert restored.axial_loads is not None
    assert restored.axial_loads.values_n["femur_r"] == -120.0
    assert restored.axial_loads.values_n["tibia_r"] is None

    # Unknown key rejected
    bad_data = dict(data)
    bad_data["unknown_key"] = 123
    with pytest.raises(ValueError, match="unknown keys"):
        ForceTorqueFrame.from_dict(bad_data)

    # Wrong schema version rejected
    bad_ver = dict(data)
    bad_ver["schema_version"] = "force-torque-frame-v2"
    with pytest.raises(ValueError, match="schema_version"):
        ForceTorqueFrame.from_dict(bad_ver)


def test_force_torque_provider_and_reader():
    class SyntheticProvider:
        def __init__(self, frame: ForceTorqueFrame | None = None) -> None:
            self._frame = frame

        def get_force_torque_frame(self) -> ForceTorqueFrame | None:
            return self._frame

    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:test",
        body="body",
        point_m=(0.0, 0.0, 0.0),
        force_n=(1.0, 0.0, 0.0),
        torque_nm=None,
        source="synthetic",
    )
    frame = ForceTorqueFrame(time_s=2.0, engine="test_engine", wrenches=(w,))
    provider = SyntheticProvider(frame)

    assert isinstance(provider, ForceTorqueProvider)
    assert read_force_torque_frame(provider, time_s=2.0) == frame
    assert read_force_torque_frame(provider, time_s=2.0001) is None
    assert read_force_torque_frame(provider, time_s=None) == frame

    empty_provider = SyntheticProvider(None)
    assert read_force_torque_frame(empty_provider, time_s=2.0) is None
    assert read_force_torque_frame("not_a_provider", time_s=2.0) is None

    class BadProvider:
        def get_force_torque_frame(self) -> str:
            return "not_a_frame"

    with pytest.raises(TypeError, match="ForceTorqueFrame or None"):
        read_force_torque_frame(BadProvider(), time_s=2.0)

    # Non-finite time_s raises ValueError
    with pytest.raises(ValueError, match="time_s must be finite"):
        read_force_torque_frame(provider, time_s=float("nan"))


def test_contracts_edge_cases():
    # String kind conversion
    w = OverlayWrench(
        kind="contact",
        label="contact:test",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 1.0),
        source="test",
    )
    assert w.kind == WrenchKind.CONTACT

    with pytest.raises(ValueError, match="Invalid WrenchKind"):
        OverlayWrench(
            kind="nonexistent_kind",
            label="contact:test",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=(0.0, 0.0, 1.0),
            source="test",
        )

    with pytest.raises(TypeError, match="WrenchKind"):
        OverlayWrench(
            kind=123,  # type: ignore[arg-type]
            label="contact:test",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=(0.0, 0.0, 1.0),
            source="test",
        )

    # OverlayWrench.from_dict unknown keys
    with pytest.raises(ValueError, match="unknown keys"):
        OverlayWrench.from_dict({"extra": 1})

    # ForceTorqueFrame invalid time_s
    with pytest.raises(ValueError, match="time_s must be finite"):
        ForceTorqueFrame(time_s=float("inf"), engine="test")

    # ForceTorqueFrame invalid engine
    with pytest.raises(ValueError, match="engine must be a non-empty string"):
        ForceTorqueFrame(time_s=0.0, engine="")

    # ForceTorqueFrame invalid wrench element
    with pytest.raises(TypeError, match="All wrenches must be OverlayWrench"):
        ForceTorqueFrame(time_s=0.0, engine="test", wrenches=("not_a_wrench",))  # type: ignore[arg-type]

    # ForceTorqueFrame invalid axial_loads type
    with pytest.raises(TypeError, match="axial_loads must be AxialLoadFrame"):
        ForceTorqueFrame(time_s=0.0, engine="test", axial_loads="not_axial_frame")  # type: ignore[arg-type]

    # ForceTorqueFrame invalid units type
    with pytest.raises(TypeError, match="units must be a Mapping"):
        ForceTorqueFrame(time_s=0.0, engine="test", units=["not", "a", "map"])  # type: ignore[arg-type]
