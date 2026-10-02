"""Unit tests for force_overlay contracts (ADR-0052, #11286)."""

from __future__ import annotations

import dataclasses
import math
import pytest

from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    ForceTorqueProvider,
    OverlayWrench,
    WrenchKind,
    read_force_torque_frame,
)
from src.shared.python.motion_matching.force_torque import SpatialWrench, validate_vec3

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_wrench_kind_enum_values() -> None:
    """WrenchKind must define all 7 required kind strings."""
    expected = {
        "joint_actuator",
        "joint_reaction",
        "contact",
        "grip",
        "external",
        "gravity",
        "muscle",
    }
    assert {k.value for k in WrenchKind} == expected
    assert WrenchKind.JOINT_ACTUATOR == "joint_actuator"
    assert WrenchKind.JOINT_REACTION == "joint_reaction"
    assert WrenchKind.CONTACT == "contact"
    assert WrenchKind.GRIP == "grip"
    assert WrenchKind.EXTERNAL == "external"
    assert WrenchKind.GRAVITY == "gravity"
    assert WrenchKind.MUSCLE == "muscle"


def test_validate_vec3_promotion() -> None:
    """validate_vec3 must validate length 3 and finite coordinates."""
    assert validate_vec3((1.0, 2.0, 3.0), "vec") == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="exactly 3"):
        validate_vec3((1.0, 2.0), "vec")
    with pytest.raises(ValueError, match="finite"):
        validate_vec3((1.0, float("nan"), 3.0), "vec")
    with pytest.raises(ValueError, match="finite"):
        validate_vec3((1.0, float("inf"), 3.0), "vec")


def test_overlay_wrench_creation_and_immutability() -> None:
    """OverlayWrench is a frozen dataclass with world frame and applied_to_body convention."""
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:right_foot:0",
        body="right_foot",
        point_m=(0.1, 0.2, 0.0),
        force_n=(0.0, 0.0, 500.0),
        torque_nm=(0.0, 0.0, 10.0),
        source="mujoco:contact",
    )
    assert w.kind == WrenchKind.CONTACT
    assert w.label == "contact:right_foot:0"
    assert w.body == "right_foot"
    assert w.point_m == (0.1, 0.2, 0.0)
    assert w.force_n == (0.0, 0.0, 500.0)
    assert w.torque_nm == (0.0, 0.0, 10.0)
    assert w.source == "mujoco:contact"
    assert w.APPLICATION_FRAME == "world"
    assert w.DIRECTION_CONVENTION == "applied_to_body"

    # Immutability check
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.body = "other_body"  # type: ignore[misc]


def test_overlay_wrench_optional_halves() -> None:
    """OverlayWrench accepts force-only or torque-only, but rejects both None."""
    # Force-only
    w_force = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        source="test",
    )
    assert w_force.force_n == (0.0, 0.0, 100.0)
    assert w_force.torque_nm is None

    # Torque-only
    w_torque = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:elbow",
        body="forearm",
        point_m=(0.1, 0.2, 0.3),
        torque_nm=(0.0, 5.0, 0.0),
        source="test",
    )
    assert w_torque.force_n is None
    assert w_torque.torque_nm == (0.0, 5.0, 0.0)

    # Both None -> ValueError
    with pytest.raises(ValueError, match="At least one"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:ground",
            body="foot",
            point_m=(0.0, 0.0, 0.0),
            force_n=None,
            torque_nm=None,
            source="test",
        )


def test_overlay_wrench_label_validation() -> None:
    """Labels must match ^[a-z_]+:[A-Za-z0-9_.:-]+$."""
    valid_labels = [
        "joint:left_elbow",
        "contact:right_foot:0",
        "external:wind.gust-1",
        "muscle:biceps_lh",
        "grip:club_handle:top",
    ]
    for lbl in valid_labels:
        w = OverlayWrench(
            kind=WrenchKind.CONTACT,
            label=lbl,
            body="body",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 0.0, 0.0),
            source="test",
        )
        assert w.label == lbl

    invalid_labels = [
        "",
        "no_colon",
        "Joint:uppercase_prefix",
        "123:starts_with_number",
        ":empty_prefix",
        "joint:",
        "joint:invalid space",
        "joint:invalid@symbol",
    ]
    for lbl in invalid_labels:
        with pytest.raises(ValueError, match="label"):
            OverlayWrench(
                kind=WrenchKind.CONTACT,
                label=lbl,
                body="body",
                point_m=(0.0, 0.0, 0.0),
                force_n=(1.0, 0.0, 0.0),
                source="test",
            )


def test_overlay_wrench_field_validation() -> None:
    """Non-empty strings, valid kind, and finite 3-vectors are enforced."""
    # Empty body
    with pytest.raises(ValueError, match="body"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:1",
            body="",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 0.0, 0.0),
            source="test",
        )

    # Empty source
    with pytest.raises(ValueError, match="source"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:1",
            body="body",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 0.0, 0.0),
            source="",
        )

    # Non-finite point
    with pytest.raises(ValueError, match="point_m"):
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:1",
            body="body",
            point_m=(0.0, float("nan"), 0.0),
            force_n=(1.0, 0.0, 0.0),
            source="test",
        )

    # Invalid kind
    with pytest.raises(ValueError, match="kind"):
        OverlayWrench(
            kind="not_a_kind",  # type: ignore[arg-type]
            label="contact:1",
            body="body",
            point_m=(0.0, 0.0, 0.0),
            force_n=(1.0, 0.0, 0.0),
            source="test",
        )


def test_overlay_wrench_to_spatial_wrench() -> None:
    """to_spatial_wrench returns SpatialWrench only when both halves are present."""
    # Both halves present
    w_both = OverlayWrench(
        kind=WrenchKind.EXTERNAL,
        label="external:push",
        body="torso",
        point_m=(0.1, 0.2, 0.3),
        force_n=(10.0, 20.0, 30.0),
        torque_nm=(1.0, 2.0, 3.0),
        source="test",
    )
    sw = w_both.to_spatial_wrench()
    assert isinstance(sw, SpatialWrench)
    assert sw.application_frame == "world"
    assert sw.point_m == (0.1, 0.2, 0.3)
    assert sw.force_n == (10.0, 20.0, 30.0)
    assert sw.torque_nm == (1.0, 2.0, 3.0)
    assert sw.direction_convention == "applied_to_body"

    # Force only -> raises naming torque_nm
    w_force = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        source="test",
    )
    with pytest.raises(ValueError, match="torque_nm"):
        w_force.to_spatial_wrench()

    # Torque only -> raises naming force_n
    w_torque = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:motor",
        body="link",
        point_m=(0.0, 0.0, 0.0),
        torque_nm=(0.0, 5.0, 0.0),
        source="test",
    )
    with pytest.raises(ValueError, match="force_n"):
        w_torque.to_spatial_wrench()


def test_overlay_wrench_dict_roundtrip() -> None:
    """OverlayWrench to_dict / from_dict preserves values and null halves."""
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.1, 0.2, 0.3),
        force_n=(0.0, 0.0, 100.0),
        torque_nm=None,
        source="test_src",
    )
    d1 = w1.to_dict()
    assert d1["kind"] == "contact"
    assert d1["force_n"] == [0.0, 0.0, 100.0]
    assert d1["torque_nm"] is None
    w1_rt = OverlayWrench.from_dict(d1)
    assert w1_rt == w1

    w2 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:shoulder",
        body="arm",
        point_m=(0.0, 1.0, 2.0),
        force_n=None,
        torque_nm=(3.0, 4.0, 5.0),
        source="test_src",
    )
    d2 = w2.to_dict()
    assert d2["torque_nm"] == [3.0, 4.0, 5.0]
    assert d2["force_n"] is None
    w2_rt = OverlayWrench.from_dict(d2)
    assert w2_rt == w2


def test_force_torque_frame_creation_and_validation() -> None:
    """ForceTorqueFrame validates unique wrench labels, finite time, non-empty engine."""
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot_l",
        body="foot_l",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 500.0),
        source="test",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot_r",
        body="foot_r",
        point_m=(0.5, 0.0, 0.0),
        force_n=(0.0, 0.0, 500.0),
        source="test",
    )
    frame = ForceTorqueFrame(
        time_s=1.23,
        engine="mujoco",
        wrenches=(w1, w2),
    )
    assert frame.time_s == 1.23
    assert frame.engine == "mujoco"
    assert frame.world_frame == "world_Zup"
    assert frame.units["force"] == "N"
    assert frame.units["torque"] == "N*m"
    assert frame.units["length"] == "m"
    assert frame.wrenches == (w1, w2)
    assert frame.axial_loads is None

    # Duplicate labels -> ValueError
    w_dup = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot_l",
        body="other_body",
        point_m=(1.0, 0.0, 0.0),
        force_n=(1.0, 0.0, 0.0),
        source="test",
    )
    with pytest.raises(ValueError, match="Duplicate wrench label"):
        ForceTorqueFrame(
            time_s=0.0,
            engine="mujoco",
            wrenches=(w1, w_dup),
        )

    # Non-finite time_s
    with pytest.raises(ValueError, match="time_s"):
        ForceTorqueFrame(time_s=float("nan"), engine="mujoco", wrenches=())

    # Empty engine
    with pytest.raises(ValueError, match="engine"):
        ForceTorqueFrame(time_s=0.0, engine="", wrenches=())


def test_force_torque_frame_axial_loads_validation() -> None:
    """AxialLoadFrame must match frame time_s within 1e-12."""
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        source="test",
    )
    ax_match = AxialLoadFrame(time_s=1.0, values_n={"tibia": 200.0}, source="test_ax")
    frame = ForceTorqueFrame(
        time_s=1.0,
        engine="mujoco",
        wrenches=(w,),
        axial_loads=ax_match,
    )
    assert frame.axial_loads is ax_match

    # Mismatched time_s
    ax_mismatch = AxialLoadFrame(
        time_s=1.001, values_n={"tibia": 200.0}, source="test_ax"
    )
    with pytest.raises(ValueError, match="time_s"):
        ForceTorqueFrame(
            time_s=1.0,
            engine="mujoco",
            wrenches=(w,),
            axial_loads=ax_mismatch,
        )

    # Invalid type for axial_loads
    with pytest.raises(TypeError, match="AxialLoadFrame"):
        ForceTorqueFrame(
            time_s=1.0,
            engine="mujoco",
            wrenches=(w,),
            axial_loads="not_an_axial_load_frame",  # type: ignore[arg-type]
        )


def test_force_torque_frame_by_kind() -> None:
    """by_kind filters wrenches by WrenchKind."""
    w_act = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:elbow",
        body="arm",
        point_m=(0.0, 0.0, 0.0),
        torque_nm=(1.0, 0.0, 0.0),
        source="test",
    )
    w_cont = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        source="test",
    )
    frame = ForceTorqueFrame(
        time_s=0.0,
        engine="mujoco",
        wrenches=(w_act, w_cont),
    )
    assert frame.by_kind(WrenchKind.JOINT_ACTUATOR) == (w_act,)
    assert frame.by_kind(WrenchKind.CONTACT) == (w_cont,)
    assert frame.by_kind(WrenchKind.MUSCLE) == ()


def test_force_torque_frame_dict_roundtrip() -> None:
    """to_dict and from_dict preserve wire schema and reject invalid inputs."""
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        source="test",
    )
    ax = AxialLoadFrame(time_s=0.5, values_n={"link": 50.0}, source="ax_src")
    frame = ForceTorqueFrame(
        time_s=0.5,
        engine="drake",
        wrenches=(w,),
        axial_loads=ax,
    )
    d = frame.to_dict()
    assert d["schema_version"] == "force-torque-frame-v1"
    assert d["time_s"] == 0.5
    assert d["engine"] == "drake"
    assert d["axial_loads"] is not None
    assert d["axial_loads"]["values_n"]["link"] == 50.0

    frame_rt = ForceTorqueFrame.from_dict(d)
    assert frame_rt.time_s == frame.time_s
    assert frame_rt.engine == frame.engine
    assert frame_rt.wrenches == frame.wrenches
    assert frame_rt.axial_loads is not None
    assert frame_rt.axial_loads.values_n == frame.axial_loads.values_n
    assert frame_rt.axial_loads.source == frame.axial_loads.source

    # Rejection of wrong schema_version
    d_bad_version = dict(d)
    d_bad_version["schema_version"] = "wrong-v2"
    with pytest.raises(ValueError, match="schema_version"):
        ForceTorqueFrame.from_dict(d_bad_version)

    # Rejection of unknown keys
    d_unknown = dict(d)
    d_unknown["unexpected_key"] = 123
    with pytest.raises(ValueError, match="Unknown key"):
        ForceTorqueFrame.from_dict(d_unknown)


def test_force_torque_provider_and_reader() -> None:
    """ForceTorqueProvider Protocol and read_force_torque_frame behavior."""
    w = OverlayWrench(
        kind=WrenchKind.GRAVITY,
        label="gravity:body",
        body="body",
        point_m=(0.0, 0.0, 1.0),
        force_n=(0.0, 0.0, -98.1),
        source="engine",
    )
    frame = ForceTorqueFrame(time_s=0.1, engine="pinocchio", wrenches=(w,))

    class ValidProvider:
        def get_force_torque_frame(self) -> ForceTorqueFrame | None:
            return frame

    provider = ValidProvider()
    assert isinstance(provider, ForceTorqueProvider)

    # Matching time
    read = read_force_torque_frame(provider, 0.1)
    assert read == frame

    # Non-matching time returns None
    assert read_force_torque_frame(provider, 0.2) is None

    # Non-finite time raises ValueError
    with pytest.raises(ValueError, match="time_s"):
        read_force_torque_frame(provider, float("nan"))

    # Non-provider returns None
    assert read_force_torque_frame("not_a_provider", 0.1) is None

    # Provider returning None
    class EmptyProvider:
        def get_force_torque_frame(self) -> ForceTorqueFrame | None:
            return None

    assert read_force_torque_frame(EmptyProvider(), 0.1) is None

    # Provider returning invalid object raises TypeError
    class BadProvider:
        def get_force_torque_frame(self) -> object:
            return "not_a_frame"

    with pytest.raises(TypeError, match="ForceTorqueFrame"):
        read_force_torque_frame(BadProvider(), 0.1)
