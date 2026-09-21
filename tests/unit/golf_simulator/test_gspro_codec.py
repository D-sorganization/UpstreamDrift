"""Unit tests for the pure GSPro JSON codec and independent golden fixtures (GS-02 #10191).

Follows TDD, DbC, LoD and DRY principles.
"""

from __future__ import annotations

import json
import math
import pytest

from src.shared.python.golf_simulator.adapters.gspro.codec import (
    decode_simulator_response,
    encode_heartbeat_payload,
    encode_shot_payload,
)
from src.shared.python.golf_simulator.adapters.gspro.profile import (
    DEFAULT_GSPRO_PROFILE,
    ResponseCategory,
)
from src.shared.python.golf_simulator.contracts import (
    AimContext,
    ClubData,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotQualification,
    SourceKind,
)

pytestmark = pytest.mark.unit


def _aim_identity() -> AimContext:
    return AimContext(
        source_to_target_rotation=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        revision=1,
        provenance="identity",
    )


def _manual_qualification() -> ShotQualification:
    return ShotQualification(
        contact=ContactStatus.NOT_APPLICABLE,
        numerical=NumericalStatus.ESTIMATED,
        scientific=ScientificStatus.NOT_APPLICABLE,
        evidence_refs=("manual_fixture",),
    )


def test_encode_straight_shot_golden() -> None:
    # 70 m/s forward (+x), 0 lateral (+y), 15 m/s upward (+z)
    # Total speed = sqrt(70^2 + 15^2) = 71.5891 m/s = 160.14028 mph
    # VLA = atan2(15, 70) * 180 / pi = 12.094757 deg
    # HLA = 0.0 deg
    # Pure backspin: angular velocity around -y = -250 rad/s
    # TotalSpin = 250 * 60 / (2 * pi) = 2387.324 RPM
    # SpinAxis tilt = 0.0 deg (pure backspin)

    speed_mps = math.sqrt(70.0**2 + 15.0**2)
    expected_mph = speed_mps * 2.2369362920544
    expected_vla = math.degrees(math.atan2(15.0, 70.0))
    expected_spin_rpm = 250.0 * 60.0 / (2.0 * math.pi)

    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot-straight-001",
        session_id="session-001",
        source_kind=SourceKind.MANUAL,
        qualification=_manual_qualification(),
        ball_velocity_m_s=(70.0, 0.0, 15.0),
        ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
        aim_context=_aim_identity(),
        created_at_utc="2026-09-15T12:00:00Z",
    )

    payload = encode_shot_payload(envelope, DEFAULT_GSPRO_PROFILE, shot_number=42)

    assert payload["DeviceID"] == "UpstreamDrift"
    assert payload["ShotNumber"] == 42
    ball = payload["BallData"]
    assert math.isclose(ball["Speed"], expected_mph, rel_tol=1e-3)
    assert math.isclose(ball["VLA"], expected_vla, abs_tol=0.01)
    assert math.isclose(ball["HLA"], 0.0, abs_tol=1e-4)
    assert math.isclose(ball["TotalSpin"], expected_spin_rpm, rel_tol=1e-3)
    assert math.isclose(ball["SpinAxis"], 0.0, abs_tol=1e-4)

    assert payload["ShotDataOptions"]["ContainsBallData"] is True
    assert payload["ShotDataOptions"]["ContainsClubData"] is False
    assert "ClubData" not in payload


def test_encode_right_launch_hla_sign() -> None:
    # 70 m/s forward (+x), -2.5 m/s right (-y in +y left frame), 15 m/s upward (+z)
    # Azimuth: atan2(-2.5, 70) is negative radians.
    # In standard golf profile with hla_sign_positive="right", HLA should be POSITIVE.
    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot-right-001",
        session_id="session-001",
        source_kind=SourceKind.MANUAL,
        qualification=_manual_qualification(),
        ball_velocity_m_s=(70.0, -2.5, 15.0),
        ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
        aim_context=_aim_identity(),
        created_at_utc="2026-09-15T12:00:00Z",
    )

    payload = encode_shot_payload(envelope, DEFAULT_GSPRO_PROFILE)
    ball = payload["BallData"]
    expected_hla_deg = math.degrees(math.atan2(2.5, 70.0))  # positive right
    assert math.isclose(ball["HLA"], expected_hla_deg, abs_tol=0.01)
    assert ball["HLA"] > 0.0


def test_encode_zero_spin() -> None:
    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot-zero-spin-001",
        session_id="session-001",
        source_kind=SourceKind.MANUAL,
        qualification=_manual_qualification(),
        ball_velocity_m_s=(60.0, 0.0, 10.0),
        ball_angular_velocity_rad_s=(0.0, 0.0, 0.0),
        aim_context=_aim_identity(),
        created_at_utc="2026-09-15T12:00:00Z",
    )

    payload = encode_shot_payload(envelope, DEFAULT_GSPRO_PROFILE)
    ball = payload["BallData"]
    assert ball["TotalSpin"] == 0.0
    assert ball["SpinAxis"] == 0.0


def test_encode_club_data_preserves_presence_without_zero_filling() -> None:
    club = ClubData(
        club_speed_m_s=45.0,  # 100.662 mph
        attack_angle_rad=math.radians(-3.5),
        # club_path, face_to_target, face_to_path omitted
    )
    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot-club-001",
        session_id="session-001",
        source_kind=SourceKind.MANUAL,
        qualification=_manual_qualification(),
        ball_velocity_m_s=(70.0, 0.0, 15.0),
        ball_angular_velocity_rad_s=(0.0, -250.0, 0.0),
        aim_context=_aim_identity(),
        created_at_utc="2026-09-15T12:00:00Z",
        club_data=club,
    )

    payload = encode_shot_payload(envelope, DEFAULT_GSPRO_PROFILE)
    assert payload["ShotDataOptions"]["ContainsClubData"] is True
    assert "ClubData" in payload
    c_data = payload["ClubData"]
    assert math.isclose(c_data["Speed"], 45.0 * 2.2369362920544, rel_tol=1e-4)
    assert math.isclose(c_data["AngleOfAttack"], -3.5, rel_tol=1e-4)
    # Unmeasured fields must NOT be filled with zero
    assert "Path" not in c_data
    assert "FaceToTarget" not in c_data


def test_encode_heartbeat_payload() -> None:
    hb = encode_heartbeat_payload(device_id="UpstreamDrift")
    assert hb["DeviceID"] == "UpstreamDrift"
    assert hb["ShotDataOptions"]["IsHeartBeat"] is True


def test_decode_simulator_responses() -> None:
    # 200 OK
    resp_200 = decode_simulator_response(
        {"Code": 200, "Message": "Shot accepted"}, DEFAULT_GSPRO_PROFILE
    )
    assert resp_200.code == 200
    assert resp_200.category == ResponseCategory.CONFIRMED_ACCEPTED
    assert resp_200.message == "Shot accepted"

    # 201 Player update
    resp_201 = decode_simulator_response(
        {
            "Code": 201,
            "Message": "Player info",
            "Player": {"Handedness": "RH", "Club": "DR"},
        },
        DEFAULT_GSPRO_PROFILE,
    )
    assert resp_201.code == 201
    assert resp_201.category == ResponseCategory.PLAYER_UPDATE
    assert resp_201.player_data == {"Handedness": "RH", "Club": "DR"}

    # 501 Error
    resp_501 = decode_simulator_response(
        {"Code": 501, "Message": "Invalid lie"}, DEFAULT_GSPRO_PROFILE
    )
    assert resp_501.code == 501
    assert resp_501.category == ResponseCategory.ERROR_REJECTED

    # Unknown code
    resp_unknown = decode_simulator_response(
        {"Code": 777, "Message": "Vendor custom"}, DEFAULT_GSPRO_PROFILE
    )
    assert resp_unknown.code == 777
    assert resp_unknown.category == ResponseCategory.UNKNOWN


def test_shared_tools_codec_integration() -> None:
    """Verify that gspro_connect from Tools is imported and used by codec."""
    from shared.python.launch_monitor import gspro_connect
    from src.shared.python.golf_simulator.adapters.gspro import codec

    assert hasattr(gspro_connect, "encode_shot_payload")
    assert hasattr(gspro_connect, "encode_heartbeat_payload")
    assert hasattr(gspro_connect, "parse_reply")
    assert hasattr(codec, "gspro_connect") or hasattr(
        codec, "_tools_encode_shot_payload"
    )
