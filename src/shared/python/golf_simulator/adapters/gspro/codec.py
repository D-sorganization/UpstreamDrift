"""Pure JSON codec and field transformations for GSPro Open Connect v1 (GS-02 #10191).

Follows TDD, DbC, Law of Demeter, and DRY.
All functions are pure and have zero network or GUI side-effects.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.golf_simulator.adapters.gspro.profile import (
    GSProProfile,
    ResponseCategory,
)
from src.shared.python.golf_simulator.contracts import ShotEnvelope

_MPS_TO_MPH: float = 2.2369362920544
_RAD_TO_DEG: float = 180.0 / math.pi
_RAD_S_TO_RPM: float = 60.0 / (2.0 * math.pi)


@dataclass(frozen=True)
class ResponseReceipt:
    """Decoded response receipt from GSPro."""

    code: int
    category: ResponseCategory
    message: str = ""
    player_data: dict[str, Any] | None = None
    raw_payload: dict[str, Any] | None = None


def encode_heartbeat_payload(device_id: str = "UpstreamDrift") -> dict[str, Any]:
    """Create a minimal heartbeat message payload."""
    return {
        "DeviceID": device_id,
        "ShotNumber": 0,
        "ShotDataOptions": {
            "ContainsBallData": False,
            "ContainsClubData": False,
            "LaunchMonitorIsReady": True,
            "LaunchMonitorBallDetected": False,
            "IsHeartBeat": True,
        },
    }


def _compute_spin_axis_tilt(angular_vel: tuple[float, float, float]) -> float:
    """Compute spin axis tilt angle in degrees.

    Pure backspin vector is along -y (tilt = 0 deg).
    Lateral tilt angle is atan2(omega_x, -omega_y).
    """
    ox, oy, _ = angular_vel
    denom = -oy
    if abs(ox) < 1e-9 and abs(denom) < 1e-9:
        return 0.0
    return float(math.degrees(math.atan2(ox, denom)))


def encode_shot_payload(
    shot: ShotEnvelope,
    profile: GSProProfile,
    shot_number: int = 1,
    device_id: str = "UpstreamDrift",
) -> dict[str, Any]:
    """Encode a canonical ShotEnvelope into a GSPro Open Connect JSON dictionary.

    Converts SI units to profile units (m/s -> mph, radians -> degrees, rad/s -> RPM).
    """
    vx, vy, vz = shot.ball_velocity_m_s
    speed_mps = math.sqrt(vx**2 + vy**2 + vz**2)
    if speed_mps <= 0.0 or not math.isfinite(speed_mps):
        raise ValueError(f"Invalid speed {speed_mps} in shot {shot.shot_id}")

    v_horiz = math.sqrt(vx**2 + vy**2)
    vla_deg = math.degrees(math.atan2(vz, v_horiz))

    # Canonical azimuth in +y left frame: atan2(vy, vx)
    azimuth_deg = math.degrees(math.atan2(vy, vx))
    # If profile defines HLA positive right, invert azimuth:
    hla_deg = -azimuth_deg if profile.hla_sign_positive == "right" else azimuth_deg

    # Spin magnitude and tilt
    ox, oy, oz = shot.ball_angular_velocity_rad_s
    omega_mag = math.sqrt(ox**2 + oy**2 + oz**2)
    total_spin_rpm = omega_mag * _RAD_S_TO_RPM

    spin_axis_deg = _compute_spin_axis_tilt((ox, oy, oz)) if omega_mag > 1e-6 else 0.0

    speed_val = speed_mps * _MPS_TO_MPH if profile.speed_unit == "mph" else speed_mps

    ball_data: dict[str, Any] = {
        "Speed": round(speed_val, 4),
        "TotalSpin": round(total_spin_rpm, 2),
        "VLA": round(vla_deg, 4),
        "HLA": round(hla_deg, 4),
        "SpinAxis": round(spin_axis_deg, 4),
    }

    # Optional club data
    club_dict: dict[str, Any] = {}
    contains_club = False
    if shot.club_data is not None:
        cd = shot.club_data
        if cd.club_speed_m_s is not None:
            c_spd = (
                cd.club_speed_m_s * _MPS_TO_MPH
                if profile.speed_unit == "mph"
                else cd.club_speed_m_s
            )
            club_dict["Speed"] = round(c_spd, 3)
        if cd.attack_angle_rad is not None:
            club_dict["AngleOfAttack"] = round(math.degrees(cd.attack_angle_rad), 2)
        if cd.club_path_rad is not None:
            # Invert path if HLA positive right
            path_deg = math.degrees(cd.club_path_rad)
            club_dict["Path"] = round(
                -path_deg if profile.hla_sign_positive == "right" else path_deg, 2
            )
        if cd.face_to_target_rad is not None:
            ftt_deg = math.degrees(cd.face_to_target_rad)
            club_dict["FaceToTarget"] = round(
                -ftt_deg if profile.hla_sign_positive == "right" else ftt_deg, 2
            )
        if cd.face_to_path_rad is not None:
            ftp_deg = math.degrees(cd.face_to_path_rad)
            club_dict["FaceToPath"] = round(
                -ftp_deg if profile.hla_sign_positive == "right" else ftp_deg, 2
            )
        if club_dict:
            contains_club = True

    payload: dict[str, Any] = {
        "DeviceID": device_id,
        "Units": profile.distance_unit,
        "ShotNumber": shot_number,
        "BallData": ball_data,
        "ShotDataOptions": {
            "ContainsBallData": True,
            "ContainsClubData": contains_club,
            "LaunchMonitorIsReady": True,
            "LaunchMonitorBallDetected": True,
            "IsHeartBeat": False,
        },
    }
    if contains_club:
        payload["ClubData"] = club_dict

    return payload


def decode_simulator_response(
    raw: str | dict[str, Any], profile: GSProProfile
) -> ResponseReceipt:
    """Decode a raw simulator JSON response into a typed ResponseReceipt."""
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception as exc:
            return ResponseReceipt(
                code=500,
                category=ResponseCategory.ERROR_REJECTED,
                message=f"Invalid JSON response: {exc}",
                raw_payload={"raw_string": raw},
            )
    elif isinstance(raw, dict):
        data = raw
    else:
        raise TypeError(f"Expected str or dict, got {type(raw)}")

    code = int(data.get("Code", 0))
    category = profile.categorize_code(code)
    message = str(data.get("Message", ""))
    player = data.get("Player")

    return ResponseReceipt(
        code=code,
        category=category,
        message=message,
        player_data=player if isinstance(player, dict) else None,
        raw_payload=data,
    )
