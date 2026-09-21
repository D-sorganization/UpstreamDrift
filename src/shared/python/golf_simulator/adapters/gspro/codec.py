"""Pure JSON codec and field transformations for GSPro Open Connect v1 (GS-02 #10191).

Consumes shared protocol codec from Tools launch_monitor (Tools#5228).
Follows TDD, DbC, Law of Demeter, and DRY.
All functions are pure and have zero network or GUI side-effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shared.python.launch_monitor import gspro_connect
from shared.python.launch_monitor.gspro_connect import (
    GSProBallData,
    GSProClubData,
    GSProShot,
)
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
    """Create a minimal heartbeat message payload via shared Tools codec."""
    return gspro_connect.encode_heartbeat_payload(device_id=device_id)


def _compute_spin_axis_tilt(angular_vel: tuple[float, float, float]) -> float:
    """Compute spin axis tilt angle in degrees.

    Pure backspin vector is along -y (tilt = 0 deg).
    Lateral tilt angle is atan2(ox, -oy).
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

    Converts SI units to profile units (m/s -> mph, radians -> degrees, rad/s -> RPM)
    and constructs wire payloads via shared.python.launch_monitor.gspro_connect.
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

    ball_data = GSProBallData(
        speed_mph=speed_val,
        spin_axis_deg=spin_axis_deg,
        total_spin_rpm=total_spin_rpm,
        hla_deg=hla_deg,
        vla_deg=vla_deg,
    )

    club_data: GSProClubData | None = None
    if shot.club_data is not None:
        cd = shot.club_data
        club_speed = (
            cd.club_speed_m_s * _MPS_TO_MPH
            if cd.club_speed_m_s is not None and profile.speed_unit == "mph"
            else cd.club_speed_m_s
        )
        aoa = (
            math.degrees(cd.attack_angle_rad)
            if cd.attack_angle_rad is not None
            else None
        )
        path = (
            (
                -math.degrees(cd.club_path_rad)
                if profile.hla_sign_positive == "right"
                else math.degrees(cd.club_path_rad)
            )
            if cd.club_path_rad is not None
            else None
        )
        ftt = (
            (
                -math.degrees(cd.face_to_target_rad)
                if profile.hla_sign_positive == "right"
                else math.degrees(cd.face_to_target_rad)
            )
            if cd.face_to_target_rad is not None
            else None
        )
        club_data = GSProClubData(
            speed_mph=club_speed,
            angle_of_attack_deg=aoa,
            path_deg=path,
            face_to_target_deg=ftt,
        )

    gspro_shot = GSProShot(ball_data=ball_data, club_data=club_data)
    return gspro_connect.encode_shot_payload(
        gspro_shot,
        device_id=device_id,
        shot_number=shot_number,
        units=profile.distance_unit,
    )


def decode_simulator_response(
    raw: str | dict[str, Any] | bytes, profile: GSProProfile
) -> ResponseReceipt:
    """Decode a raw simulator JSON response into a typed ResponseReceipt."""
    try:
        reply = gspro_connect.parse_reply(raw)
    except Exception as exc:
        raw_payload = (
            {"raw_string": raw}
            if isinstance(raw, str)
            else (raw if isinstance(raw, dict) else None)
        )
        return ResponseReceipt(
            code=500,
            category=ResponseCategory.ERROR_REJECTED,
            message=f"Invalid JSON response: {exc}",
            raw_payload=raw_payload,
        )

    category = profile.categorize_code(reply.code)
    player_dict = reply.raw.get("Player")
    player_data = player_dict if isinstance(player_dict, dict) else None

    return ResponseReceipt(
        code=reply.code,
        category=category,
        message=reply.message,
        player_data=player_data,
        raw_payload=reply.raw,
    )
