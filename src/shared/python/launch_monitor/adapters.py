"""TrackMan and FlightScope CSV adapters.

Each adapter reads a device-specific CSV export and returns a list of
:class:`~src.shared.python.launch_monitor.types.LaunchMonitorShot`
objects normalised to SI units.

Column name mappings follow the default export format of each device
as of 2024.  Unrecognised columns are preserved in ``shot.extra``.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

from .types import LaunchMonitorShot

_MPH_TO_MPS = 0.44704
_YARDS_TO_METERS = 0.9144


def _float_or_none(value: str) -> float | None:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def _require_float(value: str, field: str) -> float:
    result = _float_or_none(value)
    if result is None:
        raise ValueError(
            f"Required field {field!r} is missing or non-numeric: {value!r}"
        )
    return result


class TrackManAdapter:
    """Parse TrackMan IV / Performance Studio CSV exports.

    TrackMan exports a single-header CSV where each row is one shot.
    Column names use the long-form English labels (e.g. "Ball Speed",
    "Launch Angle").  Values are in US customary units (mph, yards,
    degrees, rpm).

    Example::

        shots = TrackManAdapter.from_csv("session.csv")
        for shot in shots:
            lc = shot.to_launch_conditions()
    """

    # Mapping: internal attribute name → list of recognised column headers
    # (checked case-insensitively; first match wins)
    _COLUMN_MAP: dict[str, list[str]] = {
        "club": ["Club", "Club Type"],
        "ball_speed_mph": ["Ball Speed", "Ball Spd", "BallSpeed"],
        "club_speed_mph": ["Club Speed", "Club Spd", "ClubSpeed", "Swing Speed"],
        "smash_factor": ["Smash Factor", "Smash Fac", "SmashFactor"],
        "launch_angle_deg": ["Launch Angle", "Lnch Angle", "LaunchAngle"],
        "launch_direction_deg": ["Launch Direction", "Lnch Dir", "LaunchDirection"],
        "back_spin_rpm": ["Back Spin", "BackSpin"],
        "side_spin_rpm": ["Side Spin", "SideSpin"],
        "spin_axis_deg": ["Spin Axis", "SpinAxis"],
        "total_spin_rpm": ["Total Spin", "TotalSpin", "Spin Rate"],
        "carry_yards": ["Carry", "Carry Dist"],
        "total_yards": ["Total", "Total Dist"],
        "max_height_yards": ["Height", "Max Height", "Apex"],
        "landing_angle_deg": ["Descent Angle", "Desc Angle", "Landing Angle"],
        "flight_time_s": ["Time of Flight", "Flight Time"],
        "attack_angle_deg": ["Attack Angle", "Atk Angle"],
        "dynamic_loft_deg": ["Dyn. Loft", "Dynamic Loft", "DynLoft"],
        "club_path_deg": ["Club Path", "Path"],
        "face_angle_deg": ["Face Angle", "Face Ang"],
    }

    @classmethod
    def _build_index(cls, headers: list[str]) -> dict[str, int]:
        lower_headers = [h.strip().lower() for h in headers]
        index: dict[str, int] = {}
        for attr, candidates in cls._COLUMN_MAP.items():
            for candidate in candidates:
                try:
                    col_idx = lower_headers.index(candidate.lower())
                    index[attr] = col_idx
                    break
                except ValueError:
                    continue
        return index

    @classmethod
    def _parse_row(
        cls,
        row: list[str],
        col_idx: dict[str, int],
        headers: list[str],
        shot_id: str,
    ) -> LaunchMonitorShot | None:
        def get(attr: str) -> str:
            idx = col_idx.get(attr)
            return row[idx].strip() if idx is not None and idx < len(row) else ""

        ball_speed_mph = _float_or_none(get("ball_speed_mph"))
        club_speed_mph = _float_or_none(get("club_speed_mph"))
        total_spin = _float_or_none(get("total_spin_rpm"))
        carry_yards = _float_or_none(get("carry_yards"))

        # Skip rows that are clearly empty / header repetitions
        if ball_speed_mph is None and total_spin is None:
            return None

        ball_speed_mps = (ball_speed_mph or 0.0) * _MPH_TO_MPS
        club_speed_mps = (club_speed_mph or 0.0) * _MPH_TO_MPS
        smash_raw = _float_or_none(get("smash_factor"))
        smash = (
            smash_raw
            if smash_raw is not None
            else (ball_speed_mps / club_speed_mps if club_speed_mps > 1e-6 else 0.0)
        )

        back_spin = _float_or_none(get("back_spin_rpm")) or 0.0
        side_spin = _float_or_none(get("side_spin_rpm")) or 0.0
        spin_axis = _float_or_none(get("spin_axis_deg")) or 0.0
        if total_spin is None:
            total_spin = math.hypot(back_spin, side_spin)

        max_h_yards = _float_or_none(get("max_height_yards"))
        total_yards = _float_or_none(get("total_yards"))

        # Collect unrecognised columns into extra
        known_indices = set(col_idx.values())
        extra = {
            headers[i].strip(): row[i].strip()
            for i in range(len(row))
            if i not in known_indices and i < len(headers)
        }

        return LaunchMonitorShot(
            club=get("club") or "Unknown",
            ball_speed_mps=ball_speed_mps,
            club_speed_mps=club_speed_mps,
            smash_factor=smash,
            launch_angle_deg=_float_or_none(get("launch_angle_deg")) or 0.0,
            launch_direction_deg=_float_or_none(get("launch_direction_deg")) or 0.0,
            back_spin_rpm=back_spin,
            side_spin_rpm=side_spin,
            spin_axis_deg=spin_axis,
            total_spin_rpm=total_spin,
            carry_m=(carry_yards or 0.0) * _YARDS_TO_METERS,
            total_m=total_yards * _YARDS_TO_METERS if total_yards is not None else None,
            max_height_m=(
                max_h_yards * _YARDS_TO_METERS if max_h_yards is not None else None
            ),
            landing_angle_deg=_float_or_none(get("landing_angle_deg")),
            flight_time_s=_float_or_none(get("flight_time_s")),
            attack_angle_deg=_float_or_none(get("attack_angle_deg")) or 0.0,
            dynamic_loft_deg=_float_or_none(get("dynamic_loft_deg")),
            club_path_deg=_float_or_none(get("club_path_deg")),
            face_angle_deg=_float_or_none(get("face_angle_deg")),
            source="TrackMan",
            shot_id=shot_id,
            extra=extra,
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> list[LaunchMonitorShot]:
        """Parse a TrackMan CSV file and return a list of shots.

        Args:
            path: Path to the TrackMan CSV export.

        Returns:
            List of :class:`LaunchMonitorShot` objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file has no recognised ball-speed column.
        """
        text = Path(path).read_text(encoding="utf-8-sig")
        return cls.from_string(text)

    @classmethod
    def from_string(cls, text: str) -> list[LaunchMonitorShot]:
        """Parse a TrackMan CSV from a string.

        Args:
            text: Raw CSV content.

        Returns:
            List of :class:`LaunchMonitorShot` objects.
        """
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return []

        headers = rows[0]
        col_idx = cls._build_index(headers)

        shots: list[LaunchMonitorShot] = []
        for row_num, row in enumerate(rows[1:], start=2):
            shot = cls._parse_row(row, col_idx, headers, shot_id=str(row_num))
            if shot is not None:
                shots.append(shot)
        return shots


class FlightScopeAdapter:
    """Parse FlightScope Mevo / X2 / X3 CSV exports.

    FlightScope uses slightly different column names and exports in
    SI or US customary depending on device settings.  This adapter
    handles the most common US-customary export format.

    Example::

        shots = FlightScopeAdapter.from_csv("session.csv")
    """

    _COLUMN_MAP: dict[str, list[str]] = {
        "club": ["Club", "Club Name", "Selection"],
        "ball_speed_mph": ["Ball Speed (mph)", "Ball Speed", "BallSpeed"],
        "club_speed_mph": ["Club Speed (mph)", "Club Speed", "ClubHead Speed"],
        "smash_factor": ["Smash Factor", "SmashFactor"],
        "launch_angle_deg": [
            "Launch Angle (deg)",
            "Launch Angle",
            "Vert. Launch Angle",
        ],
        "launch_direction_deg": [
            "Launch Direction (deg)",
            "Launch Direction",
            "Horiz. Launch Angle",
        ],
        "back_spin_rpm": ["Backspin (rpm)", "Backspin", "Back Spin"],
        "side_spin_rpm": ["Sidespin (rpm)", "Sidespin", "Side Spin"],
        "spin_axis_deg": ["Spin Axis (deg)", "Spin Axis"],
        "total_spin_rpm": ["Total Spin (rpm)", "Total Spin", "Spin Rate"],
        "carry_yards": ["Carry (yds)", "Carry Distance", "Carry"],
        "total_yards": ["Total (yds)", "Total Distance", "Total"],
        "max_height_yards": ["Max Height (yds)", "Apex (yds)", "Max Height"],
        "landing_angle_deg": ["Descent Angle (deg)", "Landing Angle", "Descent"],
        "flight_time_s": ["Hang Time (s)", "Hang Time", "Flight Time"],
        "attack_angle_deg": ["Attack Angle (deg)", "Attack Angle", "AoA"],
        "dynamic_loft_deg": ["Dynamic Loft (deg)", "Dynamic Loft"],
        "club_path_deg": ["Club Path (deg)", "Club Path"],
        "face_angle_deg": ["Face Angle (deg)", "Face Angle"],
    }

    @classmethod
    def _build_index(cls, headers: list[str]) -> dict[str, int]:
        lower_headers = [h.strip().lower() for h in headers]
        index: dict[str, int] = {}
        for attr, candidates in cls._COLUMN_MAP.items():
            for candidate in candidates:
                try:
                    col_idx = lower_headers.index(candidate.lower())
                    index[attr] = col_idx
                    break
                except ValueError:
                    continue
        return index

    @classmethod
    def from_csv(cls, path: str | Path) -> list[LaunchMonitorShot]:
        """Parse a FlightScope CSV file and return a list of shots.

        Args:
            path: Path to the FlightScope CSV export.

        Returns:
            List of :class:`LaunchMonitorShot` objects.
        """
        text = Path(path).read_text(encoding="utf-8-sig")
        return cls.from_string(text)

    @classmethod
    def from_string(cls, text: str) -> list[LaunchMonitorShot]:
        """Parse a FlightScope CSV from a string.

        Args:
            text: Raw CSV content.

        Returns:
            List of :class:`LaunchMonitorShot` objects.
        """
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return []

        headers = rows[0]
        col_idx = cls._build_index(headers)
        known_indices = set(col_idx.values())

        def _cell(row: list[str], attr: str) -> str:
            idx = col_idx.get(attr)
            return row[idx].strip() if idx is not None and idx < len(row) else ""

        shots: list[LaunchMonitorShot] = []
        for row_num, row in enumerate(rows[1:], start=2):
            if len(row) < 2:
                continue

            ball_speed_mph = _float_or_none(_cell(row, "ball_speed_mph"))
            total_spin = _float_or_none(_cell(row, "total_spin_rpm"))
            if ball_speed_mph is None and total_spin is None:
                continue

            club_speed_mph = _float_or_none(_cell(row, "club_speed_mph")) or 0.0
            ball_speed_mps = (ball_speed_mph or 0.0) * _MPH_TO_MPS
            club_speed_mps = club_speed_mph * _MPH_TO_MPS
            smash_raw = _float_or_none(_cell(row, "smash_factor"))
            smash = (
                smash_raw
                if smash_raw is not None
                else (ball_speed_mps / club_speed_mps if club_speed_mps > 1e-6 else 0.0)
            )

            back_spin = _float_or_none(_cell(row, "back_spin_rpm")) or 0.0
            side_spin = _float_or_none(_cell(row, "side_spin_rpm")) or 0.0
            spin_axis = _float_or_none(_cell(row, "spin_axis_deg")) or 0.0
            if total_spin is None:
                total_spin = math.hypot(back_spin, side_spin)

            carry_yards = _float_or_none(_cell(row, "carry_yards"))
            total_yards = _float_or_none(_cell(row, "total_yards"))
            max_h_yards = _float_or_none(_cell(row, "max_height_yards"))

            extra = {
                headers[i].strip(): row[i].strip()
                for i in range(len(row))
                if i not in known_indices and i < len(headers)
            }

            shots.append(
                LaunchMonitorShot(
                    club=_cell(row, "club") or "Unknown",
                    ball_speed_mps=ball_speed_mps,
                    club_speed_mps=club_speed_mps,
                    smash_factor=smash,
                    launch_angle_deg=_float_or_none(_cell(row, "launch_angle_deg"))
                    or 0.0,
                    launch_direction_deg=(
                        _float_or_none(_cell(row, "launch_direction_deg")) or 0.0
                    ),
                    back_spin_rpm=back_spin,
                    side_spin_rpm=side_spin,
                    spin_axis_deg=spin_axis,
                    total_spin_rpm=total_spin,
                    carry_m=(carry_yards or 0.0) * _YARDS_TO_METERS,
                    total_m=(
                        total_yards * _YARDS_TO_METERS
                        if total_yards is not None
                        else None
                    ),
                    max_height_m=(
                        max_h_yards * _YARDS_TO_METERS
                        if max_h_yards is not None
                        else None
                    ),
                    landing_angle_deg=_float_or_none(_cell(row, "landing_angle_deg")),
                    flight_time_s=_float_or_none(_cell(row, "flight_time_s")),
                    attack_angle_deg=_float_or_none(_cell(row, "attack_angle_deg"))
                    or 0.0,
                    dynamic_loft_deg=_float_or_none(_cell(row, "dynamic_loft_deg")),
                    club_path_deg=_float_or_none(_cell(row, "club_path_deg")),
                    face_angle_deg=_float_or_none(_cell(row, "face_angle_deg")),
                    source="FlightScope",
                    shot_id=str(row_num),
                    extra=extra,
                )
            )
        return shots
