"""Header-fingerprint launch-monitor import profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from src.shared.python.launch_monitor.schema import ColumnMapping, METRICS


def normalize_header(value: str) -> str:
    """Normalize a source header for alias matching while retaining its meaning."""
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", expanded.lower())
    text = re.sub(
        r"\b(mph|kmh|kph|mps|rpm|yards?|yds?|yd|feet|foot|ft|inches|inch|in|degrees?|deg|radians?|rad|seconds?|sec|s)\b",
        " ",
        text,
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


COMMON_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "shot_id": ("shot", "shot number", "shot no", "shot id"),
    "club": ("club", "club name"),
    "player": ("player", "player name"),
    "date": ("date", "shot date"),
    "time": ("time", "shot time"),
    "captured_at": ("timestamp", "date time", "datetime", "captured at"),
    "club_speed": ("club speed", "club head speed", "clubhead speed", "chs"),
    "ball_speed": ("ball speed", "launch speed"),
    "attack_angle": ("attack angle", "angle of attack", "aoa"),
    "club_path": ("club path", "path"),
    "face_angle": ("face angle", "face to target", "club face"),
    "face_to_path": ("face to path", "face path"),
    "dynamic_loft": ("dynamic loft",),
    "dynamic_lie": ("dynamic lie",),
    "spin_loft": ("spin loft",),
    "swing_direction": ("swing direction", "horizontal swing plane"),
    "swing_plane": ("swing plane", "vertical swing plane"),
    "low_point_distance": ("low point", "low point distance"),
    "impact_height": ("impact height", "face impact height"),
    "impact_offset": ("impact offset", "face impact offset"),
    "launch_angle": ("launch angle", "vertical launch", "vla"),
    "launch_direction": (
        "launch direction",
        "horizontal launch",
        "horizontal launch angle",
        "hla",
        "side angle",
    ),
    "spin_rate": ("spin rate", "total spin", "spin"),
    "back_spin": ("back spin", "backspin"),
    "side_spin": ("side spin", "sidespin"),
    "spin_axis": ("spin axis", "axis tilt"),
    "smash_factor": ("smash factor", "smash"),
    "carry_distance": ("carry", "carry distance"),
    "total_distance": ("total", "total distance"),
    "roll_distance": ("roll", "roll distance", "run"),
    "lateral_carry": (
        "carry side",
        "lateral landing",
        "carry deviation distance",
    ),
    "lateral_total": (
        "side",
        "side distance",
        "offline",
        "total side",
        "total deviation distance",
    ),
    "apex_height": ("height", "apex", "apex height", "peak height", "max height"),
    "flight_time": ("flight time", "time aloft"),
    "descent_angle": ("descent angle", "landing angle", "vertical descent"),
    "curve": ("curve",),
    "putt_distance": ("putt distance", "distance"),
    "skid_distance": ("skid distance",),
    "roll_speed": ("roll speed",),
}


@dataclass(frozen=True)
class ImportProfile:
    """Vendor header profile and unit defaults."""

    profile_id: str
    vendor: str
    signatures: tuple[str, ...]
    aliases: dict[str, tuple[str, ...]]
    default_units: dict[str, str]

    def mappings_for(self, headers: list[str]) -> tuple[ColumnMapping, ...]:
        """Return all unambiguous mappings found in ``headers``."""
        normalized_to_source = {normalize_header(header): header for header in headers}
        mappings: list[ColumnMapping] = []
        claimed_sources: set[str] = set()
        for target, aliases in self.aliases.items():
            for alias in aliases:
                source = normalized_to_source.get(normalize_header(alias))
                if source is None or source in claimed_sources:
                    continue
                mappings.append(ColumnMapping(source, target))
                claimed_sources.add(source)
                break
        return tuple(mappings)


def _defaults() -> dict[str, str]:
    values: dict[str, str] = {}
    for name, definition in METRICS.items():
        values[name] = {
            "m/s": "mph",
            "m": "yd",
            "rad": "deg",
            "rad/s": "rpm",
            "s": "s",
            "1": "1",
        }[definition.canonical_unit]
    return values


def _aliases(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    aliases = dict(COMMON_ALIASES)
    aliases.update(overrides)
    return aliases


PROFILES: Final[dict[str, ImportProfile]] = {
    item.profile_id: item
    for item in (
        ImportProfile(
            "trackman",
            "TrackMan",
            ("attack angle", "club path", "face angle", "dynamic loft"),
            _aliases(),
            _defaults(),
        ),
        ImportProfile(
            "foresight",
            "Foresight Sports",
            ("hla", "vla", "total spin", "offline"),
            _aliases(),
            _defaults(),
        ),
        ImportProfile(
            "flightscope",
            "FlightScope",
            ("vertical launch", "horizontal launch", "lateral landing", "flight time"),
            _aliases(),
            _defaults(),
        ),
        ImportProfile(
            "garmin",
            "Garmin",
            ("total deviation distance", "launch direction", "spin axis"),
            _aliases(),
            _defaults(),
        ),
        ImportProfile(
            "skytrak",
            "SkyTrak",
            ("back spin", "side spin", "max height"),
            _aliases(),
            _defaults(),
        ),
        ImportProfile(
            "uneekor",
            "Uneekor",
            ("side angle", "side distance", "apex"),
            _aliases(),
            _defaults(),
        ),
        ImportProfile(
            "full_swing",
            "Full Swing",
            ("club speed", "ball speed", "face to path", "carry distance"),
            _aliases(),
            _defaults(),
        ),
        ImportProfile(
            "rapsodo",
            "Rapsodo",
            ("smash factor", "launch direction", "shot type"),
            _aliases(),
            _defaults(),
        ),
        ImportProfile(
            "gspro",
            "GSPro / Open Connect",
            (
                "ball data speed",
                "ball data hla",
                "ball data vla",
                "ball data back spin",
            ),
            _aliases(
                ball_speed=("ball data speed", "ball speed"),
                launch_direction=("ball data hla", "hla"),
                launch_angle=("ball data vla", "vla"),
                spin_rate=("ball data total spin", "total spin"),
                back_spin=("ball data back spin", "back spin"),
                side_spin=("ball data side spin", "side spin"),
                spin_axis=("ball data spin axis", "spin axis"),
                club_speed=("club data speed", "club speed"),
                attack_angle=("club data angle of attack", "angle of attack"),
                face_angle=("club data face to target", "face to target"),
                club_path=("club data path", "club path"),
                dynamic_loft=("club data loft", "dynamic loft"),
                dynamic_lie=("club data lie", "dynamic lie"),
                impact_height=("club data vertical face impact", "impact height"),
                impact_offset=(
                    "club data horizontal face impact",
                    "impact offset",
                ),
            ),
            _defaults(),
        ),
        ImportProfile("generic", "Generic", (), _aliases(), _defaults()),
    )
}


@dataclass(frozen=True)
class ProfileDetection:
    """Ranked profile detection result."""

    profile_id: str
    confidence: float
    matched_signatures: tuple[str, ...]
    alternatives: tuple[tuple[str, float], ...]


def detect_profile(headers: list[str]) -> ProfileDetection:
    """Detect the most likely vendor profile from header fingerprints."""
    if not headers:
        raise ValueError("headers must contain at least one column")
    normalized = {normalize_header(header) for header in headers}
    ranked: list[tuple[str, float, tuple[str, ...]]] = []
    for profile_id, profile in PROFILES.items():
        if profile_id == "generic":
            continue
        matches = tuple(
            signature
            for signature in profile.signatures
            if normalize_header(signature) in normalized
        )
        score = len(matches) / len(profile.signatures) if profile.signatures else 0.0
        ranked.append((profile_id, score, matches))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    best_id, best_score, matches = ranked[0]
    if best_score < 0.5:
        best_id, best_score, matches = "generic", 0.0, ()
    alternatives = tuple((profile_id, score) for profile_id, score, _ in ranked[:4])
    return ProfileDetection(best_id, best_score, matches, alternatives)
