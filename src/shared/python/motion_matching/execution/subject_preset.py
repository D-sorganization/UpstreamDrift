"""Named subject presets for the anthropometric spec builder (#15296).

A ``subject-preset/1`` file states one fitted subject as settings, each with
its provenance, so a build is reproducible by name (``--subject-preset``).
Kinematic settings only: a preset is not physical or scientific acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching.club_models import (
    CLUBS,
    ClubSpec,
    sole_offset_m,
    sole_point_m,
    with_butt_to_sole_length,
)
from src.shared.python.motion_matching.execution.assets import DEFAULT_FULL_BODY
from src.shared.python.motion_matching.execution.marker_seed_overrides import (
    validate_marker_seed_overrides,
)

SCHEMA = "subject-preset/1"
PRESET_FOLDER = DEFAULT_FULL_BODY / "evidence/subject_presets"
_POSITIVE = ("stature_m", "mass_kg", "trunk_scale", "arm_scale", "shoulder_scale")
#: Settings a preset may leave out: the builder then keeps its own value.
OPTIONAL = ("thigh_length_m", "shank_length_m", "hip_half_spacing_m")
_TEXT = ("name", "club", "club_markers")


@dataclass(frozen=True)
class SubjectPreset:
    """One named subject: lengths in m, mass in kg, angles in deg.

    ``club_butt_to_sole_m`` is the retail club length; the two club angles
    describe the soled head at address (``club_models.sole_offset_m``).
    ``marker_seeds`` (``marker_seed_overrides``) and ``dropped_markers`` (label
    to reason) are optional; every entry needs a provenance line.
    Raises ``ValueError`` / ``TypeError`` for a value outside its domain.
    """

    name: str
    stature_m: float
    mass_kg: float
    trunk_scale: float
    arm_scale: float
    shoulder_scale: float
    shoulder_below_cervicale_m: float
    club: str
    club_butt_to_sole_m: float
    club_shaft_elevation_deg: float
    club_head_roll_deg: float
    club_markers: str
    #: Hip to knee and knee to ankle joint centres, both legs, m (optional).
    thigh_length_m: float | None = None
    shank_length_m: float | None = None
    #: Half the distance between the hip joint centres, m (optional).
    hip_half_spacing_m: float | None = None
    provenance: Mapping[str, str] = field(default_factory=dict, compare=False)
    marker_seeds: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    dropped_markers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("name", *SETTINGS):
            value = getattr(self, key)
            if value is None and key in OPTIONAL:
                continue
            if key in _TEXT:
                if not isinstance(value, str) or not value:
                    raise TypeError(f"{key} must be a non-empty string")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be a number, got {value!r}")
            elif not math.isfinite(value) or (
                key in (*_POSITIVE, *OPTIONAL) and value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive, got {value}")
        if self.shoulder_below_cervicale_m < 0:
            raise ValueError("shoulder_below_cervicale_m must be >= 0")
        if self.club not in CLUBS:
            raise ValueError(f"club must be one of {sorted(CLUBS)}, got {self.club!r}")
        if not 0.0 < self.club_shaft_elevation_deg <= 90.0:
            raise ValueError("club_shaft_elevation_deg must be in (0, 90]")
        self.club_spec()  # refuses a butt to sole length the head cannot take
        validate_marker_seed_overrides(self.marker_seeds, self.dropped_markers)
        lines = {k: v.get("provenance") for k, v in self.marker_seeds.items()}
        lines.update(self.dropped_markers)
        if blank := sorted(k for k, v in lines.items() if not str(v or "").strip()):
            raise ValueError(f"markers without a provenance line: {blank}")

    def club_spec(self) -> ClubSpec:
        """The catalogue club at this preset's butt to sole length."""
        base = CLUBS[self.club]
        offset = sole_offset_m(
            base, self.club_shaft_elevation_deg, self.club_head_roll_deg
        )
        return with_butt_to_sole_length(base, self.club_butt_to_sole_m, offset)

    def club_sole_point_m(self) -> tuple[float, float, float]:
        """The sole in the club-body frame at this preset's address lie."""
        lie = (self.club_shaft_elevation_deg, self.club_head_roll_deg)
        return sole_point_m(CLUBS[self.club], *lie)

    def builder_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``compose_anthropometric_document``."""
        kwargs = {k: getattr(self, k) for k in SETTINGS if not k.startswith("club_")}
        return {
            **kwargs,
            "club_markers_path": self.club_markers,
            "club_spec": self.club_spec(),
            "club_butt_to_sole_m": self.club_butt_to_sole_m,
            "club_sole_point_m": self.club_sole_point_m(),
            "subject_preset": self.name,
            "marker_seed_overrides": dict(self.marker_seeds) or None,
            "dropped_markers": sorted(self.dropped_markers) or None,
        }


_BLOCKS = ("name", "provenance", "marker_seeds", "dropped_markers")
SETTINGS = tuple(f.name for f in fields(SubjectPreset) if f.name not in _BLOCKS)


def subject_preset_path(name: Path | str) -> Path:
    """An existing file, or ``evidence/subject_presets/<name>.json``."""
    for path in (Path(name), PRESET_FOLDER / f"{name}.json"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Subject preset not found: {name}")


def load_subject_preset(name: Path | str) -> SubjectPreset:
    """A validated ``subject-preset/1`` file by name or path (``ValueError`` for a
    wrong schema, a missing or unknown setting or one without provenance).
    The ``marker_seeds`` and ``dropped_markers`` blocks are optional."""
    raw = json.loads(subject_preset_path(name).read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}, got {raw.get('schema')!r}")
    settings, provenance = raw.get("settings", {}), raw.get("provenance", {})
    absent = set(OPTIONAL) - set(settings)
    provenance = {**dict.fromkeys(absent, "not set"), **provenance}
    if missing := sorted(set(SETTINGS) - absent - set(settings)):
        raise ValueError(f"settings missing: {missing}")
    if unknown := sorted(set(settings) - set(SETTINGS)):
        raise ValueError(f"unknown settings: {unknown}")
    if blank := sorted(k for k in SETTINGS if not str(provenance.get(k, "")).strip()):
        raise ValueError(f"settings without a provenance line: {blank}")
    return SubjectPreset(
        name=str(raw.get("name", "")),
        provenance={k: str(provenance[k]) for k in SETTINGS},
        marker_seeds=raw.get("marker_seeds", {}),
        dropped_markers=raw.get("dropped_markers", {}),
        **settings,
    )
