"""Model card written into every clip manifest (#15150, E2).

The card says what a clip is made of: the replay kind, the fit or tracking
error where the bundle records one, the golfer model, the club and shaft, the
shot setup and the impact model. Every field carries a provenance ``tag``
(``measured``, ``fitted``, ``assumed`` or ``unknown``) and the ``source`` it
was read from. Missing data is an explicit ``unknown``, never dropped. The
schema string, tags and provenance keys live in ``model_card_settings.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.motion_matching.club_shaft_chain import shaft_model
from src.tools.native_viewer_export.replay_mode import ReplayMode, replay_mode_of

if TYPE_CHECKING:
    from src.tools.native_viewer_export.core import ExportSettings, SwingInput

SETTINGS_PATH = Path(__file__).with_name("model_card_settings.json")
TAGS = ("measured", "fitted", "assumed", "unknown")
UNKNOWN = "unknown"
_FIELD_KEYS = ("value", "tag", "source")


@lru_cache(maxsize=1)
def load_card_settings(path: Path | None = None) -> dict[str, Any]:
    """The card settings; ``ValueError`` for a tag outside :data:`TAGS`."""
    settings = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    tags = [*settings["replay_kind_tags"].values()]
    tags += [entry["tag"] for entry in settings["fit_error_keys"]]
    tags += [settings[f"{name}_tag"] for name in ("golfer_model", "club")]
    tags += [settings[f"{name}_tag"] for name in ("shot_setup", "impact_model")]
    bad = sorted({t for t in tags if t not in TAGS})
    if bad:
        raise ValueError(f"model card tags {bad} not in {TAGS}")
    return settings


@dataclass(frozen=True)
class CardField:
    """One card entry: a JSON value, its provenance tag and its source."""

    value: Any
    tag: str
    source: str

    def __post_init__(self) -> None:
        if self.tag not in TAGS:
            raise ValueError(f"tag must be one of {TAGS}, got {self.tag!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")

    @classmethod
    def unknown(cls, source: str) -> CardField:
        """An explicit unknown, saying what was looked for in ``source``."""
        return cls(UNKNOWN, UNKNOWN, source)


@dataclass(frozen=True)
class ModelCard:
    """Schema-versioned model card of one clip (JSON round trip)."""

    schema: str
    replay_kind: CardField
    fit_error: CardField
    golfer_model: CardField
    club: CardField
    shot_setup: CardField
    impact_model: CardField

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"schema": self.schema}
        for item in fields(self)[1:]:
            entry = getattr(self, item.name)
            doc[item.name] = {k: getattr(entry, k) for k in _FIELD_KEYS}
        return doc

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> ModelCard:
        """Rebuild a card; ``ValueError`` for another schema or a missing key."""
        schema = load_card_settings()["schema"]
        if doc.get("schema") != schema:
            raise ValueError(f"model card schema must be {schema!r}")
        names = [item.name for item in fields(cls)[1:]]
        missing = [n for n in names if set(doc.get(n) or {}) != set(_FIELD_KEYS)]
        if missing:
            raise ValueError(f"model card entries missing or malformed: {missing}")
        entries = {n: CardField(**doc[n]) for n in names}
        return cls(schema=schema, **entries)


def _source(name: str) -> str:
    return str(load_card_settings()["sources"][name])


def _replay_kind(bundle: Any) -> CardField:
    mode = replay_mode_of(bundle)
    if mode is ReplayMode.UNKNOWN:
        return CardField.unknown("replay_mode: bundle provenance does not decide it")
    tag = load_card_settings()["replay_kind_tags"][mode.value]
    return CardField(mode.value, tag, f"replay_mode: {_source('replay_kind_tags')}")


def _lookup(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _fit_error(provenance: dict[str, Any]) -> CardField:
    entries = load_card_settings()["fit_error_keys"]
    found = {
        e["key"]: _lookup(provenance, e["key"])
        for e in entries
        if _lookup(provenance, e["key"]) is not None
    }
    keys = ", ".join(e["key"] for e in entries)
    if not found:
        return CardField.unknown(f"bundle provenance records none of: {keys}")
    tag = next(e["tag"] for e in entries if e["key"] in found)
    return CardField(found, tag, f"bundle provenance {', '.join(found)}")


def _golfer(bundle: Any, spec: dict[str, Any]) -> CardField:
    revision = (spec.get("spec_revision") or {}).get("configuration", UNKNOWN)
    value = {"spec_sha256": bundle.spec_sha256, "spec_revision": revision}
    tag = load_card_settings()["golfer_model_tag"]
    return CardField(value, tag, f"bundle spec: {_source('golfer_model_tag')}")


def _club(club_name: str, spec: dict[str, Any]) -> CardField:
    club = spec.get("club")
    if not isinstance(club, dict):
        return CardField.unknown("bundle spec has no club record")
    try:
        model = shaft_model(spec)
    except ValueError as exc:
        return CardField.unknown(f"bundle spec club.shaft: {exc}")
    shaft = club.get("shaft") or {}
    value = {"club": club_name, "shaft_model": model}
    value.update({k: shaft.get(k, UNKNOWN) for k in ("flex", "flex_cpm", "n_links")})
    status = shaft.get("profile_status", "no shaft profile (rigid club)")
    source = f"bundle spec club.shaft ({status}): {_source('club_tag')}"
    return CardField(value, load_card_settings()["club_tag"], source)


def _ball(ball_record: dict[str, Any] | None) -> Any:
    if not ball_record or "position_m" not in ball_record:
        return UNKNOWN
    return {
        "position_m": ball_record["position_m"],
        "source": ball_record.get("source", UNKNOWN),
        "ground_height_m": ball_record.get("ground_height_m", 0.0),
    }


def _shot_setup(
    settings: ExportSettings, ball_record: dict[str, Any] | None
) -> CardField:
    value = {
        "club_roll_in_hands_deg": settings.club_roll_in_hands_deg,
        "ball_position_shift_m": settings.ball_position_shift_m,
        "shot_shape": settings.shot_shape or UNKNOWN,
        "ball": _ball(ball_record),
    }
    tag = load_card_settings()["shot_setup_tag"]
    return CardField(value, tag, f"export settings: {_source('shot_setup_tag')}")


def _impact(provenance: dict[str, Any], settings: ExportSettings) -> CardField:
    record = provenance.get("ball_impact")
    bundle_impact = (
        UNKNOWN
        if not isinstance(record, dict)
        else {k: record.get(k, UNKNOWN) for k in ("frame", "cor", "ball_mass_kg")}
    )
    flight = settings.ball_flight is not None
    launch = load_card_settings()["flight_launch_model"] if flight else UNKNOWN
    if bundle_impact == UNKNOWN and launch == UNKNOWN:
        return CardField.unknown("no bundle ball_impact and ball flight off")
    value = {"bundle_impact": bundle_impact, "flight_launch": launch}
    tag = load_card_settings()["impact_model_tag"]
    return CardField(value, tag, f"bundle and export: {_source('impact_model_tag')}")


def model_card_of(
    swing: SwingInput,
    settings: ExportSettings,
    ball_record: dict[str, Any] | None,
) -> ModelCard:
    """The model card of ``swing`` exported with ``settings``.

    ``ball_record`` is the ball manifest record of the export (``None`` when
    the ball is off). Postcondition: every field is present; data that is not
    recorded is :data:`UNKNOWN` with the ``unknown`` tag.
    """
    bundle = swing.bundle
    provenance = dict(bundle.provenance or {})
    spec = json.loads(bundle.spec_bytes)
    return ModelCard(
        schema=load_card_settings()["schema"],
        replay_kind=_replay_kind(bundle),
        fit_error=_fit_error(provenance),
        golfer_model=_golfer(bundle, spec),
        club=_club(swing.club, spec),
        shot_setup=_shot_setup(settings, ball_record),
        impact_model=_impact(provenance, settings),
    )
