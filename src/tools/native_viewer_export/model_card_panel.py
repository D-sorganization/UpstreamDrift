"""The model card drawn on request in the clip (#15150, E2).

Every clip manifest carries the model card (``model_card.py``); with
``ExportSettings.model_card_panel`` on, each frame also shows a compact panel:
a row per card field with its value and its provenance tag (measured, fitted,
assumed or unknown), the row in the tag's colour. The panel is drawn by the
shared text panel renderer (``compositor.draw_text_panel``) like the impact
card and the colour key. Position, font size, tag colours, the field list and
the row format live in ``model_card_panel_settings.json``; the option is off
unless that file's ``enabled_by_default`` says otherwise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from src.tools.native_viewer_export import compositor
from src.tools.native_viewer_export.model_card import TAGS, UNKNOWN, ModelCard

if TYPE_CHECKING:
    from src.tools.native_viewer_export.core import ClipOverlays, ExportSettings

SETTINGS_PATH = Path(__file__).with_name("model_card_panel_settings.json")
POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")
CARD_FIELDS = tuple(f.name for f in fields(ModelCard) if f.name != "schema")
Box = tuple[int, int, int, int]


class PanelLine(NamedTuple):
    """One panel row: the card ``field``, its row ``text`` and its ``tag``."""

    field: str
    text: str
    tag: str


@dataclass(frozen=True)
class CardPanelSettings:
    """Validated ``model_card_panel_settings.json`` (``ValueError``)."""

    enabled_by_default: bool
    position: str
    font_px: int
    opacity: float
    title: str
    fields: tuple[dict[str, Any], ...]
    line_format: str
    tag_format: str
    unknown_text: str
    number_format: str
    tag_colours: dict[str, tuple[int, int, int]]
    max_value_chars: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled_by_default, bool):
            raise TypeError("enabled_by_default must be bool")
        if self.position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}")
        if min(self.font_px, self.max_value_chars) < 1:
            raise ValueError("font_px and max_value_chars must be positive")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must lie in [0, 1]")
        bad = [e.get("field") for e in self.fields if e.get("field") not in CARD_FIELDS]
        if bad or not all(str(e.get("label", "")).strip() for e in self.fields):
            raise ValueError(f"fields need a label and a card field, got {bad}")
        colours = self.tag_colours
        ok = set(colours) == set(TAGS) and all(
            len(c) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in c)
            for c in colours.values()
        )
        if not ok:
            raise ValueError(f"tag_colours must give three 0-255 integers per {TAGS}")

    def as_dict(self) -> dict[str, Any]:
        """The settings as keyword arguments of this class."""
        return asdict(self)


@lru_cache(maxsize=1)
def load_panel_settings(path: Path | None = None) -> CardPanelSettings:
    """Read the panel settings (``ValueError``/``TypeError`` when invalid)."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    raw.pop("sources", None)
    raw["fields"] = tuple(raw["fields"])
    raw["tag_colours"] = {k: tuple(v) for k, v in raw["tag_colours"].items()}
    return CardPanelSettings(**raw)


def default_enabled() -> bool:
    """Default of the exporter option (``enabled_by_default`` of the settings)."""
    return load_panel_settings().enabled_by_default


def _lookup(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _scalar(value: Any, cfg: CardPanelSettings) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}={_scalar(v, cfg)}" for k, v in value.items())
    if isinstance(value, float):
        return cfg.number_format.format(value)
    return str(value).replace("_", " ")


def _value_text(value: Any, tag: str, keys: list[str], cfg: CardPanelSettings) -> str:
    if tag == UNKNOWN or value == UNKNOWN:
        return cfg.unknown_text
    if isinstance(value, dict):
        found = [_lookup(value, k) for k in keys]
        value = next((v for v in found if v not in (None, UNKNOWN)), UNKNOWN)
        if value == UNKNOWN:
            return cfg.unknown_text
    text = _scalar(value, cfg)
    limit = cfg.max_value_chars
    return text if len(text) <= limit else text[: limit - 3] + "..."


def panel_lines(
    card: ModelCard, settings: CardPanelSettings | None = None
) -> list[PanelLine]:
    """The panel rows of ``card``, in the order of the settings' field list.

    Raises ``TypeError`` unless ``card`` is a :class:`ModelCard`. Postcondition:
    one row per listed field, carrying that field's tag; a field tagged
    unknown (or a record with none of its keys) shows ``unknown_text``.
    """
    if not isinstance(card, ModelCard):
        raise TypeError(f"card must be a ModelCard, got {type(card).__name__}")
    cfg = settings or load_panel_settings()
    out: list[PanelLine] = []
    for entry in cfg.fields:
        name = entry["field"]
        item = getattr(card, name)
        value = _value_text(item.value, item.tag, list(entry.get("keys", [])), cfg)
        text = cfg.line_format.format(label=entry["label"], value=value, tag=item.tag)
        out.append(PanelLine(name, text, item.tag))
    return out


@dataclass(frozen=True)
class ModelCardPanel:
    """The panel of one export: its settings and its rows."""

    settings: CardPanelSettings
    lines: tuple[PanelLine, ...]

    def box(self, shape: tuple[int, ...], limit_y: int) -> Box:
        """Pixel box in the settings' corner; a bottom corner ends above
        ``limit_y`` (the top of the bottom HUD rows), clipped to the frame."""
        cfg = self.settings
        vertical, side = cfg.position.split("-")
        rows = self._rows()
        x0, y0, x1, y1 = compositor.text_panel_box(
            shape, cfg.title, rows, side, cfg.font_px
        )[0]
        if vertical == "bottom":
            bottom = max(y1 - y0, limit_y - compositor.HUD_LAYOUT.panel_gap_px)
            y0, y1 = bottom - (y1 - y0), min(int(shape[0]), bottom)
        return x0, max(0, y0), x1, y1

    def _rows(self) -> list[tuple[str, tuple[str, ...], bool]]:
        tag_format = self.settings.tag_format
        return [(ln.text, (tag_format.format(tag=ln.tag),), False) for ln in self.lines]

    def draw(self, img: compositor.Image8, limit_y: int) -> compositor.Image8:
        """Copy of ``img`` with the panel; only pixels of :meth:`box` change."""
        cfg = self.settings
        colours = [cfg.tag_colours[ln.tag] for ln in self.lines]
        return compositor.draw_text_panel(
            img,
            self.box(img.shape, limit_y),
            cfg.font_px,
            cfg.title,
            self._rows(),
            cfg.opacity,
            colours,
        )


def with_panel(
    overlays: ClipOverlays | None, export: ExportSettings, card: ModelCard
) -> ClipOverlays | None:
    """``overlays`` with the card panel when the option and the HUD are on."""
    if not (export.model_card_panel and export.hud):
        return overlays
    from src.tools.native_viewer_export.core import ClipOverlays

    cfg = load_panel_settings()
    panel = ModelCardPanel(cfg, tuple(panel_lines(card, cfg)))
    return replace(overlays or ClipOverlays(), card_panel=panel)


def draw_all(
    overlays: ClipOverlays | None,
    hud: compositor.HudInfo,
    outputs: dict[str, compositor.Image8],
) -> dict[str, compositor.Image8]:
    """``outputs`` with the card panel drawn clear of the bottom HUD rows."""
    from src.tools.native_viewer_export.ellipsoid_panel import hud_top_px

    panel = None if overlays is None else overlays.card_panel
    if panel is None:
        return outputs
    return {n: panel.draw(f, hud_top_px(f.shape, hud)) for n, f in outputs.items()}
