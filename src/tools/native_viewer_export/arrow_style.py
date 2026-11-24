"""Force-arrow looks for the native viewer export, read from settings (#15150).

The owner asked for slimmer, cleaner arrows for shared high-resolution
renders. Each named style (shaft radius, the lowest output height it is
legible at) and the fixed head length live in
``src/config/force_arrow_style.json``, so they are settings, never literals
in the drawing code. ``standard`` keeps the NV-9 720p legibility floor
(#11697); ``slim`` is for 1080p and above.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = [
    "ARROW_STYLE_PATH",
    "ArrowStyle",
    "arrow_style_names",
    "head_span_shaft_radii",
    "load_arrow_style",
]

ARROW_STYLE_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "force_arrow_style.json"
)


@dataclass(frozen=True)
class ArrowStyle:
    """One arrow look: shaft radius (m) and its legibility floor."""

    name: str
    shaft_radius_m: float
    min_height_px: float
    min_shaft_px: float


def _positive(doc: dict, key: str, where: str) -> float:
    value = doc.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{where}: {key} must be a positive number, got {value!r}")
    return float(value)


def _document(path: Path) -> dict:
    # Keyed on the file's stamp so an edited settings file is re-read.
    stat = path.stat()
    return _parse(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _parse(path: Path, _mtime_ns: int, _size: int) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    styles = doc.get("styles") if isinstance(doc, dict) else None
    if not isinstance(styles, dict) or not styles:
        raise ValueError(f"{path}: styles must be a non-empty object")
    if doc.get("default_style") not in styles:
        raise ValueError(f"{path}: default_style must name one of {sorted(styles)}")
    _positive(doc, "head_span_shaft_radii", str(path))
    return doc


def arrow_style_names(path: Path | None = None) -> tuple[str, ...]:
    """Names of the configured arrow styles, default first."""
    doc = _document(Path(path) if path is not None else ARROW_STYLE_PATH)
    return (doc["default_style"], *sorted(set(doc["styles"]) - {doc["default_style"]}))


def head_span_shaft_radii(path: Path | None = None) -> float:
    """Arrow head length as a multiple of the arrow's shaft radius."""
    doc = _document(Path(path) if path is not None else ARROW_STYLE_PATH)
    return float(doc["head_span_shaft_radii"])


def load_arrow_style(name: str | None = None, path: Path | None = None) -> ArrowStyle:
    """Return the named arrow style (default: the configured default style).

    Raises ``ValueError`` for an unknown name or a setting that is missing,
    non-numeric or not positive and finite.
    """
    path = Path(path) if path is not None else ARROW_STYLE_PATH
    doc = _document(path)
    key = doc["default_style"] if name is None else name
    if key not in doc["styles"]:
        raise ValueError(f"unknown arrow style {key!r}; choose {sorted(doc['styles'])}")
    entry = doc["styles"][key]
    where = f"{path} style {key!r}"
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: must be an object")
    return ArrowStyle(
        key,
        _positive(entry, "shaft_radius_m", where),
        _positive(entry, "min_height_px", where),
        _positive(entry, "min_shaft_px", where),
    )
