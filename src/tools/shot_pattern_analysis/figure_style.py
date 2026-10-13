"""Shared figure palette and reproducible figure-file metadata.

Every publication figure draws Straight, Draw and Fade in the same colours,
and PDF and SVG files omit their creation timestamp so that re-rendering the
same inputs produces byte-identical files.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

PUBLICATION_PATTERN_COLORS: Final = {
    "Straight": "#286090",
    "Draw": "#c66a19",
    "Fade": "#7b4a9e",
}
"""Light-background palette used by every article and report figure."""

DARK_PATTERN_COLORS: Final = {
    "Straight": "#83d7ff",
    "Draw": "#f8bf58",
    "Fade": "#e98ac0",
}
"""Dark-background palette used by the desktop tool's quick-look outputs."""

FIGURE_CREATOR: Final = "UpstreamDrift shot-pattern analysis"

_TIMESTAMP_KEYS: Final = {"pdf": "CreationDate", "svg": "Date"}


def reproducible_metadata(
    file_format: str, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return ``savefig`` metadata for ``file_format`` without a timestamp.

    Matplotlib stamps PDF and SVG files with the render time by default; that
    key is set to ``None`` here so it is omitted. ``extra`` entries are kept
    but may not set the timestamp. Raster formats get ``extra`` unchanged.
    """
    if not isinstance(file_format, str) or not file_format:
        raise ValueError("file_format must be a non-empty string such as 'pdf'")
    metadata = dict(extra or {})
    timestamp = _TIMESTAMP_KEYS.get(file_format.lower())
    if timestamp is None:
        return metadata
    if metadata.get(timestamp) is not None:
        raise ValueError(f"{timestamp} would make the {file_format} non-reproducible")
    metadata.setdefault("Creator", FIGURE_CREATOR)
    metadata[timestamp] = None
    return metadata
