"""The one ordered launcher maturity scale and its ``min_maturity`` filter.

The scale lives in ``maturity_scale.json`` (with its ``sources`` string) and
is shared by the launcher manifest loader, the launcher API and the desktop
tile chip. ``deprecated`` is a lifecycle end state: it ranks below every
rung, so it is shown only at the default floor ``deprecated`` and hidden by
any other ``min_maturity``. See the JSON for the status-to-maturity mapping.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path(__file__).with_name("maturity_scale.json")


@functools.cache
def _scale() -> Mapping[str, Any]:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def filter_order() -> tuple[str, ...]:
    """Every level from lowest rank (``deprecated``) to highest."""
    scale = _scale()
    return (scale["deprecated"], *scale["order"])


def valid_levels() -> frozenset[str]:
    """All canonical maturity levels, deprecated included."""
    return frozenset(filter_order())


def default_min_maturity() -> str:
    """The floor that shows everything currently shown."""
    return str(_scale()["default_min_maturity"])


def chip_hidden_levels() -> frozenset[str]:
    """Levels that draw no chip on a launcher tile."""
    return frozenset(_scale()["chip_hidden_levels"])


def validate_level(level: object, *, name: str = "maturity") -> str:
    """Return ``level`` if canonical.

    Raises:
        ValueError: ``level`` is not a canonical maturity level.
    """
    if not isinstance(level, str) or level not in valid_levels():
        raise ValueError(f"{name} must be one of {list(filter_order())}, got {level!r}")
    return level


def meets_min_maturity(tile_maturity: str, min_maturity: str) -> bool:
    """Whether a tile at ``tile_maturity`` passes the ``min_maturity`` floor.

    Postcondition: ``deprecated`` passes only when the floor is ``deprecated``.

    Raises:
        ValueError: either argument is not a canonical maturity level.
    """
    order = filter_order()
    tile = order.index(validate_level(tile_maturity, name="tile_maturity"))
    floor = order.index(validate_level(min_maturity, name="min_maturity"))
    return tile >= floor


def maturity_for_status(status: str | None) -> str:
    """Map a models.yaml ``status`` to a maturity (unknown statuses: stable)."""
    key = status.strip().lower() if isinstance(status, str) else ""
    return str(_scale()["status_to_maturity"].get(key, "stable"))


def resolve_model_maturity(model: Any) -> str:
    """Maturity of a registry model: declared ``launcher.maturity`` else status."""
    launcher = getattr(model, "launcher", None)
    if isinstance(launcher, dict):
        declared, status = launcher.get("maturity"), launcher.get("status")
    else:
        declared = getattr(launcher, "maturity", None)
        status = getattr(launcher, "status", None)
    if isinstance(declared, str) and declared in valid_levels():
        return declared
    return maturity_for_status(status or getattr(model, "status", None))


def filter_models_by_maturity(models: Iterable[Any], min_maturity: str) -> list[Any]:
    """Keep registry models whose resolved maturity meets ``min_maturity``."""
    validate_level(min_maturity, name="min_maturity")
    return [
        m for m in models if meets_min_maturity(resolve_model_maturity(m), min_maturity)
    ]


__all__ = [
    "chip_hidden_levels",
    "default_min_maturity",
    "filter_models_by_maturity",
    "filter_order",
    "maturity_for_status",
    "meets_min_maturity",
    "resolve_model_maturity",
    "valid_levels",
    "validate_level",
]
