"""Curated theme preset library for :class:`PlotStyleSet`.

Bundles four built-in JSON presets shipped under
``src/shared/python/plot_style/presets/``:

* ``default`` — muted matplotlib palette, sphere markers.
* ``scientific_violet`` — violet-purple gradient suited for publication.
* ``monochrome`` — greys with outline emphasis.
* ``high_contrast`` — saturated complementary colors.

Each preset is a fully valid v1 :class:`PlotStyleSet` JSON document and
round-trips through :meth:`PlotStyleSet.save` / :meth:`PlotStyleSet.load`
without loss.

Design-by-Contract
------------------
The library is immutable after construction. All public accessors raise
:class:`KeyError` (with a helpful message listing valid names) when an
unknown preset is requested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, cast

from .persistence import PlotStyleSet

__all__ = ["BUILTIN_PRESET_NAMES", "PresetLibrary"]

# Order is preserved when listing names. Update when a new built-in preset
# JSON is added under ``plot_style/presets/``.
BUILTIN_PRESET_NAMES: tuple[str, ...] = (
    "default",
    "scientific_violet",
    "monochrome",
    "high_contrast",
)

_PRESETS_PACKAGE = "src.shared.python.plot_style.presets"


def _load_builtin_preset(name: str) -> PlotStyleSet:
    """Load one packaged preset JSON via :mod:`importlib.resources`."""
    resource = resources.files(_PRESETS_PACKAGE).joinpath(f"{name}.json")
    with resource.open("r", encoding="utf-8") as handle:
        payload = cast(dict[str, Any], json.load(handle))
    return PlotStyleSet.from_json(payload)


@dataclass(frozen=True)
class PresetLibrary:
    """Immutable, name-indexed collection of :class:`PlotStyleSet` themes.

    Construct via :meth:`default` to load every built-in preset, or pass a
    pre-built mapping for tests / dependency injection.

    Attributes
    ----------
    presets:
        Ordered mapping of preset name to :class:`PlotStyleSet`.
    """

    presets: dict[str, PlotStyleSet] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.presets, dict):
            raise TypeError(
                "presets must be a dict[str, PlotStyleSet]; "
                f"got {type(self.presets).__name__}"
            )
        for key, value in self.presets.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"preset names must be non-empty strings; got {key!r}")
            if not isinstance(value, PlotStyleSet):
                raise TypeError(
                    f"preset {key!r} must be a PlotStyleSet; got {type(value).__name__}"
                )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> PresetLibrary:
        """Load every built-in preset from the package's ``presets/`` dir."""
        loaded = {name: _load_builtin_preset(name) for name in BUILTIN_PRESET_NAMES}
        return cls(presets=loaded)

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> PlotStyleSet:
        if not isinstance(name, str):
            raise TypeError(f"preset name must be a string; got {type(name).__name__}")
        if name not in self.presets:
            available = ", ".join(self.names()) or "(none)"
            raise KeyError(f"unknown preset {name!r}; available presets: {available}")
        return self.presets[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.presets

    def __len__(self) -> int:
        return len(self.presets)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.presets)

    def names(self) -> list[str]:
        """Return preset names in declaration order."""
        return list(self.presets.keys())
