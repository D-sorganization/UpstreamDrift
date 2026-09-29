"""ColorResolver implementations sub-package.

Each module here exposes a class implementing the
:class:`~src.shared.python.plot_style.contracts.ColorResolver` Protocol
for one of the three :data:`~src.shared.python.plot_style.colors.ColorScale`
variants:

* :class:`StaticColor` — passes a hex / RGBA through unchanged.
* :class:`PaletteColor` — categorical lookup into a named palette.
* :class:`DataDrivenColor` — channel value → vmin/vmax normalisation
  → colormap LUT sampling, with a vectorised bulk path for
  animation-rate frame queries.

The :data:`RESOLVER_REGISTRY` dispatch table maps each
:data:`ColorScale` dataclass to the matching resolver class so callers
can build a resolver from any scale without an ``isinstance`` ladder.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..colors import (
    DataDrivenColor as _DataDrivenColorScale,
    PaletteColor as _PaletteColorScale,
    StaticColor as _StaticColorScale,
)
from .data_driven import DataDrivenColor
from .palette import PaletteColor
from .static import StaticColor

__all__ = [
    "RESOLVER_REGISTRY",
    "DataDrivenColor",
    "PaletteColor",
    "StaticColor",
]


RESOLVER_REGISTRY: Mapping[type, type] = {
    _StaticColorScale: StaticColor,
    _PaletteColorScale: PaletteColor,
    _DataDrivenColorScale: DataDrivenColor,
}
