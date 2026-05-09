"""Colormap identifiers and user-defined colormap dataclass.

:class:`ColormapId` enumerates matplotlib built-ins (passed through
verbatim) plus a small set of *semantic* aliases this codebase
registers for kinematic / kinetic data (velocity, force, ...).

:class:`CustomColormap` defines a user-supplied colormap via a list of
``(position, hex)`` stops. Strict validation of monotonic positions and
parseable hex strings happens in ``__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from matplotlib.colors import is_color_like

__all__ = ["ColormapId", "CustomColormap", "SEMANTIC_COLORMAP_ALIASES"]


class ColormapId(str, Enum):
    """Built-in and semantic colormap identifiers.

    Matplotlib built-ins are passed through to ``matplotlib.cm`` by name.
    Semantic aliases (``VELOCITY``, ``FORCE``, ...) resolve to the
    matplotlib name listed in :data:`SEMANTIC_COLORMAP_ALIASES`.
    """

    # ---- Matplotlib built-ins ----
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    INFERNO = "inferno"
    CIVIDIS = "cividis"
    TURBO = "turbo"
    COOLWARM = "coolwarm"
    SPECTRAL = "spectral"

    # ---- Semantic aliases (registered by this codebase) ----
    VELOCITY = "velocity"
    FORCE = "force"
    ACCELERATION = "acceleration"
    HEIGHT = "height"
    GENERIC_DIVERGING = "generic_diverging"

    def __str__(self) -> str:
        return self.value


# Resolve a semantic alias -> built-in matplotlib name.
SEMANTIC_COLORMAP_ALIASES: dict[ColormapId, ColormapId] = {
    ColormapId.VELOCITY: ColormapId.PLASMA,
    ColormapId.FORCE: ColormapId.INFERNO,
    ColormapId.ACCELERATION: ColormapId.TURBO,
    ColormapId.HEIGHT: ColormapId.VIRIDIS,
    ColormapId.GENERIC_DIVERGING: ColormapId.COOLWARM,
}


def resolve_colormap_alias(cmap_id: ColormapId) -> ColormapId:
    """Resolve a semantic alias to its underlying matplotlib colormap.

    Built-in identifiers are returned unchanged.
    """
    if not isinstance(cmap_id, ColormapId):
        raise TypeError(f"cmap_id must be ColormapId; got {type(cmap_id).__name__}")
    return SEMANTIC_COLORMAP_ALIASES.get(cmap_id, cmap_id)


@dataclass(frozen=True)
class CustomColormap:
    """User-defined colormap built from ``(position, hex)`` stops.

    Attributes
    ----------
    name:
        Unique non-empty identifier.
    stops:
        Tuple of ``(position, hex)`` pairs. ``position`` lies in
        ``[0, 1]`` and the sequence is strictly increasing.
    """

    name: str
    stops: tuple[tuple[float, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string; got {self.name!r}")
        if not isinstance(self.stops, tuple):
            raise TypeError(
                "stops must be a tuple of (position, hex) pairs; "
                f"got {type(self.stops).__name__}"
            )
        if len(self.stops) < 2:
            raise ValueError(
                f"stops must contain at least 2 entries; got {len(self.stops)}"
            )

        last_pos = -float("inf")
        for index, stop in enumerate(self.stops):
            if not isinstance(stop, tuple) or len(stop) != 2:
                raise ValueError(
                    "each stop must be a (position, hex) 2-tuple; "
                    f"got {stop!r} at index {index}"
                )
            position, hex_value = stop
            if not isinstance(position, (int, float)) or isinstance(position, bool):
                raise TypeError(
                    f"stop position must be numeric; got {position!r} at {index}"
                )
            position_f = float(position)
            if not 0.0 <= position_f <= 1.0:
                raise ValueError(
                    "stop positions must lie in [0, 1]; "
                    f"got {position_f} at index {index}"
                )
            if position_f <= last_pos:
                raise ValueError(
                    "stop positions must be strictly increasing; "
                    f"got {position_f} after {last_pos} at index {index}"
                )
            last_pos = position_f
            if not isinstance(hex_value, str) or not hex_value:
                raise ValueError(
                    f"stop hex must be a non-empty string; got {hex_value!r}"
                )
            if not is_color_like(hex_value):
                raise ValueError(
                    f"stop hex {hex_value!r} is not a parseable matplotlib color"
                )
