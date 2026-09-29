"""Colormap registry resolving :class:`ColormapId` and custom names.

This module is the single entry point for retrieving an actual
``matplotlib.colors.Colormap`` instance from a :class:`ColormapId`
enum value, a semantic alias, or a custom-colormap name registered
at runtime via :func:`register_custom_colormap`.

The semantic-alias map (``VELOCITY``, ``FORCE``, ``ACCELERATION``,
``HEIGHT``, ``GENERIC_DIVERGING``) is immutable at module load and
delegates to the matplotlib built-in chosen for each kinematic /
kinetic data family.

Registration of :class:`CustomColormap` is *idempotent* — registering
the same colormap (matched by ``name + stops``) twice is a no-op.
Registering a *different* colormap under an existing name raises
:class:`ValueError`.
"""

from __future__ import annotations

from typing import Final

import matplotlib.colors as mcolors
from matplotlib import colormaps as _mpl_colormaps

from .colormaps import ColormapId, CustomColormap, resolve_colormap_alias

__all__ = [
    "get_colormap",
    "list_colormaps",
    "register_custom_colormap",
    "unregister_custom_colormap",
]


# ColormapId.value -> matplotlib registry name. Most ColormapId values
# already match the matplotlib name; the exception is ``SPECTRAL``
# which matplotlib spells with a capital S.
_BUILTIN_MPL_NAME: Final[dict[ColormapId, str]] = {
    ColormapId.VIRIDIS: "viridis",
    ColormapId.PLASMA: "plasma",
    ColormapId.MAGMA: "magma",
    ColormapId.INFERNO: "inferno",
    ColormapId.CIVIDIS: "cividis",
    ColormapId.TURBO: "turbo",
    ColormapId.COOLWARM: "coolwarm",
    ColormapId.SPECTRAL: "Spectral",
}


# Process-global registry for user-defined colormaps. Keyed by
# ``CustomColormap.name``; values are the original :class:`CustomColormap`
# (for idempotency comparisons) plus the materialised matplotlib
# :class:`~matplotlib.colors.LinearSegmentedColormap`.
_CUSTOM_COLORMAPS: dict[str, tuple[CustomColormap, mcolors.Colormap]] = {}


def _materialise_custom(cmap: CustomColormap) -> mcolors.Colormap:
    """Build a matplotlib colormap from ``CustomColormap`` stops."""
    return mcolors.LinearSegmentedColormap.from_list(
        cmap.name, [(pos, hex_) for pos, hex_ in cmap.stops]
    )


def _resolve_builtin(cmap_id: ColormapId) -> mcolors.Colormap:
    """Resolve a :class:`ColormapId` (or alias) to a matplotlib colormap."""
    target = resolve_colormap_alias(cmap_id)
    mpl_name = _BUILTIN_MPL_NAME.get(target)
    if mpl_name is None:  # pragma: no cover - defensive guard
        raise KeyError(f"ColormapId {target!r} has no matplotlib mapping")
    return _mpl_colormaps[mpl_name]


def get_colormap(cmap_id: ColormapId | str) -> mcolors.Colormap:
    """Return the :class:`~matplotlib.colors.Colormap` for ``cmap_id``.

    Parameters
    ----------
    cmap_id:
        Either a :class:`ColormapId` enum value (built-in or semantic
        alias) or the ``name`` of a previously registered
        :class:`CustomColormap`.

    Raises
    ------
    TypeError
        If ``cmap_id`` is neither :class:`ColormapId` nor :class:`str`.
    KeyError
        If a string ``cmap_id`` does not match any registered custom
        colormap name and is not a valid :class:`ColormapId` value.
    """
    if isinstance(cmap_id, ColormapId):
        return _resolve_builtin(cmap_id)
    if isinstance(cmap_id, str):
        if cmap_id in _CUSTOM_COLORMAPS:
            return _CUSTOM_COLORMAPS[cmap_id][1]
        # Allow the *value* of a ColormapId (e.g. "viridis") as a
        # convenience alias for the enum.
        try:
            return _resolve_builtin(ColormapId(cmap_id))
        except ValueError as exc:
            raise KeyError(
                f"colormap {cmap_id!r} is not registered and is not a ColormapId value"
            ) from exc
    raise TypeError(f"cmap_id must be ColormapId or str; got {type(cmap_id).__name__}")


def register_custom_colormap(cmap: CustomColormap) -> None:
    """Register a :class:`CustomColormap` for retrieval by name.

    Idempotent for the same logical colormap — registering a
    :class:`CustomColormap` that compares equal (by ``name`` and
    ``stops``) to one already registered is a no-op.

    Raises
    ------
    TypeError
        If ``cmap`` is not a :class:`CustomColormap`.
    ValueError
        If a different colormap is already registered under ``cmap.name``.
    """
    if not isinstance(cmap, CustomColormap):
        raise TypeError(
            f"cmap must be a CustomColormap instance; got {type(cmap).__name__}"
        )
    if cmap.name in _BUILTIN_MPL_NAME.values() or any(
        cmap.name == cid.value for cid in ColormapId
    ):
        raise ValueError(
            f"colormap name {cmap.name!r} clashes with a built-in "
            "ColormapId value; choose a different name"
        )

    existing = _CUSTOM_COLORMAPS.get(cmap.name)
    if existing is not None:
        existing_cmap = existing[0]
        if existing_cmap.stops == cmap.stops:
            return  # idempotent: same logical colormap
        raise ValueError(
            f"colormap name {cmap.name!r} is already registered with "
            f"different stops: existing={existing_cmap.stops!r} "
            f"new={cmap.stops!r}"
        )

    _CUSTOM_COLORMAPS[cmap.name] = (cmap, _materialise_custom(cmap))


def unregister_custom_colormap(name: str) -> None:
    """Remove a previously registered custom colormap by ``name``.

    Intended for tests that need to reset the global registry between
    cases. Raises :class:`KeyError` if ``name`` is not registered.
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a string; got {type(name).__name__}")
    if name not in _CUSTOM_COLORMAPS:
        raise KeyError(f"no custom colormap named {name!r} is registered")
    del _CUSTOM_COLORMAPS[name]


def list_colormaps() -> tuple[str, ...]:
    """Return the names of every available colormap.

    The result contains every :class:`ColormapId` value (built-in and
    semantic alias) followed by the names of currently registered
    :class:`CustomColormap` instances, in registration order.
    """
    builtins = tuple(cid.value for cid in ColormapId)
    customs = tuple(_CUSTOM_COLORMAPS.keys())
    return builtins + customs
