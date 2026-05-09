"""Top-level ``load_body_target`` dispatch — sibling of ``load_club_target``.

Currently only ``.c3d`` is wired in; the dispatcher shape leaves room for
future formats (``.trc``, ``.bvh``, ``.npz`` etc.) without changing the
public API.

Public API:
    load_body_target       -- format-dispatched loader returning ``BodyTarget``.
    load_body_target_c3d   -- explicit c3d loader (re-exported).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .body_target import BodyTarget
from .club_target import AlignOptions, ClubTarget
from .loaders.body_json import load_body_target_json
from .loaders.c3d_body import load_body_target_c3d

logger = logging.getLogger(__name__)

__all__ = [
    "load_body_target",
    "load_body_target_c3d",
    "load_body_target_json",
]

_C3D_SUFFIXES = frozenset({".c3d"})
_JSON_SUFFIXES = frozenset({".json"})


def load_body_target(
    path: Path | str,
    *,
    opts: AlignOptions | None = None,
    marker_set: Sequence[str] | None = None,
    impact_source: ClubTarget | None = None,
) -> BodyTarget:
    """Load a :class:`BodyTarget` from any supported source file format.

    Args:
        path:           Path to a supported source file (currently ``.c3d``).
        opts:           Resampling / impact-alignment options. Defaults to
                        :class:`AlignOptions` defaults (1 kHz, 0.3 s, impact-aligned).
        marker_set:     Optional explicit marker subset to extract.
        impact_source:  Optional :class:`ClubTarget` to share the timegrid with.

    Returns:
        Validated :class:`BodyTarget` on the simulation timegrid.

    Raises:
        ValueError:        If the file extension is unsupported or downstream
                           validation fails.
        FileNotFoundError: Propagated from the underlying loader.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    options = opts if opts is not None else AlignOptions()
    if suffix in _C3D_SUFFIXES:
        logger.debug("Dispatching to load_body_target_c3d for %s", p.name)
        return load_body_target_c3d(
            p, options, marker_set=marker_set, impact_source=impact_source
        )
    if suffix in _JSON_SUFFIXES:
        logger.debug("Dispatching to load_body_target_json for %s", p.name)
        return load_body_target_json(
            p, options, marker_set=marker_set, impact_source=impact_source
        )
    raise ValueError(
        f"Unsupported file format {suffix!r} for {p.name}; "
        f"expected one of {sorted(_C3D_SUFFIXES | _JSON_SUFFIXES)}"
    )
