"""Top-level ``load_club_target`` dispatch -- mirror of the MATLAB loaders.

The source-format-specific loaders are hoisted (xlsx, c3d) under a
single dispatch entry point so engine code never has to special-case the
input file format.

Routing:
    *.xlsx, *.xlsm, *.xls -> :func:`load_club_target_excel`
    *.c3d                 -> :func:`load_club_target_c3d`
    *.mat                 -> :func:`load_club_target_mat`

The Excel loader mirrors ``load_club_target_excel.m`` including the cm-units
convention and the row-1 event-marker header (A/T/I/F/CHS) that pins
``impact_idx`` to the documented sample number rather than the speed-argmax
heuristic.

The MATLAB ``.mat`` loader (issue #4477) consumes a stamped-impact dataset
with full 3-DOF clubface rotation matrices.

Public API:
    load_club_target -- format-dispatched loader returning :class:`ClubTarget`.
    load_club_target_excel -- explicit xlsx loader (re-exported).
    load_club_target_c3d   -- explicit c3d loader (re-exported).
    load_club_target_mat   -- explicit .mat loader (re-exported).
"""

from __future__ import annotations

import logging
from pathlib import Path

from .loaders.c3d import load_club_target_c3d
from .loaders.excel import ALLOWED_SHEETS, load_club_target_excel
from .loaders.matlab_dataset import load_club_target_mat
from .target import AlignOptions, ClubTarget

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_SHEETS",
    "load_club_target",
    "load_club_target_c3d",
    "load_club_target_excel",
    "load_club_target_mat",
]

_EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls"})
_C3D_SUFFIXES = frozenset({".c3d"})
_MAT_SUFFIXES = frozenset({".mat"})


def load_club_target(
    path: Path | str,
    *,
    sheet: str | None = None,
    opts: AlignOptions | None = None,
) -> ClubTarget:
    """Load a :class:`ClubTarget` from any supported source file format.

    Args:
        path:  Path to a Wiffle-style xlsx workbook or a cluster-marker C3D file.
        sheet: Required for xlsx inputs. One of :data:`ALLOWED_SHEETS`.
        opts:  Resampling / impact-alignment options. Defaults to
               :class:`AlignOptions` defaults (1 kHz, 0.3 s, impact-aligned).

    Returns:
        Validated :class:`ClubTarget` on the simulation timegrid.

    Raises:
        ValueError: If the file extension is not recognised, the sheet
            argument is missing for xlsx, or the loader's own validation
            fails.
        FileNotFoundError: Propagated from the underlying loader.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    options = opts if opts is not None else AlignOptions()
    if suffix in _EXCEL_SUFFIXES:
        if sheet is None:
            raise ValueError(
                f"sheet= is required for Excel inputs; allowed: {sorted(ALLOWED_SHEETS)}"
            )
        logger.debug("Dispatching to load_club_target_excel for %s", p.name)
        return load_club_target_excel(p, sheet, options)
    if suffix in _C3D_SUFFIXES:
        logger.debug("Dispatching to load_club_target_c3d for %s", p.name)
        return load_club_target_c3d(p, options)
    if suffix in _MAT_SUFFIXES:
        logger.debug("Dispatching to load_club_target_mat for %s", p.name)
        return load_club_target_mat(p, options)
    raise ValueError(
        f"Unsupported file format {suffix!r} for {p.name}; "
        f"expected one of "
        f"{sorted(_EXCEL_SUFFIXES | _C3D_SUFFIXES | _MAT_SUFFIXES)}"
    )
