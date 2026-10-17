"""Fail-closed readers of open-fp swing files (OF-02, epic #13061).

A swing is a ``<id>.json`` + ``<id>.npz`` pair (or the CSV export of one) in the
documented open-fp layout.  The readers validate everything they can, convert
the CoP from millimetres to metres, and return the engine-neutral
:class:`~src.shared.python.force_plates.contracts.VerticalPlateTrial`, whose own
constructor enforces the CoP/threshold/net-CoP source rules (those errors
propagate).  Physical declarations that a file cannot carry (uncertainty,
sensing depth, capacity) are required arguments, never guessed.

open-fp is AGPL-3.0; this code is written from the documented file format only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import numpy as np

from src.shared.python.force_plates.contracts import (
    Handedness,
    PlateProvenance,
    PlateUncertainty,
    VerticalPlateTrial,
    finite_non_negative,
    finite_positive,
)
from src.shared.python.force_plates.open_fp_format import _arrays, _files, _meta
from src.shared.python.force_plates.open_fp_format import format as fmt

__all__ = ["PlateDeclarations", "list_swings", "read_swing", "read_swing_csv"]

_DEFAULT_DEVICE = "open-fp"


@dataclass(frozen=True)
class PlateDeclarations:
    """Physical facts the swing files cannot carry; required, with no defaults.

    ``uncertainty`` is the declared 1-sigma measurement uncertainty,
    ``sensing_depth_m`` the depth of the force-sensing plane below the plate top
    [m] (finite, non-negative), ``capacity_n`` the total capacity of the pair [N]
    (positive), ``device`` a non-empty label for the provenance record.
    """

    uncertainty: PlateUncertainty
    sensing_depth_m: float
    capacity_n: float
    device: str = _DEFAULT_DEVICE

    def __post_init__(self) -> None:
        if not isinstance(self.uncertainty, PlateUncertainty):
            raise TypeError("uncertainty must be a PlateUncertainty")
        finite_positive(self.capacity_n, "capacity_n")
        finite_non_negative(self.sensing_depth_m, "sensing_depth_m")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError(f"device must be a non-empty string, got {self.device!r}")


@dataclass(frozen=True)
class _Header:
    """Per-trial scalars not held in the arrays."""

    handedness: Handedness
    body_weight_n: float
    events_s: Mapping[str, float | None]


def _require_declared(declared: object) -> PlateDeclarations:
    if not isinstance(declared, PlateDeclarations):
        raise TypeError("declared must be a PlateDeclarations")
    return declared


def _events_on_time_axis(
    rel: Mapping[str, float | None], t: np.ndarray, path: Path
) -> dict[str, float | None]:
    """Shift events (seconds from ``t[0]``) onto the ``t`` clock, checking range."""
    start, span = float(t[0]), float(t[-1] - t[0])
    out: dict[str, float | None] = {}
    for name, value in rel.items():
        if value is not None and not 0.0 <= value <= span + fmt.EVENT_EDGE_TOL_S:
            raise ValueError(
                f"{path}: events[{name!r}] = {value} lies outside the trial [0, {span}] s"
            )
        out[name] = None if value is None else min(start + value, float(t[-1]))
    return out


def _build_trial(
    arrays: _arrays.SwingArrays,
    header: _Header,
    declared: PlateDeclarations,
    digests: Mapping[str, str],
) -> VerticalPlateTrial:
    per_m = fmt.MM_PER_M
    provenance = PlateProvenance(
        device=declared.device,
        source_format=fmt.SOURCE_FORMAT,
        source_revision=fmt.SOURCE_REVISION,
        file_sha256=digests,
        calibration_id=None,
        qualification=None,
    )
    return VerticalPlateTrial(
        times_s=arrays.t,
        load_n={"left": arrays.fz_left, "right": arrays.fz_right},
        cop_m={
            "left": arrays.cop_left / per_m,
            "right": arrays.cop_right / per_m,
            "net": arrays.cop_combined / per_m,
        },
        body_weight_n=header.body_weight_n,
        handedness=header.handedness,
        events_s=header.events_s,
        cop_min_load_n=fmt.SOURCE_COP_MIN_LOAD_N,
        capacity_n=declared.capacity_n,
        sensing_depth_m=declared.sensing_depth_m,
        provenance=provenance,
        uncertainty=declared.uncertainty,
        corner_forces_n=arrays.forces,
    )


def read_swing(
    json_path: str | Path, *, declared: PlateDeclarations
) -> VerticalPlateTrial:
    """Read the swing ``<id>.json`` and its sibling ``<id>.npz``.

    Preconditions: ``json_path`` is a regular (non-symlink) ``<id>.json`` file
    whose stem is a swing id equal to the JSON ``id``, with a regular sibling
    ``.npz``; both within the size caps; ``declared`` a
    :class:`PlateDeclarations`.  Events (seconds from ``t[0]`` in the file) are
    returned on the time axis of ``times_s``.  ``impact_source`` is only used to
    require that an ``impact`` event has a source.

    Only the stem of ``json_path`` is validated (not its directory components):
    callers pass trusted directories.  Raises ``TypeError`` for wrong argument
    types and ``ValueError`` (naming the
    file and field) for anything malformed, inconsistent or out of range,
    including violations found by the trial contract.
    """
    decl = _require_declared(declared)
    json_file = _files.checked_file(
        _files.as_path(json_path, "json_path"), ".json", swing_id_stem=True
    )
    npz_file = _files.checked_file(
        json_file.with_suffix(".npz"), ".npz", swing_id_stem=True
    )
    json_bytes = _files.read_limited(json_file, fmt.MAX_JSON_BYTES)
    npz_bytes = _files.read_limited(npz_file, fmt.MAX_NPZ_BYTES)
    meta = _meta.parse_meta(json_bytes, json_file)
    arrays = _arrays.load_npz(npz_bytes, npz_file)
    _arrays.check_lead_pct(arrays, meta.handedness, _arrays.NPZ_LEAD_CHECK, npz_file)
    header = _Header(
        meta.handedness,
        meta.body_weight_n,
        _events_on_time_axis(meta.events_rel_s, arrays.t, json_file),
    )
    digests = {
        "json": _files.sha256_hex(json_bytes),
        "npz": _files.sha256_hex(npz_bytes),
    }
    return _build_trial(arrays, header, decl, digests)


# ------------------------------------------------------------------- CSV --
_CSV_COLUMN_COUNT = len(fmt.CSV_HEADER.split(","))
_CSV_NUMBER = re.compile(r"-?\d{1,20}(\.\d{1,20})?")


def _csv_cell(cell: str, path: Path, row: int) -> float:
    if cell == "":
        return math.nan
    if not _CSV_NUMBER.fullmatch(cell):
        raise ValueError(f"{path}: row {row} has a malformed number {cell[:40]!r}")
    return float(cell)


def _csv_matrix(rows: Iterable[str], path: Path) -> np.ndarray:
    parsed = []
    for number, line in enumerate(rows, start=2):
        cells = line.split(",")
        if len(cells) != _CSV_COLUMN_COUNT:
            raise ValueError(
                f"{path}: row {number} has {len(cells)} cells, expected {_CSV_COLUMN_COUNT}"
            )
        parsed.append([_csv_cell(cell, path, number) for cell in cells])
    if len(parsed) < 2:
        raise ValueError(f"{path}: a swing needs at least 2 data rows")
    return np.array(parsed, dtype=np.float64)


def _arrays_from_csv(matrix: np.ndarray) -> _arrays.SwingArrays:
    col = matrix.T
    return _arrays.SwingArrays(
        t=col[0],
        lead_pct=col[1],
        fz_left=col[2],
        fz_right=col[3],
        cop_left=col[4:6].T,
        cop_right=col[6:8].T,
        cop_combined=col[8:10].T,
        forces=col[10:18].T,
    )


def _split_csv(data: bytes, path: Path) -> list[str]:
    """Data lines of the export; it has no quoting, so no csv parser is needed."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not valid UTF-8 ({exc})") from exc
    lines = text.splitlines()
    if not lines or lines[0] != fmt.CSV_HEADER:
        raise ValueError(f"{path}: CSV header does not match the documented export")
    return lines[1:]


def _check_csv_arguments(handedness: object, body_weight_n: object) -> None:
    if handedness not in get_args(Handedness):
        raise ValueError(f"handedness must be 'RH' or 'LH', got {handedness!r}")
    finite_positive(body_weight_n, "body_weight_n")


def read_swing_csv(
    csv_path: str | Path,
    *,
    handedness: Handedness,
    body_weight_n: float,
    declared: PlateDeclarations,
) -> VerticalPlateTrial:
    """Read the CSV export of a swing.

    The export carries neither handedness, body weight nor events, so the first
    two are required arguments (``handedness`` is verified against ``lead_pct``)
    and the events mapping is empty.  Values are rounded in the file (0.01 N,
    0.01 mm, 0.0001 s); the CSV cross-check tolerances in
    :mod:`~src.shared.python.force_plates.open_fp_format.format` cover that, and
    the contract's net-CoP tolerance (0.1 mm) covers the CoP rounding.  A load
    rounded across the 20 N CoP threshold is rejected by the contract.

    Raises ``TypeError`` / ``ValueError`` as :func:`read_swing` does.
    """
    decl = _require_declared(declared)
    _check_csv_arguments(handedness, body_weight_n)
    path = _files.checked_file(
        _files.as_path(csv_path, "csv_path"), ".csv", swing_id_stem=False
    )
    data = _files.read_limited(path, fmt.MAX_CSV_BYTES)
    arrays = _arrays_from_csv(_csv_matrix(_split_csv(data, path), path))
    _arrays.validate_arrays(arrays, path)
    _arrays.check_lead_pct(arrays, handedness, _arrays.CSV_LEAD_CHECK, path)
    header = _Header(handedness, float(body_weight_n), {})
    return _build_trial(arrays, header, decl, {"csv": _files.sha256_hex(data)})


def list_swings(directory: str | Path) -> tuple[str, ...]:
    """Sorted ids of the swings in ``directory``.

    An id is listed when ``<id>.json`` and ``<id>.npz`` are both regular files
    (symlinks are never followed) and the stem matches the swing id pattern.
    Other files are ignored; nothing is opened or parsed.  ``ValueError`` if
    ``directory`` is not a directory.
    """
    root = _files.as_path(directory, "directory")
    if not root.is_dir():
        raise ValueError(f"{root}: not a directory")

    def regular(candidate: Path) -> bool:
        return candidate.is_file() and not candidate.is_symlink()

    ids = {
        entry.stem
        for entry in root.glob("*.json")
        if fmt.ID_PATTERN.match(entry.stem)
        and regular(entry)
        and regular(entry.with_suffix(".npz"))
    }
    return tuple(sorted(ids))
