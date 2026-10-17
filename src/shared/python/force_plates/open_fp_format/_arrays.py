"""Strict loading and checking of the swing arrays (``.npz``, shared with CSV)."""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.force_plates.contracts import Handedness
from src.shared.python.force_plates.open_fp_format import format as fmt

__all__ = [
    "CSV_LEAD_CHECK",
    "NPZ_LEAD_CHECK",
    "LeadCheck",
    "SwingArrays",
    "check_lead_pct",
    "load_npz",
    "validate_arrays",
]

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class SwingArrays:
    """Swing arrays in file units (N, mm, percent, s), all length ``n``."""

    t: Array
    forces: Array
    fz_left: Array
    fz_right: Array
    cop_left: Array
    cop_right: Array
    cop_combined: Array
    lead_pct: Array


@dataclass(frozen=True)
class LeadCheck:
    """Tolerances of the ``lead_pct`` cross-check: percentage points and newtons."""

    tol_pct: float
    slack_n: float


NPZ_LEAD_CHECK = LeadCheck(fmt.LEAD_PCT_TOL, fmt.LOAD_SLACK_N)
CSV_LEAD_CHECK = LeadCheck(fmt.LEAD_PCT_TOL_CSV, fmt.LOAD_SLACK_N_CSV)

_FULL_SCALE_PCT = 100.0
_MAX_NDIM = 2


class _Refused(ValueError):
    """An npz member this reader declines to load (already names the file)."""


_ALLOWED_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
_ZIP_ENCRYPTED_FLAG = 0x1
_READ_ERRORS = (
    ValueError,
    OSError,
    EOFError,
    zipfile.BadZipFile,
    NotImplementedError,
    RuntimeError,
)


def _read_npy_header(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> tuple[tuple[int, ...], np.dtype, int]:
    """Shape, dtype and header length of the ``.npy`` member ``info``."""
    with archive.open(info) as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(member)
        else:
            raise ValueError(f"unsupported npy format version {version}")
        return tuple(shape), dtype, member.tell()


def _check_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, path: Path) -> None:
    """Refuse unsupported/encrypted members and headers that lie about size."""
    name = info.filename
    if info.flag_bits & _ZIP_ENCRYPTED_FLAG:
        raise _Refused(f"{path}: npz member {name!r} is encrypted")
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise _Refused(
            f"{path}: npz member {name!r} uses unsupported compression "
            f"{info.compress_type}"
        )
    try:
        shape, dtype, header_len = _read_npy_header(archive, info)
    except _READ_ERRORS as exc:
        raise _Refused(
            f"{path}: npz member {name!r} has a bad npy header ({exc})"
        ) from exc
    if dtype.hasobject or len(shape) > _MAX_NDIM or any(n < 0 for n in shape):
        raise _Refused(f"{path}: npz member {name!r} has an unacceptable npy header")
    size = math.prod(shape) * dtype.itemsize
    if size > fmt.MAX_NPZ_BYTES or header_len + size != info.file_size:
        raise _Refused(
            f"{path}: npz member {name!r} header claims {size} data bytes, which does "
            f"not fit the {info.file_size}-byte member"
        )


def _open_npz(data: bytes, path: Path) -> np.lib.npyio.NpzFile:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if sum(info.file_size for info in infos) > fmt.MAX_NPZ_BYTES:
                raise _Refused(f"{path}: npz expands to more than the size limit")
            for info in infos:
                _check_member(archive, info, path)
        return np.load(io.BytesIO(data), allow_pickle=False)
    except _Refused:
        raise
    except _READ_ERRORS as exc:
        raise ValueError(f"{path}: not a valid npz archive ({exc})") from exc


def _member(archive: np.lib.npyio.NpzFile, key: str, path: Path) -> Array:
    try:
        raw = archive[key]
    except _READ_ERRORS as exc:
        raise ValueError(f"{path}: npz array {key!r} is unreadable ({exc})") from exc
    if not isinstance(raw, np.ndarray) or raw.dtype.kind != "f":
        raise ValueError(f"{path}: npz array {key!r} must be a float array")
    return raw.astype(np.float64)


def _check_shapes(arrays: dict[str, Array], path: Path) -> None:
    t = arrays["t"]
    if t.ndim != 1 or t.shape[0] < 2:
        raise ValueError(f"{path}: array 't' must be 1-D with at least 2 samples")
    for key, width in fmt.ARRAY_KEYS.items():
        expected = (t.shape[0],) if width is None else (t.shape[0], width)
        if arrays[key].shape != expected:
            raise ValueError(
                f"{path}: array {key!r} has shape {arrays[key].shape}, expected {expected}"
            )


def load_npz(data: bytes, path: Path) -> SwingArrays:
    """Load the documented arrays from npz ``data`` read from ``path``.

    Pickles are never loaded.  Raises ``ValueError`` for an unreadable archive,
    a key set other than the documented one, a non-float dtype, inconsistent
    shapes, or values failing :func:`validate_arrays`.
    """
    with _open_npz(data, path) as archive:
        missing = sorted(set(fmt.ARRAY_KEYS) - set(archive.files))
        extra = sorted(set(archive.files) - set(fmt.ARRAY_KEYS))
        if missing or extra:
            raise ValueError(
                f"{path}: npz missing arrays {missing}, unexpected {extra}"
            )
        arrays = {key: _member(archive, key, path) for key in fmt.ARRAY_KEYS}
    _check_shapes(arrays, path)
    swing = SwingArrays(
        t=arrays["t"],
        forces=arrays["forces"],
        fz_left=arrays["fz_L"],
        fz_right=arrays["fz_R"],
        cop_left=arrays["cop_L"],
        cop_right=arrays["cop_R"],
        cop_combined=arrays["cop_C"],
        lead_pct=arrays["lead_pct"],
    )
    validate_arrays(swing, path)
    return swing


def validate_arrays(arrays: SwingArrays, path: Path) -> None:
    """Check time, load and corner-force values (shared by NPZ and CSV).

    Time, loads and corner forces must be finite (the check_lead_pct and
    event steps need that); time ordering  Load sign (the noise-floor policy), CoP and net-CoP rules belong
    to the trial contract and are checked there, once.
    """
    if not np.isfinite(arrays.t).all():
        raise ValueError(f"{path}: 't' must be finite")
    for key, load in (("fz_L", arrays.fz_left), ("fz_R", arrays.fz_right)):
        if not np.isfinite(load).all():
            raise ValueError(f"{path}: {key!r} must be finite")
    if not np.isfinite(arrays.forces).all():
        raise ValueError(f"{path}: 'forces' must be finite")


def check_lead_pct(
    arrays: SwingArrays, handedness: Handedness, check: LeadCheck, path: Path
) -> None:
    """Cross-check ``lead_pct`` against the loads and ``handedness``.

    ``lead_pct`` must be NaN exactly where the total load is below the source
    threshold (outside a ``check.slack_n`` band, where rounding makes it
    ambiguous), and elsewhere equal ``100 * lead / total`` within
    ``check.tol_pct``.  A golfer entered with the wrong handedness fails here.
    """
    total = arrays.fz_left + arrays.fz_right
    lead = arrays.fz_left if handedness == "RH" else arrays.fz_right
    pct = arrays.lead_pct
    thr, slack = fmt.SOURCE_COP_MIN_LOAD_N, check.slack_n
    if np.any((total >= thr + slack) & np.isnan(pct)) or np.any(
        (total < thr - slack) & ~np.isnan(pct)
    ):
        raise ValueError(f"{path}: lead_pct NaN pattern does not match the total load")
    known = ~np.isnan(pct)
    expected = _FULL_SCALE_PCT * lead[known] / total[known]
    if np.any(np.abs(pct[known] - expected) > check.tol_pct):
        raise ValueError(
            f"{path}: lead_pct is inconsistent with the loads for handedness "
            f"{handedness!r}"
        )
