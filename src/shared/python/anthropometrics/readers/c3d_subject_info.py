"""Reader for C3D ``SUBJECT_INFO`` / ``PROCESSING`` parameter groups.

The C3D file format reserves the ``PROCESSING`` parameter group for
subject anthropometry written by motion-capture software (Vicon Nexus,
Qualisys QTM, ...). Common fields:

* ``PROCESSING:Bodymass``    — body mass in kilograms
* ``PROCESSING:Height``      — stature in millimetres
* ``PROCESSING:LeftLegLength`` / ``RightLegLength``  — millimetres
* ``PROCESSING:LeftArmLength`` / ``RightArmLength``  — millimetres

The ``SUBJECTS`` group holds session-level identifiers:

* ``SUBJECTS:NAMES``  — session-display names (treated as identifier
  only — generic-naming policy)
* ``SUBJECTS:AGE``    — age in years
* ``SUBJECTS:SEX``    — ``"M"`` / ``"F"`` (anything else maps to
  :data:`anthropometrics.Sex.UNSPECIFIED`)

Every field is **optional**: when a key is absent or non-finite the
corresponding attribute on :class:`C3DSubjectMetadata` is ``None``
(or :data:`Sex.UNSPECIFIED` for sex). Missing files raise
:class:`FileNotFoundError`; that is the only hard failure mode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upstream_drift_tools.utils.logging import get_logger

from .._types import Sex

logger = get_logger(__name__)

_MM_TO_M: float = 1.0e-3


@dataclass(frozen=True)
class C3DSubjectMetadata:
    """Subject metadata extracted from a single C3D file.

    Every field is optional — fields not present (or non-finite) in
    the source file are ``None`` (or :data:`Sex.UNSPECIFIED` for
    :attr:`sex`). All length fields are stored in **metres** and mass
    in **kilograms** to match the rest of the anthropometrics package.
    """

    subject_id: str | None
    height_m: float | None
    mass_kg: float | None
    age_years: float | None
    sex: Sex
    leg_length_m: float | None
    arm_length_m: float | None


# --------------------------------------------------------------------------- #
# Public entry point.                                                         #
# --------------------------------------------------------------------------- #
def read_c3d_subject_metadata(path: Path | str) -> C3DSubjectMetadata:
    """Open a C3D file and return its :class:`C3DSubjectMetadata`.

    Args:
        path: Filesystem path to a ``.c3d`` file.

    Returns:
        A fully-populated :class:`C3DSubjectMetadata`. Fields whose
        underlying parameter is missing or non-finite are ``None``
        (or :data:`Sex.UNSPECIFIED`).

    Raises:
        FileNotFoundError: When *path* does not exist.
        ImportError: When the optional ``ezc3d`` dependency is not
            installed in the active environment.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"C3D file not found: {file_path}")

    parameters = _load_parameters(file_path)
    return _extract_subject_metadata(parameters)


# --------------------------------------------------------------------------- #
# I/O.                                                                        #
# --------------------------------------------------------------------------- #
def _load_parameters(file_path: Path) -> Mapping[str, Any]:
    """Open *file_path* via ``ezc3d`` and return the parameters mapping."""
    try:
        import ezc3d  # local import: keep package importable without ezc3d
    except ImportError as error:  # pragma: no cover - exercised via mocks
        raise ImportError(
            "ezc3d is required to read C3D subject metadata. "
            "Install it with: pip install ezc3d"
        ) from error

    c3d_data = ezc3d.c3d(str(file_path))
    return c3d_data["parameters"]  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# Pure extraction (testable without ezc3d).                                   #
# --------------------------------------------------------------------------- #
def _extract_subject_metadata(
    parameters: Mapping[str, Any],
) -> C3DSubjectMetadata:
    """Return :class:`C3DSubjectMetadata` from a C3D parameters mapping.

    This function is pure: it does no I/O. It accepts the structure
    returned by ``ezc3d.c3d(...)["parameters"]`` and is the seam used
    by the unit tests to inject synthetic fixtures.
    """
    processing = _get_group(parameters, "PROCESSING")
    subjects = _get_group(parameters, "SUBJECTS")

    mass_kg = _read_scalar(processing, "Bodymass")
    height_mm = _read_scalar(processing, "Height")
    height_m = height_mm * _MM_TO_M if height_mm is not None else None

    leg_length_m = _read_average_length_m(processing, "LeftLegLength", "RightLegLength")
    arm_length_m = _read_average_length_m(processing, "LeftArmLength", "RightArmLength")

    subject_id = _read_first_string(subjects, "NAMES")
    if subject_id is not None:
        # Generic-naming policy: SUBJECTS:NAMES is informational only.
        logger.info("C3D SUBJECTS:NAMES present (session id only): %s", subject_id)

    age_years = _read_scalar(subjects, "AGE")
    sex = _read_sex(subjects)

    return C3DSubjectMetadata(
        subject_id=subject_id,
        height_m=height_m,
        mass_kg=mass_kg,
        age_years=age_years,
        sex=sex,
        leg_length_m=leg_length_m,
        arm_length_m=arm_length_m,
    )


# --------------------------------------------------------------------------- #
# Parameter helpers.                                                          #
# --------------------------------------------------------------------------- #
def _get_group(parameters: Mapping[str, Any], group_name: str) -> Mapping[str, Any]:
    """Return *group_name* from *parameters* with case-insensitive lookup.

    Returns an empty dict when the group is absent so downstream
    helpers can treat "missing group" identically to "missing field".
    """
    for key, value in parameters.items():
        if key.upper() == group_name.upper() and isinstance(value, Mapping):
            return value
    return {}


def _get_param_value(group: Mapping[str, Any], key: str) -> Any | None:
    """Return ``group[key]['value']`` with case-insensitive key matching."""
    for candidate, entry in group.items():
        if candidate.upper() != key.upper():
            continue
        if isinstance(entry, Mapping):
            return entry.get("value")
        return entry
    return None


def _read_scalar(group: Mapping[str, Any], key: str) -> float | None:
    """Read the first scalar from ``group[key]``; ``None`` if missing/non-finite."""
    value = _get_param_value(group, key)
    if value is None:
        return None
    try:
        flat = list(_iter_flat(value))
    except TypeError:
        return None
    if not flat:
        return None
    try:
        scalar = float(flat[0])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _read_average_length_m(
    group: Mapping[str, Any],
    left_key: str,
    right_key: str,
) -> float | None:
    """Average left/right millimetre lengths and convert to metres.

    If only one side is present, that single side is returned. If
    both sides are absent or non-finite, the result is ``None``.
    """
    left = _read_scalar(group, left_key)
    right = _read_scalar(group, right_key)
    available = [v for v in (left, right) if v is not None]
    if not available:
        return None
    avg_mm = sum(available) / len(available)
    return avg_mm * _MM_TO_M


def _read_first_string(group: Mapping[str, Any], key: str) -> str | None:
    """Read the first string from ``group[key]``; ``None`` if blank/missing."""
    value = _get_param_value(group, key)
    if value is None:
        return None
    try:
        flat = list(_iter_flat(value))
    except TypeError:
        return None
    if not flat:
        return None
    text = str(flat[0]).strip()
    return text or None


def _read_sex(group: Mapping[str, Any]) -> Sex:
    """Parse ``SUBJECTS:SEX`` with ``"M"`` / ``"F"`` mapping; else unspecified."""
    raw = _read_first_string(group, "SEX")
    if raw is None:
        return Sex.UNSPECIFIED
    upper = raw.upper()
    if upper == "M":
        return Sex.MALE
    if upper == "F":
        return Sex.FEMALE
    return Sex.UNSPECIFIED


def _iter_flat(value: Any) -> Any:
    """Yield scalars from *value*, descending one level into list-likes.

    ezc3d returns parameter values as either Python scalars, Python
    lists, or numpy arrays. This helper normalises the three so that
    callers can pull a "first scalar" without branching.
    """
    if isinstance(value, (str, bytes)):
        yield value
        return
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep elsewhere
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.ndarray):
        yield from value.ravel().tolist()
        return
    try:
        iterator = iter(value)
    except TypeError:
        yield value
        return
    for item in iterator:
        if isinstance(item, (list, tuple)):
            yield from item
        else:
            yield item
