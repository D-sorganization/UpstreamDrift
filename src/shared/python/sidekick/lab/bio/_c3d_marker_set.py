"""Marker-set detection for C3D files.

Reads ``POINT.LABELS``, ``SUBJECTS.MARKER_SETS``, and ``MODEL`` parameter
groups (when present) and pattern-matches against a registry of known
marker sets used by the project. The detection is deterministic: an
explicit ``SUBJECTS.MARKER_SETS`` or ``MODEL`` parameter takes priority,
falling back to label-coverage heuristics over the ``POINT.LABELS`` set.

Implements issue #4710.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ...utils.logging import get_logger

logger = get_logger(__name__)


class MarkerSet(str, Enum):
    """Recognised marker-set conventions.

    The string values double as the canonical name reported by detection
    logs and surfaced via :class:`C3DMetadata`. ``UNKNOWN`` is the
    sentinel for files whose marker set cannot be classified.
    """

    CGM2_4 = "CGM2.4"
    PLUG_IN_GAIT_41 = "Plug-in-Gait-41"
    IOR = "IOR"
    GOLF_CLUSTER = "GolfCluster"
    GOLF_TOUR_AVERAGE_BODY = "GolfTourAverageBody"
    UNKNOWN = "Unknown"


class MarkerSetMismatchError(ValueError):
    """Raised when a C3D file's marker set cannot be reconciled with the loader.

    Carries the detected ``MarkerSet`` plus the file's full label list so
    callers can decide whether to retry with an explicit override or fail
    fast instead of silently producing NaN-filled outputs.
    """

    def __init__(
        self,
        message: str,
        *,
        detected: MarkerSet,
        labels: list[str],
    ) -> None:
        super().__init__(message)
        self.detected = detected
        self.labels = labels


@dataclass(frozen=True)
class _MarkerSetSignature:
    """Detection signature for a known marker set.

    Attributes:
        marker_set: The :class:`MarkerSet` this signature classifies.
        name_aliases: Substrings (case-insensitive) that match against
            ``SUBJECTS.MARKER_SETS`` or ``MODEL`` parameter values.
        required: Marker labels that MUST all be present (case-insensitive)
            for the heuristic match to fire. Empty when name-only matching
            is sufficient.
        optional: Marker labels that boost confidence but are not required.
        min_required_hits: Minimum count from ``required`` that must match.
            Defaults to ``len(required)`` when zero.
    """

    marker_set: MarkerSet
    name_aliases: tuple[str, ...]
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    min_required_hits: int = 0


# Detection priority order: name-based aliases first, then heuristic by
# required-marker coverage. Entries earlier in the list win ties.
_SIGNATURES: tuple[_MarkerSetSignature, ...] = (
    _MarkerSetSignature(
        marker_set=MarkerSet.CGM2_4,
        name_aliases=("CGM2.4", "CGM 2.4", "CGM24"),
        # CGM2.4 adds medial knee/ankle markers and a head cluster on top of
        # the Plug-in-Gait core.
        required=(
            "LASI",
            "RASI",
            "LPSI",
            "RPSI",
            "LKNE",
            "RKNE",
            "LKNM",
            "RKNM",
            "LANK",
            "RANK",
            "LMED",
            "RMED",
        ),
        min_required_hits=10,
    ),
    _MarkerSetSignature(
        marker_set=MarkerSet.PLUG_IN_GAIT_41,
        name_aliases=("Plug-in-Gait", "PluginGait", "PiG", "VICON PIG"),
        required=(
            "LASI",
            "RASI",
            "LPSI",
            "RPSI",
            "LTHI",
            "RTHI",
            "LKNE",
            "RKNE",
            "LTIB",
            "RTIB",
            "LANK",
            "RANK",
            "LHEE",
            "RHEE",
            "LTOE",
            "RTOE",
        ),
        optional=("LFHD", "RFHD", "LBHD", "RBHD", "C7", "T10", "CLAV", "STRN"),
        min_required_hits=12,
    ),
    _MarkerSetSignature(
        marker_set=MarkerSet.IOR,
        name_aliases=("IOR", "Rizzoli", "Cappozzo"),
        required=(
            "R_ASIAS",
            "L_ASIAS",
            "R_AISPS",
            "L_AISPS",
            "R_KNEE",
            "L_KNEE",
            "R_ANKLE",
            "L_ANKLE",
        ),
        min_required_hits=6,
    ),
    _MarkerSetSignature(
        marker_set=MarkerSet.GOLF_CLUSTER,
        name_aliases=("GolfCluster", "Golf Cluster", "ClubCluster"),
        # The golf cluster uses redundant markers on the clubhead and grip;
        # any one of these label-pairs is enough to recognise the set.
        required=(),
        optional=(
            "CH",
            "ClubHead",
            "CLUBHEAD",
            "BUTT",
            "GRIP",
            "GripButt",
            "ClubButt",
        ),
        min_required_hits=0,
    ),
    _MarkerSetSignature(
        marker_set=MarkerSet.GOLF_TOUR_AVERAGE_BODY,
        name_aliases=("Tour Average", "TourAverage", "GolfTourAverage"),
        required=(
            "WaistLeft",
            "WaistRight",
            "WaistLBack",
            "WaistRBack",
            "BackTop",
            "BackLeft",
            "BackRight",
            "HeadTop",
            "HeadFront",
            "HeadSide",
            "LShoulderTop",
            "LShoulderBack",
            "LElbowOut",
            "LUArmHigh",
            "LWristTop",
            "RShoulderTop",
            "RShoulderBack",
            "RElbowOut",
            "RUArmHigh",
            "RWristTop",
            "LKneeOut",
            "LToeIn",
            "LToeOut",
            "LAnkleOut",
            "RKneeOut",
            "RToeIn",
            "RToeOut",
            "RAnkleOut",
        ),
        min_required_hits=24,
    ),
)


def _normalise_labels(labels: list[str]) -> set[str]:
    """Return a case-insensitive label set with whitespace stripped."""
    return {label.strip().upper() for label in labels if label and label.strip()}


def _coerce_string_list(value: Any) -> list[str]:
    """Coerce an ezc3d parameter value into a list of stripped strings."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        return [text.strip()] if text.strip() else []
    try:
        items = list(value)
    except TypeError:
        return [str(value).strip()]
    out: list[str] = []
    for item in items:
        if isinstance(item, bytes):
            item = item.decode(errors="replace")
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _read_param_strings(parameters: dict[str, Any], group: str, key: str) -> list[str]:
    """Read a string-valued parameter, returning ``[]`` when absent."""
    grp = parameters.get(group)
    if not grp:
        return []
    entry = grp.get(key)
    if not entry:
        return []
    if isinstance(entry, dict):
        return _coerce_string_list(entry.get("value"))
    return _coerce_string_list(entry)


def _golf_cluster_match(labels_upper: set[str]) -> bool:
    """Return True when the label set has both grip and clubhead markers."""
    butt_tokens = {"BUTT", "GRIP", "GRIPBUTT", "CLUBBUTT", "BUTT_END", "CLUB_BUTT"}
    head_tokens = {
        "CH",
        "CLUBHEAD",
        "CLUB_HEAD",
        "CLUBFACE",
    }
    has_butt = any(tok in lbl for lbl in labels_upper for tok in butt_tokens)
    has_head = any(
        lbl == "CH" or any(tok in lbl for tok in head_tokens) for lbl in labels_upper
    )
    return has_butt and has_head


def _name_matches(declared_names: list[str], aliases: tuple[str, ...]) -> bool:
    """Return True when any declared marker-set/model name contains an alias."""
    if not declared_names or not aliases:
        return False
    upper_names = [n.upper() for n in declared_names]
    upper_aliases = [a.upper() for a in aliases]
    return any(alias in name for name in upper_names for alias in upper_aliases)


def detect_marker_set(
    point_labels: list[str],
    parameters: dict[str, Any] | None = None,
) -> MarkerSet:
    """Detect the marker-set convention of a C3D file.

    Detection priority (deterministic):

    1. ``SUBJECTS.MARKER_SETS`` matches a known alias.
    2. ``MODEL.NAME`` (or ``MODEL`` group bare value) matches a known alias.
    3. Heuristic coverage of required markers from ``POINT.LABELS``.

    Args:
        point_labels: ``POINT.LABELS`` list from the C3D file.
        parameters: Full parameter mapping (the ``c3d_data["parameters"]``
            dict); ``None`` is treated as missing groups.

    Returns:
        The detected :class:`MarkerSet`. ``MarkerSet.UNKNOWN`` is returned
        when no signature fires.
    """
    params = parameters or {}
    declared_marker_sets = _read_param_strings(params, "SUBJECTS", "MARKER_SETS")
    declared_models: list[str] = []
    declared_models.extend(_read_param_strings(params, "MODEL", "NAME"))
    declared_models.extend(_read_param_strings(params, "MODEL", "USED"))
    # Some files store the model name as a bare scalar under MODEL.
    model_group = params.get("MODEL")
    if isinstance(model_group, dict):
        for key, entry in model_group.items():
            if key in {"NAME", "USED"}:
                continue
            if isinstance(entry, dict):
                declared_models.extend(_coerce_string_list(entry.get("value")))

    declared_names = declared_marker_sets + declared_models
    labels_upper = _normalise_labels(point_labels)

    # Pass 1: name-based detection.
    for sig in _SIGNATURES:
        if _name_matches(declared_names, sig.name_aliases):
            logger.info(
                "Detected marker set %s via declared name (%s)",
                sig.marker_set.value,
                ", ".join(declared_names),
            )
            return sig.marker_set

    # Pass 2: heuristic detection on POINT.LABELS.
    # Pass 2: heuristic detection on POINT.LABELS. Iterate in declared
    # priority order — the first signature that meets its threshold wins,
    # so more-specific sets (e.g. CGM2.4) must be listed before their
    # supersets (e.g. Plug-in-Gait).
    for sig in _SIGNATURES:
        if sig.marker_set is MarkerSet.GOLF_CLUSTER:
            if _golf_cluster_match(labels_upper):
                logger.info(
                    "Detected marker set %s via grip/clubhead label heuristic",
                    sig.marker_set.value,
                )
                return sig.marker_set
            continue
        required_upper = {r.upper() for r in sig.required}
        if not required_upper:
            continue
        hits = len(required_upper & labels_upper)
        threshold = sig.min_required_hits or len(required_upper)
        if hits >= threshold:
            logger.info(
                "Detected marker set %s via label coverage (%d required hits)",
                sig.marker_set.value,
                hits,
            )
            return sig.marker_set

    logger.warning(
        "Could not classify marker set; %d POINT.LABELS, declared names=%r",
        len(point_labels),
        declared_names,
    )
    return MarkerSet.UNKNOWN


__all__ = [
    "MarkerSet",
    "MarkerSetMismatchError",
    "detect_marker_set",
]
