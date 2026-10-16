"""Thoracolumbar spinal level enumeration, regions and adjacency (SPN-1a).

Each :class:`SpinalLevel` is the intervertebral disc space between an
inferior (caudal) and a superior (cranial) vertebra. Members are declared in
caudal-to-cranial order, so the enum order is the spinal stacking order from
L5-S1 up to T1-T2. Adjacent members are contiguous: the superior vertebra of
one level is the inferior vertebra of the next.

Cervical extension point: the cervical levels (C1-C2 through C7-T1) are out of
scope for the segmental-spine epic (#12735). A future change adds a
``SpinalRegion.CERVICAL`` level family by appending members at the cranial end
of this enum, preserving the caudal-to-cranial enum order, and extending
``_VERTEBRA_PAIRS`` accordingly. No cervical members exist yet, so nothing here
depends on them.
"""

from __future__ import annotations

import enum

from src.shared.python.contracts import ensure

__all__ = ["SpinalLevel", "SpinalRegion", "levels_between", "parse_level"]


class SpinalRegion(enum.Enum):
    """Anatomical region of the spine."""

    CERVICAL = "cervical"
    THORACIC = "thoracic"
    LUMBAR = "lumbar"
    SACRAL = "sacral"


class SpinalLevel(enum.Enum):
    """Intervertebral level, declared caudal to cranial.

    Each value is ``(inferior vertebra, superior vertebra, region of the
    superior vertebra)``. The enum order is the caudal-to-cranial order.
    """

    L5_S1 = ("S1", "L5", SpinalRegion.LUMBAR)
    L4_L5 = ("L5", "L4", SpinalRegion.LUMBAR)
    L3_L4 = ("L4", "L3", SpinalRegion.LUMBAR)
    L2_L3 = ("L3", "L2", SpinalRegion.LUMBAR)
    L1_L2 = ("L2", "L1", SpinalRegion.LUMBAR)
    T12_L1 = ("L1", "T12", SpinalRegion.THORACIC)
    T11_T12 = ("T12", "T11", SpinalRegion.THORACIC)
    T10_T11 = ("T11", "T10", SpinalRegion.THORACIC)
    T9_T10 = ("T10", "T9", SpinalRegion.THORACIC)
    T8_T9 = ("T9", "T8", SpinalRegion.THORACIC)
    T7_T8 = ("T8", "T7", SpinalRegion.THORACIC)
    T6_T7 = ("T7", "T6", SpinalRegion.THORACIC)
    T5_T6 = ("T6", "T5", SpinalRegion.THORACIC)
    T4_T5 = ("T5", "T4", SpinalRegion.THORACIC)
    T3_T4 = ("T4", "T3", SpinalRegion.THORACIC)
    T2_T3 = ("T3", "T2", SpinalRegion.THORACIC)
    T1_T2 = ("T2", "T1", SpinalRegion.THORACIC)

    def __init__(self, inferior: str, superior: str, region: SpinalRegion) -> None:
        self._inferior = inferior
        self._superior = superior
        self._region = region

    @property
    def inferior(self) -> str:
        """Caudal vertebra of the level, e.g. ``"L5"`` for ``L4_L5``."""
        return self._inferior

    @property
    def superior(self) -> str:
        """Cranial vertebra of the level, e.g. ``"L4"`` for ``L4_L5``."""
        return self._superior

    @property
    def region(self) -> SpinalRegion:
        """Region of the superior vertebra, e.g. ``THORACIC`` for ``T12_L1``."""
        return self._region

    @property
    def ordinal(self) -> int:
        """Zero-based caudal-to-cranial index (``L5_S1`` is 0, ``T1_T2`` is 16)."""
        return _CAUDAL_TO_CRANIAL.index(self)


_CAUDAL_TO_CRANIAL: tuple[SpinalLevel, ...] = tuple(SpinalLevel)
_BY_NAME: dict[str, SpinalLevel] = {member.name: member for member in SpinalLevel}


def levels_between(
    caudal: SpinalLevel, cranial: SpinalLevel
) -> tuple[SpinalLevel, ...]:
    """Return the inclusive run of levels from ``caudal`` up to ``cranial``.

    Preconditions: both arguments are :class:`SpinalLevel`; ``caudal`` is the
    same as or caudal to ``cranial``.

    Postconditions: the first element is ``caudal``, the last is ``cranial``,
    and the tuple is in caudal-to-cranial order.

    Raises:
        TypeError: if either argument is not a :class:`SpinalLevel`.
        ValueError: if ``caudal`` lies cranial to ``cranial``.
    """
    if not isinstance(caudal, SpinalLevel):
        raise TypeError(f"caudal must be a SpinalLevel, got {type(caudal).__name__}")
    if not isinstance(cranial, SpinalLevel):
        raise TypeError(f"cranial must be a SpinalLevel, got {type(cranial).__name__}")
    low, high = caudal.ordinal, cranial.ordinal
    if low > high:
        raise ValueError(
            f"caudal level {caudal.name} lies cranial to {cranial.name}; "
            "arguments must be given caudal first"
        )
    result = _CAUDAL_TO_CRANIAL[low : high + 1]
    ensure(
        result[0] is caudal and result[-1] is cranial,
        "levels_between must return an inclusive caudal-to-cranial run",
        result,
    )
    return result


def parse_level(name: str) -> SpinalLevel:
    """Parse a level name such as ``"L4_L5"``, ``"L4-L5"`` or ``"l4/l5"``.

    Matching is case-insensitive and ignores surrounding whitespace.

    Raises:
        TypeError: if ``name`` is not a ``str``.
        ValueError: if ``name`` is empty or does not name a known level.
    """
    if not isinstance(name, str):
        raise TypeError(f"level name must be a str, got {type(name).__name__}")
    key = name.strip().upper().replace("-", "_").replace("/", "_")
    if not key:
        raise ValueError("level name must not be empty")
    member = _BY_NAME.get(key)
    if member is None:
        raise ValueError(f"unknown spinal level {name!r}")
    return member
