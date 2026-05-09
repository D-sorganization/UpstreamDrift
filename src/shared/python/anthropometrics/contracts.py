"""Public Protocols for the anthropometrics subsystem.

Every concrete implementation in downstream child issues of the
anthropometrics EPIC must satisfy one of these Protocols. They
are intentionally minimal so that estimators, persistence
layers, and engine adapters can evolve independently.

All Protocols are :func:`~typing.runtime_checkable` so callers
can use ``isinstance`` checks for fail-fast validation without
incurring an import of every concrete subclass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from ._subject_anthropometrics import SubjectAnthropometrics
    from .segment_properties import SegmentProperties


@runtime_checkable
class Estimator(Protocol):
    """Produce :class:`SubjectAnthropometrics` from raw subject metadata.

    Implementations apply published ratio tables (de Leva,
    Dempster, Zatsiorsky-Seluyanov) or scan-derived models.
    """

    def estimate(
        self,
        *,
        subject_id: str,
        height_m: float,
        mass_kg: float,
        sex: str = "unspecified",
        age_years: float | None = None,
    ) -> SubjectAnthropometrics:
        """Return a fully-populated :class:`SubjectAnthropometrics`."""
        ...


@runtime_checkable
class Reader(Protocol):
    """Load a previously-persisted :class:`SubjectAnthropometrics`."""

    def read(self, path: Path) -> SubjectAnthropometrics:
        """Return the :class:`SubjectAnthropometrics` stored at *path*."""
        ...


@runtime_checkable
class Writer(Protocol):
    """Persist a :class:`SubjectAnthropometrics` to a backing store."""

    def write(self, anthro: SubjectAnthropometrics, path: Path) -> None:
        """Write *anthro* to *path*."""
        ...


@runtime_checkable
class EngineAdapter(Protocol):
    """Translate canonical segments into engine-specific structures.

    Each physics engine (Pinocchio, Drake, MuJoCo, Bullet, ...)
    has its own representation of inertial parameters; concrete
    adapters bridge between :class:`SegmentProperties` and that
    representation.
    """

    def to_engine_segment(self, props: SegmentProperties) -> object:
        """Return the engine-native representation of *props*."""
        ...
