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
    """Export and re-import a :class:`SubjectAnthropometrics` for a physics engine.

    Each physics engine (Drake, Pinocchio, MyoSuite/MuJoCo, OpenSim,
    Simscape, ...) has its own native description format. Concrete
    adapters serialise a canonical :class:`SubjectAnthropometrics`
    into that format via :meth:`export` and recover an equivalent
    record via :meth:`import_back`. Adapters guarantee numerically
    exact round-trips on all inertial fields (``rtol=1e-9,
    atol=1e-12``) so that downstream pipelines can move between
    engines without information loss.
    """

    engine_name: str
    """Lower-case engine identifier, e.g. ``"drake"``, ``"pinocchio"``."""

    def export(
        self, anthropometrics: SubjectAnthropometrics, output_path: Path
    ) -> None:
        """Serialise *anthropometrics* to *output_path* in the engine format."""
        ...

    def import_back(self, input_path: Path) -> SubjectAnthropometrics:
        """Re-load a previously-exported subject from *input_path*."""
        ...
