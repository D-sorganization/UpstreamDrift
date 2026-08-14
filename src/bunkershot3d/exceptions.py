"""Exception hierarchy for BunkerShot3D (issue #8608, ADR-0032).

Everything this package refuses to do raises something that descends from
:class:`BunkerShot3DError`, which in turn descends from the platform root
``core.error_utils.GolfSuiteError`` (CLAUDE.md, "Error handling"). A caller can
therefore catch "BunkerShot3D rejected this" in one clause without catching
every ``ValueError`` in the process.

**The standard-library bases are kept.** Before this issue the package raised a
mix of bare ``ValueError``, ``RuntimeError``, ``NotImplementedError`` and
``FileNotFoundError``, and code outside this package catches those. Each root
below therefore inherits *both* :class:`BunkerShot3DError` and the built-in it
replaces, so existing ``except ValueError`` sites keep working while new code
can be specific.

Which root to pick:

============================  ==============================================
Situation                     Base
============================  ==============================================
A value or invariant is bad   :class:`BunkerShot3DValueError`
An operation is out of order  :class:`BunkerShot3DStateError`
An optional backend is absent :class:`BackendNotImplementedError`
An input file is missing      :class:`BunkerShot3DFileNotFoundError`
============================  ==============================================

Safety-critical checks ``raise`` these; they never ``assert``. ``python -O``
strips ``assert`` statements, and a guard that disappears under optimisation is
not a guard.
"""

from __future__ import annotations

from src.shared.python.core.error_utils import GolfSuiteError

__all__ = [
    "BackendNotImplementedError",
    "BunkerShot3DError",
    "BunkerShot3DFileNotFoundError",
    "BunkerShot3DStateError",
    "BunkerShot3DValueError",
    "ConfigurationInvalidError",
    "DomainInvariantError",
    "UnitConventionError",
    "UnitConversionError",
]


class BunkerShot3DError(GolfSuiteError):
    """Root of every error raised by :mod:`bunkershot3d`."""


class BunkerShot3DValueError(BunkerShot3DError, ValueError):
    """A value, or a combination of values, is inadmissible.

    Also a :class:`ValueError`, so callers written before #8608 keep working.
    """


class BunkerShot3DStateError(BunkerShot3DError, RuntimeError):
    """An operation was attempted in a state that cannot support it.

    Also a :class:`RuntimeError`, so callers written before #8608 keep working.
    """


class BunkerShot3DFileNotFoundError(BunkerShot3DError, FileNotFoundError):
    """A required input file does not exist.

    Also a :class:`FileNotFoundError`, so callers written before #8608 keep
    working.
    """


class ConfigurationInvalidError(BunkerShot3DValueError):
    """A configuration document cannot be loaded or assembled."""


class DomainInvariantError(BunkerShot3DValueError):
    """A domain value object was handed values that violate its invariants."""


class UnitConversionError(BunkerShot3DValueError):
    """A quantity cannot be converted: it is non-finite, or out of domain."""


class UnitConventionError(BunkerShot3DValueError):
    """A name carries no recognised unit suffix.

    Raised by :func:`bunkershot3d.units.si_unit_for`. The package convention is
    that every physical quantity is named with its unit
    (``dt_s``, ``v_mps``, ``theta_rad``); an unsuffixed physical name is a
    defect, not a style preference.
    """


class BackendNotImplementedError(BunkerShot3DError, NotImplementedError):
    """A requested simulation backend is unavailable or deliberately refuses.

    Also a :class:`NotImplementedError`: several backends are optional
    dependencies and callers test for that with the built-in.
    """

    def __init__(self, backend: str, feature: str = "") -> None:
        """Build the error.

        Args:
            backend: Backend identifier, or a full explanatory message.
            feature: Optional detail about what specifically is unavailable.
        """
        msg = f"Backend '{backend}' is not implemented."
        if feature:
            msg += f" Feature: {feature}"
        super().__init__(msg)
        self.backend = backend
        self.feature = feature
