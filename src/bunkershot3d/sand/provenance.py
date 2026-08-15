"""Provenance records for sand properties (issue #8610).

Why this module exists
----------------------

**No published bulk density, internal friction angle or angle of repose
specific to *golf bunker* sand was found.** The nearest well-characterised
analogue -- and the source of the DRFT constants used elsewhere in this
package -- is Quikrete medium sand, 0.3-0.8 mm, whose size band overlaps the
USGA bunker window:

===========================  ==========================
quantity                     value
===========================  ==========================
internal friction angle Phi  34 deg
packing (solid) fraction     0.60
grain density rho_grain      2600 kg/m^3
===========================  ==========================

Those numbers are **borrowed from an analogue, not measured on bunker sand**.
Presenting a borrowed constant as a measurement is the exact failure mode that
issue #7999 already corrected once in this package (a hand-written shear-cell
line and a uniform-random restitution shipped as a "calibration"). Every sand
preset therefore carries a :class:`SandProvenance` record naming the basis of
each honesty-critical property, and
``tests/bunkershot3d/sand/test_presets.py`` fails if a preset claims a
measured bunker-sand friction angle.

Particle size distributions are different: the USGA / Turf & Soil Diagnostics
sieve tables *are* published, so those carry
:attr:`ProvenanceBasis.SPECIFICATION`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .exceptions import ProvenanceError

__all__ = [
    "BUNKER_SAND_MEASUREMENT_GAP",
    "QUIKRETE_ANALOGUE_SOURCE",
    "QUIKRETE_FRICTION_ANGLE_DEG",
    "QUIKRETE_PACKING_SOLID_FRACTION",
    "QUIKRETE_PARTICLE_DENSITY_KG_M3",
    "REQUIRED_PROVENANCE_KEYS",
    "USGA_SIEVE_TABLE_SOURCE",
    "PropertyProvenance",
    "ProvenanceBasis",
    "SandProvenance",
    "borrowed_from_quikrete",
    "published_specification",
]


class ProvenanceBasis(StrEnum):
    """How a property value came to have the value it has."""

    MEASURED = "measured"
    """Measured on golf bunker sand itself. Nothing in this package is."""

    SPECIFICATION = "specification"
    """Read off a published specification or recommendation table."""

    BORROWED_ANALOGUE = "borrowed_analogue"
    """Taken from a different, better-characterised material."""

    ESTIMATED = "estimated"
    """Derived from first principles or a documented rule of thumb."""

    CONVENTION = "convention"
    """A modelling convention chosen for reproducibility, not a measurement."""


QUIKRETE_ANALOGUE_SOURCE = (
    "Quikrete medium sand, 0.3-0.8 mm (Agarwal, Karsai, Goldman & Kamrin, "
    "J. Terramechanics 2019, arXiv:1901.10667; Science Advances 2021, "
    "arXiv:2005.10976)"
)

BUNKER_SAND_MEASUREMENT_GAP = (
    "No published bulk density, internal friction angle or angle of repose "
    "specific to golf bunker sand was found; this value is borrowed from the "
    "Quikrete medium-sand analogue and is not measured on bunker sand."
)

USGA_SIEVE_TABLE_SOURCE = (
    "USGA / Turf & Soil Diagnostics, Evaluating Bunker Sands (PSD per "
    "ASTM F1632-99, USDA size classes); USGA Green Section Record 58(11), "
    "June 2020"
)

QUIKRETE_FRICTION_ANGLE_DEG = 34.0
QUIKRETE_PACKING_SOLID_FRACTION = 0.60
QUIKRETE_PARTICLE_DENSITY_KG_M3 = 2600.0

REQUIRED_PROVENANCE_KEYS: tuple[str, ...] = (
    "friction_angle_deg",
    "moisture",
    "packing",
    "particle_density_kg_m3",
    "particle_size_distribution",
)
"""Properties a :class:`~bunkershot3d.sand.state.SandState` must account for."""


@dataclass(frozen=True, slots=True)
class PropertyProvenance:
    """Where one property value came from."""

    basis: ProvenanceBasis
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ProvenanceError("provenance source must not be empty")

    @property
    def is_borrowed(self) -> bool:
        """True when the value came from an analogue material."""
        return self.basis is ProvenanceBasis.BORROWED_ANALOGUE

    @property
    def is_measured(self) -> bool:
        """True only when measured on golf bunker sand itself."""
        return self.basis is ProvenanceBasis.MEASURED

    def describe(self, name: str) -> str:
        """Return a one-line human-readable provenance statement."""
        detail = f" -- {self.note}" if self.note else ""
        return f"{name}: {self.basis.value} from {self.source}{detail}"


def borrowed_from_quikrete(note: str = "") -> PropertyProvenance:
    """Return the standard borrowed-analogue record for the Quikrete constants."""
    combined = BUNKER_SAND_MEASUREMENT_GAP
    if note:
        combined = f"{note} {BUNKER_SAND_MEASUREMENT_GAP}"
    return PropertyProvenance(
        basis=ProvenanceBasis.BORROWED_ANALOGUE,
        source=QUIKRETE_ANALOGUE_SOURCE,
        note=combined,
    )


def published_specification(note: str = "") -> PropertyProvenance:
    """Return the standard record for values read off the USGA sieve tables."""
    return PropertyProvenance(
        basis=ProvenanceBasis.SPECIFICATION,
        source=USGA_SIEVE_TABLE_SOURCE,
        note=note,
    )


@dataclass(frozen=True, slots=True)
class SandProvenance:
    """The provenance of every honesty-critical property of a sand state."""

    entries: Mapping[str, PropertyProvenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, entry in self.entries.items():
            if not isinstance(entry, PropertyProvenance):
                raise ProvenanceError(
                    f"provenance for '{name}' must be a PropertyProvenance, "
                    f"got {type(entry).__name__}"
                )
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def entry(self, name: str) -> PropertyProvenance:
        """Return the provenance record for ``name``.

        Raises:
            ProvenanceError: if no record was supplied for ``name``.
        """
        try:
            return self.entries[name]
        except KeyError:
            raise ProvenanceError(
                f"no provenance recorded for '{name}'; "
                f"recorded properties are {sorted(self.entries)}"
            ) from None

    def require_keys(self, keys: Iterable[str] = REQUIRED_PROVENANCE_KEYS) -> None:
        """Raise unless a record exists for every key in ``keys``.

        Raises:
            ProvenanceError: naming the missing keys.
        """
        missing = sorted(set(keys) - set(self.entries))
        if missing:
            raise ProvenanceError(
                "sand state is missing provenance for: "
                + ", ".join(missing)
                + ". Every honesty-critical property must record whether its "
                "value was measured, read off a specification, or borrowed "
                "from an analogue material (issue #7999)."
            )

    def borrowed_properties(self) -> tuple[str, ...]:
        """Return the names of properties whose values came from an analogue."""
        return tuple(sorted(name for name, e in self.entries.items() if e.is_borrowed))

    def measured_properties(self) -> tuple[str, ...]:
        """Return the names of properties measured on golf bunker sand."""
        return tuple(sorted(name for name, e in self.entries.items() if e.is_measured))

    def summary(self) -> str:
        """Return a multi-line provenance statement for reports and manifests."""
        return "\n".join(
            entry.describe(name) for name, entry in sorted(self.entries.items())
        )
