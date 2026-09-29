"""USGA bunker-sand specification tables and compliance reporting (issue #8610).

Two published tables, both carried because they disagree and a preset must say
which one it targets.

**Turf & Soil Diagnostics lab table** (method per *Golf Course Management*
1986, 54:64-70; PSD per ASTM F1632-99; USDA size classes), by weight:

==================  ==========  ==============
fraction            sieve (mm)  bunker sand
==================  ==========  ==============
gravel              2.00        <= 2 %
very coarse         1.00        <= 15 %
coarse + medium     0.25-1.00   78-100 %
very fine           0.05-0.10   <= 5 %
silt + clay         < 0.05      <= 3 %
uniformity Cu                   2.0-5.0
==================  ==========  ==============

**USGA Green Section Record 58(11), June 2020**, by volume, is tighter:
gravel (2-4 mm) <= 3 %; very coarse including gravel (1-4 mm) <= 7 %;
coarse + medium (0.25-1 mm) >= 65 %; fine + very fine (0.05-0.25 mm) <= 25 %;
silt + clay <= 3 %. Windy sites are an explicit exception: > 80 % between
0.25-1 mm with 10-20 % in the 1-2 mm fraction to resist erosion, which
breaches that table's own very-coarse cap.

Compliance is **reported, not enforced**. A sand outside the band is a real
sand; refusing to model it would be worse than saying it is out of band.
"""

from __future__ import annotations

from dataclasses import dataclass

from .psd import ParticleSizeDistribution

__all__ = [
    "USGA_GSR_2020_SPECIFICATION",
    "USGA_LAB_SPECIFICATION",
    "ComplianceReport",
    "SieveBand",
    "Specification",
    "evaluate_compliance",
]


@dataclass(frozen=True, slots=True)
class SieveBand:
    """One row of a specification table.

    Attributes:
        name: Size-class label, used as the report key.
        lower_m: Lower size bound, metres.
        upper_m: Upper size bound, metres.
        min_fraction: Minimum permitted mass fraction.
        max_fraction: Maximum permitted mass fraction.
    """

    name: str
    lower_m: float
    upper_m: float
    min_fraction: float = 0.0
    max_fraction: float = 1.0


@dataclass(frozen=True, slots=True)
class Specification:
    """A named, cited specification table."""

    name: str
    citation: str
    bands: tuple[SieveBand, ...]
    uniformity_coefficient_range: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    """The outcome of checking one distribution against one specification."""

    specification_name: str
    measurements: tuple[tuple[str, float], ...]
    uniformity_coefficient: float
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True when no band or uniformity check was breached."""
        return not self.violations

    def summary(self) -> str:
        """Return a human-readable compliance statement."""
        if self.passed:
            return f"complies with {self.specification_name}"
        return f"does not comply with {self.specification_name}: " + "; ".join(
            self.violations
        )


USGA_LAB_SPECIFICATION = Specification(
    name="USGA / Turf & Soil Diagnostics bunker sand (by weight)",
    citation=(
        "Turf & Soil Diagnostics, Evaluating Bunker Sands; PSD per "
        "ASTM F1632-99, USDA size classes"
    ),
    bands=(
        SieveBand("gravel", 2.0e-3, 4.0e-3, max_fraction=0.02),
        SieveBand("very coarse", 1.0e-3, 2.0e-3, max_fraction=0.15),
        SieveBand(
            "coarse + medium", 2.5e-4, 1.0e-3, min_fraction=0.78, max_fraction=1.0
        ),
        SieveBand("very fine", 5.0e-5, 1.0e-4, max_fraction=0.05),
        SieveBand("silt + clay", 0.0, 5.0e-5, max_fraction=0.03),
    ),
    uniformity_coefficient_range=(2.0, 5.0),
)

USGA_GSR_2020_SPECIFICATION = Specification(
    name="USGA Green Section Record 58(11) 2020 bunker sand (by volume)",
    citation="USGA Green Section Record 58(11), June 2020",
    bands=(
        SieveBand("gravel", 2.0e-3, 4.0e-3, max_fraction=0.03),
        SieveBand("very coarse", 1.0e-3, 4.0e-3, max_fraction=0.07),
        SieveBand(
            "coarse + medium", 2.5e-4, 1.0e-3, min_fraction=0.65, max_fraction=1.0
        ),
        SieveBand("fine + very fine", 5.0e-5, 2.5e-4, max_fraction=0.25),
        SieveBand("silt + clay", 0.0, 5.0e-5, max_fraction=0.03),
    ),
    uniformity_coefficient_range=(2.0, 5.0),
)


def evaluate_compliance(
    psd: ParticleSizeDistribution,
    specification: Specification,
) -> ComplianceReport:
    """Check a distribution against a specification table.

    Args:
        psd: The sieve analysis to check.
        specification: The table to check it against.

    Returns:
        A :class:`ComplianceReport` listing every measured band fraction and
        every violation. Nothing is raised: an out-of-band sand is reported,
        because out-of-band sand exists on real golf courses.
    """
    measurements: list[tuple[str, float]] = []
    violations: list[str] = []
    for band in specification.bands:
        lower = max(band.lower_m, psd.sieve_openings_m[0])
        fraction = (
            psd.fraction_finer_than(band.upper_m) - psd.fraction_finer_than(lower)
            if band.upper_m > lower
            else 0.0
        )
        measurements.append((band.name, fraction))
        if fraction < band.min_fraction - 1e-12:
            violations.append(
                f"{band.name} is {fraction * 100:.3g}%, below the minimum "
                f"{band.min_fraction * 100:.3g}%"
            )
        elif fraction > band.max_fraction + 1e-12:
            violations.append(
                f"{band.name} is {fraction * 100:.3g}%, above the maximum "
                f"{band.max_fraction * 100:.3g}%"
            )
    cu = psd.uniformity_coefficient
    cu_range = specification.uniformity_coefficient_range
    if cu_range is not None and not cu_range[0] <= cu <= cu_range[1]:
        violations.append(
            f"uniformity coefficient Cu is {cu:.3g}, outside the specified "
            f"{cu_range[0]:.3g}-{cu_range[1]:.3g}"
        )
    return ComplianceReport(
        specification_name=specification.name,
        measurements=tuple(measurements),
        uniformity_coefficient=cu,
        violations=tuple(violations),
    )
