"""Reference data for validation -- and the register of what does not exist.

Issue #8616.  Read the negative half of this module first.

There is essentially no literature to validate a bunker shot against
-------------------------------------------------------------------

An exhaustive enumeration of ISEA / *Procedia Engineering* / *The
Engineering of Sport* volumes 2, 13, 32, 34, 60, 72, 112 and 147 found no
paper on bunkers, sand, wedges, club-turf interaction or divot mechanics:
all 35 golf papers are swing dynamics, shafts, ball aerodynamics, putting
or acoustics.  *Sports Engineering* (~72 golf papers) and the *Journal of
Sports Sciences* return the same result.  **This is a real gap in the
field, not a search failure**, and :data:`UNMEASURED_QUANTITIES` records
it as data so that :func:`require_measurable` can refuse rather than
letting a plot of a model output against nothing be called validation.

What can be validated
---------------------

* :data:`WIVOU_2016` -- the only primary greenside-bunker dataset located.
  Entry distance, divot depth, carry, and two carry correlations.  It
  contains **no** clubhead speed, launch angle, ball speed or spin, and
  :meth:`ReferenceDataset.value_range` refuses to be asked for them.
* :data:`GRANULAR_INTRUSION_BENCHMARK` -- the wheel-in-sand benchmark the
  DRFT constants come from.  Reproducing it validates the *solver*,
  independent of golf, and it is where the real leverage is.

Provenance discipline
---------------------

Every entry carries its source string.  This package has been burnt once
already by presenting a borrowed constant as a measurement (#7999), and a
validation suite that repeats that mistake is worse than none, because it
launders the error through a metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .exceptions import NoReferenceDataError

__all__ = [
    "GRANULAR_INTRUSION_BENCHMARK",
    "REFERENCE_DATASETS",
    "UNMEASURED_QUANTITIES",
    "WIVOU_2016",
    "DomainOverlap",
    "IntrusionBenchmark",
    "ReferenceDataset",
    "ReferenceRange",
    "domain_overlap",
    "reference_dataset",
    "require_measurable",
]


@dataclass(frozen=True, slots=True)
class ReferenceRange:
    """A measured range of one quantity, with the source it came from.

    Attributes:
        quantity: Name of the quantity, SI-suffixed.
        low: Smallest measured value.
        high: Largest measured value.
        unit: SI unit symbol.
        source: Publication the range was read from.
    """

    quantity: str
    low: float
    high: float
    unit: str
    source: str

    def __post_init__(self) -> None:
        """Validate the range.

        Raises:
            ValueError: If the bounds are not ordered.
        """
        if not self.high > self.low:
            raise ValueError(
                f"reference range {self.quantity!r} must be ordered, got "
                f"{self.low} to {self.high}"
            )

    @property
    def span(self) -> float:
        """Width of the measured range."""
        return self.high - self.low

    @property
    def bounds(self) -> tuple[float, float]:
        """``(low, high)`` as a plain tuple."""
        return (self.low, self.high)


@dataclass(frozen=True)
class ReferenceDataset:
    """One published dataset, including what it does **not** contain.

    Recording the absent quantities in the dataset itself is the point: a
    dataset object that will happily be indexed for a quantity its source
    never measured is how a citation drifts onto a number it does not
    support.

    Attributes:
        key: Short identifier used by :func:`reference_dataset`.
        citation: Full citation.
        url: Where the source can be obtained, if it is open access.
        n_samples: Number of measured trials, where the source states one.
        ranges: Measured range per quantity.
        correlations: Published correlation coefficients, keyed by the
            factor carry was correlated against.
        absent_quantities: Quantities the source explicitly does not
            contain, so that citing it for them raises.
        notes: Caveats a reader of the numbers needs.
    """

    key: str
    citation: str
    url: str
    n_samples: int | None
    ranges: Mapping[str, ReferenceRange]
    correlations: Mapping[str, float] = field(default_factory=dict)
    absent_quantities: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the mappings so a dataset cannot be edited in place."""
        object.__setattr__(self, "ranges", MappingProxyType(dict(self.ranges)))
        object.__setattr__(
            self, "correlations", MappingProxyType(dict(self.correlations))
        )

    def value_range(self, quantity: str) -> tuple[float, float]:
        """Return the measured ``(low, high)`` bounds of ``quantity``.

        Args:
            quantity: SI-suffixed quantity name.

        Returns:
            The measured bounds.

        Raises:
            NoReferenceDataError: If the source does not contain the
                quantity, whether or not it is in the general register of
                unmeasured quantities.
        """
        entry = self.ranges.get(quantity)
        if entry is not None:
            return entry.bounds
        if quantity in self.absent_quantities:
            raise NoReferenceDataError(
                f"{self.citation} does not contain {quantity!r}; the source "
                "measured delivery and outcome geometry only. Citing it for "
                "this quantity would attribute a number to a paper that never "
                "reported one."
            )
        raise NoReferenceDataError(
            f"{self.citation} does not contain {quantity!r}; available "
            f"quantities are {sorted(self.ranges)}"
        )


@dataclass(frozen=True)
class IntrusionBenchmark:
    """Published errors of intrusion models against the same experiments.

    Attributes:
        citation: Where the errors are published.
        sinkage_mae_m: Mean absolute sinkage error per method.
        max_speed_m_s: Fastest intrusion anywhere in the corpus.
        natural_sand_bias: Fractional over-prediction on natural sand,
            attributed to grain angularity.
        notes: Caveats.
    """

    citation: str
    sinkage_mae_m: Mapping[str, float]
    max_speed_m_s: float
    natural_sand_bias: float
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the error mapping."""
        object.__setattr__(
            self, "sinkage_mae_m", MappingProxyType(dict(self.sinkage_mae_m))
        )


_WIVOU_CITATION = (
    "Wivou, Udawatta & Pathirana (2016), 'Analysis of Greenside Bunker Shots "
    "in Golf', ISBS 2016, pp. 1147-1150"
)

WIVOU_2016 = ReferenceDataset(
    key="wivou_2016",
    citation=_WIVOU_CITATION,
    url="https://ojs.ub.uni-konstanz.de/cpa/article/download/6986/6281",
    n_samples=55,
    ranges={
        "entry_distance_behind_ball_m": ReferenceRange(
            "entry_distance_behind_ball_m", 0.080, 0.280, "m", _WIVOU_CITATION
        ),
        "divot_depth_m": ReferenceRange(
            "divot_depth_m", 0.025, 0.052, "m", _WIVOU_CITATION
        ),
        "carry_m": ReferenceRange("carry_m", 1.0, 12.0, "m", _WIVOU_CITATION),
    },
    correlations={
        "entry_distance_behind_ball_m": -0.98,
        "divot_depth_m": -0.91,
    },
    absent_quantities=(
        "ball_launch_angle_rad",
        "ball_speed_m_s",
        "ball_spin_rad_s",
        "clubhead_speed_m_s",
    ),
    notes=(
        "Golfers of handicap 5-25, about 55-60 shots, lob wedge.",
        "Players were asked to enter 25-50 mm behind the ball and entered "
        "80-280 mm; the measured range is the delivered one, not the target.",
        "r = -0.98 for carry against entry distance is reported with other "
        "variables held constant; the unfiltered value is -0.42. The degrees "
        "of freedom behind the controlled figure are not stated.",
        "A 2019 follow-on reports peak downswing angular velocity only, never "
        "converted to a linear clubhead speed.",
    ),
)
"""The only primary greenside-bunker dataset located (issue #8616 research)."""

GRANULAR_INTRUSION_BENCHMARK = IntrusionBenchmark(
    citation=(
        "Agarwal, Senatore, Zhang, Kingsbury, Iagnemma, Goldman & Kamrin, "
        "J. Terramechanics (2019), arXiv:1901.10667; wheel-in-sand sinkage"
    ),
    sinkage_mae_m={
        "rft": 2.7e-3,
        "mpm": 3.2e-3,
        "bekker_wong_reece": 26.1e-3,
    },
    max_speed_m_s=1.44,
    natural_sand_bias=0.35,
    notes=(
        "RFT is not a fidelity compromise: its sinkage error is comparable "
        "with MPM's and an order of magnitude better than classical "
        "terramechanics, at roughly 1e6 times lower cost.",
        "The whole corpus tops out at 1.44 m/s. Greenside delivery is "
        "20-27 m/s, so this benchmark validates the solver at about 1/17 of "
        "the speed it is used at.",
        "RFT over-predicts by about 35% on natural sand, attributed to grain "
        "angularity. That bias is a known offset, not an uncertainty.",
    ),
)
"""The wheel-in-sand benchmark the DRFT constants are drawn from."""

REFERENCE_DATASETS: Mapping[str, ReferenceDataset] = MappingProxyType(
    {WIVOU_2016.key: WIVOU_2016}
)
"""Every published dataset this package may validate against."""

_FIELD_GAP = (
    "No published value exists anywhere. An exhaustive enumeration of the "
    "ISEA / Procedia Engineering / Engineering of Sport volumes, of Sports "
    "Engineering and of the Journal of Sports Sciences found no bunker, sand "
    "or wedge-interaction paper at all. This is a real gap in the field and "
    "not a search failure, so it cannot be closed by looking harder."
)

UNMEASURED_QUANTITIES: Mapping[str, str] = MappingProxyType(
    {
        "ball_launch_angle_rad": _FIELD_GAP,
        "ball_speed_m_s": _FIELD_GAP,
        "ball_spin_rad_s": _FIELD_GAP,
        "clubhead_deceleration_m_s2": _FIELD_GAP,
        "clubhead_speed_m_s": (
            _FIELD_GAP + " Wivou et al. (2019) report peak downswing angular velocity "
            "only, and never convert it to a linear clubhead speed."
        ),
        "energy_split_fraction": _FIELD_GAP,
        "ejecta_mass_kg": _FIELD_GAP,
        "coefficient_of_restitution_through_sand": _FIELD_GAP,
        "sand_energy_loss_j": _FIELD_GAP,
    }
)
"""Quantities for which no published measurement exists, and why.

Every entry is a quantity this model produces and which therefore looks
validatable.  None of them is."""


def reference_dataset(key: str) -> ReferenceDataset:
    """Look up a dataset by key.

    Args:
        key: Short dataset identifier, e.g. ``"wivou_2016"``.

    Returns:
        The dataset.

    Raises:
        NoReferenceDataError: If no dataset with that key is registered.
    """
    dataset = REFERENCE_DATASETS.get(key)
    if dataset is None:
        raise NoReferenceDataError(
            f"no reference dataset {key!r}; registered datasets are "
            f"{sorted(REFERENCE_DATASETS)}"
        )
    return dataset


def require_measurable(quantity: str) -> None:
    """Refuse to validate a quantity nobody has measured.

    Call this at the top of any routine that is about to compare a model
    output against a "reference" value.  It is the executable form of the
    issue's instruction not to invent a validation that cannot be
    performed.

    Args:
        quantity: SI-suffixed quantity name.

    Raises:
        NoReferenceDataError: If the quantity is in
            :data:`UNMEASURED_QUANTITIES`.
    """
    reason = UNMEASURED_QUANTITIES.get(quantity)
    if reason is not None:
        raise NoReferenceDataError(
            f"there is no published measurement of {quantity!r} to validate "
            f"against. {reason}"
        )


@dataclass(frozen=True, slots=True)
class DomainOverlap:
    """How much of a swept range a published measurement actually covers.

    Attributes:
        swept: The range the model is exercised over.
        measured: The range the reference data covers.
        overlap: The intersection, or ``None`` when they are disjoint.
        covered_fraction: Share of the swept range inside the measurement.
    """

    swept: tuple[float, float]
    measured: tuple[float, float]
    overlap: tuple[float, float] | None
    covered_fraction: float

    @property
    def is_extrapolation(self) -> bool:
        """True when any part of the sweep falls outside the measurement."""
        return self.covered_fraction < 1.0

    def describe(self) -> str:
        """One line stating the coverage, fit for the credibility statement."""
        if self.overlap is None:
            return (
                f"swept {self.swept[0]:g} to {self.swept[1]:g}, measured "
                f"{self.measured[0]:g} to {self.measured[1]:g}: disjoint, "
                "the whole sweep is extrapolation"
            )
        return (
            f"swept {self.swept[0]:g} to {self.swept[1]:g}, measured "
            f"{self.measured[0]:g} to {self.measured[1]:g}: "
            f"{self.covered_fraction:.0%} of the sweep is inside the "
            "measured domain"
        )


def domain_overlap(
    swept: tuple[float, float], measured: tuple[float, float]
) -> DomainOverlap:
    """Report what share of a swept range the reference data covers.

    "Over what domain of applicability" is one of the four questions a
    credibility statement has to answer, and it is the one most often
    answered by assertion.  This answers it by arithmetic.

    Args:
        swept: ``(low, high)`` the model is exercised over.
        measured: ``(low, high)`` covered by the reference data.

    Returns:
        The overlap and the covered fraction.

    Raises:
        ValueError: If either range is not ordered low to high.
    """
    for name, bounds in (("swept", swept), ("measured", measured)):
        if not bounds[1] > bounds[0]:
            raise ValueError(
                f"{name} range must be ordered low to high, got {bounds!r}"
            )
    low = max(swept[0], measured[0])
    high = min(swept[1], measured[1])
    if high <= low:
        return DomainOverlap(swept, measured, None, 0.0)
    return DomainOverlap(
        swept, measured, (low, high), (high - low) / (swept[1] - swept[0])
    )
