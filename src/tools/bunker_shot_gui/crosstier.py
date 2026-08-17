"""F0 against F1 on the quantities both produce (issue #8713, epic #8699).

This module computes; it draws nothing. No Qt, no matplotlib, in keeping
with the split the workbench established in issue #8618.

What the comparison is for
--------------------------

Track B's field pictures come from a tier that cannot be validated against
reality at true scale. What it *can* be is checked for consistency against
F0 on the quantities both tiers produce -- and where the two disagree, that
disagreement is the most useful output in the epic, because it localises
where F0's constitutive shortcut stops describing the physics.

So the arithmetic here is built to surface divergence rather than to
average it away. Every compared quantity carries a ratio and a verdict on
that ratio against a **declared** band; stretches of the record where the
two part company become :class:`DivergenceSpan` objects a renderer can
shade; and the sharpest single result -- the crossing of F0's inertial
share and F1's momentum-flux share -- is located by interpolation and
reported with its mechanism rather than as a number on its own.

What agreement licenses, and what it does not
---------------------------------------------

Nothing here is validation. ADR-0033 states it plainly and
:func:`licence_statement` restates it inside every view built on this
module, computed from :mod:`bunkershot3d.vandv.credibility` and the
solver's own envelope constants rather than quoted, so the sentence cannot
drift away from the code. Two uncalibrated models agreeing is two
uncalibrated models agreeing. The comparison can **falsify** -- a
disagreement beyond the declared band means at least one tier is wrong --
and that asymmetry is the whole of its value.

What F1 could not be asked
--------------------------

F1's :meth:`~bunkershot3d.solvers.mpm.solver.PlaneStrainMPMSolver.solve`
answers an *instantaneous* query by building a declared straight-line
constant-speed approach to the queried pose, and raises on a static one;
whole-shot marching is deferred to issue #8733. So there is no F1 force
*history* to overlay on F0's. What there is instead is a set of probes:
individual instants along F0's record, each handed to both tiers unchanged,
each conditional on its own declared approach. The view draws them as
points on F0's continuous curve, and says so. Two consequences follow and
both are carried explicitly rather than hidden:

* **Speed lost** is F0's alone. F1's section is driven kinematically, so it
  loses no speed at all. What is reported is what F1's force *would* have
  removed from the same head over the same window -- one-way coupled, since
  F1 was never asked what the slower head would have done.
* **The divot** is not the same measurement on the two sides. F0 moves no
  sand, so its divot is the swept lower envelope of the head; F1's is where
  the sand actually ended up. Comparing them is worth doing and is not a
  like-for-like check, so :attr:`ComparedQuantity.note` says which is which
  wherever the number is drawn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.solvers import EnvelopeStatus, FidelityTier
from bunkershot3d.solvers.envelope import MAX_VALIDATED_SPEED_M_S
from bunkershot3d.solvers.mpm.verification import F0CrossCheck
from bunkershot3d.vandv.credibility import envelope_exceedance

from .agreement import (
    DECLARED_AGREEMENT_BAND,
    AgreementClass,
    ComparedQuantity,
    DivergenceSpan,
    QuantityAgreement,
    licence_statement,
)
from .traces import ValidityBand

__all__ = [
    "DECLARED_AGREEMENT_BAND",
    "AgreementClass",
    "ComparedQuantity",
    "CrossTierComparison",
    "CrossTierProbe",
    "DivergenceSpan",
    "InertialCrossover",
    "QuantityAgreement",
    "inertial_share_crossover",
    "licence_statement",
]
"""Re-exported from :mod:`~src.tools.bunker_shot_gui.agreement` so a caller
comparing two tiers has one import site rather than two."""

_STAMP_MAX_CHARS = 240

_ADR_BULK_RESOLUTION_M = (0.001, 0.002)
"""The cell-size band ADR-0033 specifies for the F1 tier."""

_TIER_F0 = "f0"
_TIER_F1 = "f1"

_GRAZING_FRACTION = 0.05
"""Below this share of the peak, F0's force is its engagement criterion."""


@dataclass(frozen=True)
class CrossTierProbe:
    """One instant, handed to both tiers unchanged.

    The F0 half is the tier's answer at the recorded pose, which is the
    same query :func:`~bunkershot3d.solvers.shot.simulate_shot` solved at
    that sample. The F1 half is a whole march: F1 has no instantaneous
    answer, so it reverses the section along its own velocity until it is
    clear of the bed and drives it back at constant speed. That declared
    approach is the modelling assumption the F1 number rests on, and it is
    what makes the two comparable at all.

    Attributes:
        frame: Index into the F0 record this probe was taken at. Zero for a
            probe from a declared speed sweep, which indexes no record.
        time_s: The moment [s].
        check: The paired result, from the F1 package's own cross-check.
        f0_divot_section_area_m2: F0's swept-envelope section at this
            sample [m^2].
        f0_sole_depth_m: F0's sole depth at this sample [m].
        declared_width_m: The out-of-plane width **both** divot masses are
            formed at, so the mass comparison rests on one assumption
            rather than two.
        bulk_density_kg_m3: The bed's bulk density, for the same masses.
    """

    frame: int
    time_s: float
    check: F0CrossCheck
    f0_divot_section_area_m2: float
    f0_sole_depth_m: float
    declared_width_m: float
    bulk_density_kg_m3: float

    def __post_init__(self) -> None:
        """Validate the probe.

        Raises:
            ValueError: If the frame is negative, the time is not finite,
                or a declared width or density is not positive.
        """
        if int(self.frame) < 0:
            raise ValueError(f"frame must be non-negative, got {self.frame!r}")
        if not math.isfinite(self.time_s):
            raise ValueError(f"time_s must be finite, got {self.time_s!r}")
        for name, value in (
            ("declared_width_m", self.declared_width_m),
            ("bulk_density_kg_m3", self.bulk_density_kg_m3),
            ("f0_divot_section_area_m2", self.f0_divot_section_area_m2),
            ("f0_sole_depth_m", self.f0_sole_depth_m),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value!r}"
                )
        if self.declared_width_m <= 0.0 or self.bulk_density_kg_m3 <= 0.0:
            raise ValueError(
                "a divot mass needs a positive declared width and density; "
                f"got {self.declared_width_m!r} m and "
                f"{self.bulk_density_kg_m3!r} kg/m^3"
            )

    # ------------------------------------------------------------- readings

    @property
    def speed_m_s(self) -> float:
        """Intrusion speed the pair was run at [m/s]."""
        return self.check.speed_m_s

    @property
    def f0_force_magnitude_n(self) -> float:
        """``|F0|`` [N]."""
        return float(np.linalg.norm(self.check.f0_force_n))

    @property
    def f1_force_magnitude_n(self) -> float:
        """``|F1|`` [N], at the declared effective width."""
        return float(np.linalg.norm(self.check.f1_force_n))

    @property
    def direction_agreement(self) -> float:
        """Cosine between the two resultants.

        The one number in the comparison that does not depend on the
        declared width at all, and therefore the one that is not
        conditional on a modelling assumption.
        """
        return self.check.direction_agreement

    @property
    def f0_inertial_fraction(self) -> float:
        """Share of F0's force carried by its ``lambda rho v_n^2`` term."""
        return self.check.f0_inertial_fraction

    @property
    def f1_flux_fraction(self) -> float:
        """Share of F1's reaction carried by momentum flux."""
        return self.check.f1_flux_fraction

    @property
    def inertial_share_gap(self) -> float:
        """F0's inertial share minus F1's flux share.

        Negative below the crossover, positive above it. The sign change is
        the sharpest single result in the comparison; see
        :func:`inertial_share_crossover`.
        """
        return self.f0_inertial_fraction - self.f1_flux_fraction

    @property
    def submerged_depth_m(self) -> float:
        """Deepest submerged element of the shared query [m].

        The depth **both** tiers were handed, read off the geometry. The
        compared depth row uses this rather than either tier's own
        ``SolverResult.max_depth_m``, because those two fields do not mean
        the same thing across the tiers -- see
        :attr:`~bunkershot3d.solvers.mpm.verification.F0CrossCheck.submerged_depth_m`.
        """
        return self.check.submerged_depth_m

    @property
    def f0_engaged_depth_m(self) -> float:
        """F0's deepest **engaged** element [m] -- a contact diagnostic.

        Reported beside the compared depth rather than as it. On a lofted
        head most elements never lead, so this runs far shallower than the
        geometry; issue #8701 is the history of that distinction being
        conflated once already.
        """
        return self.check.f0_max_depth_m

    @property
    def f1_divot_section_area_m2(self) -> float:
        """Section of sand F1 removed from below the original surface [m^2]."""
        return self.check.f1_divot_section_area_m2

    @property
    def f0_divot_mass_kg(self) -> float:
        """F0's swept section carried to a mass at the declared width."""
        return (
            self.f0_divot_section_area_m2
            * self.declared_width_m
            * self.bulk_density_kg_m3
        )

    @property
    def f1_divot_mass_kg(self) -> float:
        """F1's removed section carried to a mass at the same width."""
        return self.check.f1_divot.displaced_mass_kg(
            width_m=self.declared_width_m,
            bulk_density_kg_m3=self.bulk_density_kg_m3,
        )

    @property
    def divot_fully_resolved(self) -> bool:
        """Whether every bin of F1's free surface held sand."""
        return self.check.f1_divot.fully_resolved

    def divot_caveat(self) -> str:
        """What has to be said beside F1's divot numbers, or an empty string."""
        if self.divot_fully_resolved:
            return ""
        return (
            f"{self.check.f1_divot.n_empty_bins} of "
            f"{self.check.f1_divot.n_bins} surface bins held no sand at "
            f"{self.time_s * 1e3:.2f} ms, so F1's divot section is a lower bound"
        )

    # ------------------------------------------------------------ agreement

    def agreement(
        self, quantity: ComparedQuantity, *, band: float = DECLARED_AGREEMENT_BAND
    ) -> QuantityAgreement:
        """Compare one instantaneous quantity at this probe.

        Args:
            quantity: Which quantity.
            band: Half-width of the agreement band on ``|ln ratio|``.

        Returns:
            The agreement.

        Raises:
            ValueError: If asked for :attr:`ComparedQuantity.SPEED_LOST`,
                which is a window integral and has no value at an instant.
                Returning zero, or the instantaneous force, would put a
                number in a row that has no meaning here.
        """
        if quantity is ComparedQuantity.SPEED_LOST:
            raise ValueError(
                "speed lost is measured over a window, not at an instant; ask "
                "the comparison, which owns the probed window"
            )
        pairs = {
            ComparedQuantity.WRENCH: (
                self.f0_force_magnitude_n,
                self.f1_force_magnitude_n,
            ),
            ComparedQuantity.SOLE_DEPTH: (
                self.check.submerged_depth_m,
                self.check.f1_max_depth_m,
            ),
            ComparedQuantity.DIVOT_SECTION: (
                self.f0_divot_section_area_m2,
                self.f1_divot_section_area_m2,
            ),
            ComparedQuantity.DIVOT_MASS: (
                self.f0_divot_mass_kg,
                self.f1_divot_mass_kg,
            ),
        }
        f0_value, f1_value = pairs[quantity]
        return QuantityAgreement(
            quantity=quantity, f0_value=f0_value, f1_value=f1_value, band=band
        )


@dataclass(frozen=True, slots=True)
class InertialCrossover:
    """Where F0's inertial share crosses F1's momentum-flux share.

    The sharpest single result the comparison produces, because the two
    shares diverge *by construction* rather than by numerical accident:
    F0's ``lambda rho v_n^2`` term grows quadratically with nothing to
    bound it, while the continuum's reaction is limited by how fast the
    yield surface lets sand be accelerated out of the way. Below the
    crossing F0 attributes less of its force to inertia than the continuum
    does; above it, more, and the gap keeps widening.

    Attributes:
        speed_m_s: Intrusion speed at which the two shares are equal.
        shared_share: The share both tiers report there.
        lower_speed_m_s: Speed of the probe below the crossing.
        upper_speed_m_s: Speed of the probe above it.
        time_s: The moment, when the crossing was bracketed along a shot
            rather than a declared speed sweep; ``None`` otherwise.
    """

    speed_m_s: float
    shared_share: float
    lower_speed_m_s: float
    upper_speed_m_s: float
    time_s: float | None = None

    def summary(self) -> str:
        """The sentence the crossover panel is captioned with."""
        moment = "" if self.time_s is None else f" ({self.time_s * 1e3:.2f} ms)"
        return (
            f"F0's inertial share crosses F1's momentum-flux share at "
            f"{self.speed_m_s:.3g} m/s{moment}, both at "
            f"{self.shared_share:.2f}, bracketed between "
            f"{self.lower_speed_m_s:.3g} and {self.upper_speed_m_s:.3g} m/s. "
            "Below it F0 credits less of its force to inertia than the "
            "continuum does; above it, more, and the gap widens with speed "
            "because F0's dynamic term is unbounded in v while the "
            "continuum's reaction is limited by how fast the yield surface "
            "lets sand accelerate out of the way"
        )


def inertial_share_crossover(
    probes: Sequence[CrossTierProbe],
) -> InertialCrossover | None:
    """Locate the crossing of the two inertial shares, or report none.

    Linear interpolation between the two probes that bracket the sign
    change of :attr:`CrossTierProbe.inertial_share_gap`. Nothing is
    extrapolated: a probe set that never brackets the crossing returns
    ``None`` rather than a number invented outside its own range.

    Args:
        probes: Probes, in any order; they are sorted by speed here.

    Returns:
        The crossing, or ``None`` when the probed range does not contain
        one.
    """
    ordered = sorted(probes, key=lambda item: item.speed_m_s)
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        low_gap = lower.inertial_share_gap
        high_gap = upper.inertial_share_gap
        if low_gap == high_gap or (low_gap > 0.0) == (high_gap > 0.0):
            continue
        span = upper.speed_m_s - lower.speed_m_s
        fraction = low_gap / (low_gap - high_gap)
        share = lower.f0_inertial_fraction + fraction * (
            upper.f0_inertial_fraction - lower.f0_inertial_fraction
        )
        moment = (
            lower.time_s + fraction * (upper.time_s - lower.time_s)
            if lower.time_s != upper.time_s
            else None
        )
        return InertialCrossover(
            speed_m_s=lower.speed_m_s + fraction * span,
            shared_share=float(share),
            lower_speed_m_s=lower.speed_m_s,
            upper_speed_m_s=upper.speed_m_s,
            time_s=moment,
        )
    return None


@dataclass(frozen=True)
class CrossTierComparison:
    """F0's record, F1's probes on it, and where the two part company.

    Attributes:
        shot_probes: Probes taken along the F0 record, on the shared
            cursor. At least one is required.
        time_s: ``(T,)`` F0's own sample times [s] -- the axis the
            workbench's single transport scrubs.
        f0_force_n: ``(T, 3)`` sand force on the head [N].
        f0_sole_depth_m: ``(T,)`` sole depth below the free surface [m].
        f0_velocity_m_s: ``(T, 3)`` head velocity [m/s]. The vector, not
            the speed, because the compared speed-lost quantity is each
            tier's resultant *projected on the direction of travel*.
        f0_divot_section_area_m2: ``(T,)`` swept-envelope section [m^2].
        band: F0's per-sample validity band, inherited unchanged. Nothing
            here improves it.
        head_mass_kg: The head, for the derived speed-lost integral.
        declared_width_m: The out-of-plane width both divot masses use.
        bulk_density_kg_m3: The bed's bulk density, for the same masses.
        f1_cell_size_m: The grid F1 was solved on [m]. Carried because it
            is provenance, not a setting: ADR-0033 specifies bulk
            resolution of 1-2 mm, and a comparison run coarser than that
            has to say so in the picture rather than in whoever ran it.
        sweep_probes: Probes from a **declared speed sweep** at a fixed
            pose, which is a separate experiment from the shot and is
            labelled as one. Present so the inertial-share crossover can be
            bracketed even when the shot itself never slows through it --
            a greenside delivery does not.
        agreement_band: Half-width of the agreement band on ``|ln ratio|``.
    """

    shot_probes: tuple[CrossTierProbe, ...]
    time_s: NDArray[np.float64]
    f0_force_n: NDArray[np.float64]
    f0_sole_depth_m: NDArray[np.float64]
    f0_velocity_m_s: NDArray[np.float64]
    f0_divot_section_area_m2: NDArray[np.float64]
    band: ValidityBand
    head_mass_kg: float
    declared_width_m: float
    bulk_density_kg_m3: float
    f1_cell_size_m: float
    sweep_probes: tuple[CrossTierProbe, ...] = ()
    agreement_band: float = DECLARED_AGREEMENT_BAND

    def __post_init__(self) -> None:
        """Validate the comparison against the record it is drawn on.

        Raises:
            ValueError: If there are no probes, if any array disagrees with
                the time axis, if a probe indexes outside the record, or if
                a probe's own time disagrees with the sample it names. The
                last is the important one: a probe drawn at the wrong
                moment is worse than a missing probe, because it looks like
                a measurement of a moment it did not measure.
        """
        probes = tuple(self.shot_probes)
        if not probes:
            raise ValueError(
                "a cross-tier comparison needs at least one probe; an empty "
                "overlay would read as two tiers that were never compared"
            )
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        if times.size < 2:
            raise ValueError(
                f"the F0 record needs at least 2 samples, got {times.size}"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        forces = np.asarray(self.f0_force_n, dtype=np.float64)
        velocities = np.asarray(self.f0_velocity_m_s, dtype=np.float64)
        for name, block in (("f0_force_n", forces), ("f0_velocity_m_s", velocities)):
            if block.shape != (times.size, 3):
                raise ValueError(
                    f"{name} must have shape {(times.size, 3)}, got {block.shape}"
                )
        columns = {
            "f0_sole_depth_m": np.asarray(self.f0_sole_depth_m, dtype=np.float64),
            "f0_divot_section_area_m2": np.asarray(
                self.f0_divot_section_area_m2, dtype=np.float64
            ),
        }
        for name, column in columns.items():
            if column.shape != (times.size,):
                raise ValueError(
                    f"{name} must have shape {(times.size,)}, got {column.shape}"
                )
        if self.band.n_frames != times.size:
            raise ValueError(
                "the validity band describes a different record: "
                f"{self.band.n_frames} samples against {times.size}"
            )
        for probe in probes:
            if not 0 <= probe.frame < times.size:
                raise ValueError(
                    f"probe frame {probe.frame} is outside the F0 record, which "
                    f"has {times.size} samples"
                )
            if not math.isclose(
                probe.time_s, float(times[probe.frame]), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"probe at frame {probe.frame} carries t="
                    f"{probe.time_s * 1e3:.4f} ms while that sample is at "
                    f"t={float(times[probe.frame]) * 1e3:.4f} ms; a probe drawn "
                    "at the wrong moment reads as a measurement of a moment it "
                    "did not measure"
                )
        for name, value in (
            ("head_mass_kg", self.head_mass_kg),
            ("declared_width_m", self.declared_width_m),
            ("bulk_density_kg_m3", self.bulk_density_kg_m3),
            ("f1_cell_size_m", self.f1_cell_size_m),
            ("agreement_band", self.agreement_band),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        object.__setattr__(self, "shot_probes", probes)
        object.__setattr__(self, "sweep_probes", tuple(self.sweep_probes))
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "f0_force_n", forces)
        object.__setattr__(self, "f0_velocity_m_s", velocities)
        for name, column in columns.items():
            object.__setattr__(self, name, column)

    # ------------------------------------------------------------ the record

    @property
    def n_frames(self) -> int:
        """Samples in the F0 record."""
        return int(self.time_s.size)

    @property
    def n_probes(self) -> int:
        """Probes taken along it."""
        return len(self.shot_probes)

    @property
    def tiers(self) -> tuple[FidelityTier, FidelityTier]:
        """The pair being compared, in the order every readout names them."""
        return (FidelityTier.F0, FidelityTier.F1)

    @property
    def worst_status(self) -> EnvelopeStatus:
        """The worst verdict anywhere in the record, inherited from F0."""
        return self.band.worst

    @property
    def f0_force_magnitude_n(self) -> NDArray[np.float64]:
        """``(T,)`` resultant magnitude of F0's recorded force [N]."""
        return np.linalg.norm(self.f0_force_n, axis=1)

    @property
    def f0_speed_m_s(self) -> NDArray[np.float64]:
        """``(T,)`` head speed [m/s], from the recorded velocity."""
        return np.linalg.norm(self.f0_velocity_m_s, axis=1)

    @property
    def probe_frames(self) -> tuple[int, ...]:
        """Sample indices the probes sit on."""
        return tuple(probe.frame for probe in self.shot_probes)

    @property
    def probe_times_s(self) -> tuple[float, ...]:
        """Moments the probes sit at [s]."""
        return tuple(probe.time_s for probe in self.shot_probes)

    @property
    def probe_speeds_m_s(self) -> tuple[float, ...]:
        """Intrusion speeds the probes were run at [m/s]."""
        return tuple(probe.speed_m_s for probe in self.shot_probes)

    @property
    def peak_probe(self) -> CrossTierProbe:
        """The probe at which F0's own force was largest.

        The shot-level agreement is quoted here rather than averaged over
        the probes, because a mean over a window that is mostly free flight
        would report a divergence smaller than the one at the moment a
        designer is looking at.
        """
        return max(self.shot_probes, key=lambda probe: probe.f0_force_magnitude_n)

    def probe_values(
        self, quantity: ComparedQuantity
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(f0, f1)`` values of one instantaneous quantity at every probe.

        Args:
            quantity: Which quantity.

        Returns:
            Two ``(n_probes,)`` arrays, in SI.

        Raises:
            ValueError: For :attr:`ComparedQuantity.SPEED_LOST`, which has
                no per-probe value.
        """
        pairs = [
            probe.agreement(quantity, band=self.agreement_band)
            for probe in self.shot_probes
        ]
        return (
            np.array([item.f0_value for item in pairs]),
            np.array([item.f1_value for item in pairs]),
        )

    # ---------------------------------------------------------- speed lost

    @property
    def probe_window(self) -> tuple[int, int]:
        """``(first, last)`` sample indices the probes bracket.

        Raises:
            ValueError: If there is only one probe, so there is no window.
        """
        frames = self.probe_frames
        if len(set(frames)) < 2:
            raise ValueError(
                "speed lost is measured over the window the probes bracket, and "
                f"{len(frames)} probe(s) at {sorted(set(frames))} bracket none; "
                "probe at least two distinct samples"
            )
        return (min(frames), max(frames))

    @property
    def f0_recorded_speed_lost_m_s(self) -> float:
        """Speed F0's own record shows the head losing over the probed window.

        The truth for F0, and *not* the number the agreement row compares:
        it is read off every sample while F1 is known at a handful, so
        putting the two side by side would report the quadrature as a
        divergence. It is stated beside the compared pair so that
        quadrature error is visible rather than absorbed.
        """
        first, last = self.probe_window
        speed = self.f0_speed_m_s
        return float(speed[first] - speed[last])

    @property
    def f0_speed_lost_m_s(self) -> float:
        """What F0's force removes from the head over the probed window.

        Integrated the same way as :attr:`f1_speed_lost_m_s`: the resultant
        projected on the direction of travel, sampled at the probes,
        interpolated between them and integrated. Same quadrature on both
        sides, because comparing an exact number against a coarsely
        sampled one measures the sampling.
        """
        return self._projected_speed_lost_m_s(_TIER_F0)

    @property
    def f1_speed_lost_m_s(self) -> float:
        """What F1's force would have taken off the same head, same window.

        The resultant **projected on the direction of travel**, so a force
        that does not oppose the motion does not decelerate. The measured
        direction cosines run from 0.14 at a grazing sample to 0.997 at the
        peak, and a magnitude-only estimate would credit the first of those
        with as much braking as the second.

        One-way coupled, and that is not a detail: F1 was queried at the
        poses and speeds F0's head actually reached, so this is what F1's
        force would have removed from a head that never slowed down more
        than F0 says it did.
        """
        return self._projected_speed_lost_m_s(_TIER_F1)

    def _projected_force_n(self, tier: str) -> NDArray[np.float64]:
        """Braking force at each probe: the resultant against the motion.

        Args:
            tier: ``"f0"`` or ``"f1"``.

        Returns:
            ``(n_probes,)`` the component of that tier's resultant opposing
            the travel direction [N], in probe order. Negative where the
            resultant pushes the head along, which is a real thing a
            grazing sole can do and is not clipped away.
        """
        ordered = sorted(self.shot_probes, key=lambda item: item.frame)
        braking = np.zeros(len(ordered))
        for index, probe in enumerate(ordered):
            velocity = self.f0_velocity_m_s[probe.frame]
            speed = float(np.linalg.norm(velocity))
            if speed <= 0.0:
                continue
            force = (
                probe.check.f0_force_n if tier == _TIER_F0 else probe.check.f1_force_n
            )
            braking[index] = -float(np.asarray(force) @ velocity) / speed
        return braking

    def _projected_speed_lost_m_s(self, tier: str) -> float:
        """Integrate one tier's braking force over the probed window.

        Raises:
            ValueError: If the probes bracket no window.
        """
        _ = self.probe_window
        ordered = sorted(self.shot_probes, key=lambda item: item.frame)
        times = np.array([probe.time_s for probe in ordered])
        return float(
            np.trapezoid(self._projected_force_n(tier), x=times) / self.head_mass_kg
        )

    def implied_speed_m_s(self, tier: str = _TIER_F1) -> NDArray[np.float64]:
        """``(T,)`` the speed history one tier's braking force implies.

        Starts from F0's own recorded speed at the first probe and removes
        the running integral of the braking force, so both tiers' curves
        leave the same point and the gap between them at the end is the
        disagreement in speed lost. ``nan`` outside the probed window,
        because outside it nothing was compared and a continued line would
        assert a comparison that was never made.

        Args:
            tier: ``"f0"`` or ``"f1"``.

        Returns:
            The implied speed [m/s].

        Raises:
            ValueError: If the probes bracket no window.
        """
        first, last = self.probe_window
        ordered = sorted(self.shot_probes, key=lambda item: item.frame)
        times = np.array([probe.time_s for probe in ordered])
        window = self.time_s[first : last + 1]
        sampled = np.interp(window, times, self._projected_force_n(tier))
        lost = np.concatenate(
            ([0.0], np.cumsum(0.5 * (sampled[:-1] + sampled[1:]) * np.diff(window)))
        )
        implied = np.full(self.n_frames, np.nan)
        implied[first : last + 1] = (
            float(self.f0_speed_m_s[first]) - lost / self.head_mass_kg
        )
        return implied

    def engagement_caveat(self) -> str:
        """Name the probes where F0 reported almost no force, or say nothing.

        F0 reports zero the moment no element is both submerged *and*
        leading-edge, which happens while the sole is still geometrically
        in the divot -- the disengaged tail issue #8702 documents. A ratio
        formed at such a sample is a division by an engagement criterion
        rather than by a physical force, and it is by far the largest ratio
        on the page, so it is named rather than left to dominate.

        Returns:
            One sentence, or an empty string when every probe carried load.
        """
        peak = self.peak_probe.f0_force_magnitude_n
        if peak <= 0.0:
            return ""
        grazing = [
            probe
            for probe in sorted(self.shot_probes, key=lambda item: item.frame)
            if probe.f0_force_magnitude_n < _GRAZING_FRACTION * peak
        ]
        if not grazing:
            return ""
        moments = ", ".join(f"{probe.time_s * 1e3:.2f} ms" for probe in grazing)
        return (
            f"At {moments} F0 reports under {_GRAZING_FRACTION:.0%} of its peak "
            "force while the sole is still in the divot: it reports zero the "
            "moment no element is both submerged and leading-edge (#8702). "
            "Ratios at those probes divide by an engagement criterion, not by "
            "a physical force, and they are the largest ratios on this page."
        )

    # ----------------------------------------------------------- agreement

    def agreement(self, quantity: ComparedQuantity) -> QuantityAgreement:
        """The shot-level agreement on one quantity.

        Args:
            quantity: Which quantity.

        Returns:
            The agreement. Instantaneous quantities are quoted at
            :attr:`peak_probe`; speed lost is the window integral.

        Raises:
            ValueError: If speed lost is asked for and the probes bracket
                no window.
        """
        if quantity is ComparedQuantity.SPEED_LOST:
            return QuantityAgreement(
                quantity=quantity,
                f0_value=max(self.f0_speed_lost_m_s, 0.0),
                f1_value=max(self.f1_speed_lost_m_s, 0.0),
                band=self.agreement_band,
            )
        return self.peak_probe.agreement(quantity, band=self.agreement_band)

    def agreements(self) -> tuple[QuantityAgreement, ...]:
        """Every quantity that could be compared, in reporting order.

        Speed lost is omitted rather than faked when fewer than two
        distinct samples were probed.
        """
        results: list[QuantityAgreement] = []
        for quantity in ComparedQuantity:
            try:
                results.append(self.agreement(quantity))
            except ValueError:
                continue
        return tuple(results)

    def agreement_series(
        self, quantity: ComparedQuantity
    ) -> tuple[QuantityAgreement, ...]:
        """One agreement per probe, in record order."""
        return tuple(
            probe.agreement(quantity, band=self.agreement_band)
            for probe in sorted(self.shot_probes, key=lambda item: item.frame)
        )

    def divergence_spans(
        self, quantity: ComparedQuantity
    ) -> tuple[DivergenceSpan, ...]:
        """Stretches of the record over which one quantity left the band.

        A run of consecutive divergent probes spans from the first to the
        last of them. A **lone** divergent probe is widened to the midpoints
        of its neighbours, so a single instant of disagreement is still
        visible when it is shaded rather than collapsing to a zero-width
        line nobody can see.

        Args:
            quantity: Which quantity to scan.

        Returns:
            The spans, in time order.
        """
        ordered = sorted(self.shot_probes, key=lambda item: item.frame)
        flags = [
            probe.agreement(quantity, band=self.agreement_band).diverged
            for probe in ordered
        ]
        spans: list[DivergenceSpan] = []
        start: int | None = None
        for index in range(len(ordered) + 1):
            live = index < len(ordered) and flags[index]
            if live and start is None:
                start = index
            elif not live and start is not None:
                spans.append(self._span(quantity, ordered, start, index - 1))
                start = None
        return tuple(spans)

    def _span(
        self,
        quantity: ComparedQuantity,
        ordered: list[CrossTierProbe],
        first: int,
        last: int,
    ) -> DivergenceSpan:
        """Build one span from a run of divergent probes."""
        start_s = ordered[first].time_s
        end_s = ordered[last].time_s
        if first == last:
            before = ordered[first - 1].time_s if first > 0 else float(self.time_s[0])
            after = (
                ordered[last + 1].time_s
                if last + 1 < len(ordered)
                else float(self.time_s[-1])
            )
            start_s = 0.5 * (before + start_s)
            end_s = 0.5 * (end_s + after)
        ratios = [
            ordered[index].agreement(quantity, band=self.agreement_band).ratio
            for index in range(first, last + 1)
        ]
        finite = [value for value in ratios if math.isfinite(value)]
        worst = (
            max(finite, key=lambda value: abs(math.log(value)) if value > 0.0 else 0.0)
            if finite
            else math.nan
        )
        return DivergenceSpan(
            quantity=quantity,
            start_s=start_s,
            end_s=end_s,
            worst_ratio=worst,
            n_probes=last - first + 1,
        )

    # ----------------------------------------------------------- crossover

    @property
    def crossover_probes(self) -> tuple[CrossTierProbe, ...]:
        """Which probe set the crossover is read from.

        The declared speed sweep when there is one, because a greenside
        shot does not slow through the crossing and the shot probes alone
        would report only that the whole record sits above it.
        """
        return self.sweep_probes or self.shot_probes

    def crossover(self) -> InertialCrossover | None:
        """Locate the inertial-share crossing, or ``None`` if unbracketed."""
        return inertial_share_crossover(self.crossover_probes)

    def crossover_summary(self) -> str:
        """State the crossing either way, since "not here" is also a result."""
        crossing = self.crossover()
        if crossing is not None:
            return crossing.summary()
        probes = sorted(self.crossover_probes, key=lambda item: item.speed_m_s)
        gaps = [probe.inertial_share_gap for probe in probes]
        side = "above" if gaps and gaps[0] > 0.0 else "below"
        f0_range = (
            f"{probes[0].f0_inertial_fraction:.2f}-"
            f"{probes[-1].f0_inertial_fraction:.2f}"
        )
        f1_range = f"{probes[0].f1_flux_fraction:.2f}-{probes[-1].f1_flux_fraction:.2f}"
        return (
            f"No crossing inside the probed range: every probe sits {side} it. "
            f"Over {probes[0].speed_m_s:.3g}-{probes[-1].speed_m_s:.3g} m/s, F0 "
            f"credits {f0_range} of its force to its dynamic term while F1 "
            f"credits {f1_range} to momentum flux. F0's term grows "
            "quadratically in speed with nothing to bound it; the continuum's "
            "reaction is limited by how fast the yield surface lets sand "
            "accelerate out of the way, so the two disagree by construction in "
            "exactly this regime"
        )

    # ------------------------------------------------------------- licence

    def resolution_note(self) -> str:
        """State the grid F1 was solved on against ADR-0033's own figure.

        Returns:
            One sentence, which says the resolution is coarser than the
            specified band when it is. ADR-0033 bars this tier from
            quoting club force at *any* resolution, so this is provenance
            rather than a caveat that could be lifted by refining.
        """
        size_mm = self.f1_cell_size_m * 1e3
        band = (
            "inside"
            if _ADR_BULK_RESOLUTION_M[0]
            <= self.f1_cell_size_m
            <= _ADR_BULK_RESOLUTION_M[1]
            else "coarser than"
            if self.f1_cell_size_m > _ADR_BULK_RESOLUTION_M[1]
            else "finer than"
        )
        return (
            f"F1 solved at dx = {size_mm:.3g} mm, {band} the "
            f"{_ADR_BULK_RESOLUTION_M[0] * 1e3:g}-"
            f"{_ADR_BULK_RESOLUTION_M[1] * 1e3:g} mm bulk resolution ADR-0033 "
            "specifies; refining does not make the tier quotable for club "
            "force, which stays F0's."
        )

    def licence(self) -> str:
        """What agreement on this page does and does not license."""
        return licence_statement(
            speed_m_s=float(self.f0_speed_m_s.max()),
            effective_width_m=self.peak_probe.check.effective_width_m,
        )

    def licence_stamp(self) -> str:
        """The short form, for an in-frame stamp beside the validity verdict."""
        exceedance = envelope_exceedance(speed_m_s=float(self.f0_speed_m_s.max()))
        text = (
            "F0 vs F1 consistency check, not validation: two uncalibrated "
            "models, NASA-STD-7009B validation 0 of 4, "
            f"{exceedance.speed_exceedance:.0f}x past the "
            f"{MAX_VALIDATED_SPEED_M_S:g} m/s published ceiling"
        )
        return text[:_STAMP_MAX_CHARS]

    def summary(self) -> str:
        """The whole comparison as text, licence first.

        Returns:
            The block a report or a headless run prints. The licence comes
            first on purpose: a reader who stops after one paragraph should
            have read the caveat rather than the ratios.
        """
        rows = "\n".join(f"  {item.summary()}" for item in self.agreements())
        caveats = [
            probe.divot_caveat() for probe in self.shot_probes if probe.divot_caveat()
        ]
        spans = [
            span.label
            for quantity in ComparedQuantity
            for span in self._safe_spans(quantity)
        ]
        parts = [
            self.licence(),
            "",
            f"Probed {self.n_probes} instant(s) of a {self.n_frames}-sample F0 "
            f"record; each F1 point is its own march to that pose under a "
            f"declared straight-line approach, not a marched shot (issue #8733). "
            f"{self.resolution_note()}",
            "",
            "Agreement, quantity by quantity:",
            rows,
            "",
            self.crossover_summary(),
        ]
        if spans:
            parts += ["", "Divergence:", "\n".join(f"  {line}" for line in spans)]
        engagement = self.engagement_caveat()
        if engagement:
            caveats = [engagement, *caveats]
        if caveats:
            parts += ["", "Caveats:", "\n".join(f"  {line}" for line in caveats)]
        return "\n".join(parts)

    def _safe_spans(self, quantity: ComparedQuantity) -> tuple[DivergenceSpan, ...]:
        """Spans for one quantity, or none when it cannot be compared."""
        try:
            return self.divergence_spans(quantity)
        except ValueError:
            return ()
