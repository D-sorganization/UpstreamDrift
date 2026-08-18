"""Bounce utilisation map -- where the sole actually carried load (issue #8614).

This is the metric that makes the tool **prescriptive rather than merely
evaluative**. Every fidelity tier that integrates a traction over the sole
already knows each element's own contribution -- DRFT builds the club force as
exactly that integral -- so resolving the load spatially costs almost nothing
and tells the designer *where to grind*: sole area that never carried load is
material that can be removed for free.

Input contract
--------------

Result schema v2 stores the **resultant** wrench, not the per-element tractions,
so this metric consumes a second, explicitly structure-of-arrays artifact that a
solver emits alongside it:

* ``element_centroid_body_m`` ``(E, 3)`` -- element centroids in head body axes
* ``element_area_m2`` ``(E,)`` -- element areas
* ``element_normal_force_N`` ``(T, E)`` -- normal load carried per element, per
  sample; compressive and therefore non-negative

Definitions
-----------

============================ ==============================================================
Quantity                     Definition
============================ ==============================================================
Element impulse              ``integral of F_e dt`` [N.s]. Impulse rather than peak force,
                             because a sliver that spikes for one sample carried nothing.
Impulse density              ``element impulse / element area`` [Pa.s]. Per-area, so a
                             large element cannot look important merely by being large.
Loaded                       Impulse density at or above ``threshold_fraction`` of the
                             largest element density. A fraction, not an absolute, so the
                             map is invariant to overall load scale.
Utilised area                Summed area of loaded elements [m^2].
Utilisation fraction         Utilised area / total sole area; dimensionless.
Removable area               Total minus utilised [m^2] -- the free-material number.
Centre of pressure           Impulse-weighted mean of the element centroids, in body axes.
                             Its position relative to the CG is the pitching couple.
============================ ==============================================================

The map is reported in **body axes**, not world, because the answer is a
grinding instruction on a physical head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import ensure

__all__ = [
    "DEFAULT_LOAD_THRESHOLD_FRACTION",
    "BounceUtilisation",
    "LoadProfile",
    "SoleLoadTrace",
    "bounce_utilisation",
]

#: An element is "loaded" once its impulse density reaches this fraction of the
#: most heavily loaded element's. 1 % keeps numerical dust out of the utilised
#: area without discarding a genuinely lightly loaded trailing edge.
DEFAULT_LOAD_THRESHOLD_FRACTION = 0.01


@dataclass(frozen=True)
class SoleLoadTrace:
    """Per-element sole loading over a strike, structure-of-arrays.

    Attributes:
        time_s: ``(T,)`` strictly increasing sample times [s].
        element_centroid_body_m: ``(E, 3)`` element centroids in body axes [m].
        element_area_m2: ``(E,)`` element areas [m^2]; strictly positive.
        element_normal_force_N: ``(T, E)`` normal load carried by each element
            at each sample [N]; non-negative, because sand pushes.
    """

    time_s: np.ndarray
    element_centroid_body_m: np.ndarray
    element_area_m2: np.ndarray
    element_normal_force_N: np.ndarray

    def __post_init__(self) -> None:
        """Validate the load trace.

        Raises:
            ValueError: If shapes disagree, a value is not finite, time is not
                strictly increasing, an area is not positive, or an element
                force is negative -- sand cannot pull on a sole, so a negative
                normal load is a sign error in the emitting solver.
        """
        times = np.asarray(self.time_s, dtype=float).reshape(-1)
        if times.size < 2:
            raise ValueError(
                f"a sole load trace needs at least 2 samples, got {times.size}"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        centroids = np.asarray(self.element_centroid_body_m, dtype=float)
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                f"element_centroid_body_m must have shape (E, 3), got {centroids.shape}"
            )
        count = centroids.shape[0]
        if count == 0:
            raise ValueError("a sole load trace needs at least one element")
        areas = np.asarray(self.element_area_m2, dtype=float).reshape(-1)
        if areas.shape != (count,):
            raise ValueError(
                f"element_area_m2 must have shape {(count,)}, got {areas.shape}"
            )
        forces = np.asarray(self.element_normal_force_N, dtype=float)
        if forces.shape != (times.size, count):
            raise ValueError(
                "element_normal_force_N must have shape "
                f"{(times.size, count)}, got {forces.shape}"
            )
        for name, array in (
            ("time_s", times),
            ("element_centroid_body_m", centroids),
            ("element_area_m2", areas),
            ("element_normal_force_N", forces),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite; found NaN or inf")
        if np.any(areas <= 0.0):
            raise ValueError("element_area_m2 must be strictly positive")
        if np.any(forces < 0.0):
            raise ValueError(
                "element_normal_force_N must be non-negative: a sole carries "
                "compression, so a negative element load is a sign error"
            )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "element_centroid_body_m", centroids)
        object.__setattr__(self, "element_area_m2", areas)
        object.__setattr__(self, "element_normal_force_N", forces)

    @property
    def n_elements(self) -> int:
        """Number of sole elements."""
        return int(self.element_area_m2.size)

    @property
    def total_area_m2(self) -> float:
        """Total sole area carried in the trace [m^2]."""
        return float(self.element_area_m2.sum())


@dataclass(frozen=True)
class LoadProfile:
    """Impulse binned along one body axis -- the one-dimensional grind chart.

    Attributes:
        axis_index: Body axis the profile runs along (0, 1 or 2).
        bin_edges_m: ``(n_bins + 1,)`` bin edges in body coordinates [m].
        impulse_Ns: ``(n_bins,)`` summed element impulse in each bin.
        area_m2: ``(n_bins,)`` summed element area in each bin.
    """

    axis_index: int
    bin_edges_m: np.ndarray
    impulse_Ns: np.ndarray
    area_m2: np.ndarray

    @property
    def impulse_fraction(self) -> np.ndarray:
        """Each bin's share of the total impulse; sums to one."""
        total = float(self.impulse_Ns.sum())
        if total == 0.0:
            return np.zeros_like(self.impulse_Ns)
        return self.impulse_Ns / total


@dataclass(frozen=True)
class BounceUtilisation:
    """Which parts of the sole carried the strike.

    Attributes:
        element_impulse_Ns: ``(E,)`` per-element impulse.
        element_peak_force_N: ``(E,)`` per-element peak load.
        element_impulse_density_Pa_s: ``(E,)`` impulse per unit area.
        loaded_mask: ``(E,)`` elements at or above the load threshold.
        total_area_m2: Sole area supplied.
        utilised_area_m2: Area of the loaded elements.
        utilisation_fraction: ``utilised_area_m2 / total_area_m2``.
        removable_area_m2: Unloaded area -- material removable for free.
        total_impulse_Ns: Summed element impulse.
        centre_of_pressure_body_m: ``(3,)`` impulse-weighted centroid, body axes.
        threshold_fraction: Threshold the mask was built with.
    """

    element_impulse_Ns: np.ndarray
    element_peak_force_N: np.ndarray
    element_impulse_density_Pa_s: np.ndarray
    loaded_mask: np.ndarray
    total_area_m2: float
    utilised_area_m2: float
    utilisation_fraction: float
    removable_area_m2: float
    total_impulse_Ns: float
    centre_of_pressure_body_m: np.ndarray
    threshold_fraction: float

    def profile(
        self, load: SoleLoadTrace, *, axis_index: int, n_bins: int
    ) -> LoadProfile:
        """Bin the element impulse along one body axis.

        Heel-to-toe and leading-edge-to-trailing-edge are the two a designer
        asks for; which body index those are is a property of the head frame the
        elements were emitted in, so the axis is named by index here.

        Args:
            load: The load trace these metrics were computed from.
            axis_index: Body axis to bin along (0, 1 or 2).
            n_bins: Number of equal-width bins.

        Returns:
            The binned profile.

        Raises:
            ValueError: If the axis index or bin count is invalid, or the trace
                does not match the element count these metrics were built from.
        """
        if axis_index not in (0, 1, 2):
            raise ValueError(f"axis_index must be 0, 1 or 2, got {axis_index}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be at least 1, got {n_bins}")
        if load.n_elements != self.element_impulse_Ns.size:
            raise ValueError(
                "the load trace has "
                f"{load.n_elements} elements but these metrics describe "
                f"{self.element_impulse_Ns.size}"
            )
        stations = load.element_centroid_body_m[:, axis_index]
        low, high = float(stations.min()), float(stations.max())
        if high == low:
            high = low + 1.0  # degenerate extent: one bin covering everything
        edges = np.linspace(low, high, n_bins + 1)
        index = np.clip(np.digitize(stations, edges[1:-1]), 0, n_bins - 1)
        impulse = np.zeros(n_bins)
        area = np.zeros(n_bins)
        np.add.at(impulse, index, self.element_impulse_Ns)
        np.add.at(area, index, load.element_area_m2)
        return LoadProfile(
            axis_index=axis_index, bin_edges_m=edges, impulse_Ns=impulse, area_m2=area
        )


def bounce_utilisation(
    load: SoleLoadTrace,
    *,
    threshold_fraction: float = DEFAULT_LOAD_THRESHOLD_FRACTION,
) -> BounceUtilisation:
    """Resolve which sole elements carried the strike.

    Args:
        load: Per-element sole loading over the strike.
        threshold_fraction: Fraction of the peak impulse **density** at or above
            which an element counts as loaded. Must be in ``(0, 1]``.

    Returns:
        The utilisation map.

    Raises:
        ValueError: If the threshold is outside ``(0, 1]``, or the sole carried
            no load at all, in which case there is nothing to resolve and a
            utilisation fraction would be a division by zero dressed up as an
            answer.
    """
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError(
            f"threshold_fraction must be in (0, 1], got {threshold_fraction}"
        )
    impulse = np.trapezoid(load.element_normal_force_N, load.time_s, axis=0)
    total_impulse = float(impulse.sum())
    if total_impulse <= 0.0:
        raise ValueError(
            "the sole carried no load in this trace, so there is no bounce "
            "utilisation to report; check that the strike is inside the window"
        )
    density = impulse / load.element_area_m2
    loaded = density >= threshold_fraction * float(density.max())
    utilised_area_m2 = float(load.element_area_m2[loaded].sum())
    total_area_m2 = load.total_area_m2
    centre_of_pressure = (impulse @ load.element_centroid_body_m) / total_impulse
    ensure(
        0.0 <= utilised_area_m2 <= total_area_m2,
        "utilised sole area cannot exceed the sole area supplied",
        value=(utilised_area_m2, total_area_m2),
    )
    return BounceUtilisation(
        element_impulse_Ns=impulse,
        element_peak_force_N=load.element_normal_force_N.max(axis=0),
        element_impulse_density_Pa_s=density,
        loaded_mask=loaded,
        total_area_m2=total_area_m2,
        utilised_area_m2=utilised_area_m2,
        utilisation_fraction=utilised_area_m2 / total_area_m2,
        removable_area_m2=total_area_m2 - utilised_area_m2,
        total_impulse_Ns=total_impulse,
        centre_of_pressure_body_m=centre_of_pressure,
        threshold_fraction=float(threshold_fraction),
    )
