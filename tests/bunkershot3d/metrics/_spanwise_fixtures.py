"""Synthetic soles for the spanwise-load tests (issue #8699).

Not a ``conftest``: these builders belong to one test module, and the metrics
package's shared ``conftest`` is a busy file that several strands of work
prepend to. Kept private (leading underscore) so pytest does not collect it.

The reference sole is twelve spanwise stations across a 60 mm span, three
chordwise elements each. See ``test_spanwise`` for the arithmetic that layout
was chosen to make exact.
"""

from __future__ import annotations

import numpy as np

from bunkershot3d.metrics import SoleLoadTrace
from bunkershot3d.solvers import (
    EnvelopeStatus,
    ValidityVerdict,
    dimensionless_groups,
)

#: Spanwise stations across the reference sole.
N_SPAN_STATIONS = 12

#: Half-span of the reference sole [m], so the span is 60 mm.
SPAN_HALF_M = 0.030

#: Chordwise elements at each spanwise station.
N_CHORD_ELEMENTS = 3

#: Half-chord of the reference sole [m].
CHORD_HALF_M = 0.010

#: Area of every element of the reference sole [m^2].
ELEMENT_AREA_M2 = 1.0e-5


def build_span_load(
    *,
    station_force_N: np.ndarray,  # noqa: N803 - the unit belongs in the name
    station_m: np.ndarray | None = None,
    duration_s: float = 0.010,
) -> SoleLoadTrace:
    """Build a sole whose load is prescribed per spanwise station.

    Each station's load is shared equally across its chordwise elements, so a
    per-station force of ``F`` puts ``F / 3`` on each of three elements and the
    station total is exactly ``F``.

    Args:
        station_force_N: ``(S,)`` constant load per station, or ``(T, S)`` load
            per station per sample. A ``(S,)`` block is held over three
            samples, so the impulse is ``F * duration_s``.
        station_m: ``(S,)`` spanwise stations [m]; defaults to
            :data:`N_SPAN_STATIONS` evenly spaced across the reference span.
        duration_s: Window length [s].

    Returns:
        The load trace, with body ``x`` chordwise and body ``y`` spanwise.

    Raises:
        ValueError: If the force block is neither 1-D nor 2-D, or its station
            count disagrees with ``station_m``.
    """
    forces = np.asarray(station_force_N, dtype=float)
    if forces.ndim == 1:
        forces = np.tile(forces, (3, 1))
    if forces.ndim != 2:
        raise ValueError(f"station_force_N must be 1-D or 2-D, got {forces.shape}")
    n_stations = forces.shape[1]
    stations = (
        np.linspace(-SPAN_HALF_M, SPAN_HALF_M, n_stations)
        if station_m is None
        else np.asarray(station_m, dtype=float).reshape(-1)
    )
    if stations.size != n_stations:
        raise ValueError(
            f"station_m has {stations.size} stations but the force block has "
            f"{n_stations}"
        )
    chord = np.linspace(-CHORD_HALF_M, CHORD_HALF_M, N_CHORD_ELEMENTS)
    chord_grid, station_grid = np.meshgrid(chord, stations, indexing="xy")
    centroids = np.column_stack(
        [chord_grid.ravel(), station_grid.ravel(), np.zeros(station_grid.size)]
    )
    element_force = np.repeat(forces, N_CHORD_ELEMENTS, axis=1) / N_CHORD_ELEMENTS
    return SoleLoadTrace(
        time_s=np.linspace(0.0, duration_s, forces.shape[0]),
        element_centroid_body_m=centroids,
        element_area_m2=np.full(centroids.shape[0], ELEMENT_AREA_M2),
        element_normal_force_N=element_force,
    )


def _groups():
    """Return one feature scale's dimensionless groups, for a verdict."""
    return dimensionless_groups(
        speed_m_s=25.0,
        feature_length_m=0.020,
        grain_diameter_m=0.0005,
        element_size_m=0.002,
        name="sole",
    )


def within_verdict() -> ValidityVerdict:
    """A verdict inside the stated envelope."""
    return ValidityVerdict(status=EnvelopeStatus.WITHIN, groups=(_groups(),))


def refused_verdict() -> ValidityVerdict:
    """A verdict on which no number may be reported."""
    return ValidityVerdict(
        status=EnvelopeStatus.REFUSED,
        groups=(_groups(),),
        reasons=("synthetic refusal for the spanwise tests",),
    )
