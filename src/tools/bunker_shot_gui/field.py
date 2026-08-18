"""The per-element sole load field, resolved in time (issues #8705, #8707).

The workbench already reports a :class:`~.bridge.SoleLoadMap`: the strike
binned onto a 12x12 sole grid. That map answers *where* the sole carried load
over the whole shot, which is the grinding question, and it is not replaced
here. What it cannot answer is *when*, or **which of the two DRFT terms** did
the carrying -- and both are already in the solver's per-element response,
collapsed to a scalar wrench and discarded.

This module is the headless half of raising that view. It computes; it draws
nothing. No Qt, no matplotlib, no display, in keeping with the split the
workbench established in issue #8618.

The two terms
-------------

3D-RFT builds the traction on an element from a depth-linear term and an
inertial (dynamic) term. They are physically different things -- one is a
quasi-static stress that grows with burial, the other is momentum flux that
grows with the square of the normal speed -- and a sole feature fights one or
the other. Carrying them separately is what lets a designer see which.

The two are *summed and then clamped*, never clamped and then summed:

    total = max(depth + inertial, 0)

which is exactly the load :func:`~.bridge.sole_load_trace` has always handed
to :func:`~bunkershot3d.metrics.bounce_utilisation`. Sand cannot pull on a
sole, but a single term can point outward on a steeply raked element while the
resultant is still compressive, so the individual terms are carried **signed**
and the clamp is applied once, to their sum. Splitting the load must not
change the number the existing metric is built from.

The scale
---------

A field animated across a shot, or compared across two grinds, is worthless if
each frame is normalised to its own maximum: the eye reads relative colour, so
per-frame scaling makes every frame look like peak load and makes two designs
look identical. :class:`LoadScale` is therefore fixed over the whole shot and
merges across designs. Each *component* keeps its own scale, because at
greenside delivery the inertial term is orders of magnitude larger than the
depth term and one shared ramp would paint the depth field uniformly blank --
so the peak of each scale is stated, in Pa, wherever the field is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.metrics import SoleLoadTrace
from bunkershot3d.solvers import EnvelopeStatus, FidelityTier, ValidityVerdict

__all__ = [
    "ContactPatch",
    "LoadComponent",
    "LoadScale",
    "SoleLoadField",
    "contact_patch",
]


class LoadComponent(str, Enum):
    """Which part of the DRFT traction a field is showing.

    Attributes:
        DEPTH: The depth-linear term alone.
        INERTIAL: The dynamic term alone.
        TOTAL: The compressive resultant of the two, which is the load the
            bounce-utilisation map is built from.
    """

    DEPTH = "depth"
    INERTIAL = "inertial"
    TOTAL = "total"

    @property
    def label(self) -> str:
        """A short heading for this component."""
        return _COMPONENT_LABEL[self]

    @property
    def description(self) -> str:
        """One line saying what the component is, for a legend."""
        return _COMPONENT_DESCRIPTION[self]


_COMPONENT_LABEL: dict[LoadComponent, str] = {
    LoadComponent.DEPTH: "Depth-dependent term",
    LoadComponent.INERTIAL: "Inertial term",
    LoadComponent.TOTAL: "Total compressive load",
}

_COMPONENT_DESCRIPTION: dict[LoadComponent, str] = {
    LoadComponent.DEPTH: (
        "quasi-static stress, grows with how deeply the element is buried"
    ),
    LoadComponent.INERTIAL: (
        "momentum flux, grows with the square of the element's normal speed"
    ),
    LoadComponent.TOTAL: "max(depth + inertial, 0); sand pushes, it cannot pull",
}

PRESSURE_UNIT = "Pa"
"""The unit every load field is reported in. Stated, never assumed."""


def _as_component(component: LoadComponent | str) -> LoadComponent:
    """Coerce a component name, naming the valid ones when it is not.

    Args:
        component: The requested component.

    Returns:
        The enum member.

    Raises:
        ValueError: If the name is not a load component. A ``raise`` rather
            than a contract decorator: a mistyped component would otherwise
            silently paint the wrong field under ``python -O``.
    """
    try:
        return LoadComponent(component)
    except ValueError as error:
        valid = ", ".join(item.value for item in LoadComponent)
        raise ValueError(
            f"unknown load component {component!r}; valid: {valid}"
        ) from error


@dataclass(frozen=True)
class LoadScale:
    """A fixed pressure scale, shared across frames and across designs.

    Attributes:
        component: Which component the scale belongs to; two scales only
            merge when they describe the same one.
        floor_pa: Bottom of the scale [Pa]; never above zero, so an unloaded
            element always sits at the pale end of the ramp.
        peak_pa: Top of the scale [Pa]; never below zero.
    """

    component: LoadComponent
    floor_pa: float
    peak_pa: float

    def __post_init__(self) -> None:
        """Validate the scale.

        Raises:
            ValueError: If a bound is not finite, or the bounds do not bracket
                zero. A scale that excludes zero would colour "no load" as
                though it were load.
        """
        for name, value in (("floor_pa", self.floor_pa), ("peak_pa", self.peak_pa)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.floor_pa > 0.0 or self.peak_pa < 0.0:
            raise ValueError(
                "a load scale must bracket zero so an unloaded element reads as "
                f"unloaded; got floor {self.floor_pa} Pa, peak {self.peak_pa} Pa"
            )
        object.__setattr__(self, "component", _as_component(self.component))
        object.__setattr__(self, "floor_pa", float(self.floor_pa))
        object.__setattr__(self, "peak_pa", float(self.peak_pa))

    @property
    def unit(self) -> str:
        """The unit the scale is expressed in."""
        return PRESSURE_UNIT

    @property
    def diverging(self) -> bool:
        """Whether the scale has to show outward-pointing traction as well."""
        return self.floor_pa < 0.0

    @property
    def limits_pa(self) -> tuple[float, float]:
        """``(low, high)`` colour limits [Pa], symmetric when diverging."""
        if self.diverging:
            extent = max(abs(self.floor_pa), self.peak_pa)
            return (-extent, extent)
        return (0.0, self.peak_pa)

    @property
    def colormap_name(self) -> str:
        """A matplotlib colormap suited to this scale's sign range."""
        return "RdBu_r" if self.diverging else "YlOrBr"

    def normalise(self, values: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Map pressures onto ``[0, 1]`` against the fixed limits.

        Args:
            values: Pressure(s) [Pa].

        Returns:
            The same shape, clipped into ``[0, 1]``. A degenerate scale (an
            all-zero field) maps everything to zero rather than dividing by a
            zero span.
        """
        low, high = self.limits_pa
        span = high - low
        array = np.asarray(values, dtype=np.float64)
        if span <= 0.0:
            return np.zeros_like(array)
        return np.clip((array - low) / span, 0.0, 1.0)

    def merged(self, other: LoadScale) -> LoadScale:
        """Return the scale covering both this one and ``other``.

        Args:
            other: The scale to merge with.

        Returns:
            The covering scale.

        Raises:
            ValueError: If the two describe different components. Merging a
                depth scale with an inertial one would put two quantities
                three orders of magnitude apart on one ramp.
        """
        if other.component is not self.component:
            raise ValueError(
                "two load scales merge only when they describe the same "
                f"component; got {self.component.value} and {other.component.value}"
            )
        return LoadScale(
            component=self.component,
            floor_pa=min(self.floor_pa, other.floor_pa),
            peak_pa=max(self.peak_pa, other.peak_pa),
        )

    @classmethod
    def from_values(
        cls, component: LoadComponent | str, values: NDArray[np.float64]
    ) -> LoadScale:
        """Build the scale covering an array of pressures.

        Args:
            component: Which component the values belong to.
            values: Pressures [Pa].

        Returns:
            The scale, always bracketing zero.
        """
        array = np.asarray(values, dtype=np.float64)
        low = float(array.min()) if array.size else 0.0
        high = float(array.max()) if array.size else 0.0
        return cls(
            component=_as_component(component),
            floor_pa=min(0.0, low),
            peak_pa=max(0.0, high),
        )

    @classmethod
    def covering(
        cls, component: LoadComponent | str, fields: tuple[SoleLoadField, ...]
    ) -> LoadScale:
        """Build the one scale two or more designs are compared on.

        Args:
            component: Which component to scale.
            fields: The load fields to cover.

        Returns:
            The covering scale.

        Raises:
            ValueError: If no field was supplied; there is nothing to scale.
        """
        chosen = _as_component(component)
        scales = [field.scale(chosen) for field in fields]
        if not scales:
            raise ValueError(
                "a shared load scale needs at least one field to cover; "
                "comparing designs on separate scales is what this prevents"
            )
        merged = scales[0]
        for scale in scales[1:]:
            merged = merged.merged(scale)
        return merged


@dataclass(frozen=True)
class SoleLoadField:
    """Per-element sole load over a strike, with the two terms separated.

    Attributes:
        time_s: ``(T,)`` strictly increasing sample times [s].
        element_centroid_body_m: ``(E, 3)`` sole element centroids in body
            axes [m]; ``x`` runs leading edge to trailing edge, ``y`` heel to
            toe.
        element_area_m2: ``(E,)`` element areas [m^2]; strictly positive.
        depth_normal_force_N: ``(T, E)`` inward normal load from the
            depth-linear term [N]; signed, see the module docstring.
        inertial_normal_force_N: ``(T, E)`` the same for the dynamic term.
        verdict: The validity statement the whole field must be read under.
        fidelity_tier: Which rung of the ADR-0032 ladder produced it.
    """

    time_s: NDArray[np.float64]
    element_centroid_body_m: NDArray[np.float64]
    element_area_m2: NDArray[np.float64]
    depth_normal_force_N: NDArray[np.float64]
    inertial_normal_force_N: NDArray[np.float64]
    verdict: ValidityVerdict
    fidelity_tier: FidelityTier

    def __post_init__(self) -> None:
        """Validate the field.

        Raises:
            ValueError: If shapes disagree, a value is not finite, time is not
                strictly increasing, or an area is not positive. These are
                ``raise`` rather than ``assert`` because ``python -O`` strips
                asserts and a malformed field would then be drawn rather than
                rejected.
        """
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        if times.size < 2:
            raise ValueError(
                f"a sole load field needs at least 2 samples, got {times.size}"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        centroids = np.asarray(self.element_centroid_body_m, dtype=np.float64)
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                f"element_centroid_body_m must have shape (E, 3), got {centroids.shape}"
            )
        count = centroids.shape[0]
        if count == 0:
            raise ValueError("a sole load field needs at least one element")
        areas = np.asarray(self.element_area_m2, dtype=np.float64).reshape(-1)
        if areas.shape != (count,):
            raise ValueError(
                f"element_area_m2 must have shape {(count,)}, got {areas.shape}"
            )
        expected = (times.size, count)
        blocks = {
            "depth_normal_force_N": np.asarray(
                self.depth_normal_force_N, dtype=np.float64
            ),
            "inertial_normal_force_N": np.asarray(
                self.inertial_normal_force_N, dtype=np.float64
            ),
        }
        for name, block in blocks.items():
            if block.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {block.shape}"
                )
        for name, array in (
            ("time_s", times),
            ("element_centroid_body_m", centroids),
            ("element_area_m2", areas),
            *blocks.items(),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite; found NaN or inf")
        if np.any(areas <= 0.0):
            raise ValueError("element_area_m2 must be strictly positive")
        if not isinstance(self.verdict, ValidityVerdict):
            raise ValueError(
                "a sole load field travels with the verdict it must be read "
                "under; a field drawn without its validity statement reads as "
                "though it had been measured"
            )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "element_centroid_body_m", centroids)
        object.__setattr__(self, "element_area_m2", areas)
        object.__setattr__(self, "depth_normal_force_N", blocks["depth_normal_force_N"])
        object.__setattr__(
            self, "inertial_normal_force_N", blocks["inertial_normal_force_N"]
        )

    # ------------------------------------------------------------- geometry

    @property
    def n_frames(self) -> int:
        """Number of samples in the field."""
        return int(self.time_s.size)

    @property
    def n_elements(self) -> int:
        """Number of sole elements resolved."""
        return int(self.element_area_m2.size)

    @property
    def total_area_m2(self) -> float:
        """Total sole area the field covers [m^2]."""
        return float(self.element_area_m2.sum())

    @property
    def leading_edge_body_m(self) -> float:
        """Body ``x`` of the leading-most sole element [m]."""
        return float(self.element_centroid_body_m[:, 0].min())

    @property
    def trailing_edge_body_m(self) -> float:
        """Body ``x`` of the trailing-most sole element [m]."""
        return float(self.element_centroid_body_m[:, 0].max())

    @property
    def status(self) -> EnvelopeStatus:
        """How much of this field may be believed."""
        return self.verdict.status

    # ------------------------------------------------------------ the field

    def component_force_N(  # noqa: N802 - the unit belongs in the name
        self, component: LoadComponent | str
    ) -> NDArray[np.float64]:
        """Return one component's per-element load [N].

        Args:
            component: Which term to return.

        Returns:
            ``(T, E)`` normal load. ``TOTAL`` is the clamped sum, which is
            exactly what the bounce-utilisation metric consumes.

        Raises:
            ValueError: If the component is not one of the three.
        """
        chosen = _as_component(component)
        if chosen is LoadComponent.DEPTH:
            return self.depth_normal_force_N
        if chosen is LoadComponent.INERTIAL:
            return self.inertial_normal_force_N
        return np.maximum(self.depth_normal_force_N + self.inertial_normal_force_N, 0.0)

    def component_pressure_pa(
        self, component: LoadComponent | str
    ) -> NDArray[np.float64]:
        """Return one component's per-element pressure [Pa].

        Per unit area, so a large element cannot look important merely by
        being large -- the same normalisation the bounce map uses.

        Args:
            component: Which term to return.

        Returns:
            ``(T, E)`` pressure [Pa].

        Raises:
            ValueError: If the component is not one of the three.
        """
        return self.component_force_N(component) / self.element_area_m2

    def resultant_force_N(  # noqa: N802 - the unit belongs in the name
        self, component: LoadComponent | str
    ) -> NDArray[np.float64]:
        """Return one component's resultant over the sole, per sample [N].

        Args:
            component: Which term to return.

        Returns:
            ``(T,)`` summed normal load.

        Raises:
            ValueError: If the component is not one of the three.
        """
        return self.component_force_N(component).sum(axis=1)

    def peak_resultant_force_N(  # noqa: N802 - the unit belongs in the name
        self, component: LoadComponent | str
    ) -> float:
        """Largest resultant this component reached [N].

        Args:
            component: Which term to return.

        Returns:
            The peak.

        Raises:
            ValueError: If the component is not one of the three.
        """
        return float(self.resultant_force_N(component).max())

    def peak_time_s(self, component: LoadComponent | str) -> float:
        """When this component's resultant peaked [s].

        The two terms need not peak together -- the depth term grows with
        burial while the inertial term follows the square of the normal speed
        -- and which one a sole feature fights is exactly the question #8705
        was filed to answer. When they do coincide, that is a finding too, so
        the moment is reported rather than a difference asserted.

        Args:
            component: Which term to return.

        Returns:
            The sample time of the peak.

        Raises:
            ValueError: If the component is not one of the three.
        """
        return float(self.time_s[int(self.resultant_force_N(component).argmax())])

    @property
    def peak_inertial_share(self) -> float:
        """The dynamic term's share of the load at the peak of the strike.

        Distinct from ``ShotOutcome.peak_inertial_fraction``, which the solver
        computes over the whole body: this is measured on the sole elements
        alone, at the sample where the sole's own compressive resultant is
        largest.

        Returns:
            The share, in ``[0, 1]``, or 0.0 when the sole carried nothing.
        """
        total = self.resultant_force_N(LoadComponent.TOTAL)
        frame = int(total.argmax())
        depth = float(self.resultant_force_N(LoadComponent.DEPTH)[frame])
        inertial = float(self.resultant_force_N(LoadComponent.INERTIAL)[frame])
        combined = depth + inertial
        if combined <= 0.0:
            return 0.0
        return float(np.clip(inertial / combined, 0.0, 1.0))

    @property
    def engaged_mask(self) -> NDArray[np.bool_]:
        """``(T, E)`` elements carrying compressive load at each sample."""
        return self.component_force_N(LoadComponent.TOTAL) > 0.0

    def scale(self, component: LoadComponent | str) -> LoadScale:
        """Return the fixed colour scale for one component of this field.

        Args:
            component: Which term to scale.

        Returns:
            The scale, covering every frame.

        Raises:
            ValueError: If the component is not one of the three.
        """
        chosen = _as_component(component)
        return LoadScale.from_values(chosen, self.component_pressure_pa(chosen))

    def load_trace(self) -> SoleLoadTrace:
        """Return the field in the shape the W7 metrics already consume.

        Returns:
            The per-element sole loading, non-negative by construction, which
            is what :func:`~bunkershot3d.metrics.bounce_utilisation` and the
            binned :class:`~.bridge.SoleLoadMap` are built from.
        """
        return SoleLoadTrace(
            time_s=self.time_s,
            element_centroid_body_m=self.element_centroid_body_m,
            element_area_m2=self.element_area_m2,
            element_normal_force_N=self.component_force_N(LoadComponent.TOTAL),
        )


@dataclass(frozen=True)
class ContactPatch:
    """The engaged element set through the shot, and where it sits (#8707).

    The initial patch alone was enough to spot the mechanism behind the
    counterintuitive bounce result -- more bounce shrank the patch, which
    raised the pressure and so drove the head *deeper* rather than skidding.
    Following it through the shot is what explains it.

    Attributes:
        time_s: ``(T,)`` sample times [s].
        engaged: ``(T, E)`` elements carrying compressive load.
        element_centroid_body_m: ``(E, 3)`` element centroids, body axes [m].
        element_area_m2: ``(E,)`` element areas [m^2].
    """

    time_s: NDArray[np.float64]
    engaged: NDArray[np.bool_]
    element_centroid_body_m: NDArray[np.float64]
    element_area_m2: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the patch series.

        Raises:
            ValueError: If the shapes disagree, or no element ever engaged --
                a patch that never formed is not a patch, and every derived
                number would be a division by zero dressed as an answer.
        """
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        engaged = np.asarray(self.engaged, dtype=bool)
        centroids = np.asarray(self.element_centroid_body_m, dtype=np.float64)
        areas = np.asarray(self.element_area_m2, dtype=np.float64).reshape(-1)
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                f"element_centroid_body_m must have shape (E, 3), got {centroids.shape}"
            )
        expected = (times.size, centroids.shape[0])
        if engaged.shape != expected:
            raise ValueError(f"engaged must have shape {expected}, got {engaged.shape}")
        if areas.shape != (centroids.shape[0],):
            raise ValueError(
                f"element_area_m2 must have shape {(centroids.shape[0],)}, "
                f"got {areas.shape}"
            )
        if not engaged.any():
            raise ValueError(
                "no sole element carried load at any sample, so there is no "
                "contact patch to follow; check that the strike is inside the "
                "recorded window"
            )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "engaged", engaged)
        object.__setattr__(self, "element_centroid_body_m", centroids)
        object.__setattr__(self, "element_area_m2", areas)

    @property
    def n_frames(self) -> int:
        """Number of samples."""
        return int(self.time_s.size)

    @property
    def n_elements(self) -> int:
        """Number of sole elements the patch is drawn from."""
        return int(self.element_area_m2.size)

    @property
    def leading_edge_m(self) -> float:
        """Body ``x`` of the leading-most sole element [m]."""
        return float(self.element_centroid_body_m[:, 0].min())

    @property
    def trailing_edge_m(self) -> float:
        """Body ``x`` of the trailing-most sole element [m]."""
        return float(self.element_centroid_body_m[:, 0].max())

    @property
    def area_m2(self) -> NDArray[np.float64]:
        """``(T,)`` engaged area at each sample [m^2]; zero before contact."""
        return np.asarray(self.engaged @ self.element_area_m2, dtype=np.float64)

    @property
    def centroid_body_m(self) -> NDArray[np.float64]:
        """``(T, 3)`` area-weighted patch centroid; NaN with no contact."""
        weights = self.engaged * self.element_area_m2
        totals = weights.sum(axis=1)
        centroid = np.full((self.n_frames, 3), np.nan, dtype=np.float64)
        live = totals > 0.0
        if live.any():
            centroid[live] = (weights[live] @ self.element_centroid_body_m) / totals[
                live, None
            ]
        return centroid

    @property
    def reach_m(self) -> NDArray[np.float64]:
        """``(T,)`` gap from the leading edge to the nearest engaged element.

        The practical dig question is how quickly load reaches the leading
        edge, so this is measured to the *closest* engaged element rather than
        to the patch centroid. NaN where nothing is engaged: a frame with no
        contact has no distance to report, and zero would read as contact
        exactly on the edge.
        """
        stations = self.element_centroid_body_m[:, 0]
        gaps = np.full(self.n_frames, np.nan, dtype=np.float64)
        for frame in range(self.n_frames):
            mask = self.engaged[frame]
            if mask.any():
                gaps[frame] = float(stations[mask].min()) - self.leading_edge_m
        return gaps

    @property
    def centroid_offset_m(self) -> NDArray[np.float64]:
        """``(T,)`` gap from the leading edge to the patch centroid [m]."""
        return self.centroid_body_m[:, 0] - self.leading_edge_m

    @property
    def engaged_frames(self) -> NDArray[np.bool_]:
        """``(T,)`` samples at which anything was engaged."""
        return np.asarray(self.engaged.any(axis=1), dtype=bool)

    @property
    def initial_frame(self) -> int:
        """Index of the first sample carrying load."""
        return int(np.argmax(self.engaged_frames))

    @property
    def initial_area_m2(self) -> float:
        """The patch at first contact [m^2] -- the number #8707 starts from."""
        return float(self.area_m2[self.initial_frame])

    @property
    def initial_time_s(self) -> float:
        """When the sole first carried load [s]."""
        return float(self.time_s[self.initial_frame])

    @property
    def peak_area_m2(self) -> float:
        """The largest the patch ever got [m^2]."""
        return float(self.area_m2.max())

    @property
    def peak_area_time_s(self) -> float:
        """When the patch was largest [s]."""
        return float(self.time_s[int(self.area_m2.argmax())])

    @property
    def closest_approach_m(self) -> float:
        """The nearest the load ever got to the leading edge [m]."""
        return float(np.nanmin(self.reach_m))

    @property
    def time_of_closest_approach_s(self) -> float:
        """When the load was nearest the leading edge [s]."""
        return float(self.time_s[int(np.nanargmin(self.reach_m))])


def contact_patch(field: SoleLoadField) -> ContactPatch:
    """Follow the engaged element set through a shot.

    Args:
        field: The per-element load field.

    Returns:
        The patch series.

    Raises:
        ValueError: If the sole never carried load.
    """
    return ContactPatch(
        time_s=field.time_s,
        engaged=field.engaged_mask,
        element_centroid_body_m=field.element_centroid_body_m,
        element_area_m2=field.element_area_m2,
    )
