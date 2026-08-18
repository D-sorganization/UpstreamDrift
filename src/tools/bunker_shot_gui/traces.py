"""The scalar traces and the validity band (issue #8708, epic #8699).

The 3-D scene in :mod:`~.shot3d` localises: it says *where* on the sole, and
*where* in the sand. This module quantifies: it says *how much*, and *when*.
Separately each is half an answer, which is why #8708 puts them on one time
cursor -- a designer seeing a force peak at 6.2 ms wants to know instantly
which part of the sole was loaded then.

Headless. It computes; it draws nothing.

Units live on the data
----------------------

Every :class:`ScalarTrace` carries the unit its own values are expressed in,
and the values are stored **already in that unit**. Nothing downstream
rescales. The alternative -- SI on the object and a scale factor applied by
whatever plots it -- puts the conversion in the renderer, where a second
renderer gets it wrong and nobody notices because both pictures look
plausible.

The band, not the badge
-----------------------

:class:`ValidityBand` is the point of this module. A panel stamped
``BEYOND_VALIDATION`` in one corner invites the reader to apply that evenly
across the whole record, and it is not even: a shot sits inside 3D-RFT's
stated envelope during the free-flight lead-in, where the sole is above the
sand and nothing is loaded, and leaves it the moment the sole engages at
speed. That transition is exactly the moment the numbers stop meaning what
they appear to mean, so it is drawn as a *band over time*.

The band is a reconstruction. ``simulate_shot`` evaluates the envelope at
every step and then keeps only ``worst_of`` those verdicts, so the per-sample
statuses exist in the solver and are discarded before the workbench sees
them. :func:`~.bridge.validity_band` recovers them by asking the same solver
the same question through its public
:meth:`~bunkershot3d.solvers.drft.DRFTSolver.envelope`, which judges a state
without integrating any force. The one thing that method cannot know is the
share of active area whose orientation had to be clamped -- and that quantity
attaches a caveat, never a status -- so the reconstructed statuses are the
statuses the march recorded. ``test_shot_traces`` pins that against
``ShotResult.verdict`` rather than trusting this paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.solvers import EnvelopeStatus, FidelityTier, ShotResult

from .field import ContactPatch

__all__ = [
    "TIME_UNIT",
    "ScalarTrace",
    "ShotTraces",
    "TraceGroup",
    "ValidityBand",
    "ValiditySpan",
    "shot_traces",
]

TIME_UNIT = "ms"
"""The unit the shared time cursor is expressed in. Stated, never assumed."""


class TraceGroup(str, Enum):
    """Which panel a trace belongs on.

    A group *is* a panel, and a panel has one y-axis, so a group has one
    unit -- millimetres of depth and square centimetres of patch area do not
    share an axis however related the two quantities are. Grouping by unit
    rather than by topic is also what keeps six wrench components off one
    axis in newtons. Members are declared in the order the panels stack.

    Attributes:
        SAND_FORCE: The sand force on the head, by world component [N].
        SAND_TORQUE: The sand torque about the body origin [N.m].
        SOLE_DEPTH: How deep the sole reference point is [mm].
        CONTACT_PATCH: How much sole is carrying load [cm^2].
        SPEED_LOSS: What the strike costs the head [m/s].
    """

    SAND_FORCE = "sand_force"
    SAND_TORQUE = "sand_torque"
    SOLE_DEPTH = "sole_depth"
    CONTACT_PATCH = "contact_patch"
    SPEED_LOSS = "speed_loss"

    @classmethod
    def _missing_(cls, value: object) -> TraceGroup:
        """Name the valid groups when coercion fails.

        Args:
            value: Whatever was offered.

        Returns:
            Never; this always raises.

        Raises:
            ValueError: Always.
        """
        valid = ", ".join(item.value for item in cls)
        raise ValueError(f"unknown trace group {value!r}; valid: {valid}")

    @property
    def label(self) -> str:
        """The heading painted above this group's panel."""
        return _GROUP_LABEL[self]


_GROUP_LABEL: dict[TraceGroup, str] = {
    TraceGroup.SAND_FORCE: "Sand force on the head",
    TraceGroup.SAND_TORQUE: "Sand torque about the body origin",
    TraceGroup.SOLE_DEPTH: "Sole depth below the free surface",
    TraceGroup.CONTACT_PATCH: "Sole area carrying load",
    TraceGroup.SPEED_LOSS: "Speed lost to the sand",
}

_STATUS_LABEL: dict[EnvelopeStatus, str] = {
    EnvelopeStatus.WITHIN: "within 3D-RFT's stated limits",
    EnvelopeStatus.EXTRAPOLATED: "extrapolated past the stated limits",
    EnvelopeStatus.BEYOND_VALIDATION: "beyond every published measurement",
    EnvelopeStatus.REFUSED: "refused: no number is reported here",
}

_STATUS_RANK: dict[EnvelopeStatus, int] = {
    EnvelopeStatus.WITHIN: 0,
    EnvelopeStatus.EXTRAPOLATED: 1,
    EnvelopeStatus.BEYOND_VALIDATION: 2,
    EnvelopeStatus.REFUSED: 3,
}
"""The solver's own ordering, mirrored so a band can name its own worst."""


@dataclass(frozen=True)
class ScalarTrace:
    """One scalar quantity over the shot, in a stated unit.

    Attributes:
        name: How the trace is referred to, and its legend entry. Unique
            within a :class:`ShotTraces`.
        unit: The unit the values are in. Never empty.
        values: ``(T,)`` the quantity, already expressed in ``unit``.
        group: Which panel it belongs on.
        description: Optional line saying what the quantity is.
    """

    name: str
    unit: str
    values: NDArray[np.float64]
    group: TraceGroup
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the trace.

        Raises:
            ValueError: If the name or unit is empty, or a value is not
                finite. A ``raise`` rather than an ``assert``: an unlabelled
                axis is the failure the demo report's unit standard exists to
                prevent, and ``python -O`` must not reintroduce it.
        """
        name = str(self.name).strip()
        unit = str(self.unit).strip()
        if not name:
            raise ValueError("a scalar trace needs a name")
        if not unit:
            raise ValueError(
                f"trace {name!r} needs a unit; an unlabelled axis is a number "
                "the reader has to guess the meaning of"
            )
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"trace {name!r} must be finite; found NaN or inf")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "group", TraceGroup(self.group))
        object.__setattr__(self, "description", str(self.description))

    @property
    def n_frames(self) -> int:
        """Number of samples in the trace."""
        return int(self.values.size)

    @property
    def axis_label(self) -> str:
        """The axis label, carrying the unit in brackets."""
        return f"{self.name} [{self.unit}]"

    @property
    def peak(self) -> float:
        """The largest absolute value the trace reached, in ``unit``."""
        return float(np.abs(self.values).max()) if self.values.size else 0.0

    def value_at(self, frame: int) -> float:
        """Return the value at one sample.

        Args:
            frame: The sample index.

        Returns:
            The value, in ``unit``.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        if not 0 <= int(frame) < self.n_frames:
            raise ValueError(
                f"frame {frame} is outside the recorded shot, which has "
                f"{self.n_frames} samples"
            )
        return float(self.values[int(frame)])


@dataclass(frozen=True)
class ValiditySpan:
    """One stretch of the record over which the verdict did not change.

    Attributes:
        start_s: When the span opens [s].
        end_s: When it closes [s].
        status: The verdict over it.
    """

    start_s: float
    end_s: float
    status: EnvelopeStatus

    @property
    def duration_s(self) -> float:
        """How long the span lasts [s]."""
        return self.end_s - self.start_s

    @property
    def label(self) -> str:
        """One line naming the span's verdict and when it applied."""
        status = self.status
        return (
            f"{status.value.replace('_', ' ').upper()} "
            f"({self.start_s * 1e3:.2f}-{self.end_s * 1e3:.2f} ms): "
            f"{_STATUS_LABEL[status]}"
        )


@dataclass(frozen=True)
class ValidityBand:
    """The envelope verdict at every sample of a shot.

    Attributes:
        time_s: ``(T,)`` strictly increasing sample times [s].
        statuses: One :class:`~bunkershot3d.solvers.EnvelopeStatus` per
            sample, in the same order.
    """

    time_s: NDArray[np.float64]
    statuses: tuple[EnvelopeStatus, ...]

    def __post_init__(self) -> None:
        """Validate the band.

        Raises:
            ValueError: If time is not strictly increasing, the counts
                disagree, or an entry is not an ``EnvelopeStatus``. A band
                one entry short of its time axis would shift every verdict
                by a sample and still draw perfectly plausibly.
        """
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        if times.size < 2:
            raise ValueError(
                f"a validity band needs at least 2 samples, got {times.size}"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        statuses = tuple(self.statuses)
        offenders = [item for item in statuses if not isinstance(item, EnvelopeStatus)]
        if offenders:
            raise ValueError(
                f"a validity band holds EnvelopeStatus values, got {offenders!r}"
            )
        if len(statuses) != times.size:
            raise ValueError(
                "a validity band carries one status per sample; got "
                f"{len(statuses)} for {times.size} samples"
            )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "statuses", statuses)

    @property
    def n_frames(self) -> int:
        """Number of samples the band covers."""
        return int(self.time_s.size)

    @property
    def worst(self) -> EnvelopeStatus:
        """The worst verdict anywhere in the record.

        The same reduction :func:`~bunkershot3d.solvers.worst_of` applies to
        the march's own per-step verdicts, so this is the status the whole
        shot is reported under.
        """
        return max(self.statuses, key=lambda status: _STATUS_RANK[status])

    @property
    def changes(self) -> bool:
        """Whether the verdict is not the same over the whole record.

        The question a badge cannot answer, and the reason #8708 asks for a
        band: ``True`` means the shot left, or entered, a different regime
        partway through.
        """
        return len(set(self.statuses)) > 1

    def status_at(self, frame: int) -> EnvelopeStatus:
        """Return the verdict at one sample.

        Args:
            frame: The sample index.

        Returns:
            The verdict the numbers at that moment must be read under.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        if not 0 <= int(frame) < self.n_frames:
            raise ValueError(
                f"frame {frame} is outside the recorded shot, which has "
                f"{self.n_frames} samples"
            )
        return self.statuses[int(frame)]

    def spans(self) -> tuple[ValiditySpan, ...]:
        """Reduce the per-sample statuses to contiguous spans.

        Returns:
            The spans, in time order. They tile the whole record without
            gaps or overlaps, and no two neighbours share a status -- which
            is what makes them drawable as one shaded band per regime rather
            than one rectangle per sample.
        """
        spans: list[ValiditySpan] = []
        start = 0
        for index in range(1, self.n_frames + 1):
            ended = index == self.n_frames
            if ended or self.statuses[index] is not self.statuses[start]:
                spans.append(
                    ValiditySpan(
                        start_s=float(self.time_s[start]),
                        end_s=float(self.time_s[min(index, self.n_frames - 1)]),
                        status=self.statuses[start],
                    )
                )
                start = index
        return tuple(spans)


@dataclass(frozen=True)
class ShotTraces:
    """Every scalar trace of one shot, on one time axis, with its band.

    Attributes:
        time_s: ``(T,)`` strictly increasing sample times [s]. The same axis
            the 3-D scene and the sole load field are on, which is what makes
            one cursor scrub all three.
        traces: The scalar traces, names unique.
        band: The envelope verdict over the same record.
        fidelity_tier: Which rung of the ADR-0032 ladder produced them. A
            verdict is per sample and a tier is not, so the tier rides on the
            set while the statuses ride on the band -- and the panel that
            stamps both reads them from one object rather than guessing.
    """

    time_s: NDArray[np.float64]
    traces: tuple[ScalarTrace, ...]
    band: ValidityBand
    fidelity_tier: FidelityTier = FidelityTier.F0

    def __post_init__(self) -> None:
        """Validate the set.

        Raises:
            ValueError: If time is not strictly increasing, a trace does not
                have one value per sample, two traces share a name, or the
                band describes a different shot. A panel drawing mismatched
                arrays is worse than an empty one: it reads as a finding.
        """
        times = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        if times.size < 2:
            raise ValueError(f"a trace set needs at least 2 samples, got {times.size}")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        traces = tuple(self.traces)
        for trace in traces:
            if trace.n_frames != times.size:
                raise ValueError(
                    f"trace {trace.name!r} must carry one value per sample; got "
                    f"{trace.n_frames} for {times.size} samples"
                )
        names = [trace.name for trace in traces]
        if len(set(names)) != len(names):
            raise ValueError(f"trace names must be unique, got {sorted(names)}")
        if not isinstance(self.band, ValidityBand):
            raise ValueError(
                "a trace set travels with its validity band; traces drawn "
                "without one read as though they had been measured"
            )
        if self.band.n_frames != times.size:
            raise ValueError(
                "the traces and the validity band must come from the same shot; "
                f"got {times.size} samples against {self.band.n_frames} verdicts"
            )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "traces", traces)

    @property
    def n_frames(self) -> int:
        """Number of samples on the shared time axis."""
        return int(self.time_s.size)

    @property
    def time_unit(self) -> str:
        """The unit the displayed time axis is in."""
        return TIME_UNIT

    @property
    def time_display(self) -> NDArray[np.float64]:
        """``(T,)`` the time axis in :data:`TIME_UNIT`."""
        return self.time_s * 1e3

    @property
    def time_axis_label(self) -> str:
        """The shared x-axis label, carrying its unit."""
        return f"time from the start of the record [{self.time_unit}]"

    @property
    def names(self) -> tuple[str, ...]:
        """Every trace name, in declaration order."""
        return tuple(trace.name for trace in self.traces)

    def trace(self, name: str) -> ScalarTrace | None:
        """Return one trace by name, or ``None``.

        Args:
            name: The trace name.

        Returns:
            The trace, or ``None`` when this shot does not carry it.
        """
        for candidate in self.traces:
            if candidate.name == name:
                return candidate
        return None

    def require(self, name: str) -> ScalarTrace:
        """Return one trace by name, naming the alternatives when absent.

        Args:
            name: The trace name.

        Returns:
            The trace.

        Raises:
            KeyError: If this shot does not carry it.
        """
        found = self.trace(name)
        if found is None:
            raise KeyError(
                f"no trace named {name!r}; this shot carries: " + ", ".join(self.names)
            )
        return found

    def group(self, group: TraceGroup | str) -> tuple[ScalarTrace, ...]:
        """Return every trace on one panel.

        Args:
            group: The panel.

        Returns:
            The traces, in declaration order; empty when this shot has none.

        Raises:
            ValueError: If the group is not one of the declared ones.
        """
        chosen = TraceGroup(group)
        return tuple(trace for trace in self.traces if trace.group is chosen)

    def groups(self) -> tuple[TraceGroup, ...]:
        """Return the panels this shot actually fills, in stacking order.

        Returns:
            The non-empty groups, in :class:`TraceGroup` declaration order,
            so the panel layout is the same for every shot that carries the
            same quantities.
        """
        return tuple(group for group in TraceGroup if self.group(group))

    def values_at(self, frame: int) -> dict[str, float]:
        """Return every trace's value at one sample.

        Args:
            frame: The sample index.

        Returns:
            Trace name to value, each in that trace's own unit.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        return {trace.name: trace.value_at(frame) for trace in self.traces}


def shot_traces(
    result: ShotResult, patch: ContactPatch, band: ValidityBand
) -> ShotTraces:
    """Assemble the scalar traces #8708 names, from an already-run shot.

    No solving: the wrench, the depth and the speed are columns the march
    already recorded, the patch area comes from the same replay that built
    the sole load field, and the band is reconstructed by
    :func:`~.bridge.validity_band`.

    Args:
        result: The shot trace.
        patch: The contact-patch series over the same samples.
        band: The envelope verdict over the same samples.

    Returns:
        The trace set.

    Raises:
        ValueError: If the three do not describe one shot.
    """
    times = np.asarray(result.times_s, dtype=np.float64)
    if patch.n_frames != times.size:
        raise ValueError(
            "the contact patch and the shot must come from one record; got "
            f"{patch.n_frames} patch samples against {times.size} shot samples"
        )
    speed = np.linalg.norm(np.asarray(result.velocities_m_s, dtype=np.float64), axis=1)
    forces = np.asarray(result.forces_n, dtype=np.float64)
    torques = np.asarray(result.torques_n_m, dtype=np.float64)
    axes = ("x", "y", "z")
    traces = [
        *(
            ScalarTrace(
                name=f"sand force {axis}",
                unit="N",
                values=forces[:, index],
                group=TraceGroup.SAND_FORCE,
                description=f"world {axis} component of the sand force on the head",
            )
            for index, axis in enumerate(axes)
        ),
        *(
            ScalarTrace(
                name=f"sand torque {axis}",
                unit="N.m",
                values=torques[:, index],
                group=TraceGroup.SAND_TORQUE,
                description=(
                    f"world {axis} component of the sand torque about the body origin"
                ),
            )
            for index, axis in enumerate(axes)
        ),
        ScalarTrace(
            name="sole depth",
            unit="mm",
            # The geometric depth of the sole reference point, not the
            # engaged-element diagnostic that issue #8701 found reads zero
            # while the sole is still millimetres under the surface.
            values=np.asarray(result.sole_depths_m, dtype=np.float64) * 1e3,
            group=TraceGroup.SOLE_DEPTH,
            description="sole reference point below the free surface, positive down",
        ),
        ScalarTrace(
            name="contact patch area",
            unit="cm^2",
            values=patch.area_m2 * 1e4,
            group=TraceGroup.CONTACT_PATCH,
            description="sole area carrying compressive load",
        ),
        ScalarTrace(
            name="speed lost",
            unit="m/s",
            # Cumulative from the first sample rather than a per-step
            # difference: "where does the head lose speed" is a running
            # question, and a per-step delta at a 20 us step is noise.
            values=float(speed[0]) - speed,
            group=TraceGroup.SPEED_LOSS,
            description="head speed given up since the start of the record",
        ),
    ]
    return ShotTraces(
        time_s=times,
        traces=tuple(traces),
        band=band,
        fidelity_tier=result.fidelity_tier,
    )
