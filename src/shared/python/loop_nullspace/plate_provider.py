"""Measured vertical-plate trials as per-frame ground observations (OF-12, #13070).

:class:`PlateObservationProvider` is an ``ObservationProvider`` for the
null-space facade (``EngineContext.observation_for``) and for
:func:`~src.shared.python.loop_nullspace.plate_screen.plate_consistency`.
For a model frame it reads the plate trial at that frame's time (OF-05
:func:`~src.shared.python.force_plates.reading_at`, with its interpolation
and skew uncertainty) and builds the per-foot vertical-cell observation
(OF-11 :meth:`GroundObservation.from_vertical_plates`).

Feet are matched to model contacts by the facade's ``ground_groups``
convention: a foot label maps to a contact-name suffix.  A frame without a
model time, or whose shifted time falls outside the trial, has no data and
returns ``None``; nothing is extrapolated or fabricated.

The common-mode clock offset is profiled, not folded into sigma:
:meth:`PlateObservationProvider.with_shift` returns the provider at one shift
and :meth:`profile_offsets` the declared grid.  :func:`screen_with_plates`
refuses an alignment that was derived from the model it is asked to test.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from src.shared.python.force_plates import (
    FOOT_LABELS,
    FootSample,
    PlacedTrial,
    PlateTimeAlignment,
    ReadingOptions,
    reading_at,
)
from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.ground_contact import (
    FootLoadReading,
    GroundObservation,
)
from src.shared.python.loop_nullspace.plate_screen import (
    PlateConsistencyReport,
    ScreenFrame,
    ScreenSetup,
    offset_grid,
    plate_consistency,
)

__all__ = ["PlateObservationProvider", "screen_with_plates"]

_CONE_ID = "H-FRICTION-CONE"


@dataclass(frozen=True, eq=False)
class PlateObservationProvider:
    """Plate trial -> ground observation for each model frame.

    ``frame_times_s`` maps a frame index to its model time [s].
    ``ground_groups`` maps each of :data:`FOOT_LABELS` to the contact-name
    suffix of that foot in the model (the facade's convention).  ``options``
    carries the reading coverage, skew and the profiled offset shift.

    Raises ``TypeError`` for wrong component types and ``ValueError`` for
    non-finite frame times or ``ground_groups`` that do not name both feet.
    """

    placed: PlacedTrial
    alignment: PlateTimeAlignment
    frame_times_s: Mapping[int, float]
    options: ReadingOptions
    ground_groups: Mapping[str, str]

    def __post_init__(self) -> None:
        for name, kind in (
            ("placed", PlacedTrial),
            ("alignment", PlateTimeAlignment),
            ("options", ReadingOptions),
        ):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be a {kind.__name__}")
        times = {operator.index(k): float(v) for k, v in self.frame_times_s.items()}
        if not all(math.isfinite(t) for t in times.values()):
            raise ValueError("frame_times_s must be finite")
        groups = dict(self.ground_groups)
        if set(groups) != set(FOOT_LABELS) or not all(groups.values()):
            raise ValueError(
                f"ground_groups must map exactly {FOOT_LABELS} to suffixes"
            )
        object.__setattr__(self, "frame_times_s", MappingProxyType(times))
        object.__setattr__(self, "ground_groups", MappingProxyType(groups))

    def __call__(
        self, index: int, snapshot: ClosedLoopSnapshot
    ) -> GroundObservation | None:
        """The frame's vertical-plate observation, or ``None`` without data."""
        model_time = self.frame_times_s.get(index)
        if model_time is None or not self._covers(model_time):
            return None
        reading = reading_at(self.placed, model_time, self.alignment, self.options)
        readings = {foot: _foot_reading(reading.feet[foot]) for foot in FOOT_LABELS}
        return GroundObservation.from_vertical_plates(
            snapshot,
            readings,
            self._groups(snapshot),
            coverage=self.options.coverage,
        )

    def with_shift(
        self, shift_s: float, coverage: float, *, half_step_s: float = 0.0
    ) -> PlateObservationProvider:
        """This provider at offset shift ``shift_s``, read at per-row ``coverage``.

        ``half_step_s`` (half the profiling grid step) is added to the reading
        skew, so a true offset between grid points is covered: the band then
        widens by ``k * rate * half_step >= rate * half_step`` (conservative).
        """
        if not math.isfinite(half_step_s) or half_step_s < 0.0:
            raise ValueError(
                f"half_step_s must be finite and >= 0, got {half_step_s!r}"
            )
        options = dataclasses.replace(
            self.options,
            offset_shift_s=shift_s,
            coverage=coverage,
            skew_s=self.options.skew_s + half_step_s,
        )
        return dataclasses.replace(self, options=options)

    def profile_offsets(self, *, reach_sigmas: float) -> tuple[float, ...]:
        """Shifts spanning ``+/- reach_sigmas`` offset sigmas, step <= sample interval."""
        step = self.placed.min_sample_interval_s()
        return offset_grid(
            sigma_offset_s=self.alignment.sigma_offset_s,
            reach_sigmas=reach_sigmas,
            max_step_s=step,
        )

    def depth_uncertain(self) -> bool:
        """True when the plate declares a non-zero sensing-depth sigma.

        The moment-row sigma then bounds the depth error by ``mu * N``, which
        holds only where the friction cones are imposed.
        """
        trial = self.placed.trial
        return trial.uncertainty.sensing_depth_sigma_m > 0.0

    def validates(self, model_identity: str) -> bool:
        """False when the alignment was derived from ``model_identity`` itself."""
        return self.alignment.validates(model_identity)

    def _covers(self, model_time_s: float) -> bool:
        start, end = self.placed.span_s()
        trial_time = self.alignment.trial_time(
            model_time_s, self.options.offset_shift_s
        )
        return start <= trial_time <= end

    def _groups(self, snapshot: ClosedLoopSnapshot) -> dict[str, list[str]]:
        names = [c.name for c in snapshot.foot_contacts]
        return {
            foot: [n for n in names if n.endswith(suffix)]
            for foot, suffix in self.ground_groups.items()
        }


def _foot_reading(sample: FootSample) -> FootLoadReading:
    return FootLoadReading(
        normal_load_n=sample.load_n,
        cop_m=sample.cop_m,
        sigma_force_n=sample.sigma_load_n,
        sigma_cop_m=sample.sigma_cop_m,
        sigma_sensing_m=sample.sigma_sensing_m,
    )


def screen_with_plates(
    frames: Sequence[ScreenFrame],
    provider: PlateObservationProvider,
    setup: ScreenSetup,
    *,
    model_identity: str,
) -> PlateConsistencyReport:
    """Screen a candidate named ``model_identity`` against ``provider``'s trial.

    The plates are read at the screen's Bonferroni coverage, and the reading
    skew is widened by half the largest gap of ``setup.offsets_s`` so that a
    true offset between grid points is covered.  Raises ``ValueError`` when
    the provider's alignment was derived from the same model (an offset
    fitted to a model cannot then test that model), or when the plate declares
    a sensing-depth sigma but ``setup`` does not declare ``H-FRICTION-CONE``
    (that sigma is only a bound on the moment-row error inside the cones).
    """
    if not isinstance(provider, PlateObservationProvider):
        raise TypeError("provider must be a PlateObservationProvider")
    if not provider.validates(model_identity):
        raise ValueError(
            f"the alignment was derived from {model_identity!r}; it cannot test it"
        )
    if provider.depth_uncertain() and not setup.declares(_CONE_ID):
        raise ValueError(
            f"a sensing-depth sigma is bounded by mu * N only inside the "
            f"friction cones; declare {_CONE_ID}"
        )
    half_step = _half_grid_step(setup.offsets_s)

    def factory(shift_s: float, coverage: float) -> PlateObservationProvider:
        return provider.with_shift(shift_s, coverage, half_step_s=half_step)

    return plate_consistency(frames, factory, setup)


def _half_grid_step(shifts: Sequence[float]) -> float:
    """Half the largest gap between neighbouring profiled shifts (0 for one shift)."""
    ordered = sorted(shifts)
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    return 0.5 * max(gaps, default=0.0)
