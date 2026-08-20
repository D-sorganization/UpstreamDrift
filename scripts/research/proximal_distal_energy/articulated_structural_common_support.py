"""Cell-identity-safe comparisons for articulated structural sensitivities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Pathway = Literal["shaft", "ground"]
CellIdentity = tuple[int, int, float, float, str, float]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class HeadlineCells:
    """Normalized headline cells with stable scientific identities."""

    pathway: Pathway
    identities: tuple[CellIdentity, ...]
    matched: BoolArray
    final_speed_difference_m_s: FloatArray
    load_match_relative_error: FloatArray
    work_match_relative_error: FloatArray

    def __post_init__(self) -> None:
        size = len(self.identities)
        fields = (
            self.matched,
            self.final_speed_difference_m_s,
            self.load_match_relative_error,
            self.work_match_relative_error,
        )
        if any(value.shape != (size,) for value in fields):
            raise ValueError("headline cell fields must align with cell identities")
        if len(set(self.identities)) != size:
            raise ValueError("headline cell identities must be unique")
        if self.matched.dtype != np.bool_:
            raise ValueError("matched must be a Boolean array")
        if any(not np.all(np.isfinite(value)) for value in fields[1:]):
            raise ValueError("headline cell numerical fields must be finite")


@dataclass(frozen=True, slots=True)
class CommonSupportComparison:
    """Matching-support transitions and paired outcomes on common execution."""

    pathway: Pathway
    common_executed_cell_count: int
    nominal_only_executed_cell_count: int
    corner_only_executed_cell_count: int
    persistent_identities: tuple[CellIdentity, ...]
    entered_identities: tuple[CellIdentity, ...]
    exited_identities: tuple[CellIdentity, ...]
    persistent_speed_change_m_s: FloatArray

    @property
    def has_paired_outcome(self) -> bool:
        """Return whether at least one persistent cell supports paired inference."""

        return bool(self.persistent_identities)


def _headline_array(arrays: Mapping[str, Any], *names: str, dtype: Any) -> NDArray[Any]:
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name], dtype=dtype)
    raise ValueError(f"headline arrays are missing one of: {', '.join(names)}")


def extract_headline_cells(
    pathway: Pathway,
    arrays: Mapping[str, Any],
) -> HeadlineCells:
    """Normalize shaft or ground headline arrays without losing cell identity."""

    if pathway not in ("shaft", "ground"):
        raise ValueError("pathway must be shaft or ground")
    cases = _headline_array(arrays, "state_case_index", dtype=np.int64)
    phases = _headline_array(arrays, "state_sample_index", dtype=np.int64)
    velocities = _headline_array(arrays, "velocity_factors", dtype=float)
    steps = _headline_array(arrays, "time_steps_s", dtype=float)
    engines = _headline_array(arrays, "engine_names", dtype=str)
    horizons = _headline_array(arrays, "horizons_s", dtype=float)
    if cases.ndim != 1 or phases.shape != cases.shape:
        raise ValueError("state case and phase identities must be aligned vectors")
    expected_shape = (cases.size, 2, 2, 2, 4)
    if (
        velocities.shape != (2,)
        or steps.shape != (2,)
        or engines.shape != (2,)
        or horizons.shape != (4,)
    ):
        raise ValueError("coordinate arrays do not match the registered headline shape")

    matched = _headline_array(arrays, "matched_load_work", "matched", dtype=bool)
    speed = _headline_array(
        arrays,
        "matched_final_speed_difference_m_s",
        "matched_speed_difference",
        dtype=float,
    )
    load_error = _headline_array(arrays, "load_match_relative_error", dtype=float)
    work_error = _headline_array(arrays, "work_match_relative_error", dtype=float)
    if any(
        value.shape != expected_shape
        for value in (matched, speed, load_error, work_error)
    ):
        raise ValueError("headline arrays do not match the registered headline shape")

    identities = tuple(
        (
            int(cases[state_slot]),
            int(phases[state_slot]),
            float(velocities[velocity_slot]),
            float(steps[step_slot]),
            str(engines[engine_slot]),
            float(horizons[horizon_slot]),
        )
        for state_slot in range(cases.size)
        for velocity_slot in range(2)
        for step_slot in range(2)
        for engine_slot in range(2)
        for horizon_slot in range(4)
    )
    return HeadlineCells(
        pathway=pathway,
        identities=identities,
        matched=np.ravel(matched),
        final_speed_difference_m_s=np.ravel(speed),
        load_match_relative_error=np.ravel(load_error),
        work_match_relative_error=np.ravel(work_error),
    )


def compare_common_support(
    nominal: HeadlineCells,
    corner: HeadlineCells,
) -> CommonSupportComparison:
    """Compare matching only where both corners executed the same identity."""

    if nominal.pathway != corner.pathway:
        raise ValueError("common-support pathways must agree")
    nominal_index = {
        identity: index for index, identity in enumerate(nominal.identities)
    }
    corner_index = {identity: index for index, identity in enumerate(corner.identities)}
    common = set(nominal_index).intersection(corner_index)

    persistent = tuple(
        identity
        for identity in nominal.identities
        if identity in common
        and nominal.matched[nominal_index[identity]]
        and corner.matched[corner_index[identity]]
    )
    exited = tuple(
        identity
        for identity in nominal.identities
        if identity in common
        and nominal.matched[nominal_index[identity]]
        and not corner.matched[corner_index[identity]]
    )
    entered = tuple(
        identity
        for identity in corner.identities
        if identity in common
        and corner.matched[corner_index[identity]]
        and not nominal.matched[nominal_index[identity]]
    )
    speed_change = np.asarray(
        [
            corner.final_speed_difference_m_s[corner_index[identity]]
            - nominal.final_speed_difference_m_s[nominal_index[identity]]
            for identity in persistent
        ],
        dtype=float,
    )
    return CommonSupportComparison(
        pathway=nominal.pathway,
        common_executed_cell_count=len(common),
        nominal_only_executed_cell_count=len(nominal_index) - len(common),
        corner_only_executed_cell_count=len(corner_index) - len(common),
        persistent_identities=persistent,
        entered_identities=entered,
        exited_identities=exited,
        persistent_speed_change_m_s=speed_change,
    )


__all__ = [
    "CellIdentity",
    "CommonSupportComparison",
    "HeadlineCells",
    "compare_common_support",
    "extract_headline_cells",
]
