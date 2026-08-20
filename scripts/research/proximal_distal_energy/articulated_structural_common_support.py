"""Cell-identity-safe comparisons for articulated structural sensitivities."""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
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
    two_engine_speed_difference_discrepancy_m_s: FloatArray
    time_step_speed_difference_discrepancy_m_s: FloatArray

    def __post_init__(self) -> None:
        size = len(self.identities)
        fields = (
            self.matched,
            self.final_speed_difference_m_s,
            self.load_match_relative_error,
            self.work_match_relative_error,
            self.two_engine_speed_difference_discrepancy_m_s,
            self.time_step_speed_difference_discrepancy_m_s,
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
    resolution_threshold_m_s: FloatArray
    resolved_outcome_change: BoolArray

    def __post_init__(self) -> None:
        size = len(self.persistent_identities)
        if self.persistent_speed_change_m_s.shape != (size,) or (
            self.resolution_threshold_m_s.shape != (size,)
        ):
            raise ValueError(
                "persistent outcomes must align with persistent identities"
            )
        if self.resolved_outcome_change.shape != (size,) or (
            self.resolved_outcome_change.dtype != np.bool_
        ):
            raise ValueError("resolved outcome status must be an aligned Boolean array")
        if np.any(self.resolution_threshold_m_s < 0.0) or not np.all(
            np.isfinite(self.resolution_threshold_m_s)
        ):
            raise ValueError("resolution thresholds must be finite and nonnegative")

    @property
    def has_paired_outcome(self) -> bool:
        """Return whether at least one persistent cell supports paired inference."""

        return bool(self.persistent_identities)


@dataclass(frozen=True, slots=True)
class OneSidedEngineeringSecants:
    """Separate engineering secants without pooling unlike support sets."""

    axis_name: str
    pathway: Pathway
    low_scale: float
    nominal_scale: float
    high_scale: float
    low_to_nominal_identities: tuple[CellIdentity, ...]
    nominal_to_high_identities: tuple[CellIdentity, ...]
    low_to_nominal_m_s_per_unit_scale: FloatArray
    nominal_to_high_m_s_per_unit_scale: FloatArray
    low_to_nominal_resolution_per_unit_scale: FloatArray
    nominal_to_high_resolution_per_unit_scale: FloatArray

    @property
    def are_averaged(self) -> bool:
        """Declare that the two one-sided evidence sets remain unpooled."""

        return False


@dataclass(frozen=True, slots=True)
class SecantClassification:
    """Resolution-aware classification on shared persistent support only."""

    shared_persistent_identities: tuple[CellIdentity, ...]
    cell_classification: tuple[str, ...]
    overall: str


@dataclass(frozen=True, slots=True)
class CornerSupportSummary:
    """Explicit planned, feasible, executed, and matched support accounting."""

    corner_id: str
    pathway: Pathway
    requested_state_count: int
    feasible_state_count: int
    retained_failures: tuple[dict[str, int | str], ...]
    planned_headline_cell_count: int
    feasible_headline_cell_count: int
    executed_headline_cell_count: int
    matched_cell_count: int
    matched_fraction_of_feasible: float
    all_registered_gates_passed: bool
    authority: dict[str, Any]

    @property
    def complete_execution(self) -> bool:
        """Return whether every feasible state supplied all 32 headline cells."""

        return self.executed_headline_cell_count == self.feasible_headline_cell_count

    @property
    def qualifies_as_release_evidence(self) -> bool:
        """Require complete execution and every registered gate."""

        return self.complete_execution and self.all_registered_gates_passed


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
    engine_discrepancy = np.abs(speed[:, :, :, 0, :] - speed[:, :, :, 1, :])
    engine_discrepancy = np.repeat(engine_discrepancy[:, :, :, None, :], 2, axis=3)
    step_discrepancy = np.abs(speed[:, :, 0, :, :] - speed[:, :, 1, :, :])
    step_discrepancy = np.repeat(step_discrepancy[:, :, None, :, :], 2, axis=2)
    return HeadlineCells(
        pathway=pathway,
        identities=identities,
        matched=np.ravel(matched),
        final_speed_difference_m_s=np.ravel(speed),
        load_match_relative_error=np.ravel(load_error),
        work_match_relative_error=np.ravel(work_error),
        two_engine_speed_difference_discrepancy_m_s=np.ravel(engine_discrepancy),
        time_step_speed_difference_discrepancy_m_s=np.ravel(step_discrepancy),
    )


def compare_common_support(
    nominal: HeadlineCells,
    corner: HeadlineCells,
    *,
    absolute_resolution_floor_m_s: float = 0.001,
) -> CommonSupportComparison:
    """Compare matching only where both corners executed the same identity."""

    if nominal.pathway != corner.pathway:
        raise ValueError("common-support pathways must agree")
    if (
        not np.isfinite(absolute_resolution_floor_m_s)
        or absolute_resolution_floor_m_s < 0.0
    ):
        raise ValueError("absolute resolution floor must be finite and nonnegative")
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
    resolution_threshold = np.asarray(
        [
            max(
                absolute_resolution_floor_m_s,
                nominal.two_engine_speed_difference_discrepancy_m_s[
                    nominal_index[identity]
                ],
                corner.two_engine_speed_difference_discrepancy_m_s[
                    corner_index[identity]
                ],
                nominal.time_step_speed_difference_discrepancy_m_s[
                    nominal_index[identity]
                ],
                corner.time_step_speed_difference_discrepancy_m_s[
                    corner_index[identity]
                ],
            )
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
        resolution_threshold_m_s=resolution_threshold,
        resolved_outcome_change=np.abs(speed_change) > resolution_threshold,
    )


def build_one_sided_engineering_secants(
    axis_name: str,
    low_vs_nominal: CommonSupportComparison,
    high_vs_nominal: CommonSupportComparison,
    *,
    low_scale: float,
    nominal_scale: float,
    high_scale: float,
) -> OneSidedEngineeringSecants:
    """Scale paired changes on each side while retaining separate support."""

    if not axis_name.strip():
        raise ValueError("axis_name must be nonempty")
    if low_vs_nominal.pathway != high_vs_nominal.pathway:
        raise ValueError("one-sided secant pathways must agree")
    scales = np.asarray([low_scale, nominal_scale, high_scale], dtype=float)
    if not np.all(np.isfinite(scales)) or not low_scale < nominal_scale < high_scale:
        raise ValueError("scales must be finite and satisfy low < nominal < high")
    low_span = nominal_scale - low_scale
    high_span = high_scale - nominal_scale
    return OneSidedEngineeringSecants(
        axis_name=axis_name,
        pathway=low_vs_nominal.pathway,
        low_scale=low_scale,
        nominal_scale=nominal_scale,
        high_scale=high_scale,
        low_to_nominal_identities=low_vs_nominal.persistent_identities,
        nominal_to_high_identities=high_vs_nominal.persistent_identities,
        low_to_nominal_m_s_per_unit_scale=(
            -low_vs_nominal.persistent_speed_change_m_s / low_span
        ),
        nominal_to_high_m_s_per_unit_scale=(
            high_vs_nominal.persistent_speed_change_m_s / high_span
        ),
        low_to_nominal_resolution_per_unit_scale=(
            low_vs_nominal.resolution_threshold_m_s / low_span
        ),
        nominal_to_high_resolution_per_unit_scale=(
            high_vs_nominal.resolution_threshold_m_s / high_span
        ),
    )


def classify_one_sided_engineering_secants(
    secants: OneSidedEngineeringSecants,
) -> SecantClassification:
    """Flag resolved opposition or material inequality without averaging."""

    low_index = {
        identity: index
        for index, identity in enumerate(secants.low_to_nominal_identities)
    }
    high_index = {
        identity: index
        for index, identity in enumerate(secants.nominal_to_high_identities)
    }
    shared = tuple(
        identity
        for identity in secants.low_to_nominal_identities
        if identity in high_index
    )
    classes: list[str] = []
    for identity in shared:
        low_slot = low_index[identity]
        high_slot = high_index[identity]
        low = secants.low_to_nominal_m_s_per_unit_scale[low_slot]
        high = secants.nominal_to_high_m_s_per_unit_scale[high_slot]
        low_resolution = secants.low_to_nominal_resolution_per_unit_scale[low_slot]
        high_resolution = secants.nominal_to_high_resolution_per_unit_scale[high_slot]
        if abs(low) <= low_resolution or abs(high) <= high_resolution:
            classes.append("resolution_limited")
        elif low * high < 0.0:
            classes.append("resolved_opposing")
        elif abs(low - high) > low_resolution + high_resolution:
            classes.append("resolved_materially_unequal")
        else:
            classes.append("resolved_direction_consistent")

    if not classes:
        overall = "insufficient_shared_persistent_support"
    elif "resolved_opposing" in classes:
        overall = "resolved_opposing_on_shared_support"
    elif "resolved_materially_unequal" in classes:
        overall = "resolved_materially_unequal_on_shared_support"
    elif "resolution_limited" in classes:
        overall = "resolution_limited_on_shared_support"
    else:
        overall = "resolved_direction_consistent_on_shared_support"
    return SecantClassification(
        shared_persistent_identities=shared,
        cell_classification=tuple(classes),
        overall=overall,
    )


def build_corner_support_summary(
    corner_id: str,
    cells: HeadlineCells,
    *,
    requested_state_count: int,
    feasible_state_count: int,
    retained_failures: tuple[Mapping[str, Any], ...],
    planned_headline_cell_count: int,
    all_registered_gates_passed: bool,
    authority: Mapping[str, Any],
) -> CornerSupportSummary:
    """Account for every state and cell without changing the match denominator."""

    if not corner_id.strip():
        raise ValueError("corner_id must be nonempty")
    if (
        not isinstance(requested_state_count, int)
        or not isinstance(feasible_state_count, int)
        or not 0 <= feasible_state_count <= requested_state_count
        or requested_state_count < 1
    ):
        raise ValueError("state counts must satisfy 0 <= feasible <= requested")
    expected_planned = requested_state_count * 32
    if planned_headline_cell_count != expected_planned:
        raise ValueError(
            "planned headline denominator must equal requested states times 32"
        )
    normalized_failures = tuple(
        {
            "case_index": int(failure["case_index"]),
            "phase_index": int(failure["phase_index"]),
            "failure_class": str(failure["failure_class"]),
        }
        for failure in retained_failures
    )
    failure_states = {
        (failure["case_index"], failure["phase_index"])
        for failure in normalized_failures
    }
    if len(failure_states) != requested_state_count - feasible_state_count:
        raise ValueError("retained failure states must account for infeasible states")
    state_counts = Counter((identity[0], identity[1]) for identity in cells.identities)
    if any(count != 32 for count in state_counts.values()):
        raise ValueError("every executed state must contain exactly 32 headline cells")
    if failure_states.intersection(state_counts):
        raise ValueError("retained failure states cannot also be executed")
    if len(state_counts) > feasible_state_count:
        raise ValueError("executed states cannot exceed feasible states")
    required_authority = {"authority_sha256", "scales", "model_sha256"}
    if not required_authority.issubset(authority):
        raise ValueError("authority record is incomplete")
    executed = len(cells.identities)
    matched = int(np.count_nonzero(cells.matched))
    return CornerSupportSummary(
        corner_id=corner_id,
        pathway=cells.pathway,
        requested_state_count=requested_state_count,
        feasible_state_count=feasible_state_count,
        retained_failures=normalized_failures,
        planned_headline_cell_count=planned_headline_cell_count,
        feasible_headline_cell_count=feasible_state_count * 32,
        executed_headline_cell_count=executed,
        matched_cell_count=matched,
        matched_fraction_of_feasible=matched / executed if executed else 0.0,
        all_registered_gates_passed=bool(all_registered_gates_passed),
        authority=dict(authority),
    )


def require_corner_release_evidence(summary: CornerSupportSummary) -> None:
    """Fail closed when partial execution or a registered gate remains."""

    if not summary.qualifies_as_release_evidence:
        raise RuntimeError(
            f"corner {summary.corner_id} does not qualify as release evidence"
        )


def corner_support_summary_record(summary: CornerSupportSummary) -> dict[str, Any]:
    """Serialize one release-qualified corner using registered denominator names."""

    require_corner_release_evidence(summary)
    return {
        "corner_id": summary.corner_id,
        "pathway": summary.pathway,
        "requested_state_count": summary.requested_state_count,
        "feasible_state_count": summary.feasible_state_count,
        "retained_failures": list(summary.retained_failures),
        "planned_headline_cell_count": summary.planned_headline_cell_count,
        "feasible_headline_cell_count": summary.feasible_headline_cell_count,
        "executed_headline_cell_count": summary.executed_headline_cell_count,
        "matched_cell_count": summary.matched_cell_count,
        "matched_fraction_of_feasible": summary.matched_fraction_of_feasible,
        "all_registered_gates_passed": summary.all_registered_gates_passed,
        "authority": summary.authority,
    }


def build_axis_summary_record(
    secants: OneSidedEngineeringSecants,
    classification: SecantClassification,
) -> dict[str, Any]:
    """Summarize only shared persistent identities without pooling either side."""

    low_index = {
        identity: index
        for index, identity in enumerate(secants.low_to_nominal_identities)
    }
    high_index = {
        identity: index
        for index, identity in enumerate(secants.nominal_to_high_identities)
    }
    expected_shared = tuple(
        identity
        for identity in secants.low_to_nominal_identities
        if identity in high_index
    )
    if classification.shared_persistent_identities != expected_shared or len(
        classification.cell_classification
    ) != len(expected_shared):
        raise ValueError("secant classification does not match shared support")
    low = np.asarray(
        [
            secants.low_to_nominal_m_s_per_unit_scale[low_index[value]]
            for value in expected_shared
        ],
        dtype=float,
    )
    high = np.asarray(
        [
            secants.nominal_to_high_m_s_per_unit_scale[high_index[value]]
            for value in expected_shared
        ],
        dtype=float,
    )

    def median(values: FloatArray) -> float | None:
        return float(np.median(values)) if values.size else None

    def value_range(values: FloatArray) -> list[float] | None:
        return [float(np.min(values)), float(np.max(values))] if values.size else None

    return {
        "axis_name": secants.axis_name,
        "pathway": secants.pathway,
        "low_scale": secants.low_scale,
        "nominal_scale": secants.nominal_scale,
        "high_scale": secants.high_scale,
        "shared_persistent_cell_count": len(expected_shared),
        "summary_statistic": "unweighted median on identities persistent in both one-sided comparisons",
        "low_to_nominal_secant_m_s_per_unit_scale": median(low),
        "nominal_to_high_secant_m_s_per_unit_scale": median(high),
        "low_to_nominal_secant_range_m_s_per_unit_scale": value_range(low),
        "nominal_to_high_secant_range_m_s_per_unit_scale": value_range(high),
        "cell_classification_counts": dict(Counter(classification.cell_classification)),
        "nonmonotonic_classification": classification.overall,
    }


__all__ = [
    "CellIdentity",
    "CommonSupportComparison",
    "CornerSupportSummary",
    "HeadlineCells",
    "OneSidedEngineeringSecants",
    "SecantClassification",
    "build_axis_summary_record",
    "build_corner_support_summary",
    "build_one_sided_engineering_secants",
    "classify_one_sided_engineering_secants",
    "compare_common_support",
    "corner_support_summary_record",
    "extract_headline_cells",
    "require_corner_release_evidence",
]
