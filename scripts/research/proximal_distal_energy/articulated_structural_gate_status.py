"""Derive fail-closed per-cell gate evidence for structural comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Pathway = Literal["shaft", "ground"]
BoolArray = NDArray[np.bool_]
StringArray = NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class StructuralCellGateStatus:
    """Flattened gate state aligned with normalized headline-cell identity."""

    pathway: Pathway
    gate_status: BoolArray
    failure_class: StringArray

    def __post_init__(self) -> None:
        if self.gate_status.ndim != 1 or self.failure_class.shape != (
            self.gate_status.size,
        ):
            raise ValueError("gate status and failure class must be aligned vectors")
        if self.gate_status.dtype != np.bool_:
            raise ValueError("gate status must be Boolean")
        failures = np.asarray(self.failure_class, dtype=str)
        if np.any(self.gate_status & (failures != "none")):
            raise ValueError("passing cells cannot retain gate failure classes")
        if np.any((~self.gate_status) & (failures == "none")):
            raise ValueError("failed cells require explicit gate failure classes")


def _array(arrays: Mapping[str, Any], name: str, dtype: Any) -> NDArray[Any]:
    if name not in arrays:
        raise ValueError(f"gate evidence is missing {name}")
    return np.asarray(arrays[name], dtype=dtype)


def _registered_shape(arrays: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    cases = _array(arrays, "state_case_index", np.int64)
    phases = _array(arrays, "state_sample_index", np.int64)
    coordinates = (
        _array(arrays, "velocity_factors", float),
        _array(arrays, "time_steps_s", float),
        _array(arrays, "engine_names", str),
        _array(arrays, "horizons_s", float),
    )
    if cases.ndim != 1 or phases.shape != cases.shape:
        raise ValueError("state identities must be aligned vectors")
    if tuple(value.shape for value in coordinates) != ((2,), (2,), (2,), (4,)):
        raise ValueError("coordinates do not match the registered headline design")
    return (cases.size, 2, 2, 2, 4)


def _branch_slots(
    names: NDArray[np.str_], required: tuple[str, str]
) -> tuple[int, int]:
    if names.ndim != 1 or len(set(names.tolist())) != names.size:
        raise ValueError("activation names must be a unique vector")
    missing = [name for name in required if np.count_nonzero(names == name) != 1]
    if missing:
        raise ValueError(
            f"activation names require exactly one of: {', '.join(required)}"
        )
    return tuple(int(np.flatnonzero(names == name)[0]) for name in required)  # type: ignore[return-value]


def _combine_branches(array: BoolArray, slots: tuple[int, int]) -> BoolArray:
    return np.asarray(array[:, slots[0], ...] & array[:, slots[1], ...], dtype=bool)


def _failure_classes(components: tuple[tuple[str, BoolArray], ...]) -> StringArray:
    shape = components[0][1].shape
    if any(value.shape != shape for _, value in components):
        raise ValueError("gate components must share the headline-cell shape")
    flattened = [(name, np.ravel(value)) for name, value in components]
    failures = np.asarray(
        [
            "+".join(name for name, value in flattened if not bool(value[index]))
            or "none"
            for index in range(flattened[0][1].size)
        ],
        dtype=str,
    )
    return failures


def _parity_with_engine_axis(parity: BoolArray) -> BoolArray:
    return np.repeat(parity[:, :, :, None, :], 2, axis=3)


def derive_structural_cell_gate_status(
    pathway: Pathway,
    arrays: Mapping[str, Any],
) -> StructuralCellGateStatus:
    """Combine both compared branches for every registered headline gate."""

    expected = _registered_shape(arrays)
    if pathway == "shaft":
        names = _array(arrays, "activation_names", str)
        slots = _branch_slots(names, ("rigid", "coupled"))
        full_names = (
            ("numerical_gate_failure", "numerical_gates_passed"),
            ("small_deflection_gate_failure", "small_deflection_gate_passed"),
            ("twist_gate_failure", "twist_gate_passed"),
        )
        components: list[tuple[str, BoolArray]] = []
        for failure, array_name in full_names:
            values = _array(arrays, array_name, bool)
            if values.shape != (expected[0], names.size, *expected[1:]):
                raise ValueError(f"{array_name} does not match the registered design")
            components.append((failure, _combine_branches(values, slots)))
        parity = _array(arrays, "parity_gates_passed", bool)
        if parity.shape != (expected[0], names.size, 2, 2, 4):
            raise ValueError("parity_gates_passed does not match the registered design")
        components.insert(
            1,
            (
                "parity_gate_failure",
                _parity_with_engine_axis(_combine_branches(parity, slots)),
            ),
        )
    elif pathway == "ground":
        names = _array(arrays, "ground_activation_names", str)
        slots = _branch_slots(names, ("fixed", "coupled"))
        numerical = _array(arrays, "primary_numerical", bool)
        if numerical.shape != (expected[0], names.size, *expected[1:]):
            raise ValueError("primary_numerical does not match the registered design")
        parity = _array(arrays, "primary_parity", bool)
        if parity.shape != (expected[0], names.size, 2, 2, 4):
            raise ValueError("primary_parity does not match the registered design")
        components = [
            ("numerical_gate_failure", _combine_branches(numerical, slots)),
            (
                "parity_gate_failure",
                _parity_with_engine_axis(_combine_branches(parity, slots)),
            ),
        ]
    else:
        raise ValueError("pathway must be shaft or ground")

    failures = _failure_classes(tuple(components))
    gates = failures == "none"
    return StructuralCellGateStatus(
        pathway=pathway,
        gate_status=np.asarray(gates, dtype=bool),
        failure_class=failures,
    )


__all__ = ["StructuralCellGateStatus", "derive_structural_cell_gate_status"]
