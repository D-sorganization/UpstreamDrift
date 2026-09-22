"""Structured perturbations for teacher episode generation (NM-04)."""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["perturbation_for_attempt", "stratified_perturbation_axes"]


def stratified_perturbation_axes(*, dim: int) -> tuple[str, ...]:
    """Name perturbation axes for geometry, initial state, control and duration."""
    if dim < 4:
        raise ValueError("dim must be >= 4 for stratified teacher perturbations")
    base = ("geometry", "initial_state", "control_profile", "duration")
    extra = tuple(f"axis_{idx}" for idx in range(dim - len(base)))
    return base + extra


def perturbation_for_attempt(
    *,
    master_seed: int,
    stage_index: int,
    attempt_index: int,
    dim: int,
) -> np.ndarray:
    """Low-discrepancy stratified perturbation in [-1, 1]^dim (deterministic)."""
    if dim < 1:
        raise ValueError("dim must be >= 1")
    axes = (
        stratified_perturbation_axes(dim=dim)
        if dim >= 4
        else tuple(f"axis_{idx}" for idx in range(dim))
    )
    values: list[float] = []
    for axis_idx, _axis in enumerate(axes[:dim]):
        key = f"{master_seed}:{stage_index}:{attempt_index}:{axis_idx}".encode()
        digest = hashlib.sha256(key).digest()
        # Map first 8 bytes to [0,1) then affine to [-1,1].
        integer = int.from_bytes(digest[:8], "big", signed=False)
        unit = integer / float(2**64 - 1)
        values.append(2.0 * unit - 1.0)
    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("perturbation must be finite")
    return vector
