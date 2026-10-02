"""Discrete authored repairs using existing bounded native pose IK.

Targets come from an earlier model hypothesis, not measured 3D motion. This
operation neither publishes a fit nor qualifies interpolation or dynamics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from copy import deepcopy

from src.shared.python.motion_matching.full_body_ik import SolvePoseOptions
from .necromatcher_fit_jobs import fit_execution_stamp
from .necromatcher_native import NativeFitBinding

_CONSTRAINT_WEIGHT = 1e6
_PRIOR_WEIGHT = 1e-4


def _authored_ranges(binding: NativeFitBinding) -> dict[str, tuple[float, float]]:
    definition = binding.fit["provenance"]["native_definition"]
    ranges = definition.get("coordinate_ranges_deg")
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError("Repair requires authored coordinate ranges")
    order = binding.plant.coordinate_order
    bounds = {}
    for name, values in ranges.items():
        if name not in order or binding.coordinate_units[order.index(name)] != "rad":
            raise ValueError("Authored degree ranges require named angular coordinates")
        limits = np.asarray(values, dtype=float)
        if (
            limits.shape != (2,)
            or not np.isfinite(limits).all()
            or limits[0] >= limits[1]
        ):
            raise ValueError("Authored ranges require finite increasing limits")
        low, high = np.deg2rad(limits)
        bounds[name] = (float(low), float(high))
    return bounds


def repair_native_motion(
    binding: NativeFitBinding, *, iterations: int = 150
) -> dict[str, Any]:
    """Repair each stored pose; report bounds and soft-constraint tradeoffs.

    The unchanged source pose is the prior for each independent solve. No
    spline, physical rates, effort profile, or historical accuracy is inferred.
    """
    if type(iterations) is not int or not 1 <= iterations <= 1000:
        raise ValueError("iterations must be an integer between 1 and 1000")
    stamp = fit_execution_stamp()
    bounds = _authored_ranges(binding)
    camera, attachments = binding.review_inputs()
    ik = binding.plant.create_ik(attachments)
    valid = np.ones(len(attachments), dtype=bool)
    options = SolvePoseOptions(
        iterations=iterations,
        solver="trf",
        bounds=bounds,
        closure_weight=_CONSTRAINT_WEIGHT,
        closure_rotation_weight=_CONSTRAINT_WEIGHT,
        ground_weight=_CONSTRAINT_WEIGHT,
        prior_weight=_PRIOR_WEIGHT,
    )
    samples, diagnostics = [], []
    for index, row in enumerate(binding.fit["q"]):
        source = np.asarray(row, dtype=float)
        targets = binding.plant.marker_positions(source, attachments)
        before = ik.closure_error(source)
        result = ik.solve_pose(
            targets, valid, source, ground=binding.plant.ground_plane, options=options
        )
        points = binding.plant.marker_positions(result.q, attachments)
        pixel_changes = np.linalg.norm(
            camera.project(points) - camera.project(targets), axis=1
        )
        if not np.isfinite(pixel_changes).all() or not np.isfinite(result.q).all():
            raise ValueError("Repaired states and projections must be finite")
        for name, (low, high) in bounds.items():
            value = result.q[binding.plant.coordinate_order.index(name)]
            if not low <= value <= high:
                raise ValueError("Native repair violated authored coordinate bounds")
        samples.append(result.q.tolist())
        diagnostics.append(
            {
                "frame_index": binding.fit["frame_indices"][index],
                "iterations": result.iterations,
                "iteration_budget_reached": result.iterations >= iterations,
                "marker_rms_m": result.marker_rms_m,
                "max_pixel_change": float(pixel_changes.max()),
                "grip_before_m": before[0],
                "grip_before_rad": before[1],
                "grip_after_m": result.closure_error_m,
                "grip_after_rad": result.closure_error_rad,
                "lowest_sphere_height_m": result.lowest_sphere_height_m,
            }
        )
    final_stamp = fit_execution_stamp()
    if any(
        final_stamp[key] != stamp[key] for key in ("source_sha256", "runtime_sha256")
    ):
        raise ValueError("Implementation or runtime changed during native repair")
    return _repair_record(binding, bounds, stamp, iterations, samples, diagnostics)


def _repair_record(
    binding: NativeFitBinding,
    bounds: dict[str, tuple[float, float]],
    stamp: dict[str, Any],
    iterations: int,
    samples: list[list[float]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep research provenance distinct from fit publication and acceptance."""
    return {
        "schema": "necromatcher/authored-discrete-repair/1",
        "source_fit_id": binding.fit_id,
        "source_fit_hash": binding.fit_hash,
        "model_id": binding.model_id,
        "model_hash": binding.model_hash,
        "native_model_hash": binding.plant.plant_sha,
        "coordinate_order": list(binding.plant.coordinate_order),
        "coordinate_units": list(binding.coordinate_units),
        "frame_indices": list(binding.fit["frame_indices"]),
        "frames": deepcopy(binding.fit["frames"]),
        "target_kind": "inferred_native_world_markers",
        "interpolation": "none_discrete_samples_only",
        "scientifically_qualified": False,
        "physical_time_qualified": False,
        "bounds_rad": bounds,
        "unbounded_coordinates": [
            name for name in binding.plant.coordinate_order if name not in bounds
        ],
        "solver_options": {
            "iterations": iterations,
            "solver": "trf",
            "budget_kind": "residual_evaluations",
            "constraint_weight": _CONSTRAINT_WEIGHT,
            "prior_weight": _PRIOR_WEIGHT,
        },
        "execution_stamp": stamp,
        "q": samples,
        "diagnostics": diagnostics,
    }
