"""Independent research replay on authored seconds, with explicit SI metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.full_body_forward_dynamics import (
    SimulationRecord,
)
from src.shared.python.motion_matching.pipeline.plant import ForwardSimulationPlant
from src.shared.python.simulation_backends import Trace

from .necromatcher_efforts import AuthoredEffortProfile
from .necromatcher_native import NativeFitBinding, load_native_fit_binding

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary


@dataclass(frozen=True)
class ReplayOptions:
    """Bounded authored run; initial rates are supplied rather than inferred."""

    source_frame_index: int
    initial_rates: tuple[float, ...]
    duration_s: float
    dt_s: float
    record_every: int = 1
    refinement: int = 4
    max_steps: int = 4096

    def __post_init__(self) -> None:
        for name in ("source_frame_index", "record_every", "refinement", "max_steps"):
            value = getattr(self, name)
            minimum = 0 if name == "source_frame_index" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not 2 <= self.refinement <= 16:
            raise ValueError("refinement must be between 2 and 16")
        for value in (self.duration_s, self.dt_s):
            if type(value) not in (float, int) or not np.isfinite(value) or value <= 0:
                raise ValueError("Replay duration and step must be finite and positive")
        if (
            not isinstance(self.initial_rates, tuple)
            or not self.initial_rates
            or any(
                type(value) not in (float, int) or not np.isfinite(value)
                for value in self.initial_rates
            )
        ):
            raise ValueError("Initial rates require an explicit finite numeric tuple")
        steps = self.duration_s / self.dt_s
        if not np.isfinite(steps) or not 1 <= steps <= self.max_steps / self.refinement:
            raise ValueError("Replay exceeds the bounded step budget")
        if not np.isclose(steps, round(steps), rtol=0, atol=1e-9):
            raise ValueError("Duration must be an integer number of integration steps")


def _run(
    binding: NativeFitBinding,
    controls: AuthoredEffortProfile,
    options: ReplayOptions,
    refinement: int,
) -> SimulationRecord:
    """Integrate from one exact saved pose without resets or target feedback."""
    if not isinstance(binding.plant, ForwardSimulationPlant):
        raise ValueError("Native plant lacks independent forward simulation")
    try:
        position = binding.fit["frame_indices"].index(options.source_frame_index)
    except ValueError as exc:
        raise IndexError("Source frame has no stored fit sample") from exc
    q0 = np.asarray(binding.fit["q"][position], dtype=np.float64)
    v0 = np.asarray(options.initial_rates, dtype=np.float64)
    if q0.shape != v0.shape:
        raise ValueError("Initial rate count must match native coordinates")

    def command(
        time: float, _q: NDArray[np.float64], _v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        authored = time + controls.start_s
        # Snap only floating-point endpoint noise; never extrapolate commands.
        tolerance = 8 * np.finfo(float).eps * max(1.0, abs(controls.end_s))
        if controls.end_s < authored <= controls.end_s + tolerance:
            authored = controls.end_s
        return np.asarray(list(binding.efforts(controls, authored).values()))

    return binding.plant.create_forward_simulator().run(
        q0,
        v0,
        command,
        duration_s=options.duration_s,
        dt_s=options.dt_s / refinement,
        record_every=options.record_every * refinement,
    )


def replay_authored_profile(
    library: NecromatcherLibrary, profile_id: str, options: ReplayOptions
) -> Trace:
    """Replay twice with fresh native resources; report numerical and grip errors."""
    asset = library.load_asset(profile_id)
    controls = library.load_effort_profile(profile_id, asset.metadata["model_id"])
    binding = load_native_fit_binding(library, controls.fit_id)
    if not controls.channels_are_zero(tuple(binding.plant.coordinate_order[:6])):
        raise ValueError("Unactuated root requires identically zero root commands")
    if controls.start_s + options.duration_s > controls.end_s:
        raise ValueError("Replay duration exceeds the authored profile horizon")
    coarse = _run(binding, controls, options, 1)
    fresh = load_native_fit_binding(library, controls.fit_id)
    fine = _run(fresh, controls, options, options.refinement)
    if not np.allclose(coarse.time_s, fine.time_s, rtol=0, atol=1e-12):
        raise ValueError("Refined replay did not preserve the recording clock")
    times = coarse.time_s + controls.start_s
    efforts = np.asarray(
        [list(binding.efforts(controls, float(t)).values()) for t in times]
    )
    gaps = np.asarray(
        [np.linalg.norm(binding.plant.closure_residuals(q)) for q in coarse.q]
    )
    difference = np.abs(coarse.q - fine.q)
    meta: dict[str, object] = {
        "schema": "necromatcher/authored-replay/1",
        "profile_id": profile_id,
        "profile_hash": asset.metadata["hash"],
        "fit_id": binding.fit_id,
        "fit_hash": binding.fit_hash,
        "model_id": binding.model_id,
        "model_hash": binding.model_hash,
        "source_frame_index": options.source_frame_index,
        "capture_id": binding.fit["capture_id"],
        "capture_hash": binding.fit["capture_hash"],
        "source_frame_json": json.dumps(
            binding.fit["frames"][
                binding.fit["frame_indices"].index(options.source_frame_index)
            ]
        ),
        "initial_state_policy": "exact_saved_pose_and_authored_rates",
        "root_policy": "unactuated",
        "initial_rates_json": json.dumps(options.initial_rates),
        "coordinate_order_json": json.dumps(controls.dofs),
        "coordinate_units_json": json.dumps(controls.coordinate_units),
        "effort_units_json": json.dumps(controls.effort_units),
        "physical_source_time_qualified": False,
        "scientific_qualified": False,
        "independent_replay_executed": True,
        "verification_refinement": options.refinement,
        "duration_s": options.duration_s,
        "record_every": options.record_every,
        "max_steps": options.max_steps,
        "initial_grip_gap_m": float(gaps[0]),
        "max_grip_gap_m": float(gaps.max()),
    }
    for unit, name in (
        ("m", "translation_difference_m"),
        ("rad", "rotation_difference_rad"),
    ):
        selected = [
            i for i, value in enumerate(controls.coordinate_units) if value == unit
        ]
        meta[f"verification_max_{name}"] = float(difference[:, selected].max(initial=0))
    return Trace(
        t=times,
        q=coarse.q,
        v=coarse.v,
        u=efforts,
        dt=options.dt_s,
        backend=binding.plant.engine_name,
        meta=meta,
    )
