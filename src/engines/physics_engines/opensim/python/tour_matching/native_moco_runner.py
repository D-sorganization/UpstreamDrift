"""File-driven native Moco preparation and independent muscle replay.

This runner joins existing providers without granting scientific qualification.
Synthetic fixtures can prove the software path; real-source blockers remain
visible before an optimizer is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any

import numpy as np
from defusedxml import ElementTree as SafeET

from .moco_initial_bindings import MocoInitialBindings
from .moco_tracking import MocoTrackingConfig, build_moco_study
from .native_passive_readiness import (
    PassiveReadinessPolicy,
    audit_native_passive_readiness,
)
from .native_reference_conventions import (
    ReferenceStateDeclaration,
    audit_source_reference,
)
from .registration import CaptureRegistration, register_points
from .trc import read_trc, write_trc
from src.shared.python.motion_matching.tour_capture_contract import TourCapture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class NativeMocoRequest:
    """Only the files and reviewed native bindings needed for one offline fit."""

    model_path: Path
    trc_path: Path
    states_guess_path: Path
    model_sha256: str
    trc_sha256: str
    states_guess_sha256: str
    bindings: MocoInitialBindings
    config: MocoTrackingConfig
    marker_weights: Mapping[str, float]
    marker_bindings: Mapping[str, tuple[str, tuple[float, float, float]]]
    registration: CaptureRegistration | None
    reference_frame_path: str
    passive_policy: PassiveReadinessPolicy | None
    excluded_markers: Mapping[str, str]

    def __post_init__(self) -> None:
        self.config.validate()
        for name in ("model_sha256", "trc_sha256", "states_guess_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not self.reference_frame_path.startswith("/"):
            raise ValueError("Native reference frame path must be absolute")
        weights = dict(self.marker_weights)
        bindings = dict(self.marker_bindings)
        exclusions = dict(self.excluded_markers)
        for value in weights.values():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Marker weights must be finite and positive")
        for label, placement in bindings.items():
            if (
                not label
                or len(placement) != 2
                or not isinstance(placement[0], str)
                or not placement[0].startswith("/")
                or len(placement[1]) != 3
                or not all(math.isfinite(float(value)) for value in placement[1])
            ):
                raise ValueError("Native marker placement requires frame and 3D offset")
        if any(not reason.strip() for reason in exclusions.values()):
            raise ValueError("Excluded marker reasons must be explicit")
        if set(exclusions) & set(weights):
            raise ValueError("A selected marker cannot also be excluded")
        object.__setattr__(self, "marker_weights", MappingProxyType(weights))
        object.__setattr__(self, "marker_bindings", MappingProxyType(bindings))
        object.__setattr__(self, "excluded_markers", MappingProxyType(exclusions))

    @property
    def identity_sha256(self) -> str:
        """Bind every caller policy and placement across all execution stages."""
        registration = self.registration
        payload = {
            "model_path": str(self.model_path.resolve()),
            "trc_path": str(self.trc_path.resolve()),
            "guess_path": str(self.states_guess_path.resolve()),
            "source_hashes": (
                self.model_sha256,
                self.trc_sha256,
                self.states_guess_sha256,
            ),
            "state_bounds": dict(self.bindings.state_bounds),
            "initial_state": dict(self.bindings.initial_state),
            "control_bounds": dict(self.bindings.control_bounds),
            "config": asdict(self.config),
            "marker_weights": dict(self.marker_weights),
            "marker_bindings": dict(self.marker_bindings),
            "registration": None
            if registration is None
            else {
                "rotation": registration.rotation.tolist(),
                "translation": registration.translation.tolist(),
                "source_frame": registration.source_frame,
                "target_frame": registration.target_frame,
            },
            "reference_frame_path": self.reference_frame_path,
            "passive_policy": (
                None if self.passive_policy is None else asdict(self.passive_policy)
            ),
            "excluded_markers": dict(self.excluded_markers),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class NativeMocoPreparation:
    """No private coordinates or marker labels appear in this public receipt."""

    request_sha256: str
    source_sha256: str | None
    capture_sha256: str | None
    guess_sha256: str | None
    loaded_model_sha256: str | None
    reference_observation_sha256: str | None
    passive_observation_sha256: str | None
    observation_clock_sha256: str | None
    registered_trc_sha256: str | None
    marker_count: int
    observation_count: int
    excluded_marker_count: int
    blockers: tuple[str, ...]
    wall_seconds: float
    cpu_seconds: float
    qualification: str = "not-qualified-for-muscle-matching"

    @property
    def ready_for_software_solve(self) -> bool:
        """A necessary technical gate, never anatomy or capture approval."""
        return not self.blockers


def _observed_marker_bindings(model: Any, request: NativeMocoRequest) -> list[str]:
    """Compare explicit placement to the exact loaded native MarkerSet."""
    failures = []
    if set(request.marker_bindings) != set(request.marker_weights):
        failures.append("marker-binding-coverage")
    native = model.getMarkerSet()
    by_name = {native.get(i).getName(): native.get(i) for i in range(native.getSize())}
    for label, (frame_path, offset) in request.marker_bindings.items():
        marker = by_name.get(label)
        if marker is None:
            failures.append("native-marker-correspondence")
            continue
        actual_frame = marker.getParentFrame().getAbsolutePathString()
        actual_offset = marker.get_location()
        values = tuple(float(actual_offset.get(i)) for i in range(3))
        if actual_frame != frame_path or values != tuple(offset):
            failures.append("native-marker-placement-mismatch")
    return failures


def _native_preparation(
    request: NativeMocoRequest,
) -> tuple[str | None, str | None, str | None, list[str]]:
    """Read the actual prepared state and independent provider dispositions."""
    import opensim as osim

    from .muscle_replay import _restore_continuous_state
    from .native_muscle_bundle import build_native_muscle_replay_bundle

    blockers: list[str] = []
    loaded: str | None = None
    reference_sha: str | None = None
    passive_sha: str | None = None
    model = osim.Model(str(request.model_path))
    blockers.extend(_observed_marker_bindings(model, request))
    if request.passive_policy is None:
        blockers.append("passive-policy-unavailable")
    if model.getActuators().getSize() != model.getMuscles().getSize():
        blockers.append("nonmuscle-assistance-unqualified")
    coordinates = model.getCoordinateSet()
    if model.getConstraintSet().getSize() or any(
        coordinates.get(i).getDefaultLocked() for i in range(coordinates.getSize())
    ):
        blockers.append("native-constraint-policy-unavailable")
    try:
        state, _, _ = _restore_continuous_state(
            model, request.bindings.initial_state, request.config.t_start_s
        )
    except (ValueError, RuntimeError):
        blockers.append("complete-native-state-unavailable")
        return loaded, reference_sha, passive_sha, blockers
    loaded = hashlib.sha256(model.dump().encode()).hexdigest()
    names = model.getStateVariableNames()
    native_states = {str(names.get(i)) for i in range(names.getSize())}
    if set(request.bindings.state_bounds) != native_states:
        blockers.append("native-state-binding-coverage")
    actuators = model.getActuators()
    native_controls = {
        str(actuators.get(i).getAbsolutePathString())
        for i in range(actuators.getSize())
    }
    if set(request.bindings.control_bounds) != native_controls:
        blockers.append("native-control-binding-coverage")
    try:
        passive = audit_native_passive_readiness(model, state, request.passive_policy)
        passive_sha = passive.observation_sha256
        blockers.extend(passive.blockers)
        blockers.extend(
            item
            for item in passive.native_state.blockers
            if item
            not in {
                "complete-native-state-unverified",
                "constraint-residual-acceptance-unqualified",
            }
        )
    except (ValueError, RuntimeError):
        blockers.append("native-passive-or-constraint-observation-unavailable")
    try:
        named_initial = request.bindings.initial_state
        reference = ReferenceStateDeclaration(
            "offline-moco-prepared-state",
            "caller-declared-complete-named-continuous-state",
            "complete-named-state",
            request.model_sha256,
            tuple(named_initial.items()),
            request.config.t_start_s,
        )
        observed = audit_source_reference(
            request.model_path, reference, request.reference_frame_path
        )
        reference_sha = observed.observation_sha256
    except (ValueError, RuntimeError):
        blockers.append("native-reference-convention-unavailable")
    try:
        muscles = model.getMuscles()
        preview = {
            muscles.get(i).getName(): np.zeros(2) for i in range(muscles.getSize())
        }
        build_native_muscle_replay_bundle(
            request.model_path,
            request.bindings.initial_state,
            np.array([0.0, request.config.duration_s]),
            preview,
            experiment_id="offline-moco-replay-capability-preview",
        )
    except (ValueError, RuntimeError):
        blockers.append("independent-native-replay-unavailable")
    if _sha(request.model_path) != request.model_sha256:
        blockers.append("model-sha256-changed-during-preparation")
    return loaded, reference_sha, passive_sha, blockers


def _load_observation_capture(
    request: NativeMocoRequest, directory: Path
) -> tuple[TourCapture | None, str | None, list[str]]:
    """Read the frozen observation clock and report its independent defects."""
    blockers: list[str] = []
    try:
        capture = read_trc(request.trc_path)
        clock_sha = hashlib.sha256(capture.time_s.tobytes()).hexdigest()
        (directory / "observation_clock.json").write_text(
            json.dumps(capture.time_s.tolist(), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if (
            capture.time_s[0] != request.config.t_start_s
            or capture.time_s[-1] != request.config.horizon_s
        ):
            blockers.append("observation-horizon-mismatch")
        if set(capture.labels) != set(request.marker_weights):
            blockers.append("reference-marker-coverage")
        if not bool(np.all(np.any(capture.valid, axis=0))):
            blockers.append("missing-marker-observation-support")
        if capture.frames < 2 or not np.allclose(
            np.diff(capture.time_s),
            np.diff(capture.time_s)[0],
            rtol=0,
            atol=1e-9,
        ):
            blockers.append("irregular-observation-clock")
        return capture, clock_sha, blockers
    except (OSError, ValueError):
        return None, None, ["capture-trc-invalid"]


def prepare_native_moco(
    request: NativeMocoRequest, output_dir: Path
) -> NativeMocoPreparation:
    """Collect all independent, obtainable blockers before any expensive solve."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    wall = time.perf_counter()
    cpu = time.process_time()
    blockers: list[str] = []
    if request.config.allow_unused_references:
        blockers.append("unused-reference-policy")
    if request.config.t_start_s != 0.0:
        blockers.append("nonzero-native-replay-origin")
    actual: dict[str, str | None] = {}
    for label, path, expected in (
        ("model", request.model_path, request.model_sha256),
        ("capture", request.trc_path, request.trc_sha256),
        ("guess", request.states_guess_path, request.states_guess_sha256),
    ):
        try:
            actual[label] = _sha(path)
        except OSError:
            actual[label] = None
        if actual[label] != expected:
            blockers.append(f"{label}-sha256-mismatch")
    capture = None
    clock_sha = None
    registered_sha = None
    if actual["capture"] == request.trc_sha256:
        capture, clock_sha, capture_blockers = _load_observation_capture(
            request, directory
        )
        blockers.extend(capture_blockers)
    if request.registration is None:
        blockers.append("capture-registration-unavailable")
    elif capture is not None and "irregular-observation-clock" not in blockers:
        try:
            transformed = register_points(capture.points_m, request.registration)
            registered = TourCapture(
                capture.time_s, capture.labels, transformed, capture.valid
            )
            output = write_trc(
                registered,
                directory / "registered.trc",
                rate_hz=1.0 / float(np.diff(capture.time_s)[0]),
            )
            if not np.array_equal(read_trc(output).time_s, capture.time_s):
                blockers.append("registration-changed-observation-clock")
            registered_sha = _sha(output)
        except (OSError, ValueError):
            blockers.append("capture-registration-failed")
    if set(request.marker_bindings) != set(request.marker_weights):
        blockers.append("marker-binding-coverage")
    loaded = reference_sha = passive_sha = None
    if actual["model"] == request.model_sha256:
        try:
            loaded, reference_sha, passive_sha, native_blockers = _native_preparation(
                request
            )
            blockers.extend(native_blockers)
        except (OSError, ValueError, RuntimeError):
            blockers.append("native-model-preparation-unavailable")
    report = NativeMocoPreparation(
        request.identity_sha256,
        actual["model"],
        actual["capture"],
        actual["guess"],
        loaded,
        reference_sha,
        passive_sha,
        clock_sha,
        registered_sha,
        len(request.marker_weights),
        capture.frames if capture is not None else 0,
        len(request.excluded_markers),
        tuple(dict.fromkeys(blockers)),
        time.perf_counter() - wall,
        time.process_time() - cpu,
    )
    (directory / "preparation.json").write_text(
        json.dumps(asdict(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


@dataclass(frozen=True)
class NativeMocoSolve:
    """One native optimizer attempt; success alone does not qualify a fit."""

    success: bool
    status: str
    objective: float | None
    control_knot_count: int
    control_order_sha256: str | None
    state_order_sha256: str | None
    marker_weight_sha256: str | None
    solution_sha256: str | None
    wall_seconds: float
    cpu_seconds: float
    qualification: str = "not-qualified-for-muscle-matching"


def _native_goal_weights(study: Any) -> dict[str, float]:
    """Read the native goal's serialized weights, not the Python request object."""
    import opensim as osim

    goal = osim.MocoMarkerTrackingGoal.safeDownCast(
        study.updProblem().updGoal("marker_tracking")
    )
    if goal is None:
        raise ValueError("Native marker tracking goal is unavailable")
    xml = SafeET.fromstring(goal.dump(), forbid_dtd=True, forbid_entities=True)
    items = xml.findall(".//MarkerWeight")
    weights = {item.attrib["name"]: float(item.findtext("weight")) for item in items}
    if len(weights) != len(items):
        raise ValueError("Native marker weights contain duplicate names")
    return weights


def _solution_names(vector: Any) -> tuple[str, ...]:
    if isinstance(vector, (tuple, list)):
        return tuple(str(name) for name in vector)
    return tuple(str(vector.get(i)) for i in range(vector.getSize()))


def _require_current_inputs(
    request: NativeMocoRequest, prepared: NativeMocoPreparation, directory: Path
) -> Path:
    """Reject any source, observation or preparation drift before native work."""
    if request.identity_sha256 != prepared.request_sha256:
        raise ValueError("Native Moco request changed after preparation")
    if not prepared.ready_for_software_solve:
        raise ValueError("Preparation blockers prohibit native Moco solve")
    for path, expected in (
        (request.model_path, request.model_sha256),
        (request.trc_path, request.trc_sha256),
        (request.states_guess_path, request.states_guess_sha256),
    ):
        if _sha(path) != expected:
            raise ValueError("Source, observation or guess changed after preparation")
    registered = directory / "registered.trc"
    if _sha(registered) != prepared.registered_trc_sha256:
        raise ValueError("Registered marker reference changed after preparation")
    if hashlib.sha256(read_trc(request.trc_path).time_s.tobytes()).hexdigest() != (
        prepared.observation_clock_sha256
    ):
        raise ValueError("Original observation clock changed after preparation")
    return registered


def _native_solve_once(
    request: NativeMocoRequest,
    registered: Path,
    directory: Path,
    wall: float,
    cpu: float,
) -> NativeMocoSolve:
    """Run and read back one exact native solver attempt."""
    import opensim as osim

    study = build_moco_study(
        str(request.model_path),
        str(registered),
        str(request.states_guess_path),
        request.config,
        initial_bindings=request.bindings,
        marker_weights=request.marker_weights,
    )
    weights = _native_goal_weights(study)
    if weights != dict(request.marker_weights):
        raise ValueError("Native marker goal did not preserve selected weights")
    weight_sha = hashlib.sha256(
        json.dumps(weights, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_parallel(0)
    solution = study.solve()
    success = bool(solution.success())
    status = str(solution.getStatus())
    objective = float(solution.getObjective()) if success else None
    count = 0
    controls_sha = states_sha = solution_sha = None
    if success:
        grid = np.asarray(solution.getTimeMat(), dtype=float).reshape(-1)
        controls = _solution_names(solution.getControlNames())
        states = _solution_names(solution.getStateNames())
        if (
            grid.size < 2
            or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)
            or not np.isclose(grid[0], request.config.t_start_s, atol=1e-12, rtol=0)
            or not np.isclose(grid[-1], request.config.horizon_s, atol=1e-12, rtol=0)
            or set(controls) != set(request.bindings.control_bounds)
            or set(states) != set(request.bindings.state_bounds)
        ):
            raise ValueError("Native solution names or full control horizon differ")
        count = int(grid.size)
        controls_sha = hashlib.sha256(json.dumps(controls).encode()).hexdigest()
        states_sha = hashlib.sha256(json.dumps(states).encode()).hexdigest()
        path = directory / "solution.sto"
        solution.write(str(path))
        solution_sha = _sha(path)
    result = NativeMocoSolve(
        success,
        status,
        objective,
        count,
        controls_sha,
        states_sha,
        weight_sha,
        solution_sha,
        time.perf_counter() - wall,
        time.process_time() - cpu,
    )
    return result


def solve_native_moco(
    request: NativeMocoRequest,
    prepared: NativeMocoPreparation,
    output_dir: Path,
) -> NativeMocoSolve:
    """Solve prepared inputs, retaining failed native attempts as receipts."""
    directory = Path(output_dir)
    registered = _require_current_inputs(request, prepared, directory)
    wall = time.perf_counter()
    cpu = time.process_time()
    try:
        result = _native_solve_once(request, registered, directory, wall, cpu)
    except (RuntimeError, ValueError) as error:
        result = NativeMocoSolve(
            False,
            f"native-error:{type(error).__name__}",
            None,
            0,
            None,
            None,
            None,
            None,
            time.perf_counter() - wall,
            time.process_time() - cpu,
        )
    payload = json.dumps(asdict(result), indent=2, allow_nan=False) + "\n"
    attempts = directory / "solve_attempts"
    attempts.mkdir(exist_ok=True)
    for number in range(1, 1_000_001):
        try:
            with (attempts / f"{number:06d}.json").open(
                "x", encoding="utf-8"
            ) as stream:
                stream.write(payload)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Native solve attempt receipt space exhausted")
    (directory / "solve.json").write_text(payload, encoding="utf-8")
    return result


__all__ = [
    "NativeMocoPreparation",
    "NativeMocoRequest",
    "NativeMocoSolve",
    "prepare_native_moco",
    "solve_native_moco",
]
