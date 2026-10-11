"""Bounded, process-isolated orchestration for synthetic native Moco rungs.

This module deliberately does not claim a Moco solve.  Each rung is executed
by a separately launched worker, and a rung can be marked successful only by
the worker's hash-bound receipt after it has completed fresh replay.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from defusedxml import ElementTree as ET


@dataclass(frozen=True)
class LadderRung:
    """One cumulative model feature in the synthetic complexity ladder."""

    name: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("rung name must be a simple identifier")
        if len(set(self.features)) != len(self.features):
            raise ValueError("rung features must be unique")


DEFAULT_RUNGS = (
    LadderRung("plain", ()),
    LadderRung("nonlinear_joint", ("nonlinear_custom_joint",)),
    LadderRung("moving_path", ("nonlinear_custom_joint", "nonlinear_mpp")),
    LadderRung(
        "conditional_path",
        ("nonlinear_custom_joint", "nonlinear_mpp", "conditional_path_point"),
    ),
    LadderRung(
        "wrap",
        (
            "nonlinear_custom_joint",
            "nonlinear_mpp",
            "conditional_path_point",
            "intersecting_wrap",
        ),
    ),
    LadderRung(
        "coupler",
        (
            "nonlinear_custom_joint",
            "nonlinear_mpp",
            "conditional_path_point",
            "intersecting_wrap",
            "coordinate_coupler",
        ),
    ),
)


def build_synthetic_rung(rung: LadderRung, directory: Path):
    """Build and reinitialize a cumulative synthetic native model rung.

    The existing constrained-muscle fixture supplies the two muscles and
    bodies. This builder writes a separate source file, then applies exactly
    the features named by ``rung`` before deriving a fresh named state.
    """
    from tests.opensim.test_native_constrained_muscle import _source

    _validate_rung_features(rung)
    directory.mkdir(parents=True, exist_ok=True)
    path, _ = _source(
        directory,
        locked=False,
        custom="nonlinear_custom_joint" in rung.features,
        moving="nonlinear_mpp" in rung.features,
        wrapped="intersecting_wrap" in rung.features,
    )
    _edit_synthetic_source(path, rung)
    return _derive_synthetic_cold_start(path, rung)


def _validate_rung_features(rung: LadderRung) -> None:
    allowed = {
        "nonlinear_custom_joint",
        "nonlinear_mpp",
        "conditional_path_point",
        "intersecting_wrap",
        "coordinate_coupler",
    }
    if not set(rung.features) <= allowed:
        raise ValueError("rung declares an unsupported synthetic feature")


def _edit_synthetic_source(path: Path, rung: LadderRung) -> None:
    tree = ET.parse(path)
    if "coordinate_coupler" not in rung.features:
        constraints = tree.find(".//ConstraintSet/objects")
        if constraints is None:
            raise ValueError("synthetic source has no constraint object set")
        for constraint in tuple(constraints):
            constraints.remove(constraint)
    else:
        coupler = tree.find(".//CoordinateCouplerConstraint[@name='coupler']")
        if coupler is None:
            raise ValueError("synthetic source omitted its declared coupler")

    if "nonlinear_custom_joint" in rung.features:
        axis = tree.find(
            ".//CustomJoint[@name='base']/SpatialTransform/"
            "TransformAxis[@name='rotation1']"
        )
        if axis is None:
            raise ValueError("custom-joint rung has no rotation1 transform axis")
        law = axis.find("LinearFunction")
        if law is None:
            raise ValueError("custom-joint rung has no replaceable linear law")
        axis.remove(law)
        axis.append(
            ET.fromstring(
                "<SimmSpline name='function'><x>-1 0 1</x>"
                "<y>-0.2 0 0.3</y></SimmSpline>"
            )
        )

    if "nonlinear_mpp" in rung.features:
        guide = tree.find(".//MovingPathPoint[@name='guide']")
        if guide is None:
            raise ValueError("moving-path rung has no guide point")
        x_location = guide.find("x_location")
        if x_location is None or len(x_location) != 1:
            raise ValueError("moving-path rung has no single x location law")
        x_location.remove(x_location[0])
        x_location.append(
            ET.fromstring(
                "<SimmSpline name='function'><x>-1 0 1</x>"
                "<y>-0.008 0 0.012</y></SimmSpline>"
            )
        )

    if "conditional_path_point" in rung.features:
        points = tree.find(
            ".//Millard2012EquilibriumMuscle[@name='flexor']//PathPointSet/objects"
        )
        insertion = (
            None if points is None else points.find("PathPoint[@name='insertion']")
        )
        if points is None or insertion is None:
            raise ValueError("conditional-path rung has no flexor insertion")
        points.remove(insertion)
        points.append(
            ET.fromstring(
                "<ConditionalPathPoint name='insertion'>"
                "<socket_parent_frame>/bodyset/base</socket_parent_frame>"
                "<location>0 0.15 0</location>"
                "<socket_coordinate>/jointset/base/q</socket_coordinate>"
                "<range>0.2 0.4</range></ConditionalPathPoint>"
            )
        )

    if "intersecting_wrap" in rung.features:
        # The original helper places a cylinder at the ground origin across
        # the flexor path. Move it onto the first segment and keep only that
        # path's wrap to add one ingredient.
        cylinder_translation = tree.find(
            ".//WrapCylinder[@name='flexor_cylinder']/translation"
        )
        if cylinder_translation is None:
            raise ValueError("synthetic flexor wrap has no translation")
        cylinder_translation.text = "0.03 -0.12 0"
        extensor = tree.find(
            ".//Millard2012EquilibriumMuscle[@name='extensor']/GeometryPath/PathWrapSet/objects"
        )
        if extensor is not None:
            for item in tuple(extensor):
                extensor.remove(item)
        extensor_wrap = tree.find(".//WrapCylinder[@name='extensor_cylinder']")
        if extensor_wrap is not None:
            for parent in tree.iter():
                if extensor_wrap in list(parent):
                    parent.remove(extensor_wrap)
                    break

    tree.write(path, encoding="utf-8", xml_declaration=True)


def _derive_synthetic_cold_start(path: Path, rung: LadderRung):
    import hashlib
    import opensim as osim

    from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
        DeclaredColdStart,
    )

    model = osim.Model(str(path))
    state = model.initSystem()
    if "nonlinear_mpp" in rung.features:
        _verify_native_mpp_curvature(model, state, osim)
    for index in range(model.getMuscles().getSize()):
        model.getMuscles().get(index).setActivation(state, 0.05)
    model.equilibrateMuscles(state)
    names = model.getStateVariableNames()
    named_state = {
        str(names.get(index)): float(
            model.getStateVariableValue(state, names.get(index))
        )
        for index in range(names.getSize())
    }
    source_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    enforcement = (
        {"/constraintset/coupler": True}
        if "coordinate_coupler" in rung.features
        else {}
    )
    declaration = DeclaredColdStart(
        model_path=path,
        source_sha256=source_digest,
        named_state=named_state,
        time_seconds=0.0,
        lock_targets={},
        chart_bounds={"/jointset/base/q": (0.2, 0.4)},
        constraint_enforcement=enforcement,
        residual_tolerance=1e-9,
    )
    return path, declaration


def _verify_native_mpp_curvature(model: Any, state: Any, osim: Any) -> None:
    point_set = model.getMuscles().get("flexor").getGeometryPath().getPathPointSet()
    guide = next(
        osim.MovingPathPoint.safeDownCast(point_set.get(index))
        for index in range(point_set.getSize())
        if point_set.get(index).getName() == "guide"
    )
    coordinate = model.getCoordinateSet().get("q")
    original_q = coordinate.getValue(state)
    derivatives = []
    step = 1e-4
    try:
        for center in (-0.5, 0.5):
            coordinate.setValue(state, center - step, False)
            model.realizePosition(state)
            before = guide.getLocation(state).get(0)
            coordinate.setValue(state, center + step, False)
            model.realizePosition(state)
            after = guide.getLocation(state).get(0)
            derivatives.append((after - before) / (2.0 * step))
    finally:
        coordinate.setValue(state, original_q, False)
        model.realizePosition(state)
    if not all(math.isfinite(value) and value > 0 for value in derivatives):
        raise ValueError("native moving-path spline is not strictly increasing")
    if abs(derivatives[1] - derivatives[0]) <= 1e-5:
        raise ValueError("native moving-path spline has no resolved curvature")


@dataclass(frozen=True)
class RungResult:
    name: str
    status: str
    receipt_sha256: str | None
    elapsed_seconds: float
    detail: str
    peak_sampled_tree_rss_bytes: int | None = None


class _WorkerTimeout(TimeoutError):
    def __init__(
        self, stdout: str, stderr: str, stop_reason: str, peak_rss: int
    ) -> None:
        super().__init__(f"worker process tree stopped: {stop_reason}")
        self.stdout = stdout
        self.stderr = stderr
        self.stop_reason = stop_reason
        self.peak_rss = peak_rss


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    created_at: float
    parent_pid: int
    parent_created_at: float | None
    depth: int


def _supervise_process_tree(
    command: Sequence[str],
    cwd: Path,
    timeout_seconds: float,
    memory_limit_bytes: int,
    *,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str, int]:
    """Run and account for only this rung's PID/creation-time-owned tree."""
    import psutil

    started = time.monotonic()
    process_options: dict[str, Any] = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env={**os.environ, **env} if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **process_options,
    )
    root = psutil.Process(process.pid)
    owned = {
        root.pid: _ProcessIdentity(
            root.pid,
            root.create_time(),
            root.ppid(),
            _parent_created_time(root, psutil),
            0,
        )
    }
    output: list[bytes] = []
    errors: list[bytes] = []
    readers = [
        threading.Thread(
            target=_drain_pipe, args=(process.stdout, output), daemon=True
        ),
        threading.Thread(
            target=_drain_pipe, args=(process.stderr, errors), daemon=True
        ),
    ]
    for reader in readers:
        reader.start()
    peak_rss = 0
    stop_reason = None
    while True:
        process.poll()  # Reap a launcher that exited after creating its child.
        _record_owned_descendants(owned, psutil)
        active, rss = _owned_tree_rss(owned, psutil)
        peak_rss = max(peak_rss, rss)
        if active == 0:
            break
        stop_reason = _process_stop_reason(
            time.monotonic() - started, rss, timeout_seconds, memory_limit_bytes
        )
        if stop_reason is not None:
            break
        time.sleep(0.01)
    if stop_reason is not None:
        _terminate_owned_tree(owned, psutil)
    process.wait(timeout=5)
    for reader in readers:
        reader.join(timeout=5)
    stdout = b"".join(output).decode(errors="replace")
    stderr = b"".join(errors).decode(errors="replace")
    if stop_reason is not None:
        raise _WorkerTimeout(stdout, stderr, stop_reason, peak_rss)
    return process.returncode, stdout, stderr, peak_rss


def _process_stop_reason(
    elapsed_seconds: float,
    sampled_tree_rss_bytes: int,
    timeout_seconds: float,
    sampled_rss_threshold_bytes: int,
) -> str | None:
    if sampled_tree_rss_bytes >= sampled_rss_threshold_bytes:
        return "sampled_rss_threshold"
    if elapsed_seconds >= timeout_seconds:
        return "wall_timeout"
    return None


def _parent_created_time(process: Any, psutil: Any) -> float | None:
    try:
        return psutil.Process(process.ppid()).create_time()
    except psutil.Error:
        return None


def _record_owned_descendants(owned: dict[int, _ProcessIdentity], psutil: Any) -> None:
    """Grow the PID set only through a still-verified owned parent identity."""
    for parent_pid, identity in tuple(owned.items()):
        try:
            parent = psutil.Process(parent_pid)
            if parent.create_time() != identity.created_at:
                continue
            for child in parent.children(recursive=False):
                created = child.create_time()
                if child.pid not in owned:
                    owned[child.pid] = _ProcessIdentity(
                        child.pid,
                        created,
                        parent_pid,
                        identity.created_at,
                        identity.depth + 1,
                    )
        except psutil.Error:
            continue


def _owned_tree_rss(owned: dict[int, _ProcessIdentity], psutil: Any) -> tuple[int, int]:
    active = 0
    rss = 0
    for identity in owned.values():
        try:
            process = psutil.Process(identity.pid)
            if process.create_time() == identity.created_at:
                active += 1
                rss += process.memory_info().rss
        except psutil.Error:
            continue
    return active, rss


def _terminate_owned_tree(owned: dict[int, _ProcessIdentity], psutil: Any) -> None:
    """Terminate descendants after exact PID creation-time validation."""
    pending = []
    for identity in sorted(owned.values(), key=lambda item: item.depth, reverse=True):
        try:
            process = psutil.Process(identity.pid)
            if process.create_time() == identity.created_at:
                process.kill()
                pending.append(process)
        except psutil.Error:
            continue
    _, alive = psutil.wait_procs(pending, timeout=5)
    still_owned = []
    for process in alive:
        identity = owned.get(process.pid)
        try:
            if identity is not None and process.create_time() == identity.created_at:
                still_owned.append(process.pid)
        except psutil.Error:
            continue
    if still_owned:
        raise RuntimeError(
            f"owned native worker processes survived termination: {still_owned}"
        )


def _drain_pipe(stream: Any, chunks: list[bytes]) -> None:
    if stream is None:
        return
    with stream:
        while chunk := stream.read(65536):
            chunks.append(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_success_receipt(directory: Path, rung: LadderRung) -> str:
    """Require worker evidence for solve, saved controls, and fresh replay."""
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError("worker did not write receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "rung": rung.name,
        "status": "solved_and_replayed",
        "fresh_model_replay": True,
        "nonzero_target_reached": True,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"worker receipt has invalid {key}")
    request, replay = _validate_retained_artifacts(directory, rung, receipt)
    _validate_replay_result(request, receipt, replay)
    _validate_stage_metrics(receipt)
    return _sha256(receipt_path)


def _validate_retained_artifacts(
    directory: Path, rung: LadderRung, receipt: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    for field in (
        "source_sha256",
        "worker_sha256",
        "harness_builder_sha256",
        "witness_provider_sha256",
        "witness_states_sha256",
        "witness_controls_sha256",
        "opensim_runtime_fingerprint_sha256",
        "request_sha256",
        "saved_solution_sha256",
        "saved_time_only_controls_sha256",
        "solution_serialization_audit_sha256",
        "fresh_replay_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"worker receipt is missing a valid {field}")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"worker receipt has invalid {field}") from exc
    if not (directory / "source.osim").is_file():
        raise ValueError("worker did not preserve its source model")
    if _sha256(directory / "source.osim") != receipt["source_sha256"]:
        raise ValueError("worker source hash does not match preserved model")
    file_hashes = {
        "request.json": "request_sha256",
        "witness/witness_states.sto": "witness_states_sha256",
        "witness/witness_time_only_controls.sto": "witness_controls_sha256",
        "solution.sto": "saved_solution_sha256",
        "time_only_controls.sto": "saved_time_only_controls_sha256",
        "solution_serialization_audit.json": "solution_serialization_audit_sha256",
        "fresh_replay.json": "fresh_replay_sha256",
    }
    for filename, field in file_hashes.items():
        path = directory / filename
        if not path.is_file() or _sha256(path) != receipt[field]:
            raise ValueError(f"worker {filename} is missing or hash-mismatched")
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if request.get("source_sha256") != receipt["source_sha256"]:
        raise ValueError("request does not bind the preserved source model")
    expected_worker_sha = _sha256(
        Path(__file__).with_name("native_moco_ladder_worker.py")
    )
    if receipt["worker_sha256"] != expected_worker_sha:
        raise ValueError("worker receipt does not identify this worker source")
    runtime_path = directory / "opensim_runtime_fingerprint.json"
    if not runtime_path.is_file():
        raise ValueError("bounded OpenSim runtime fingerprint is missing")
    _validate_request_rung_binding(request, receipt, rung)
    _validate_witness_request(request, receipt)
    _validate_solution_serialization_audit(
        directory / "solution_serialization_audit.json", request, receipt
    )
    _validate_bounded_runtime_fingerprint(
        runtime_path, receipt["opensim_runtime_fingerprint_sha256"]
    )
    replay = json.loads((directory / "fresh_replay.json").read_text(encoding="utf-8"))
    if (
        replay.get("fresh_model") is not True
        or replay.get("input_kind") != "time_only_controls"
    ):
        raise ValueError("fresh replay evidence is not bound to time-only controls")
    if replay.get("physical_qualification") != "unqualified":
        raise ValueError("synthetic software replay must remain physically unqualified")
    return request, replay


def _validate_replay_result(
    request: dict[str, Any], receipt: dict[str, Any], replay: dict[str, Any]
) -> None:
    if not isinstance(request.get("target_q"), (int, float)) or isinstance(
        request.get("target_q"), bool
    ):
        raise ValueError("request has no finite target coordinate")
    target_q = request["target_q"]
    if not math.isfinite(target_q):
        raise ValueError("request has no finite target coordinate")
    initial_q = request.get("initial_q")
    if (
        isinstance(initial_q, bool)
        or not isinstance(initial_q, (int, float))
        or not math.isfinite(initial_q)
        or (target_q - initial_q) == 0
    ):
        raise ValueError("request has no distinct finite initial coordinate")
    if isinstance(receipt.get("target_q"), bool) or receipt.get("target_q") != target_q:
        raise ValueError("receipt target differs from retained request")
    if (
        isinstance(receipt.get("initial_q"), bool)
        or receipt.get("initial_q") != initial_q
    ):
        raise ValueError("receipt initial coordinate differs from retained request")
    if not isinstance(receipt.get("achieved_q"), (int, float)) or isinstance(
        receipt.get("achieved_q"), bool
    ):
        raise ValueError("receipt optimized endpoint is invalid")
    if (
        not math.isfinite(receipt["achieved_q"])
        or abs(receipt["achieved_q"] - target_q) >= 2e-7
    ):
        raise ValueError("optimized solution endpoint misses requested target")
    if (
        not isinstance(receipt.get("objective"), (int, float))
        or isinstance(receipt.get("objective"), bool)
        or not math.isfinite(receipt["objective"])
    ):
        raise ValueError("worker receipt is missing its solver objective")
    replay_errors = replay.get("max_state_error_by_unit")
    expected_error_limits = {
        "coordinates_rad": 2e-7,
        "speeds_rad_s": 1e-7,
        "activations": 1e-8,
        "fiber_lengths_m": 1e-7,
    }
    if not isinstance(replay_errors, dict) or set(replay_errors) != set(
        expected_error_limits
    ):
        raise ValueError("fresh replay receipt is missing per-unit state errors")
    if any(
        isinstance(replay_errors[key], bool)
        or not isinstance(replay_errors[key], (int, float))
        or not math.isfinite(replay_errors[key])
        or replay_errors[key] >= limit
        for key, limit in expected_error_limits.items()
    ):
        raise ValueError("fresh replay state error exceeds positive-fixture limits")
    if replay.get("target_endpoint_within_tolerance") is not True:
        raise ValueError("fresh replay did not reach the requested endpoint")
    for field in ("optimized_endpoint_q", "replayed_endpoint_q"):
        value = replay.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or abs(value - target_q) >= 2e-7
        ):
            raise ValueError(f"fresh replay has invalid {field}")
        if (value - initial_q) * (target_q - initial_q) <= 0:
            raise ValueError(f"fresh replay {field} moves in the wrong direction")
        if abs(value - initial_q) <= 2e-6:
            raise ValueError(f"fresh replay {field} misses nonzero motion")


def _validate_stage_metrics(receipt: dict[str, Any]) -> None:
    metrics = receipt.get("stage_metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("worker receipt is missing per-stage measurements")
    for stage in metrics.values():
        if not isinstance(stage, dict) or any(
            isinstance(stage.get(field), bool)
            or not isinstance(stage.get(field), (int, float))
            or not math.isfinite(stage[field])
            or stage[field] < 0
            for field in ("wall_seconds", "cpu_seconds")
        ):
            raise ValueError("worker stage measurements must contain wall and CPU time")


def _validate_request_rung_binding(
    request: dict[str, Any], receipt: dict[str, Any], rung: LadderRung
) -> None:
    if (
        request.get("rung") != rung.name
        or request.get("features") != list(rung.features)
        or receipt.get("features") != list(rung.features)
    ):
        raise ValueError("request or receipt does not bind the declared rung features")


def _validate_witness_request(request: dict[str, Any], receipt: dict[str, Any]) -> None:
    cold_state = request.get("cold_start_named_state")
    units = request.get("cold_start_state_units")
    profile = request.get("native_mechanics_profile")
    if (
        not isinstance(cold_state, dict)
        or not cold_state
        or set(units or {}) != set(cold_state)
        or not isinstance(profile, dict)
        or set(profile) != {"schema_version", "source_sha256", "components"}
        or profile.get("schema_version") != "native-sdk-mechanics-v1"
        or profile.get("source_sha256") != receipt["source_sha256"]
        or not isinstance(profile.get("components"), list)
    ):
        raise ValueError("witness request omits its complete source-bound cold state")
    profile_payload = {
        "schema_version": profile.get("schema_version"),
        "source_sha256": profile.get("source_sha256"),
        "components": profile.get("components"),
    }
    profile_digest = hashlib.sha256(
        json.dumps(profile_payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    if request.get("native_mechanics_profile_sha256") != profile_digest:
        raise ValueError("witness request does not bind the exact native profile")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in cold_state.values()
    ):
        raise ValueError("witness cold state contains an invalid numeric value")
    expected_units = {
        name: (
            "rad"
            if name.endswith("/value")
            else "rad/s"
            if name.endswith("/speed")
            else "1"
            if name.endswith("/activation")
            else "m"
            if name.endswith("/fiber_length")
            else None
        )
        for name in cold_state
    }
    if any(
        units.get(name) != unit or unit is None for name, unit in expected_units.items()
    ):
        raise ValueError("witness cold-state units do not cover native state names")
    if (
        request.get("cold_start_time_seconds") != 0.0
        or not isinstance(request.get("cold_start_lock_targets"), dict)
        or not isinstance(request.get("cold_start_chart_bounds"), dict)
        or not isinstance(request.get("cold_start_constraint_enforcement"), dict)
        or request.get("cold_start_residual_tolerance") != 1e-9
        or request.get("initial_q") != cold_state.get("/jointset/base/q/value")
        or request.get("witness_profile") != "constant-0.05-forward-integrated-v1"
        or request.get("solution_serialization_max_ulps") != 8
        or request.get("witness_excitation_by_channel")
        != {"flexor": 0.05, "extensor": 0.05}
    ):
        raise ValueError("witness request omits its explicit synthetic policy")
    if request.get("witness_source_sha256") != receipt["source_sha256"]:
        raise ValueError("forward witness does not bind the request source")
    bindings = (
        ("witness_states_sha256", "witness_states_sha256"),
        ("witness_controls_sha256", "witness_controls_sha256"),
        ("witness_provider_sha256", "witness_provider_sha256"),
        ("witness_worker_sha256", "worker_sha256"),
        ("witness_builder_sha256", "harness_builder_sha256"),
    )
    if any(request.get(left) != receipt.get(right) for left, right in bindings):
        raise ValueError(
            "witness implementation or artifact hashes differ from receipt"
        )
    expected_builder = _sha256(Path(__file__))
    provider_path = (
        Path(__file__).resolve().parents[2]
        / "src/engines/physics_engines/opensim/python/tour_matching/native_constrained_muscle.py"
    )
    if receipt.get("harness_builder_sha256") != expected_builder:
        raise ValueError("witness receipt does not identify this harness builder")
    if receipt.get("witness_provider_sha256") != _sha256(provider_path):
        raise ValueError("witness receipt does not identify the replay provider")
    knots = request.get("witness_time_knots")
    if (
        not isinstance(knots, list)
        or len(knots) != 11
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in knots
        )
        or knots[0] != 0.0
        or knots[-1] != 0.01
        or any(right <= left for left, right in zip(knots, knots[1:], strict=False))
    ):
        raise ValueError("witness request does not retain its exact full 10 ms clock")


def _validate_solution_serialization_audit(
    path: Path, request: dict[str, Any], receipt: dict[str, Any]
) -> None:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError("solution STO round-trip audit is not an object")
    arrays = audit.get("arrays")
    if (
        audit.get("schema") != "synthetic-solution-sto-roundtrip-v1"
        or request.get("solution_serialization_max_ulps") != 8
        or audit.get("maximum_allowed_ulps") != 8
        or not isinstance(audit.get("maximum_ulp_error"), (int, float))
        or isinstance(audit.get("maximum_ulp_error"), bool)
        or not math.isfinite(audit["maximum_ulp_error"])
        or audit["maximum_ulp_error"] > 8
        or receipt.get("solution_serialization_audit") != audit
        or not isinstance(arrays, dict)
        or set(arrays) != {"time", "states", "controls"}
        or not all(isinstance(array, dict) for array in arrays.values())
        or not isinstance(audit.get("state_names"), list)
        or not all(isinstance(name, str) for name in audit["state_names"])
        or not isinstance(audit.get("control_names"), list)
        or not all(isinstance(name, str) for name in audit["control_names"])
        or set(audit.get("state_names", []))
        != set(request.get("cold_start_named_state", {}))
        or set(audit.get("control_names", []))
        != {
            f"/forceset/{name}"
            for name in request.get("witness_excitation_by_channel", {})
        }
    ):
        raise ValueError("solution STO round-trip audit is missing or over budget")
    if any(
        not isinstance(array.get(field), str)
        or len(array[field]) != 64
        or any(char not in "0123456789abcdef" for char in array[field])
        for array in arrays.values()
        for field in ("in_memory_sha256", "persisted_sha256")
    ):
        raise ValueError("solution STO round-trip audit hashes are invalid")
    for array in arrays.values():
        for field in ("max_absolute_error", "max_ulp_error"):
            value = array.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError("solution STO round-trip audit metrics are invalid")
        if array["max_ulp_error"] > 8:
            raise ValueError("solution STO round-trip array exceeds its ULP budget")
    if audit["maximum_ulp_error"] != max(
        array["max_ulp_error"] for array in arrays.values()
    ):
        raise ValueError("solution STO round-trip maximum does not match arrays")


def _validate_bounded_runtime_fingerprint(path: Path, expected_sha256: str) -> None:
    runtime = json.loads(path.read_text(encoding="utf-8"))
    if (
        runtime.get("schema") != "opensim-bounded-runtime-fingerprint-v1"
        or not isinstance(runtime.get("version"), str)
        or runtime.get("coverage_note")
        != "bounded Python extension and Moco DLL fingerprint; not a closed dependency-runtime identity"
        or not isinstance(runtime.get("components"), dict)
        or set(runtime["components"])
        != {"_common.pyd", "_simulation.pyd", "_moco.pyd", "osimMoco.dll"}
        or _sha256(path) != expected_sha256
    ):
        raise ValueError("bounded OpenSim runtime fingerprint does not match receipt")
    if any(
        not isinstance(component_sha, str)
        or len(component_sha) != 64
        or any(char not in "0123456789abcdef" for char in component_sha)
        for component_sha in runtime["components"].values()
    ):
        raise ValueError("runtime fingerprint component hash is invalid")


def worker_command(
    rung: LadderRung,
    directory: Path,
    python_executable: str = sys.executable,
    mesh_intervals: int = 10,
) -> tuple[str, ...]:
    """Return the isolated worker command for one rung."""
    mesh_intervals = _validate_mesh_intervals(mesh_intervals)
    command = (
        python_executable,
        "-m",
        "tests.opensim.native_moco_ladder_worker",
        "--rung",
        rung.name,
        "--output",
        str(directory),
    )
    if mesh_intervals != 10:
        command += ("--mesh-intervals", str(mesh_intervals))
    return command


def _validate_mesh_intervals(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("mesh interval count must be an integer in [1, 100]")
    return value


def run_ladder(
    rungs: Sequence[LadderRung],
    command_for: Callable[[LadderRung, Path], Sequence[str]],
    output_root: Path,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[RungResult, ...]:
    """Run workers serially and stop on the first failure or timeout.

    Each worker receives a new output directory. A nonzero exit, timeout, or
    incomplete/mismatched evidence is terminal for this ladder invocation.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > 30.0
    ):
        raise ValueError("per-rung timeout must be in (0, 30] seconds")
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[RungResult] = []
    for rung in rungs:
        directory = output_root / rung.name
        directory.mkdir(parents=False, exist_ok=False)
        command = tuple(command_for(rung, directory))
        if not command:
            raise ValueError("worker command must not be empty")
        result = _run_one_rung(rung, directory, command, timeout_seconds)
        results.append(result)
        if result.status != "passed":
            break
    return tuple(results)


def _run_one_rung(
    rung: LadderRung, directory: Path, command: Sequence[str], timeout_seconds: float
) -> RungResult:
    started = time.monotonic()
    peak_tree_rss = None
    try:
        return_code, stdout, stderr, peak_tree_rss = _supervise_process_tree(
            command,
            Path(__file__).resolve().parents[2],
            timeout_seconds,
            1024 * 1024 * 1024,
        )
    except (subprocess.TimeoutExpired, _WorkerTimeout) as exc:
        _write_worker_logs(
            directory, getattr(exc, "stdout", ""), getattr(exc, "stderr", "")
        )
        memory_stop = (
            isinstance(exc, _WorkerTimeout)
            and exc.stop_reason == "sampled_rss_threshold"
        )
        return RungResult(
            rung.name,
            "memory_limit" if memory_stop else "timeout",
            None,
            time.monotonic() - started,
            (
                "sampled aggregate worker-tree RSS reached the soft 1 GiB threshold"
                if memory_stop
                else f"worker process tree exceeded {timeout_seconds:g}s wall time"
            ),
            getattr(exc, "peak_rss", None),
        )
    elapsed = time.monotonic() - started
    _write_worker_logs(directory, stdout, stderr)
    if return_code != 0:
        return RungResult(
            rung.name,
            "worker_failed",
            None,
            elapsed,
            f"worker exited {return_code}",
            peak_tree_rss,
        )
    try:
        digest = _validate_success_receipt(directory, rung)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return RungResult(
            rung.name,
            "invalid_evidence",
            None,
            elapsed,
            str(exc),
            peak_tree_rss,
        )
    return RungResult(rung.name, "passed", digest, elapsed, "", peak_tree_rss)


def run_retained_replay_diagnostic(
    source_run: Path,
    output_root: Path,
    *,
    worker_python: str,
    worker_environment: dict[str, str],
) -> dict[str, Any]:
    """Run one retained-input replay under the existing owned-tree supervisor."""
    if output_root.exists():
        raise FileExistsError(output_root)
    allowed_environment = {
        "CASADIPATH",
        "PYTHONPATH",
        "MYPYPATH",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    }
    if set(worker_environment) - allowed_environment:
        raise ValueError("worker environment contains an unapproved variable")
    output_root.mkdir(parents=True, exist_ok=False)
    source_root, command, invocation = _retained_replay_invocation(
        source_run, output_root, worker_python, worker_environment
    )
    (output_root / "invocation.json").write_text(
        json.dumps(invocation, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result: dict[str, Any]
    try:
        return_code, stdout, stderr, peak_rss = _supervise_process_tree(
            command,
            source_root,
            30.0,
            1024**3,
            env=worker_environment,
        )
        _write_diagnostic_supervisor_logs(output_root, stdout, stderr)
        result = {
            "status": "worker_exit" if return_code else "worker_returned",
            "return_code": return_code,
            "stop_reason": None,
            "peak_sampled_tree_rss_bytes": peak_rss,
            "owned_tree_cleanup_verified": True,
        }
    except _WorkerTimeout as exc:
        _write_diagnostic_supervisor_logs(output_root, exc.stdout, exc.stderr)
        result = {
            "status": exc.stop_reason,
            "return_code": None,
            "stop_reason": exc.stop_reason,
            "peak_sampled_tree_rss_bytes": exc.peak_rss,
            "owned_tree_cleanup_verified": True,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        result = {
            "status": "supervisor_error",
            "return_code": None,
            "stop_reason": type(exc).__name__,
            "peak_sampled_tree_rss_bytes": None,
            "owned_tree_cleanup_verified": False,
        }
        _write_diagnostic_supervisor_logs(output_root, "", repr(exc))
    (output_root / "supervisor_result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _retained_replay_invocation(
    source_run: Path,
    output_root: Path,
    worker_python: str,
    worker_environment: dict[str, str],
) -> tuple[Path, list[str], dict[str, Any]]:
    worker_output = output_root / "native_audit"
    source_root = Path(__file__).resolve().parents[2]
    worker_path = source_root / "tests/opensim/native_moco_ladder_worker.py"
    input_names = (
        "source.osim",
        "request.json",
        "solution.sto",
        "time_only_controls.sto",
    )
    command = [
        worker_python,
        "-m",
        "tests.opensim.native_moco_ladder_worker",
        "--replay-existing",
        str(source_run.resolve()),
        "--output",
        str(worker_output.resolve()),
    ]
    invocation = {
        "schema": "native-retained-replay-invocation-v1",
        "worker_command": command,
        "working_directory": str(source_root),
        "source_run": str(source_run.resolve()),
        "input_sha256": {name: _sha256(source_run / name) for name in input_names},
        "worker_sha256": _sha256(worker_path),
        "harness_sha256": _sha256(Path(__file__)),
        "nonsecret_environment": dict(sorted(worker_environment.items())),
        "timeout_seconds": 30.0,
        "sampled_rss_soft_threshold_bytes": 1024**3,
        "optimizer_invoked": False,
    }
    return source_root, command, invocation


def _write_diagnostic_supervisor_logs(
    output_root: Path, stdout: str, stderr: str
) -> None:
    (output_root / "supervisor.stdout.log").write_text(stdout, encoding="utf-8")
    (output_root / "supervisor.stderr.log").write_text(stderr, encoding="utf-8")


def _write_worker_logs(directory: Path, stdout: Any, stderr: Any) -> None:
    (directory / "worker.stdout.log").write_text(
        stdout if isinstance(stdout, str) else "", encoding="utf-8"
    )
    (directory / "worker.stderr.log").write_text(
        stderr if isinstance(stderr, str) else "", encoding="utf-8"
    )
