"""File-driven native Moco preparation and independent muscle replay.

This runner joins existing providers without granting scientific qualification.
Synthetic fixtures can prove the software path; real-source blockers remain
visible before an optimizer is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
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
from .native_moco_request import NativeMocoRequest
from .native_capability_probe import (
    NativeProbeFailure,
    _native_runtime_identity,
    _native_short_capability_probe,
)
from .native_passive_readiness import (
    PassiveReadinessPolicy,
    audit_native_passive_readiness,
)
from .native_reference_conventions import (
    ReferenceStateDeclaration,
    audit_source_reference,
)
from .registration import CaptureRegistration, register_points
from .native_prepared_state import DeclaredColdStart, reconstruct_declared_cold_start
from .native_mixed_actuation import MixedActuationProfile
from .native_body_inertia import NativeBodyInertiaAudit, audit_native_body_inertia
from .native_diagnostic_contract import (
    DiagnosticCode,
    NativeDiagnosticAssessment,
    NativeDiagnosticFinding,
    NativeDiagnosticPolicy,
    NativeQualificationContext,
    RuntimePhase,
    RuntimeViability,
    diagnostic_effects,
)
from .native_wrapped_path import (
    WrappedPathDeclaration,
    audit_native_wrapped_path_source,
)
from .trc import read_trc, write_trc
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.estimation import dime_global_calibration


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    physical_body_inertia_sha256: str | None
    physical_body_inertia_qualified: bool
    marker_count: int
    observation_count: int
    excluded_marker_count: int
    blockers: tuple[str, ...]
    wall_seconds: float
    cpu_seconds: float
    qualification: str = "not-qualified-for-muscle-matching"
    allow_unqualified_passive_diagnostic: bool = False
    diagnostic_assessment: NativeDiagnosticAssessment | None = None

    @property
    def ready_for_software_solve(self) -> bool:
        """A necessary technical gate, never anatomy or capture approval."""
        blocker_receipt = hashlib.sha256(
            json.dumps(self.blockers, allow_nan=False).encode()
        ).hexdigest()
        return (
            self.physical_body_inertia_qualified
            and self.physical_body_inertia_sha256 is not None
            and self.diagnostic_assessment is not None
            and self.diagnostic_assessment.blocker_receipt_sha256 == blocker_receipt
            and not (
                DiagnosticCode.PASSIVE_POLICY_UNAVAILABLE
                in self.diagnostic_assessment.allowed_unqualified_findings
                and self.passive_observation_sha256 is None
            )
            and self.diagnostic_assessment.execution_permitted
        )


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


def _audit_native_passive(
    model: Any,
    state: Any,
    policy: Any,
    mixed: MixedActuationProfile | None = None,
    declared: DeclaredColdStart | None = None,
) -> tuple[str | None, list[str]]:
    try:
        if declared is not None:
            from .native_prepared_state import observe_declared_native_sample

            observe_declared_native_sample(model, state, declared)
        if mixed is not None:
            from .native_mixed_actuation import admit_native_mixed_profile

            admit_native_mixed_profile(model, state, mixed, declared)
        passive = audit_native_passive_readiness(model, state, policy)
        blockers = list(passive.blockers)
        blockers.extend(
            item
            for item in passive.native_state.blockers
            if not (mixed is not None and item == "nonmuscle-assistance-unqualified")
            if not (
                declared is not None
                and item
                in {
                    "native-constraint-state-unqualified",
                    "native-lock-target-unverified",
                }
            )
            if item
            not in {
                "complete-native-state-unverified",
                "constraint-residual-acceptance-unqualified",
            }
        )
        return passive.observation_sha256, blockers
    except (ValueError, RuntimeError):
        return None, ["native-passive-or-constraint-observation-unavailable"]


def _audit_reference_convention(
    request: NativeMocoRequest,
) -> tuple[str | None, list[str]]:
    try:
        reference = _reference_state_declaration(request)
        observed = audit_source_reference(
            request.model_path, reference, request.reference_frame_path
        )
        return observed.observation_sha256, []
    except (ValueError, RuntimeError):
        return None, ["native-reference-convention-unavailable"]


def _reference_state_declaration(
    request: NativeMocoRequest,
) -> ReferenceStateDeclaration:
    return ReferenceStateDeclaration(
        "offline-moco-prepared-state",
        "caller-declared-complete-named-continuous-state",
        "complete-named-state",
        request.model_sha256,
        tuple(request.bindings.initial_state.items()),
        request.config.t_start_s,
    )


def _preview_replay_capability(
    model: Any, request: NativeMocoRequest, declared: Any
) -> list[str]:
    try:
        from .native_muscle_bundle import build_native_muscle_replay_bundle

        muscles = model.getMuscles()
        preview = {
            muscles.get(i).getName(): np.zeros(2) for i in range(muscles.getSize())
        }
        if request.mixed_actuation is not None:
            from .native_mixed_replay import build_native_mixed_replay_bundle

            build_native_mixed_replay_bundle(
                request.model_path,
                request.bindings.initial_state,
                np.array([0.0, request.config.duration_s]),
                {
                    channel.path: np.full(2, channel.control_bounds[0])
                    for channel in request.mixed_actuation.channels
                },
                request.mixed_actuation,
                experiment_id="offline-moco-mixed-replay-capability-preview",
                constrained_cold_start=declared,
            )
        elif declared is None:
            build_native_muscle_replay_bundle(
                request.model_path,
                request.bindings.initial_state,
                np.array([0.0, request.config.duration_s]),
                preview,
                experiment_id="offline-moco-replay-capability-preview",
            )
        else:
            from .native_constrained_muscle import build_constrained_muscle_bundle

            build_constrained_muscle_bundle(
                declared,
                np.array([0.0, request.config.duration_s]),
                preview,
                experiment_id="offline-moco-constrained-replay-capability-preview",
                wrapped_path_declaration=request.wrapped_path_declaration,
                native_mechanics_profile=request.native_mechanics_profile,
            )
        return []
    except (ValueError, RuntimeError):
        return ["independent-native-replay-unavailable"]


def _declared_chart_blockers(request: NativeMocoRequest) -> list[str]:
    """Prove caller optimizer boxes stay inside the declared coordinate chart."""
    declared = request.constrained_cold_start
    if declared is None:
        return []
    bounds = request.bindings.state_bounds
    blockers = []
    for coordinate, (lower, upper) in declared.chart_bounds.items():
        interval = bounds.get(coordinate + "/value")
        if interval is None or interval[0] < lower or interval[1] > upper:
            blockers.append("native-moco-chart-bounds-unqualified")
    for terms, lower, upper in declared.linear_chart_bounds.values():
        implied_low, implied_high = 0.0, 0.0
        for coordinate, weight in terms.items():
            interval = bounds.get(coordinate + "/value")
            if interval is None:
                blockers.append("native-moco-linear-chart-bounds-unqualified")
                break
            endpoints = (weight * interval[0], weight * interval[1])
            implied_low += min(endpoints)
            implied_high += max(endpoints)
        else:
            if (
                not np.isfinite((implied_low, implied_high)).all()
                or implied_low < lower
                or implied_high > upper
            ):
                blockers.append("native-moco-linear-chart-bounds-unqualified")
    return blockers


def _native_preparation(
    request: NativeMocoRequest,
    *,
    run_probe: bool = True,
) -> tuple[
    str | None,
    str | None,
    str | None,
    list[str],
    str | None,
    str | None,
    bool,
    float | None,
    float | None,
    NativeProbeFailure | None,
]:
    """Read the actual prepared state and independent provider dispositions."""
    import opensim as osim

    from .muscle_replay import _restore_continuous_state

    blockers: list[str] = _declared_chart_blockers(request)
    loaded: str | None = None
    model = osim.Model(str(request.model_path))
    runtime_sha = _native_runtime_identity(osim)
    probe_sha: str | None = None
    probe_failure: NativeProbeFailure | None = None
    probe_start = probe_end = None
    blockers.extend(_observed_marker_bindings(model, request))
    if model.getControllerSet().getSize():
        blockers.append("native-source-controller-undeclared")
    if request.constrained_cold_start is not None:
        from .native_mechanics_profile import (
            declare_native_mechanics_execution_profile,
            validate_native_mechanics_execution_profile,
        )

        try:
            expected_profile = declare_native_mechanics_execution_profile(
                request.model_path, request.model_sha256
            )
            if expected_profile.components and request.native_mechanics_profile is None:
                blockers.append("native-mechanics-profile-required")
            elif request.native_mechanics_profile is not None:
                validate_native_mechanics_execution_profile(
                    model, request.native_mechanics_profile, request.model_sha256
                )
                if expected_profile != request.native_mechanics_profile:
                    blockers.append("native-mechanics-profile-invalid")
        except (OSError, ValueError, RuntimeError, TypeError):
            blockers.append("native-mechanics-profile-invalid")
    actuators = {
        str(model.getActuators().get(i).getAbsolutePathString())
        for i in range(model.getActuators().getSize())
    }
    muscles = {
        str(model.getMuscles().get(i).getAbsolutePathString())
        for i in range(model.getMuscles().getSize())
    }
    forces = model.getForceSet()
    if any(
        str(forces.get(i).getAbsolutePathString()) not in actuators | muscles
        for i in range(forces.getSize())
    ):
        blockers.append("native-external-force-uninventoried")
    if request.passive_policy is None:
        blockers.append("passive-policy-unavailable")
    if (
        request.mixed_actuation is None
        and model.getActuators().getSize() != model.getMuscles().getSize()
    ):
        blockers.append("nonmuscle-assistance-unqualified")
    coordinates = model.getCoordinateSet()
    declared = request.constrained_cold_start
    if declared is None and (
        model.getConstraintSet().getSize()
        or any(
            coordinates.get(i).getDefaultLocked() for i in range(coordinates.getSize())
        )
    ):
        blockers.append("native-constraint-policy-unavailable")
    if declared is not None and declared.lock_targets:
        blockers.append("native-moco-locked-coordinates-unsupported")
    try:
        if declared is None:
            state, _, _ = _restore_continuous_state(
                model, request.bindings.initial_state, request.config.t_start_s
            )
        else:
            with reconstruct_declared_cold_start(declared) as prepared:
                model, state = prepared.model, prepared.state
    except (ValueError, RuntimeError):
        blockers.append("complete-native-state-unavailable")
        return loaded, None, None, blockers, runtime_sha, None, False, None, None, None
    discrete_names = model.getDiscreteVariableNames()
    for index in range(discrete_names.getSize()):
        name = str(discrete_names.get(index))
        if not name.endswith("/override_actuation"):
            blockers.append("native-discrete-state-unreconstructible")
            continue
        value = float(model.getDiscreteVariableValue(state, name))
        if not math.isfinite(value) or value != 0.0:
            blockers.append("native-discrete-state-unreconstructible")
    loaded = hashlib.sha256(model.dump().encode()).hexdigest()
    names = model.getStateVariableNames()
    native_states = {str(names.get(i)) for i in range(names.getSize())}
    if set(request.bindings.state_bounds) != native_states:
        blockers.append("native-state-binding-coverage")
    native_actuators = model.getActuators()
    native_controls = {
        str(native_actuators.get(i).getAbsolutePathString())
        for i in range(native_actuators.getSize())
    }
    if set(request.bindings.control_bounds) != native_controls:
        blockers.append("native-control-binding-coverage")

    passive_sha, passive_blockers = _audit_native_passive(
        model, state, request.passive_policy, request.mixed_actuation, declared
    )
    blockers.extend(passive_blockers)

    reference_sha, ref_blockers = _audit_reference_convention(request)
    blockers.extend(ref_blockers)

    hard_before_probe = any(
        blocker != "passive-policy-unavailable" for blocker in blockers
    )
    probe_attempted = False
    if run_probe and not hard_before_probe:
        probe_attempted = True
        probe_start = float(state.getTime())
        probe_end = probe_start
        try:
            probe_sha, probe_start, probe_end = _native_short_capability_probe(
                model,
                state,
                request.config.duration_s,
                request.model_sha256,
                runtime_sha,
                request.constrained_cold_start,
            )
        except NativeProbeFailure as exc:
            probe_failure = exc
            if exc.window_seconds is not None:
                probe_start, probe_end = exc.window_seconds
            else:
                probe_start = probe_end = None
            blockers.append(f"native-short-probe-failed:{exc.exception_type}")
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            blockers.append(f"native-short-probe-failed:{type(exc).__name__}")

    if _sha(request.model_path) != request.model_sha256:
        blockers.append("model-sha256-changed-during-preparation")
    return (
        loaded,
        reference_sha,
        passive_sha,
        blockers,
        runtime_sha,
        probe_sha,
        probe_attempted,
        probe_start,
        probe_end,
        probe_failure,
    )


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


def _physical_audit_bytes(audit: NativeBodyInertiaAudit) -> bytes:
    """Serialize the exact native audit for preparation and later re-admission."""
    return (
        json.dumps(
            {
                "schema_version": "native-body-inertia-preflight/1.0.0",
                "qualified": audit.admitted,
                "audit": asdict(audit),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _physical_body_preflight(
    request: NativeMocoRequest, directory: Path
) -> tuple[str, bool, list[str]]:
    """Retain failed physical evidence without hiding other preparation blockers."""
    try:
        audit = audit_native_body_inertia(request.model_path, request.model_sha256)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        payload = {
            "schema_version": "native-body-inertia-preflight/1.0.0",
            "qualified": False,
            "source_sha256": request.model_sha256,
            "adapter_sha256": _sha(
                Path(audit_native_body_inertia.__code__.co_filename)
            ),
            "validator_sha256": _sha(Path(dime_global_calibration.__file__)),
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
        serialized = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
        blockers = ["physical-body-inertia-audit-unavailable"]
        qualified = False
    else:
        serialized = _physical_audit_bytes(audit)
        blockers = [] if audit.admitted else ["physical-body-inertia-unqualified"]
        qualified = audit.admitted
    path = directory / "physical_body_inertia.json"
    path.write_bytes(serialized)
    return hashlib.sha256(serialized).hexdigest(), qualified, blockers


def _preparation_source_digests(
    request: NativeMocoRequest,
) -> tuple[dict[str, str | None], list[str]]:
    """Read each frozen input once and retain every independent hash failure."""
    actual: dict[str, str | None] = {}
    blockers: list[str] = []
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
    return actual, blockers


def _native_diagnostic_findings(
    blockers: list[str],
    source_sha256: str,
    evidence_sha256: str,
    *,
    audit_codes: tuple[DiagnosticCode, ...],
    additional_findings: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    """Translate every preparation disposition into the closed typed registry."""
    from .native_diagnostic_contract import (
        ExecutionEffect,
        NativeDiagnosticFinding,
        QualificationEffect,
    )

    blocker_codes = {
        "passive-policy-unavailable": DiagnosticCode.PASSIVE_POLICY_UNAVAILABLE,
        "physical-body-inertia-audit-unavailable": DiagnosticCode.PHYSICAL_BODY_INERTIA_INVALID,
        "physical-body-inertia-unqualified": DiagnosticCode.PHYSICAL_BODY_INERTIA_INVALID,
        "native-source-controller-undeclared": DiagnosticCode.NATIVE_SOURCE_CONTROLLER_UNDECLARED,
        "native-external-force-uninventoried": DiagnosticCode.NATIVE_EXTERNAL_FORCE_UNINVENTORIED,
        "nonmuscle-assistance-unqualified": DiagnosticCode.NATIVE_EXTERNAL_FORCE_UNINVENTORIED,
        "passive-policy-model-mismatch": DiagnosticCode.NATIVE_PASSIVE_POLICY_INVALID,
        "passive-policy-muscle-coverage-mismatch": DiagnosticCode.NATIVE_PASSIVE_POLICY_INVALID,
        "native-constraint-policy-unavailable": DiagnosticCode.NATIVE_CONSTRAINT_POLICY_UNAVAILABLE,
        "native-moco-locked-coordinates-unsupported": DiagnosticCode.NATIVE_CONSTRAINT_REALIZATION_FAILED,
        "native-discrete-state-unreconstructible": DiagnosticCode.NATIVE_STATE_COVERAGE_INVALID,
        "complete-native-state-unavailable": DiagnosticCode.NATIVE_STATE_RECONSTRUCTION_FAILED,
        "native-state-binding-coverage": DiagnosticCode.NATIVE_STATE_COVERAGE_INVALID,
        "native-control-binding-coverage": DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID,
        "marker-binding-coverage": DiagnosticCode.NATIVE_MARKER_BINDING_INVALID,
        "native-marker-correspondence": DiagnosticCode.NATIVE_MARKER_BINDING_INVALID,
        "native-marker-placement-mismatch": DiagnosticCode.NATIVE_MARKER_BINDING_INVALID,
        "capture-trc-invalid": DiagnosticCode.NATIVE_OBSERVATION_INVALID,
        "observation-horizon-mismatch": DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID,
        "reference-marker-coverage": DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID,
        "missing-marker-observation-support": DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID,
        "irregular-observation-clock": DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID,
        "capture-registration-unavailable": DiagnosticCode.NATIVE_MARKER_BINDING_INVALID,
        "registration-changed-observation-clock": DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID,
        "capture-registration-failed": DiagnosticCode.NATIVE_MARKER_BINDING_INVALID,
        "native-guess-invalid": DiagnosticCode.NATIVE_STATE_RECONSTRUCTION_FAILED,
        "native-reference-convention-unavailable": DiagnosticCode.NATIVE_MARKER_BINDING_INVALID,
        "native-passive-or-constraint-observation-unavailable": DiagnosticCode.NATIVE_OBSERVATION_INVALID,
        "native-moco-chart-bounds-unqualified": DiagnosticCode.NATIVE_CONSTRAINT_REALIZATION_FAILED,
        "native-moco-linear-chart-bounds-unqualified": DiagnosticCode.NATIVE_CONSTRAINT_REALIZATION_FAILED,
        "model-sha256-mismatch": DiagnosticCode.NATIVE_SOURCE_IDENTITY_INVALID,
        "capture-sha256-mismatch": DiagnosticCode.NATIVE_SOURCE_IDENTITY_INVALID,
        "guess-sha256-mismatch": DiagnosticCode.NATIVE_SOURCE_IDENTITY_INVALID,
        "model-sha256-changed-during-preparation": DiagnosticCode.NATIVE_SOURCE_IDENTITY_INVALID,
        "native-model-preparation-unavailable": DiagnosticCode.NATIVE_MODEL_LOAD_FAILED,
        "native-mechanics-profile-required": DiagnosticCode.NATIVE_SOURCE_IDENTITY_INVALID,
        "native-mechanics-profile-invalid": DiagnosticCode.NATIVE_SOURCE_IDENTITY_INVALID,
        "unused-reference-policy": DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID,
        "nonzero-native-replay-origin": DiagnosticCode.NATIVE_INPUT_COVERAGE_INVALID,
    }
    findings: list[NativeDiagnosticFinding] = []
    if any(
        not isinstance(finding, NativeDiagnosticFinding)
        or finding.source_sha256 != source_sha256
        for finding in additional_findings
    ):
        raise TypeError("additional native findings must be source-bound typed values")
    findings.extend(additional_findings)
    for blocker in dict.fromkeys(blockers):
        code = blocker_codes.get(blocker)
        if blocker.startswith("native-short-probe-failed:"):
            code = DiagnosticCode.NATIVE_SHORT_PROBE_FAILED
        elif blocker.startswith(("muscle-disabled:", "muscle-overridden:")):
            code = DiagnosticCode.NATIVE_PASSIVE_POLICY_INVALID
        elif blocker.startswith(
            (
                "muscle-policy-options:",
                "normalized-fiber-limit:",
                "passive-elastic-force-limit:",
                "tendon-force-limit:",
            )
        ):
            code = DiagnosticCode.NATIVE_PASSIVE_LIMIT_EXCEEDED
        if code is None:
            # Fail closed when a future string blocker has not been classified.
            code = DiagnosticCode.NATIVE_OBSERVATION_INVALID
        disposition = _diagnostic_effects(code)
        findings.append(
            NativeDiagnosticFinding(
                code=code,
                source_sha256=source_sha256,
                evidence_sha256=evidence_sha256,
                execution_effect=disposition[0],
                qualification_effect=disposition[1],
            )
        )
    for code in audit_codes:
        execution, qualification = _diagnostic_effects(code)
        findings.append(
            NativeDiagnosticFinding(
                code=code,
                source_sha256=source_sha256,
                evidence_sha256=evidence_sha256,
                execution_effect=execution,
                qualification_effect=qualification,
            )
        )
    return tuple(findings)


def _diagnostic_effects(
    code: DiagnosticCode,
) -> tuple[Any, Any]:
    """Read the reviewed effect pair from the immutable diagnostic registry."""
    return diagnostic_effects(code)


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
    actual, source_blockers = _preparation_source_digests(request)
    blockers.extend(source_blockers)
    physical_sha = None
    physical_qualified = False
    if actual["model"] == request.model_sha256:
        physical_sha, physical_qualified, physical_blockers = _physical_body_preflight(
            request, directory
        )
        blockers.extend(physical_blockers)
    capture = None
    clock_sha = None
    registered_sha = None
    if actual["capture"] == request.trc_sha256:
        capture, clock_sha, capture_blockers = _load_observation_capture(
            request, directory
        )
        blockers.extend(capture_blockers)
    if (
        actual["guess"] == request.states_guess_sha256
        and actual["model"] == request.model_sha256
        and capture is not None
    ):
        from .native_moco_guess import audit_native_moco_guess

        try:
            audit_native_moco_guess(
                request.model_path,
                request.states_guess_path,
                request.model_sha256,
                capture.time_s,
            )
        except (OSError, ValueError, RuntimeError):
            blockers.append("native-guess-invalid")
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
    wrapped_findings: tuple[NativeDiagnosticFinding, ...] = ()
    if actual["model"] == request.model_sha256:
        try:
            wrapped_findings = audit_native_wrapped_path_source(
                request.model_path,
                request.model_sha256,
                request.wrapped_path_declaration,
                _reference_state_declaration(request),
                request.reference_frame_path,
            ).findings
        except (OSError, ValueError, RuntimeError):
            from .native_diagnostic_contract import (
                ExecutionEffect,
                QualificationEffect,
            )

            wrap_error_sha = hashlib.sha256(
                f"{request.model_sha256}:wrapped-path-source-audit-failed".encode()
            ).hexdigest()
            wrapped_findings = (
                NativeDiagnosticFinding(
                    code=DiagnosticCode.NATIVE_SOURCE_IDENTITY_INVALID,
                    source_sha256=request.model_sha256,
                    evidence_sha256=wrap_error_sha,
                    execution_effect=ExecutionEffect.BLOCK,
                    qualification_effect=QualificationEffect.NONE,
                ),
            )
    loaded = reference_sha = passive_sha = None
    runtime_sha = probe_sha = None
    probe_failure = None
    probe_attempted = False
    probe_start = probe_end = None
    if actual["model"] == request.model_sha256:
        try:
            run_probe = not blockers and not any(
                finding.execution_effect.value == "block"
                for finding in wrapped_findings
            )
            (
                loaded,
                reference_sha,
                passive_sha,
                native_blockers,
                runtime_sha,
                probe_sha,
                probe_attempted,
                probe_start,
                probe_end,
                probe_failure,
            ) = _native_preparation(request, run_probe=run_probe)
            blockers.extend(native_blockers)
        except (OSError, ValueError, RuntimeError):
            blockers.append("native-model-preparation-unavailable")
    probe_failure_sha = None
    if probe_failure is not None:
        private_receipt = probe_failure.private_receipt(
            request.identity_sha256, request.model_sha256
        )
        private_receipt_bytes = json.dumps(
            private_receipt, sort_keys=True, allow_nan=False
        ).encode()
        probe_failure_sha = hashlib.sha256(private_receipt_bytes).hexdigest()
        (directory / "native_probe_failure_private.json").write_bytes(
            private_receipt_bytes + b"\n"
        )
    evidence_payload = {
        "source_digests": actual,
        "loaded_model_sha256": loaded,
        "reference_observation_sha256": reference_sha,
        "passive_observation_sha256": passive_sha,
        "physical_body_inertia_sha256": physical_sha,
        "probe_sha256": probe_sha,
        "probe_failure_receipt_sha256": probe_failure_sha,
        "blockers": tuple(dict.fromkeys(blockers)),
    }
    diagnostic_evidence_sha = hashlib.sha256(
        json.dumps(evidence_payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    final_blockers = tuple(dict.fromkeys(blockers))
    blocker_receipt_sha = hashlib.sha256(
        json.dumps(final_blockers, allow_nan=False).encode()
    ).hexdigest()
    audit_codes: list[DiagnosticCode] = []
    if (
        not source_blockers
        and "model-sha256-changed-during-preparation" not in blockers
        and capture is not None
        and actual["guess"] == request.states_guess_sha256
    ):
        audit_codes.append(DiagnosticCode.NATIVE_SOURCE_IDENTITY_VERIFIED)
    if loaded is not None and "complete-native-state-unavailable" not in blockers:
        audit_codes.append(DiagnosticCode.NATIVE_STATE_RECONSTRUCTED)
    if loaded is not None and not any(
        blocker in blockers
        for blocker in (
            "native-constraint-policy-unavailable",
            "native-moco-locked-coordinates-unsupported",
            "native-discrete-state-unreconstructible",
        )
    ):
        audit_codes.append(DiagnosticCode.NATIVE_CONSTRAINTS_RECONSTRUCTED)
    if loaded is not None and not any(
        blocker in blockers
        for blocker in (
            "native-state-binding-coverage",
            "native-control-binding-coverage",
            "native-input-coverage-invalid",
        )
    ):
        audit_codes.append(DiagnosticCode.NATIVE_INPUT_BINDINGS_VERIFIED)
    if loaded is not None and not any(
        "marker" in blocker or "registration" in blocker for blocker in blockers
    ):
        audit_codes.append(DiagnosticCode.NATIVE_MARKER_BINDINGS_VERIFIED)
    if physical_qualified and physical_sha is not None:
        audit_codes.append(DiagnosticCode.NATIVE_PHYSICAL_INERTIA_ADMITTED)
    if probe_sha is not None:
        audit_codes.extend(
            (
                DiagnosticCode.NATIVE_MECHANICS_EVALUATED,
                DiagnosticCode.NATIVE_SHORT_PROBE_COMPLETED,
            )
        )
    from .native_diagnostic_contract import assess_native_diagnostic

    attempt_evidence = probe_sha or diagnostic_evidence_sha
    assessment = assess_native_diagnostic(
        request.identity_sha256,
        request.model_sha256,
        _native_diagnostic_findings(
            blockers,
            request.model_sha256,
            diagnostic_evidence_sha,
            audit_codes=tuple(audit_codes),
            additional_findings=wrapped_findings
            + (
                ()
                if request.qualification_context is None
                else request.qualification_context.findings
            ),
        ),
        policy=request.effective_diagnostic_policy,
        runtime_viability=(
            RuntimeViability.VIABLE
            if probe_sha is not None
            else RuntimeViability.FAILED
            if probe_attempted
            else RuntimeViability.NOT_ATTEMPTED
        ),
        runtime_phase=(
            RuntimePhase.SHORT_CAPABILITY_PROBE
            if probe_sha is not None or probe_attempted
            else None
        ),
        runtime_window_seconds=(
            (probe_start, probe_end)
            if probe_start is not None and probe_end is not None
            else None
        ),
        native_runtime_sha256=runtime_sha
        if probe_sha is not None or probe_attempted
        else None,
        runtime_evidence_sha256=attempt_evidence
        if probe_sha is not None or probe_attempted
        else None,
        blocker_receipt_sha256=blocker_receipt_sha,
        qualification_provenance_references=(
            ()
            if request.qualification_context is None
            else request.qualification_context.provenance_references
        ),
    )
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
        physical_sha,
        physical_qualified,
        len(request.marker_weights),
        capture.frames if capture is not None else 0,
        len(request.excluded_markers),
        final_blockers,
        time.perf_counter() - wall,
        time.process_time() - cpu,
        allow_unqualified_passive_diagnostic=request.allow_unqualified_passive_diagnostic,
        diagnostic_assessment=assessment,
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
    diagnostic_assessment: NativeDiagnosticAssessment | None = None


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


def _require_physical_body_audit(
    request: NativeMocoRequest, prepared: NativeMocoPreparation, directory: Path
) -> None:
    """Recompute native body physics, not just a claimed sidecar checksum."""
    expected = prepared.physical_body_inertia_sha256
    if not prepared.physical_body_inertia_qualified or expected is None:
        raise ValueError("Physical body inertia admission is missing or rejected")
    try:
        frozen = (directory / "physical_body_inertia.json").read_bytes()
    except OSError as exc:
        raise ValueError("Physical body inertia audit sidecar is missing") from exc
    if hashlib.sha256(frozen).hexdigest() != expected:
        raise ValueError("Physical body inertia audit sidecar changed")
    try:
        current = audit_native_body_inertia(request.model_path, request.model_sha256)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError("Physical body inertia native re-audit failed") from exc
    if not current.admitted or _physical_audit_bytes(current) != frozen:
        raise ValueError("Physical body inertia audit identity changed")


def _require_current_inputs(
    request: NativeMocoRequest, prepared: NativeMocoPreparation, directory: Path
) -> Path:
    """Reject any source, observation or preparation drift before native work."""
    if request.identity_sha256 != prepared.request_sha256:
        raise ValueError("Native Moco request changed after preparation")
    if (
        request.allow_unqualified_passive_diagnostic
        != prepared.allow_unqualified_passive_diagnostic
    ):
        raise ValueError("Native Moco diagnostic disposition changed after preparation")
    if (
        not prepared.physical_body_inertia_qualified
        or prepared.physical_body_inertia_sha256 is None
    ):
        raise ValueError("Physical body inertia admission is missing or rejected")
    if not prepared.ready_for_software_solve:
        raise ValueError("Preparation blockers prohibit native Moco solve")
    for path, expected in (
        (request.model_path, request.model_sha256),
        (request.trc_path, request.trc_sha256),
        (request.states_guess_path, request.states_guess_sha256),
    ):
        if _sha(path) != expected:
            raise ValueError("Source, observation or guess changed after preparation")
    _require_physical_body_audit(request, prepared, directory)
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

    progress_path = directory / "native_moco_progress.json"

    def record_progress(stage: str) -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "request_sha256": request.identity_sha256,
                    "source_sha256": request.model_sha256,
                    "stage": stage,
                    "elapsed_wall_seconds": time.perf_counter() - wall,
                    "elapsed_cpu_seconds": time.process_time() - cpu,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    record_progress("building-study")

    study = build_moco_study(
        str(request.model_path),
        str(registered),
        str(request.states_guess_path),
        request.config,
        initial_bindings=request.bindings,
        marker_weights=request.marker_weights,
        progress_callback=record_progress,
    )
    record_progress("study-built")
    weights = _native_goal_weights(study)
    if weights != dict(request.marker_weights):
        raise ValueError("Native marker goal did not preserve selected weights")
    weight_sha = hashlib.sha256(
        json.dumps(weights, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_parallel(0)
    record_progress("optimizing")
    solution = study.solve()
    record_progress("optimization-returned")
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
    preparation_assessment = prepared.diagnostic_assessment
    if preparation_assessment is None:
        raise ValueError("Native preparation has no typed diagnostic assessment")
    solve_evidence_sha = hashlib.sha256(
        json.dumps(
            {
                "preparation_assessment": preparation_assessment.to_json_dict(),
                "solve_attempt": asdict(result),
                "time_window_seconds": (
                    request.config.t_start_s,
                    request.config.horizon_s,
                ),
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    solve_assessment = replace(
        preparation_assessment,
        runtime_viability=(
            RuntimeViability.VIABLE if result.success else RuntimeViability.FAILED
        ),
        runtime_phase=RuntimePhase.OPTIMIZATION,
        runtime_window_seconds=(
            request.config.t_start_s,
            request.config.horizon_s,
        ),
        runtime_evidence_sha256=solve_evidence_sha,
    )
    result = replace(result, diagnostic_assessment=solve_assessment)
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
