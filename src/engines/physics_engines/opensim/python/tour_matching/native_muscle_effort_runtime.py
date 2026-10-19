from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_adapter import (
    NativeMuscleEffortProposal,
    NativeMuscleEffortRequest,
    NativeMuscleEffortSession,
    NativeMusclePathState,
    NativeMuscleStateSnapshot,
    NativePassiveComponent,
    NativePassiveEffortObservation,
    native_state_fingerprint,
)
from src.shared.python.motion_matching.distributed_feedback import FeedbackRequest

if TYPE_CHECKING:
    from src.engines.physics_engines.opensim.python.tour_matching.muscle_replay import (
        NativeMuscleReplayResult,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_adapter import (
        AnalyticRecruitmentConfig,
        NativeCoordinateMapping,
    )


class NativeEffortBackend(Protocol):
    """Narrow native boundary required by one controlled interval."""

    def observe(self) -> NativeMuscleStateSnapshot: ...

    def apply_raw_excitations(self, proposal: NativeMuscleEffortProposal) -> None: ...

    def advance_to(self, end_time_s: float) -> NativeMuscleStateSnapshot: ...

    def applied_raw_excitations(self) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class NativeEffortStepResult:
    proposal: NativeMuscleEffortProposal
    accepted_snapshot: NativeMuscleStateSnapshot | None
    input_profile_sha256: str | None = None
    input_profile_start_state_sha256: str | None = None


class NativeMuscleEffortExecutor:
    """Connect one stateless request to fail-stop native advance/acceptance."""

    def __init__(
        self,
        session: NativeMuscleEffortSession,
        backend: NativeEffortBackend,
    ) -> None:
        self._session = session
        self._backend = backend

    def execute_one_interval(
        self,
        request: NativeMuscleEffortRequest,
        feedback: FeedbackRequest,
        *,
        effort_tolerance_n_m: float,
        max_nfev: int = 80,
    ) -> NativeEffortStepResult:
        """Propose first; integrate only a feasible candidate and verify raw readback."""
        before = self._backend.observe()
        expected = self._session.snapshot
        if (
            self._fingerprint(before) != self._fingerprint(expected)
            or before.passive_effort != expected.passive_effort
        ):
            raise ValueError("native backend state differs from accepted session state")
        proposal = self._session.propose(
            request,
            feedback,
            effort_tolerance_n_m=effort_tolerance_n_m,
            max_nfev=max_nfev,
        )
        if not proposal.feasible:
            self._session.reject(proposal)
            return NativeEffortStepResult(proposal, None)

        token = self._session.begin_advance(proposal)
        try:
            self._backend.apply_raw_excitations(proposal)
            advanced = self._backend.advance_to(proposal.expected_end_time_s)
            self._verify_applied_raw(proposal, self._backend.applied_raw_excitations())
            self._session.accept_advanced(token, advanced)
        except Exception as error:  # noqa: BLE001 - native failures may partially advance the plant
            self._session.fail_advance(token, error)
            raise RuntimeError(
                "native advance failed and session was poisoned"
            ) from error
        return NativeEffortStepResult(
            proposal,
            advanced,
            getattr(self._backend, "input_profile_sha256", None),
            getattr(self._backend, "input_profile_start_state_sha256", None),
        )

    @staticmethod
    def _fingerprint(snapshot: NativeMuscleStateSnapshot) -> str:
        return native_state_fingerprint(
            source_sha256=snapshot.source_sha256,
            runtime_identity=snapshot.runtime_identity,
            time_s=snapshot.time_s,
            coordinate=snapshot.coordinate,
            paths=snapshot.paths,
            required_state_paths=snapshot.required_state_paths,
            complete_state=snapshot.complete_state,
        )

    @staticmethod
    def _verify_applied_raw(
        proposal: NativeMuscleEffortProposal, applied: Mapping[str, float]
    ) -> None:
        expected = dict(proposal.raw_excitations)
        if set(applied) != set(expected):
            raise RuntimeError("native raw-excitation readback is incomplete")
        if any(
            not isinstance(applied[name], (float, int)) or applied[name] != value
            for name, value in expected.items()
        ):
            raise RuntimeError("native applied raw excitation differs from proposal")


class _OpenSimNativeEffortBackend:
    """One-coordinate, Millard-only backend with strict one-interval readback.

    A candidate is applied to a fresh model with an owned constant input player.
    The accepted state is restored exactly; no equilibration is performed.
    """

    def __init__(
        self,
        *,
        osim: Any,
        model: Any,
        state: Any,
        source_sha256: str,
        runtime_identity: str,
        source_path: Path,
        coordinate: NativeCoordinateMapping,
        analytic_by_muscle: Mapping[str, AnalyticRecruitmentConfig],
        integrator_accuracy: float,
    ) -> None:

        from src.engines.physics_engines.opensim.python.tour_matching import (
            muscle_replay,
        )
        from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_adapter import (
            AnalyticRecruitmentConfig,
            NativeCoordinateMapping,
        )

        if not isinstance(coordinate, NativeCoordinateMapping):
            raise TypeError("coordinate mapping must be source-audited")
        if any(
            not isinstance(value, AnalyticRecruitmentConfig)
            for value in analytic_by_muscle.values()
        ):
            raise TypeError("each native muscle needs an analytic proposal config")
        if (
            len(source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_sha256.lower())
            or not runtime_identity
            or runtime_identity != str(osim.GetVersionAndDate())
        ):
            raise ValueError("source and runtime identities are required")
        if not math.isfinite(integrator_accuracy) or not 0 < integrator_accuracy < 1:
            raise ValueError(
                "integrator accuracy must be finite and between zero and one"
            )
        self._osim = osim
        self._model = model
        self._state = state
        self._source_sha256 = source_sha256
        self._runtime_identity = runtime_identity
        self._source_path = source_path.resolve(strict=True)
        self._coordinate_mapping = coordinate
        self._analytic_by_muscle = dict(analytic_by_muscle)
        self._integrator_accuracy = float(integrator_accuracy)
        self._muscles, self._muscle_names = muscle_replay._admit_native_muscles(
            model, self._analytic_by_muscle, ()
        )
        self._coordinate = model.getCoordinateSet().get(coordinate.coordinate_name)
        self._validate_model_boundary(state)
        self._applied: dict[str, float] = {}
        self._last_snapshot: NativeMuscleStateSnapshot | None = None
        self._pending_model: Any | None = None
        self._pending_state: Any | None = None
        self._pending_muscles: Any | None = None
        self._pending_muscle_names: tuple[str, ...] = ()
        self._pending_coordinate: Any | None = None
        self._pending_start_snapshot: NativeMuscleStateSnapshot | None = None
        self._input_profile_sha256: str | None = None
        self._input_profile_start_state_sha256: str | None = None

    @property
    def input_profile_start_state_sha256(self) -> str | None:
        return self._input_profile_start_state_sha256

    @property
    def input_profile_sha256(self) -> str | None:
        """Digest of the owned two-knot constant input profile, separate from model identity."""
        return self._input_profile_sha256

    @staticmethod
    def _state_flag(component: Any, state: Any, getter_name: str) -> bool:
        getter = getattr(component, getter_name, None)
        if not callable(getter):
            raise ValueError(f"native state flag {getter_name} is unavailable")
        value = getter(state)
        if not isinstance(value, bool):
            raise ValueError(f"native state flag {getter_name} is not boolean")
        return value

    def _validate_model_boundary(self, state: Any) -> None:
        self._validate_model_components(
            self._model,
            state,
            self._muscles,
            self._muscle_names,
            self._coordinate,
        )

    def _validate_model_components(
        self,
        model: Any,
        state: Any,
        muscles: Any,
        muscle_names: tuple[str, ...],
        coordinate: Any,
    ) -> None:
        if tuple(muscle_names) != tuple(self._analytic_by_muscle):
            raise ValueError("analytic config order must match native muscle order")
        coordinates = model.getCoordinateSet()
        if coordinates.getSize() != 1:
            raise ValueError(
                "only the audited one-coordinate native model is supported"
            )
        if (
            str(coordinate.getAbsolutePathString())
            != self._coordinate_mapping.coordinate_path
        ):
            raise ValueError("native coordinate path differs from audited mapping")
        if self._state_flag(coordinate, state, "getLocked") or self._state_flag(
            coordinate, state, "isPrescribed"
        ):
            raise ValueError("native coordinate is locked or prescribed at this state")
        if model.getConstraintSet().getSize() != 0:
            raise ValueError("native constraints are outside the supported mapping")
        for index in range(muscles.getSize()):
            muscle = muscles.get(index)
            if muscle.getConcreteClassName() != "Millard2012EquilibriumMuscle":
                raise ValueError("native muscle law has no supported state policy")
            if (
                muscle.get_ignore_activation_dynamics()
                or muscle.get_ignore_tendon_compliance()
                or self._state_flag(muscle, state, "getIgnoreActivationDynamics")
                or self._state_flag(muscle, state, "getIgnoreTendonCompliance")
            ):
                raise ValueError("ignored native muscle dynamics are unsupported")
            self._verify_analytic_parameters(
                self._osim, muscle, self._analytic_by_muscle[muscle_names[index]]
            )
        state_names_set = model.getStateVariableNames()
        state_names = tuple(
            str(state_names_set.get(i)) for i in range(state_names_set.getSize())
        )
        expected = {
            self._coordinate_mapping.coordinate_value_state_path,
            self._coordinate_mapping.coordinate_speed_state_path,
        }
        for index in range(muscles.getSize()):
            path = str(muscles.get(index).getAbsolutePathString())
            expected.update((path + "/activation", path + "/fiber_length"))
        if len(state_names) != len(set(state_names)) or set(state_names) != expected:
            raise ValueError("native continuous state schema is not fully admitted")
        if model.getNumControls() != muscles.getSize():
            raise ValueError("model controls are not exactly the admitted muscles")

    @staticmethod
    def _verify_analytic_parameters(
        osim: Any, muscle: Any, config: AnalyticRecruitmentConfig
    ) -> None:
        shared_values = (
            ("getMaxIsometricForce", config.max_isometric_force_n),
            ("getOptimalFiberLength", config.optimal_fiber_length_m),
            ("getTendonSlackLength", config.tendon_slack_length_m),
            ("getPennationAngleAtOptimalFiberLength", config.pennation_angle_rad),
        )

        def verify_parameter(native: Any, getter_name: str, expected: float) -> None:
            getter = getattr(native, getter_name, None)
            if not callable(getter):
                raise ValueError(
                    f"native analytic parameter {getter_name} is unavailable"
                )
            actual = float(getter())
            if not math.isfinite(actual) or not math.isclose(
                actual, float(expected), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"analytic config differs from native {getter_name}")

        for getter_name, expected in shared_values:
            verify_parameter(muscle, getter_name, expected)
        millard_type = getattr(osim, "Millard2012EquilibriumMuscle", None)
        downcast = getattr(millard_type, "safeDownCast", None)
        millard = downcast(muscle) if callable(downcast) else None
        if millard is None:
            raise ValueError("native muscle cannot be admitted as Millard2012")
        activation_values = (
            ("getActivationTimeConstant", config.activation_time_constant_s),
            ("getDeactivationTimeConstant", config.deactivation_time_constant_s),
            ("getMinimumActivation", config.minimum_activation),
        )
        for getter_name, expected in activation_values:
            verify_parameter(millard, getter_name, expected)

    def _observe_path(
        self, muscle: Any, state: Any, state_values: Mapping[str, float]
    ) -> tuple[NativeMusclePathState, float, tuple[NativePassiveComponent, ...]]:
        name = str(muscle.getName())
        path = str(muscle.getAbsolutePathString())
        activation_path = path + "/activation"
        fiber_path = path + "/fiber_length"
        moment_arm = float(muscle.computeMomentArm(state, self._coordinate))
        passive_along_tendon = float(muscle.getPassiveFiberForceAlongTendon(state))
        activation = float(muscle.getActivation(state))
        fiber_length = float(muscle.getFiberLength(state))
        raw_excitation = float(muscle.getExcitation(state))
        observed = (
            float(muscle.getLength(state)),
            moment_arm,
            activation,
            fiber_length,
            float(muscle.getFiberVelocity(state)),
            raw_excitation,
            passive_along_tendon,
            float(muscle.getActuation(state)),
        )
        if any(not math.isfinite(value) for value in observed):
            raise ValueError("native muscle observation contains non-finite values")
        if not math.isclose(state_values[activation_path], activation, abs_tol=1e-12):
            raise ValueError("native activation getter differs from bound state")
        if not math.isclose(state_values[fiber_path], fiber_length, abs_tol=1e-12):
            raise ValueError("native fiber length differs from bound state")
        path_state = NativeMusclePathState(
            name=name,
            absolute_path=path,
            muscle_tendon_length_m=observed[0],
            moment_arm_m_per_rad=moment_arm,
            activation=activation,
            previous_raw_excitation=raw_excitation,
            activation_state_path=activation_path,
            fiber_length_state_path=fiber_path,
            fiber_length_m=fiber_length,
            fiber_velocity_m_s=observed[4],
            analytic=self._analytic_by_muscle[name],
            native_actuation_force_n=observed[7],
        )
        return (
            path_state,
            passive_along_tendon * moment_arm,
            (NativePassiveComponent.PASSIVE_FIBER_FORCE_ALONG_TENDON,),
        )

    def observe(self) -> NativeMuscleStateSnapshot:
        state = self._state
        self._validate_model_boundary(state)
        self._model.realizeDynamics(state)
        state_set = self._model.getStateVariableNames()
        state_names = tuple(str(state_set.get(i)) for i in range(state_set.getSize()))
        complete_state = tuple(
            (name, float(self._model.getStateVariableValue(state, name)))
            for name in state_names
        )
        if any(not math.isfinite(value) for _, value in complete_state):
            raise ValueError("native state contains non-finite continuous values")
        state_values = dict(complete_state)
        records = tuple(
            self._observe_path(self._muscles.get(index), state, state_values)
            for index in range(self._muscles.getSize())
        )
        snapshot = NativeMuscleStateSnapshot(
            source_sha256=self._source_sha256,
            runtime_identity=self._runtime_identity,
            time_s=float(state.getTime()),
            coordinate=self._coordinate_mapping,
            paths=tuple(record[0] for record in records),
            required_state_paths=state_names,
            complete_state=complete_state,
            passive_effort=None,
        )
        passive = NativePassiveEffortObservation(
            source_sha256=self._source_sha256,
            runtime_identity=self._runtime_identity,
            time_s=snapshot.time_s,
            native_state_sha256=native_state_fingerprint(
                source_sha256=snapshot.source_sha256,
                runtime_identity=snapshot.runtime_identity,
                time_s=snapshot.time_s,
                coordinate=snapshot.coordinate,
                paths=snapshot.paths,
                required_state_paths=snapshot.required_state_paths,
                complete_state=snapshot.complete_state,
            ),
            coordinate_path=self._coordinate_mapping.coordinate_path,
            effort_unit=self._coordinate_mapping.effort_unit,
            path_contributions_n_m=tuple(
                (
                    record[0].absolute_path,
                    record[1],
                )
                for record in records
            ),
            component_names_by_path=tuple(
                (record[0].absolute_path, record[2]) for record in records
            ),
        )
        result = replace(snapshot, passive_effort=passive)
        self._last_snapshot = result
        return result

    def apply_raw_excitations(self, proposal: NativeMuscleEffortProposal) -> None:
        import numpy as np

        from src.engines.physics_engines.opensim.python.tour_matching import (
            muscle_replay,
        )

        accepted = self._last_snapshot
        if accepted is None:
            raise RuntimeError("native input requires a current accepted observation")
        requested = dict(proposal.raw_excitations)
        if (
            tuple(requested) != self._muscle_names
            or proposal.source_sha256 != self._source_sha256
            or proposal.runtime_identity != self._runtime_identity
            or proposal.start_time_s != accepted.time_s
            or proposal.expected_end_time_s <= proposal.start_time_s
            or self._pending_model is not None
        ):
            raise ValueError(
                "proposal identity, time, order, or pending state is invalid"
            )
        if (
            hashlib.sha256(self._source_path.read_bytes()).hexdigest()
            != self._source_sha256
        ):
            raise RuntimeError("native model source bytes changed before input setup")

        model = self._osim.Model(str(self._source_path))
        input_times = np.asarray(
            [proposal.start_time_s, proposal.expected_end_time_s], dtype=np.float64
        )
        controls = {
            name: np.asarray([value, value], dtype=np.float64)
            for name, value in proposal.raw_excitations
        }
        muscles, muscle_names = muscle_replay._admit_native_muscles(model, controls, ())
        muscle_replay._configure_input_player(
            model, muscles, muscle_names, input_times, controls
        )
        model.finalizeConnections()
        state, state_names, _ = muscle_replay._restore_continuous_state(
            model, dict(accepted.complete_state), accepted.time_s
        )
        coordinate = model.getCoordinateSet().get(
            self._coordinate_mapping.coordinate_name
        )
        self._validate_model_components(model, state, muscles, muscle_names, coordinate)
        self._verify_restored_state(model, state, state_names, accepted)
        restored_state_sha256 = native_state_fingerprint(
            source_sha256=accepted.source_sha256,
            runtime_identity=accepted.runtime_identity,
            time_s=accepted.time_s,
            coordinate=accepted.coordinate,
            paths=accepted.paths,
            required_state_paths=accepted.required_state_paths,
            complete_state=accepted.complete_state,
        )
        payload = {
            "physical_model_sha256": self._source_sha256,
            "runtime_identity": self._runtime_identity,
            "input_kind": "owned-time-only-piecewise-linear-constant-interval",
            "start_time_s": proposal.start_time_s,
            "end_time_s": proposal.expected_end_time_s,
            "muscle_order": list(muscle_names),
            "raw_excitation": [value for _, value in proposal.raw_excitations],
            "analytic_configuration_provenance": (
                "Millard physical constants are checked against loaded native getters; "
                "excitation bounds and rate are caller-declared policy values"
            ),
            "restored_state_sha256": restored_state_sha256,
        }
        self._input_profile_start_state_sha256 = restored_state_sha256
        self._input_profile_sha256 = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        self._pending_model = model
        self._pending_state = state
        self._pending_muscles = muscles
        self._pending_muscle_names = muscle_names
        self._pending_coordinate = coordinate
        self._pending_start_snapshot = accepted
        self._applied = requested

    @staticmethod
    def _verify_restored_state(
        model: Any,
        state: Any,
        state_names: tuple[str, ...],
        accepted: NativeMuscleStateSnapshot,
    ) -> None:
        restored = tuple(
            (name, float(model.getStateVariableValue(state, name)))
            for name in state_names
        )
        if (
            state_names != accepted.required_state_paths
            or float(state.getTime()) != accepted.time_s
            or restored != accepted.complete_state
        ):
            raise RuntimeError(
                "controlled model did not restore exact accepted state/time"
            )

    def advance_to(self, end_time_s: float) -> NativeMuscleStateSnapshot:
        if (
            self._pending_model is None
            or self._pending_state is None
            or self._pending_muscles is None
            or self._pending_coordinate is None
        ):
            raise RuntimeError("constant input profile was not prepared")
        manager = self._osim.Manager(self._pending_model)
        manager.setIntegratorAccuracy(self._integrator_accuracy)
        manager.initialize(self._pending_state)
        start_snapshot = self._pending_start_snapshot
        if start_snapshot is None:
            raise RuntimeError("accepted start state is unavailable")
        self._verify_restored_state(
            self._pending_model,
            self._pending_state,
            start_snapshot.required_state_paths,
            start_snapshot,
        )
        self._verify_raw_readback(
            self._pending_muscles,
            self._pending_state,
            self._pending_muscle_names,
            self._applied,
        )
        advanced_state = manager.integrate(float(end_time_s))
        self._verify_raw_readback(
            self._pending_muscles,
            advanced_state,
            self._pending_muscle_names,
            self._applied,
        )
        self._model = self._pending_model
        self._state = advanced_state
        self._muscles = self._pending_muscles
        self._muscle_names = self._pending_muscle_names
        self._coordinate = self._pending_coordinate
        self._pending_model = None
        self._pending_state = None
        self._pending_muscles = None
        self._pending_coordinate = None
        self._pending_start_snapshot = None
        return self.observe()

    @staticmethod
    def _verify_raw_readback(
        muscles: Any,
        state: Any,
        muscle_names: tuple[str, ...],
        expected: Mapping[str, float],
    ) -> None:
        actual = {
            str(muscles.get(index).getName()): float(
                muscles.get(index).getExcitation(state)
            )
            for index in range(muscles.getSize())
        }
        if tuple(actual) != muscle_names or actual != dict(expected):
            raise RuntimeError("native time-only player did not apply exact raw inputs")

    def applied_raw_excitations(self) -> Mapping[str, float]:
        return {
            str(self._muscles.get(i).getName()): float(
                self._muscles.get(i).getExcitation(self._state)
            )
            for i in range(self._muscles.getSize())
        }


def load_pinned_opensim_backend(
    *,
    osim: Any,
    source_path: str | Path,
    expected_source_sha256: str,
    expected_runtime_identity: str,
    coordinate_mapping: NativeCoordinateMapping,
    analytic_by_muscle: Mapping[str, AnalyticRecruitmentConfig],
    initial_state: Mapping[str, float],
    initial_time_s: float,
    integrator_accuracy: float,
) -> _OpenSimNativeEffortBackend:
    """Load pinned bytes and restore a complete named state without equilibration."""

    if not math.isfinite(initial_time_s):
        raise ValueError("initial native time must be finite")
    path = Path(source_path).resolve(strict=True)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError("native model source hash differs from reviewed pin")
    runtime_identity = str(osim.GetVersionAndDate())
    if runtime_identity != expected_runtime_identity:
        raise ValueError("loaded OpenSim runtime differs from reviewed pin")
    model = osim.Model(str(path))
    model.finalizeConnections()
    state = model.initSystem()
    names = tuple(
        str(model.getStateVariableNames().get(i))
        for i in range(model.getStateVariableNames().getSize())
    )
    if len(names) != len(set(names)) or set(initial_state) != set(names):
        raise ValueError("initial state must bind every native state exactly once")
    values = {name: float(value) for name, value in initial_state.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("initial native state values must be finite")
    for name in names:
        model.setStateVariableValue(state, name, values[name])
    state.setTime(float(initial_time_s))
    model.realizeDynamics(state)
    return _OpenSimNativeEffortBackend(
        osim=osim,
        model=model,
        source_path=path,
        state=state,
        source_sha256=source_sha256,
        runtime_identity=runtime_identity,
        coordinate=coordinate_mapping,
        analytic_by_muscle=analytic_by_muscle,
        integrator_accuracy=integrator_accuracy,
    )


def _validate_fresh_replay_inputs(
    initial: NativeMuscleStateSnapshot,
    proposal: NativeMuscleEffortProposal,
    advanced: NativeMuscleStateSnapshot,
    state_absolute_tolerances: Mapping[str, float],
    muscle_force_absolute_tolerances_n: Mapping[str, float],
) -> tuple[dict[str, float], tuple[str, ...], dict[str, float], dict[str, float]]:
    if (
        proposal.start_time_s != initial.time_s
        or not math.isfinite(proposal.expected_end_time_s)
        or proposal.expected_end_time_s <= proposal.start_time_s
        or proposal.expected_end_time_s != advanced.time_s
        or advanced.time_s <= initial.time_s
        or proposal.source_sha256 != initial.source_sha256
        or advanced.source_sha256 != initial.source_sha256
        or proposal.runtime_identity != initial.runtime_identity
        or advanced.runtime_identity != initial.runtime_identity
        or proposal.coordinate_path != initial.coordinate.coordinate_path
        or advanced.coordinate != initial.coordinate
        or not proposal.feasible
        or proposal.allocation_status != "feasible"
        or advanced.required_state_paths != initial.required_state_paths
    ):
        raise ValueError("replay inputs do not match the accepted native interval")
    raw = dict(proposal.raw_excitations)
    muscle_names = tuple(name for name, _ in proposal.raw_excitations)
    if (
        not raw
        or len(raw) != len(proposal.raw_excitations)
        or tuple(path.name for path in initial.paths) != muscle_names
        or tuple(path.name for path in advanced.paths) != muscle_names
    ):
        raise ValueError(
            "proposal raw controls must uniquely bind ordered native muscles"
        )
    if set(state_absolute_tolerances) != set(initial.required_state_paths):
        raise ValueError("fresh replay requires a tolerance for every native state")
    if set(muscle_force_absolute_tolerances_n) != set(muscle_names):
        raise ValueError(
            "fresh replay requires a force tolerance for every native muscle"
        )
    tolerances = {
        name: float(value) for name, value in state_absolute_tolerances.items()
    }
    force_tolerances = {
        name: float(value) for name, value in muscle_force_absolute_tolerances_n.items()
    }
    if any(not math.isfinite(value) or value <= 0 for value in tolerances.values()):
        raise ValueError("state tolerances must be finite and positive")
    if any(
        not math.isfinite(value) or value <= 0 for value in force_tolerances.values()
    ):
        raise ValueError("muscle force tolerances must be finite and positive")
    if any(path.native_actuation_force_n is None for path in advanced.paths):
        raise ValueError("accepted native snapshot omits actuator force readback")
    return raw, muscle_names, tolerances, force_tolerances


def _verify_fresh_replay_output(
    replay: NativeMuscleReplayResult,
    initial: NativeMuscleStateSnapshot,
    advanced: NativeMuscleStateSnapshot,
    muscle_names: tuple[str, ...],
    times: Any,
    expected_controls: Any,
    tolerances: Mapping[str, float],
    force_tolerances: Mapping[str, float],
) -> None:
    import numpy as np

    if replay.model_sha256 != initial.source_sha256:
        raise RuntimeError("fresh replay loaded different native model bytes")
    if replay.policy.get("provider") != initial.runtime_identity:
        raise RuntimeError("fresh replay loaded a different OpenSim runtime")
    if replay.state_names != initial.required_state_paths:
        raise RuntimeError("fresh replay state order differs from admitted schema")
    if replay.muscle_names != muscle_names or not np.array_equal(replay.times, times):
        raise RuntimeError(
            "fresh replay muscle order or time grid differs from proposal"
        )
    if not np.array_equal(replay.applied_excitations, expected_controls):
        raise RuntimeError("fresh replay changed the constant raw excitation")
    if (
        replay.states.shape != (2, len(initial.required_state_paths))
        or not np.isfinite(replay.states).all()
        or replay.muscle_forces_n.shape != (2, len(muscle_names))
        or not np.isfinite(replay.muscle_forces_n).all()
    ):
        raise RuntimeError("fresh replay omitted complete finite interval observations")
    for index, (snapshot, row) in enumerate(
        ((initial, replay.states[0]), (advanced, replay.states[-1]))
    ):
        expected = dict(snapshot.complete_state)
        for column, name in enumerate(replay.state_names):
            if not math.isclose(
                float(row[column]),
                expected[name],
                rel_tol=0.0,
                abs_tol=tolerances[name],
            ):
                endpoint = "initial" if index == 0 else "advanced"
                raise RuntimeError(f"fresh replay {endpoint} state differs at {name}")
    for column, path in enumerate(advanced.paths):
        native_force = path.native_actuation_force_n
        if native_force is None:
            raise ValueError("accepted native snapshot omits actuator force readback")
        actual_force = float(replay.muscle_forces_n[-1, column])
        if not math.isclose(
            actual_force,
            native_force,
            rel_tol=0.0,
            abs_tol=force_tolerances[path.name],
        ):
            raise RuntimeError(f"fresh replay actuation differs at {path.name}")


def verify_fresh_constant_replay(
    *,
    source_path: str | Path,
    initial: NativeMuscleStateSnapshot,
    proposal: NativeMuscleEffortProposal,
    advanced: NativeMuscleStateSnapshot,
    state_absolute_tolerances: Mapping[str, float],
    muscle_force_absolute_tolerances_n: Mapping[str, float],
    integrator_accuracy: float = 1e-8,
) -> NativeMuscleReplayResult:
    """Compare caller-bound snapshots against a fresh constant-input replay."""
    import numpy as np

    from src.engines.physics_engines.opensim.python.tour_matching.muscle_replay import (
        replay_muscle_excitations,
    )

    raw, names, state_tolerances, force_tolerances = _validate_fresh_replay_inputs(
        initial,
        proposal,
        advanced,
        state_absolute_tolerances,
        muscle_force_absolute_tolerances_n,
    )
    times = np.asarray([initial.time_s, advanced.time_s], dtype=np.float64)
    controls = {
        name: np.asarray([value, value], dtype=np.float64)
        for name, value in raw.items()
    }
    replay = replay_muscle_excitations(
        source_path,
        dict(initial.complete_state),
        times,
        controls,
        accuracy=integrator_accuracy,
    )
    expected_controls = np.asarray([list(raw.values()), list(raw.values())])
    _verify_fresh_replay_output(
        replay,
        initial,
        advanced,
        names,
        times,
        expected_controls,
        state_tolerances,
        force_tolerances,
    )
    return replay
