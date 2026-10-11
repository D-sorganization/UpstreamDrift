"""Score explicit marker observations from a fresh OpenSim mixed replay.

This bridge restores complete named native states for model-owned FK. Its
scores are diagnostic inputs to the existing observation contract, not physics
or physiological qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from src.engines.native_replay_contracts import (
    native_replay_contract_types,
    validate_native_replay_bundle,
)
from src.shared.python.motion_matching.replay_metrics import (
    NativeMarkerPositionOutput,
    ObservedMarkerPositions,
    PositionInterpolation,
    ReplayFiveMetrics,
    ReplayObservationAlignment,
    align_native_positions_to_observations,
)

from .native_constrained_muscle import (
    DeclaredColdStart,
    observe_constrained_markers,
)
from .native_mixed_actuation import MixedActuationProfile
from .native_mixed_replay import NativeMixedReplayResult, replay_native_mixed_bundle

_FRAME_ID = "opensim-ground"
_SCHEMA = "opensim-mixed-marker-score/1.0.0"


@dataclass(frozen=True)
class NativeMixedMarkerObservation:
    """Complete-state FK output and its source-bound diagnostic identity."""

    native_output: NativeMarkerPositionOutput
    replay_identity_sha256: str
    marker_binding_sha256: str
    state_output_sha256: str
    executed_loaded_model_sha256: str
    qualification: str = field(default="unqualified", init=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "opensim-mixed-marker-observation/1.0.0",
            "qualification": self.qualification,
            "replay_identity_sha256": self.replay_identity_sha256,
            "marker_binding_sha256": self.marker_binding_sha256,
            "state_output_sha256": self.state_output_sha256,
            "executed_loaded_model_sha256": self.executed_loaded_model_sha256,
            "frame_id": self.native_output.frame_id,
            "timebase_id": self.native_output.timebase_id,
            "marker_labels": list(self.native_output.marker_labels),
            "sample_count": len(self.native_output.time_s),
        }


@dataclass(frozen=True)
class NativeMixedMarkerScore:
    """Existing five-marker metrics with explicit unqualified provenance."""

    observation: NativeMixedMarkerObservation
    alignment: ReplayObservationAlignment
    metrics: ReplayFiveMetrics
    receipt_sha256: str
    scorer_provider_sha256: str
    qualification: str = field(default="unqualified", init=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA,
            "qualification": self.qualification,
            "receipt_sha256": self.receipt_sha256,
            "scorer_provider_sha256": self.scorer_provider_sha256,
            "observation": self.observation.as_dict(),
            "alignment_identity_sha256": self.alignment.alignment_identity_sha256,
            "observation_time_grid_sha256": (
                self.alignment.observation_time_grid_sha256
            ),
            "metrics": {
                name: (float(value) if math.isfinite(float(value)) else None)
                for name, value in self.metrics.as_dict().items()
            },
        }


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(values: NDArray[np.float64]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _scorer_source_sha256() -> str:
    source = Path(align_native_positions_to_observations.__code__.co_filename)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _immutable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _immutable_mask(values: NDArray[np.bool_]) -> NDArray[np.bool_]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.bool_))
    return np.frombuffer(array.tobytes(), dtype=np.bool_).reshape(array.shape)


def _immutable_alignment(
    alignment: ReplayObservationAlignment,
) -> ReplayObservationAlignment:
    return replace(
        alignment,
        native_output_time_s=_immutable(alignment.native_output_time_s),
        observation_time_s=_immutable(alignment.observation_time_s),
        predicted_positions_m=_immutable(alignment.predicted_positions_m),
        observation_positions_m=_immutable(alignment.observation_positions_m),
        observation_valid=_immutable_mask(alignment.observation_valid),
    )


def _validate_replay(
    bundle: Any,
    declaration: DeclaredColdStart,
    profile: MixedActuationProfile,
    replay: NativeMixedReplayResult,
) -> None:
    if bundle.model.engine_id != "opensim":
        raise ValueError("OpenSim mixed marker scoring requires an OpenSim bundle")
    if declaration.source_sha256 != bundle.model.source_model_sha256:
        raise ValueError("declared cold-start source differs from replay bundle")
    if replay.model_sha256 != bundle.model.source_model_sha256:
        raise ValueError("native replay source identity differs from bundle")
    if replay.provider_sha256 != bundle.model.provider_sha256:
        raise ValueError("native replay provider identity differs from bundle")
    if replay.input_sha256 != bundle.applied_input_sha256:
        raise ValueError("native replay input identity differs from bundle")
    if replay.initial_state_sha256 != bundle.integrity.initial_state_sha256:
        raise ValueError("native replay initial state differs from bundle")
    if replay.policy_sha256 != bundle.policy_sha256:
        raise ValueError("native replay policy identity differs from bundle")
    if replay.state_names != tuple(declaration.named_state):
        raise ValueError("native replay ordered state names differ from declaration")
    channel_ids = tuple(channel.channel_id for channel in bundle.input_history.channels)
    profile_paths = tuple(channel.path for channel in profile.channels)
    if replay.channel_paths != channel_ids or channel_ids != profile_paths:
        raise ValueError("native replay ordered channels differ from frozen bundle")
    if (
        replay.profile.sha256 == ""
        or tuple(channel.path for channel in replay.profile.channels) != profile_paths
    ):
        raise ValueError("native replay compiled profile differs from declaration")
    expected_times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    expected_controls = np.asarray(bundle.input_history.values, dtype=np.float64)
    if (
        replay.times.shape != expected_times.shape
        or not np.array_equal(replay.times, expected_times)
        or replay.states.shape != (len(expected_times), len(replay.state_names))
        or replay.applied_controls.shape != expected_controls.shape
        or not np.allclose(
            replay.applied_controls, expected_controls, rtol=0, atol=1e-12
        )
        or len(replay.constraint_audits) != len(expected_times)
        or not np.isfinite(replay.states).all()
    ):
        raise ValueError("native replay output differs from frozen full horizon")
    bundle_initial = {
        item.component_id: tuple(item.values) for item in bundle.initial_state
    }
    for name, value in declaration.named_state.items():
        if bundle_initial.get(name) != (float(value),):
            raise ValueError("declared initial state differs from frozen bundle")
    declared_initial = np.asarray(
        [declaration.named_state[name] for name in replay.state_names], dtype=float
    )
    if not np.allclose(replay.states[0], declared_initial, rtol=0, atol=1e-12):
        raise ValueError(
            "native replay first sample differs from complete initial state"
        )
    loaded_identities = set()
    for index, (time, audit) in enumerate(
        zip(expected_times, replay.constraint_audits, strict=True)
    ):
        if audit.time_seconds != float(time):
            raise ValueError("native replay audit clock differs from frozen sample")
        if audit.source_sha256 not in (None, bundle.model.source_model_sha256):
            raise ValueError("native replay audit source differs from frozen sample")
        audit_names = tuple(name for name, _ in audit.named_state)
        audit_values = np.asarray([value for _, value in audit.named_state])
        if audit_names != replay.state_names or not np.allclose(
            audit_values, replay.states[index], rtol=0, atol=1e-12
        ):
            raise ValueError("native replay audit differs from saved named state")
        loaded_identities.add(audit.loaded_model_sha256)
    if len(loaded_identities) != 1:
        raise ValueError("native replay loaded-model identity changed across samples")


def _replay_identity(
    bundle: Any,
    replay: NativeMixedReplayResult,
    bindings: Mapping[str, tuple[str, tuple[float, float, float]]],
    observer_sha256: str,
    scorer_sha256: str,
) -> tuple[str, str, str, str]:
    provider_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    binding_sha256 = _digest(
        tuple(
            (label, frame, tuple(offset)) for label, (frame, offset) in bindings.items()
        )
    )
    state_sha256 = _digest(
        {
            "states": _array_digest(replay.states),
            "actuations": _array_digest(replay.actuations),
            "applied_controls": _array_digest(replay.applied_controls),
            "constraint_audits": [asdict(audit) for audit in replay.constraint_audits],
        }
    )
    replay_sha256 = _digest(
        {
            "schema_version": "opensim-mixed-replay-marker-source/1.0.0",
            "source_model_sha256": bundle.model.source_model_sha256,
            "loaded_native_model_sha256": bundle.model.loaded_native_model_sha256,
            "provider_id": bundle.model.provider_id,
            "provider_sha256": bundle.model.provider_sha256,
            "state_schema_sha256": bundle.state_schema_sha256,
            "initial_state_sha256": bundle.integrity.initial_state_sha256,
            "input_channel_schema_sha256": bundle.input_channel_schema_sha256,
            "applied_input_sha256": bundle.applied_input_sha256,
            "policy_sha256": bundle.policy_sha256,
            "time_grid_sha256": bundle.time_grid_sha256,
            "profile_sha256": replay.profile.sha256,
            "state_output_sha256": state_sha256,
            "marker_binding_sha256": binding_sha256,
            "observer_provider_sha256": provider_sha256,
            "observer_sha256": observer_sha256,
            "scorer_provider_sha256": scorer_sha256,
            "executed_loaded_model_sha256": replay.constraint_audits[
                0
            ].loaded_model_sha256,
        }
    )
    return (
        replay_sha256,
        binding_sha256,
        state_sha256,
        replay.constraint_audits[0].loaded_model_sha256,
    )


def _marker_observation(
    bundle: Any,
    declaration: DeclaredColdStart,
    profile: MixedActuationProfile,
    replay: NativeMixedReplayResult,
    bindings: Mapping[str, tuple[str, tuple[float, float, float]]],
) -> NativeMixedMarkerObservation:
    _validate_replay(bundle, declaration, profile, replay)
    if bundle.input_history.timebase_id != "simulation_relative":
        raise ValueError("native marker observation requires simulation-relative time")
    indices = np.arange(len(replay.times), dtype=np.intp)
    positions, observer_sha256 = observe_constrained_markers(
        declaration, replay, bindings, indices
    )
    times = _immutable(replay.times)
    marker_positions = _immutable(positions)
    output = NativeMarkerPositionOutput(
        times,
        marker_positions,
        tuple(bindings),
        _FRAME_ID,
        bundle.input_history.timebase_id,
    )
    replay_sha256, bindings_sha256, state_sha256, loaded_model_sha256 = (
        _replay_identity(
            bundle, replay, bindings, observer_sha256, _scorer_source_sha256()
        )
    )
    return NativeMixedMarkerObservation(
        output, replay_sha256, bindings_sha256, state_sha256, loaded_model_sha256
    )


def replay_and_score_native_mixed_markers(
    bundle: Any,
    model_path: str | Path,
    profile: MixedActuationProfile,
    declaration: DeclaredColdStart,
    bindings: Mapping[str, tuple[str, tuple[float, float, float]]],
    observations: ObservedMarkerPositions,
    *,
    interpolation: PositionInterpolation = PositionInterpolation.LINEAR_POSITION,
) -> NativeMixedMarkerScore:
    """Freshly replay, observe model-owned markers, and run canonical metrics."""
    contracts = native_replay_contract_types()
    bundle = validate_native_replay_bundle(bundle, contracts)
    if not bindings:
        raise ValueError("explicit native marker attachments are required")
    if observations.frame_id != _FRAME_ID:
        raise ValueError(
            "marker observations must use the declared OpenSim ground frame"
        )
    if observations.timebase_id != bundle.input_history.timebase_id:
        raise ValueError("marker observation timebase differs from the replay bundle")
    replay = replay_native_mixed_bundle(
        bundle, model_path, profile, constrained_cold_start=declaration
    )
    marker_observation = _marker_observation(
        bundle, declaration, profile, replay, bindings
    )
    alignment = align_native_positions_to_observations(
        marker_observation.native_output,
        observations,
        interpolation=interpolation,
        source_identity_sha256=marker_observation.replay_identity_sha256,
    )
    alignment = _immutable_alignment(alignment)
    metrics = alignment.compute_replay_five_metrics()
    scorer_sha256 = _scorer_source_sha256()
    receipt = _digest(
        {
            "schema_version": _SCHEMA,
            "qualification": "unqualified",
            "observation": marker_observation.as_dict(),
            "alignment_identity_sha256": alignment.alignment_identity_sha256,
            "scorer_provider_sha256": scorer_sha256,
        }
    )
    return NativeMixedMarkerScore(
        marker_observation, alignment, metrics, receipt, scorer_sha256
    )
