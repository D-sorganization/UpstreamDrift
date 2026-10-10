"""Assistance-explicit T01 freezing and independent native scalar replay.

This restricted profile admits fixed-path muscles and CoordinateActuators.
It reports assistance, never a muscle-only match or physiological acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.native_replay_contracts import (
    native_replay_contract_types,
    validate_native_replay_bundle,
)
from . import muscle_replay, native_muscle_bundle
from .native_mixed_actuation import (
    MixedActuationProfile,
    NativeMixedProfile,
    admit_native_mixed_profile,
)
from .native_scalar_replay import integrate_native_scalar_replay

_ACCURACY = 1e-8


@dataclass(frozen=True)
class NativeMixedReplayResult:
    """Owned physical samples; work is sampled trapezoidal quadrature."""

    state_names: tuple[str, ...]
    channel_paths: tuple[str, ...]
    profile: NativeMixedProfile
    times: NDArray[np.float64]
    states: NDArray[np.float64]
    actuations: NDArray[np.float64]
    applied_controls: NDArray[np.float64]
    powers_w: NDArray[np.float64]
    work_j: NDArray[np.float64]
    model_sha256: str
    input_sha256: str
    initial_state_sha256: str
    policy_sha256: str
    provider_sha256: str
    mode: str = "native-mixed-assistance-replay"


def _inputs(
    times: NDArray[np.float64],
    controls: Mapping[str, NDArray[np.float64]],
    initial: Mapping[str, float],
    profile: MixedActuationProfile,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]], dict[str, float]]:
    # Share grid/state validation without imposing muscle excitation bounds on
    # signed dimensionless mechanical controls.
    grid, _, state = muscle_replay._validated_inputs(times, {}, initial, _ACCURACY)
    if grid[0] != 0:
        raise ValueError("mixed replay requires simulation-relative time zero")
    if not isinstance(profile, MixedActuationProfile):
        raise TypeError("mixed replay requires a typed profile")
    if set(controls) != {channel.path for channel in profile.channels}:
        raise ValueError("mixed controls must exactly cover declared channel paths")
    copied = {}
    for channel in profile.channels:
        values = np.array(controls[channel.path], dtype=np.float64, copy=True)
        low, high = channel.control_bounds
        if (
            values.shape != grid.shape
            or not np.isfinite(values).all()
            or np.any(values < low)
            or np.any(values > high)
        ):
            raise ValueError(
                "mixed controls exceed finite declared bounds or time shape"
            )
        copied[channel.path] = values
    return grid, copied, state


def _prepare(
    path: Path, initial: Mapping[str, float], profile: MixedActuationProfile
) -> tuple[
    Any, Any, tuple[str, ...], Any, NativeMixedProfile, dict[str, tuple[float, ...]]
]:
    import opensim as osim

    raw = path.read_bytes()
    native_muscle_bundle._validate_self_contained_source(raw)
    model = osim.Model(str(path))
    if path.read_bytes() != raw:
        raise ValueError("native source changed during load")
    state, names, domains = muscle_replay._restore_continuous_state(model, initial, 0.0)
    admitted = admit_native_mixed_profile(model, state, profile)
    if state.getNY() != len(names):
        raise ValueError("native continuous state coverage is incomplete")
    options = native_muscle_bundle._registered_options(model, state, model.getMuscles())
    return model, state, names, domains, admitted, options


def _identity(
    path: Path,
    model: Any,
    names: tuple[str, ...],
    admitted: NativeMixedProfile,
    options: Mapping[str, tuple[float, ...]],
    contracts: Any,
) -> Any:
    paths = tuple(channel.path for channel in admitted.channels)
    base = native_muscle_bundle._identity(path, model, names, paths, options, contracts)
    # Registered override variables carry the owning scalar actuator's native
    # output units. Unknown registered discrete semantics are rejected.
    units = {
        "registered-discrete:"
        + channel.path
        + "/override_actuation": channel.output_unit
        for channel in admitted.channels
    }
    components = []
    for component in base.state_schema.components:
        if component.component_id.startswith("registered-discrete:"):
            if component.component_id not in units:
                raise ValueError("mixed discrete state needs reviewed owner semantics")
            component = replace(component, unit=units[component.component_id])
        components.append(component)
    provider = hashlib.sha256(base.provider_sha256.encode())
    for name in (
        "native_mixed_replay.py",
        "native_mixed_actuation.py",
        "moco_initial_bindings.py",
    ):
        provider.update(name.encode())
        provider.update(Path(__file__).with_name(name).read_bytes())
    provider.update(admitted.sha256.encode())
    return replace(
        base,
        variant_id="reviewed-mixed-scalar-assistance",
        provider_id="opensim-native-mixed-bundle",
        provider_sha256=provider.hexdigest(),
        state_schema=replace(base.state_schema, components=tuple(components)),
    )


def _policy(identity: Any, contracts: Any) -> Any:
    return replace(
        native_muscle_bundle._policy(identity, contracts),
        initialization_policy_id="reviewed-mixed-scalar-cold-start",
        input_player_id="native-time-only-linear-scalar-command",
        contact_policy_id="explicit-muscle-coordinate-assistance-no-contact",
    )


def build_native_mixed_replay_bundle(
    model_path: str | Path,
    initial_state: Mapping[str, float],
    times: NDArray[np.float64],
    controls: Mapping[str, NDArray[np.float64]],
    profile: MixedActuationProfile,
    *,
    experiment_id: str = "native-opensim-mixed-replay",
) -> Any:
    """Freeze source, complete state, explicit assistance roles and saved controls."""
    contracts = native_replay_contract_types()
    grid, copied, initial = _inputs(times, controls, initial_state, profile)
    path = Path(model_path)
    raw = path.read_bytes()
    model, _, names, _, admitted, options = _prepare(path, initial, profile)
    identity = _identity(path, model, names, admitted, options, contracts)
    if path.read_bytes() != raw:
        raise ValueError("native source changed during bundle admission")
    paths = tuple(channel.path for channel in admitted.channels)
    capability = contracts.CapabilityDeclaration(
        "reviewed-native-mixed-scalar-cold-start",
        True,
        contracts.CapabilitySupport.SUPPORTED,
        contracts.CapabilityAvailability.AVAILABLE,
    )
    return contracts.build_experiment_replay_bundle(
        experiment_id,
        identity,
        (capability,),
        tuple((name, (initial[name],)) for name in names) + tuple(options.items()),
        tuple(contracts.InputChannel(name, name, "1") for name in paths),
        contracts.ActuationInputKind.ACTUATOR_COMMAND,
        contracts.InputInterpolation.LINEAR,
        tuple(grid),
        tuple(tuple(copied[name][i] for name in paths) for i in range(len(grid))),
        _policy(identity, contracts),
    )


def _immutable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


def replay_native_mixed_bundle(
    bundle: Any, model_path: str | Path, profile: MixedActuationProfile
) -> NativeMixedReplayResult:
    """Revalidate frozen T01 identity, then execute one fresh time-only replay."""
    contracts = native_replay_contract_types()
    bundle = validate_native_replay_bundle(bundle, contracts)
    initial, grid, controls = native_muscle_bundle._bundle_inputs(bundle)
    raw = Path(model_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != bundle.model.source_model_sha256:
        raise ValueError("native source model identity differs")
    with TemporaryDirectory(prefix="opensim-frozen-mixed-") as directory:
        snapshot = Path(directory) / "frozen.osim"
        snapshot.write_bytes(raw)
        expected = build_native_mixed_replay_bundle(
            snapshot,
            initial,
            grid,
            controls,
            profile,
            experiment_id=bundle.experiment_id,
        )
        if expected.capabilities[0] not in bundle.capabilities or any(
            getattr(bundle, field) != getattr(expected, field)
            for field in ("model", "initial_state", "policy", "input_history")
        ):
            raise ValueError(
                "mixed model, state, controls, roles or executed policy identity differs"
            )
        model, _, names, _, admitted, _ = _prepare(snapshot, initial, profile)
        actuators = model.getActuators()
        short_names = tuple(
            actuators.get(i).getName() for i in range(actuators.getSize())
        )
        if len(set(short_names)) != len(short_names):
            raise ValueError("native player requires unique registered actuator names")
        paths = tuple(channel.path for channel in admitted.channels)
        muscle_replay._configure_input_player(
            model,
            actuators,
            short_names,
            grid,
            dict(zip(short_names, (controls[path] for path in paths), strict=True)),
        )
        state, names, domains = muscle_replay._restore_continuous_state(
            model, initial, 0.0
        )
        samples = integrate_native_scalar_replay(
            model, state, names, paths, grid, domains, _ACCURACY
        )
        expected_controls = np.column_stack([controls[path] for path in paths])
        if not np.allclose(
            samples.applied_controls, expected_controls, rtol=0, atol=1e-12
        ):
            raise RuntimeError(
                "native applied controls differ from saved ordered controls"
            )
        work = np.zeros_like(samples.powers_w)
        work[1:] = np.cumsum(
            np.diff(grid)[:, None] * (samples.powers_w[:-1] + samples.powers_w[1:]) / 2,
            axis=0,
        )
        if snapshot.read_bytes() != raw:
            raise ValueError("executed native source identity differs")
        arrays = tuple(
            _immutable(value)
            for value in (
                grid,
                samples.states,
                samples.actuations,
                samples.applied_controls,
                samples.powers_w,
                work,
            )
        )
        return NativeMixedReplayResult(
            state_names=names,
            channel_paths=paths,
            profile=admitted,
            times=arrays[0],
            states=arrays[1],
            actuations=arrays[2],
            applied_controls=arrays[3],
            powers_w=arrays[4],
            work_j=arrays[5],
            model_sha256=digest,
            input_sha256=bundle.applied_input_sha256,
            initial_state_sha256=bundle.integrity.initial_state_sha256,
            policy_sha256=bundle.integrity.execution_policy_sha256,
            provider_sha256=bundle.model.provider_sha256,
        )
