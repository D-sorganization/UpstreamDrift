"""T01 admission for a reviewed OpenSim cold-start muscle replay subset.

Registered discrete/modeling options are bound explicitly. This is not an
arbitrary SimTK State serializer: plugins, moving paths, constraints, contact
and unreviewed component classes require separate policies.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from pyexpat import ExpatError

import numpy as np
from defusedxml.common import DefusedXmlException
from defusedxml.expatbuilder import parseString as safe_parse_string
from numpy.typing import NDArray

from src.engines.native_replay_contracts import (
    native_replay_admission_bytes,
    native_replay_contract_types,
    validate_native_replay_bundle,
)

import src.engines.physics_engines.opensim.python.tour_matching.muscle_replay as muscle_replay
from src.engines.physics_engines.opensim.python.tour_matching import (
    native_visual_resources,
)

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

_VERSION = "1.0.0"
_ACCURACY = 1e-8
_COMPONENTS = frozenset(
    [
        "Ground",
        "FrameGeometry",
        "WrapObjectSet",
        "BodySet",
        "Body",
        "JointSet",
        "SliderJoint",
        "PinJoint",
        "Coordinate",
        "PhysicalOffsetFrame",
        "ControllerSet",
        "ConstraintSet",
        "ForceSet",
        "Millard2012EquilibriumMuscle",
        "Thelen2003Muscle",
        "MuscleFixedWidthPennationModel",
        "MuscleFirstOrderActivationDynamicModel",
        "GeometryPath",
        "PathPoint",
        "Station",
        "MarkerSet",
        "Marker",
        "ContactGeometrySet",
        "ProbeSet",
        "ComponentSet",
    ]
)


def _registered_options(
    model: Any, state: Any, muscles: Any
) -> dict[str, tuple[float, ...]]:
    """Bind registered discrete state and active native force/modeling flags."""
    options: dict[str, tuple[float, ...]] = {}
    for kind, names, getter in (
        ("discrete", model.getDiscreteVariableNames(), model.getDiscreteVariableValue),
        ("option", model.getModelingOptionNames(), model.getModelingOption),
    ):
        for i in range(names.getSize()):
            name = names.get(i)
            value = float(getter(state, name))
            if not np.isfinite(value):
                raise ValueError("registered native state must be finite")
            options[f"registered-{kind}:{name}"] = (value,)
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        flags = tuple(
            float(value)
            for value in (
                muscle.isActuationOverridden(state),
                muscle.appliesForce(state),
                muscle.getIgnoreTendonCompliance(state),
                muscle.getIgnoreActivationDynamics(state),
            )
        )
        if flags[0] != 0 or flags[1] != 1:
            raise ValueError(
                "native policy forbids disabled or overridden muscle force"
            )
        options[muscle.getAbsolutePathString() + "/native-options"] = flags
    return options


def _audit_components(
    model: Any, visual_mesh_paths: frozenset[str] = frozenset()
) -> None:
    """Reject unreviewed component semantics before and after native initialization."""
    observed_visual_meshes: set[str] = set()
    for component in tuple(model.getComponentsList()):
        kind = component.getConcreteClassName()
        path = component.getAbsolutePathString()
        if kind == "Mesh" and path in visual_mesh_paths:
            observed_visual_meshes.add(path)
            continue
        if kind not in _COMPONENTS:
            raise ValueError(
                "component needs a separately reviewed native replay policy"
            )
    if observed_visual_meshes != set(visual_mesh_paths):
        raise ValueError("source-bound visual Mesh component ownership differs")


def _validate_self_contained_source(
    raw: bytes,
    visual_resources: tuple[native_visual_resources.SourceVisualResource, ...] = (),
) -> None:
    """Require an inline model and an exact source-owned visual resource list."""
    try:
        document = safe_parse_string(
            raw,
            namespaces=False,
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
    except (ExpatError, DefusedXmlException) as error:
        raise ValueError("native source XML must be self-contained") from error
    root = document.documentElement
    models = [
        child for child in root.childNodes if getattr(child, "tagName", None) == "Model"
    ]
    if root.tagName != "OpenSimDocument" or len(models) != 1:
        raise ValueError("native source must contain one inline OpenSim Model")
    elements = (root, *root.getElementsByTagName("*"))
    for element in elements:
        # This parser deliberately disables namespace processing because
        # OpenSim uses C++-qualified component names such as
        # ``HuntCrossleyForce::ContactParametersSet``. Resource checks still
        # inspect the local portion of a literal prefixed name so a namespace
        # prefix cannot hide an external reference from this gate.
        tag = element.tagName.rsplit(":", 1)[-1].lower()
        if tag in {"file", "filename", "file_name"} or tag.endswith("_file"):
            if not visual_resources or tag != "mesh_file":
                raise ValueError(
                    "external source resources need a separate identity policy"
                )
        if any(
            element.attributes.item(index).name.rsplit(":", 1)[-1].lower()
            in {"file", "filename", "href"}
            for index in range(element.attributes.length)
        ):
            raise ValueError(
                "external source references need a separate identity policy"
            )
    if visual_resources:
        references = native_visual_resources.discover_visual_meshes(raw)
        expected = tuple(
            sorted(
                (
                    item.relative_path,
                    item.component_path,
                )
                for item in visual_resources
            )
        )
        actual = tuple(
            sorted((item.relative_path, item.component_path) for item in references)
        )
        if actual != expected:
            raise ValueError(
                "external source resources need an exact source-bound visual closure"
            )


def _prepare(
    path: Path,
    initial: Mapping[str, float],
    controls: Mapping[str, NDArray[np.float64]],
    visual_resources: tuple[native_visual_resources.SourceVisualResource, ...] = (),
) -> tuple[Any, Any, tuple[str, ...], tuple[str, ...], dict[str, tuple[float, ...]]]:
    import opensim as osim

    raw = path.read_bytes()
    _validate_self_contained_source(raw, visual_resources)
    if native_visual_resources.capture_visual_resources(path, raw) != visual_resources:
        raise ValueError("native visual resource closure changed during admission")
    visual_paths = native_visual_resources.visual_mesh_component_paths(visual_resources)
    model = osim.Model(str(path))
    if raw != path.read_bytes():
        raise ValueError("source model changed during native load")
    if native_visual_resources.capture_visual_resources(path, raw) != visual_resources:
        raise ValueError("native visual resources changed during model load")
    _audit_components(model, visual_paths)
    muscles, muscle_names = muscle_replay._admit_native_muscles(model, controls, ())
    coordinates = model.getCoordinateSet()
    if any(
        coordinates.get(i).getDefaultClamped() for i in range(coordinates.getSize())
    ):
        raise ValueError("clamped coordinates require another native policy")
    state, names, _ = muscle_replay._restore_continuous_state(model, initial, 0.0)
    _audit_components(model, visual_paths)
    if state.getNY() != len(names):
        raise ValueError("native continuous state coverage is incomplete")
    model.realizeDynamics(state)
    options = _registered_options(model, state, muscles)
    return model, state, names, muscle_names, options


def _coordinate_units(
    model: Any,
    *,
    native_mechanics_profile: Any | None = None,
    coupled_rotation_paths: frozenset[str] = frozenset(),
) -> dict[str, str]:
    import opensim as osim

    units = {}
    observed_coupled: set[str] = set()
    coordinates = model.getCoordinateSet()
    for i in range(coordinates.getSize()):
        coordinate = coordinates.get(i)
        motion = coordinate.getMotionType()
        if motion == osim.Coordinate.Rotational:
            unit = "rad"
        elif motion == osim.Coordinate.Translational:
            unit = "m"
        elif (
            motion == osim.Coordinate.Coupled
            and coordinate.getAbsolutePathString() in coupled_rotation_paths
        ):
            unit = "rad"
            observed_coupled.add(coordinate.getAbsolutePathString())
        else:
            custom_unit = _custom_joint_coordinate_unit(
                model, coordinate, native_mechanics_profile, osim
            )
            if custom_unit is None:
                raise ValueError("coordinate motion units need another native policy")
            unit = custom_unit
        prefix = coordinate.getAbsolutePathString()
        units[prefix + "/value"] = unit
        units[prefix + "/speed"] = unit + "/s"
    if observed_coupled != coupled_rotation_paths:
        raise ValueError("declared coupled rotation chart differs from native source")
    return units


def _custom_joint_coordinate_unit(
    model: Any, coordinate: Any, profile: Any | None, osim: Any
) -> str | None:
    if profile is None:
        return None
    delegated_paths = {
        item.absolute_path
        for item in profile.components
        if item.concrete_class == "CustomJoint"
    }
    coordinate_path = coordinate.getAbsolutePathString()
    units: set[str] = set()
    joints = model.getJointSet()
    for index in range(joints.getSize()):
        joint = osim.CustomJoint.safeDownCast(joints.get(index))
        if joint is None or joint.getAbsolutePathString() not in delegated_paths:
            continue
        transform = joint.getSpatialTransform()
        for axis_index in range(6):
            axis = transform.getTransformAxis(axis_index)
            names = axis.getCoordinateNames()
            if any(
                str(names.getValue(name_index)) == coordinate.getName()
                for name_index in range(names.size())
            ):
                units.add("rad" if axis_index < 3 else "m")
    return next(iter(units)) if len(units) == 1 else None


def _state_specs(
    model: Any,
    names: tuple[str, ...],
    options: Mapping[str, tuple[float, ...]],
    contracts: Any,
    native_mechanics_profile: Any | None = None,
    coupled_rotation_paths: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    units = _coordinate_units(
        model,
        native_mechanics_profile=native_mechanics_profile,
        coupled_rotation_paths=coupled_rotation_paths,
    )
    roles = {
        "value": contracts.StateComponentRole.POSITION,
        "speed": contracts.StateComponentRole.VELOCITY,
        "activation": contracts.StateComponentRole.MUSCLE_ACTIVATION,
        "fiber_length": contracts.StateComponentRole.MUSCLE_FIBER_STATE,
    }
    specs = []
    for name in names:
        suffix = name.rsplit("/", 1)[-1]
        if suffix not in roles:
            raise ValueError("continuous state semantics need another native policy")
        unit = units.get(name)
        if unit is None:
            if suffix not in {"activation", "fiber_length"}:
                raise ValueError("coordinate state units are not registered")
            unit = "1" if suffix == "activation" else "m"
        specs.append(
            contracts.StateComponentSpec(
                name, roles[suffix], 1, unit, "opensim-named-continuous"
            )
        )
    specs.extend(
        contracts.StateComponentSpec(
            name,
            contracts.StateComponentRole.AUXILIARY,
            len(values),
            "N" if name.startswith("registered-discrete:") else "1",
            "native-cold-start-registered-options",
        )
        for name, values in options.items()
    )
    return tuple(specs)


def _identity(
    path: Path,
    model: Any,
    names: tuple[str, ...],
    muscles: tuple[str, ...],
    options: Mapping[str, tuple[float, ...]],
    contracts: Any,
    native_mechanics_profile: Any | None = None,
    visual_resources: tuple[native_visual_resources.SourceVisualResource, ...] = (),
    coupled_rotation_paths: frozenset[str] = frozenset(),
) -> Any:
    import opensim as osim

    provider = hashlib.sha256(native_replay_admission_bytes())
    native_extensions = tuple(sorted(Path(osim.__file__).parent.glob("*.pyd")))
    if not native_extensions:
        raise ValueError("native OpenSim binary identity is unavailable")
    artifacts = (
        Path(__file__),
        Path(muscle_replay.__file__),
        Path(__file__).with_name("native_scalar_replay.py"),
        *native_extensions,
    )
    for artifact in artifacts:
        provider.update(artifact.name.encode())
        provider.update(artifact.read_bytes())
    if visual_resources:
        artifact = Path(native_visual_resources.__file__)
        provider.update(artifact.name.encode())
        provider.update(artifact.read_bytes())
    provider.update(osim.GetVersionAndDate().encode())
    version = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:-.*)?", osim.GetVersion())
    if version is None:
        raise ValueError("native OpenSim version format is unreviewed")
    semantic_version = ".".join((version[1], version[2], version[3] or "0"))
    return contracts.ModelIdentity(
        "opensim",
        "native-model",
        "reviewed-cold-start-muscles",
        _VERSION,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "opensim-native-muscle-bundle",
        semantic_version,
        provider.hexdigest(),
        contracts.InitialStateSchema(
            "opensim-native-continuous-and-registered-options",
            _VERSION,
            _state_specs(
                model,
                names,
                options,
                contracts,
                native_mechanics_profile,
                coupled_rotation_paths,
            ),
        ),
        muscles,
        hashlib.sha256(model.dump().encode()).hexdigest(),
    )


def _policy(identity: Any, contracts: Any) -> Any:
    return contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id="opensim-native-manager",
        solver_version=identity.provider_version,
        integration_method="RungeKuttaMerson-accuracy-1e-8",
        step_policy="adaptive",
        step_size_seconds=None,
        initialization_policy_id="reviewed-fresh-model-continuous-and-registered-options",
        initialization_policy_version=_VERSION,
        input_player_id="native-time-only-linear-excitation",
        input_player_version=_VERSION,
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="muscle-and-gravity-only-no-contact",
        contact_policy_version=_VERSION,
        contact_policy_sha256=identity.loaded_native_model_sha256,
    )


def build_native_muscle_replay_bundle(
    model_path: str | Path,
    initial_state: Mapping[str, float],
    times: NDArray[np.float64],
    excitations: Mapping[str, NDArray[np.float64]],
    *,
    experiment_id: str = "native-opensim-muscle-replay",
) -> ExperimentReplayBundle:
    """Freeze complete admitted state, registered options and native excitation inputs."""
    contracts = native_replay_contract_types()
    grid, controls, initial = muscle_replay._validated_inputs(
        times, excitations, initial_state, _ACCURACY
    )
    if grid[0] != 0:
        raise ValueError("native bundle requires simulation-relative time zero")
    path = Path(model_path)
    raw = path.read_bytes()
    visual_resources = native_visual_resources.capture_visual_resources(path, raw)
    model, _, names, muscles, options = _prepare(
        path, initial, controls, visual_resources
    )
    identity = _identity(
        path, model, names, muscles, options, contracts, visual_resources
    )
    values = tuple((name, (initial[name],)) for name in names) + tuple(options.items())
    capability = contracts.CapabilityDeclaration(
        "reviewed-native-muscle-cold-start",
        True,
        contracts.CapabilitySupport.SUPPORTED,
        contracts.CapabilityAvailability.AVAILABLE,
    )
    bundle = contracts.build_experiment_replay_bundle(
        experiment_id,
        identity,
        (capability,),
        values,
        tuple(contracts.InputChannel(name, name, "1") for name in muscles),
        contracts.ActuationInputKind.MUSCLE_EXCITATION,
        contracts.InputInterpolation.LINEAR,
        tuple(grid),
        tuple(tuple(controls[name][i] for name in muscles) for i in range(len(grid))),
        _policy(identity, contracts),
    )
    if visual_resources:
        return native_visual_resources.make_resource_envelope(
            bundle, identity.source_model_sha256, visual_resources
        )
    return bundle


def _bundle_inputs(
    bundle: Any,
) -> tuple[dict[str, float], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Decode shared complete continuous state and ordered T01 histories."""
    values = {item.component_id: item.values for item in bundle.initial_state}
    initial = {
        name: value[0]
        for name, value in values.items()
        if not name.startswith("registered-") and not name.endswith("/native-options")
    }
    history = bundle.input_history
    grid = np.array(history.time_seconds, dtype=float)
    controls = {
        channel.channel_id: np.array([row[i] for row in history.values], dtype=float)
        for i, channel in enumerate(history.channels)
    }
    return initial, grid, controls


def replay_native_muscle_bundle(
    bundle: ExperimentReplayBundle
    | native_visual_resources.NativeMuscleResourceEnvelope,
    model_path: str | Path,
) -> muscle_replay.NativeMuscleReplayResult:
    """Revalidate identity and execute fresh native integration with no feedback."""
    contracts = native_replay_contract_types()
    envelope = (
        bundle
        if isinstance(bundle, native_visual_resources.NativeMuscleResourceEnvelope)
        else None
    )
    shared_bundle = envelope.replay_bundle if envelope is not None else bundle
    shared_bundle = validate_native_replay_bundle(shared_bundle, contracts)
    initial, grid, controls = _bundle_inputs(shared_bundle)
    path = Path(model_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != shared_bundle.model.source_model_sha256:
        raise ValueError("native source model identity differs")
    if envelope is None:
        _validate_self_contained_source(raw)
        with TemporaryDirectory(prefix="opensim-frozen-replay-") as directory:
            # This policy excludes external-resource component classes. Owned frozen
            # bytes bind admission and execution without reopening the caller's file.
            snapshot = Path(directory) / "frozen.osim"
            snapshot.write_bytes(raw)
            expected = build_native_muscle_replay_bundle(
                snapshot,
                initial,
                grid,
                controls,
                experiment_id=shared_bundle.experiment_id,
            )
            if expected != shared_bundle:
                raise ValueError(
                    "native model, state, registered options or executed policy identity differs"
                )
            result = muscle_replay.replay_muscle_excitations(
                snapshot, initial, grid, controls, accuracy=_ACCURACY
            )
            if (
                result.model_sha256 != shared_bundle.model.source_model_sha256
                or snapshot.read_bytes() != raw
            ):
                raise ValueError("executed native source model identity differs")
            return result

    if hashlib.sha256(raw).hexdigest() != envelope.source_model_sha256:
        raise ValueError("visual resource envelope source identity differs")
    if (
        native_visual_resources.capture_visual_resources(path, raw)
        != envelope.resources
    ):
        raise ValueError(
            "visual resource closure differs from its source-bound manifest"
        )
    with native_visual_resources.freeze_source_and_resources(
        path, raw, envelope.resources
    ) as snapshot:
        expected = build_native_muscle_replay_bundle(
            snapshot,
            initial,
            grid,
            controls,
            experiment_id=shared_bundle.experiment_id,
        )
        if expected != envelope:
            raise ValueError("native source or visual resource identity differs")
        result = muscle_replay.replay_muscle_excitations(
            snapshot, initial, grid, controls, accuracy=_ACCURACY
        )
        if (
            result.model_sha256 != shared_bundle.model.source_model_sha256
            or snapshot.read_bytes() != raw
            or native_visual_resources.capture_visual_resources(snapshot, raw)
            != envelope.resources
        ):
            raise ValueError("executed native source/resource identity differs")
        return result


def dumps_native_muscle_resource_bundle(
    envelope: native_visual_resources.NativeMuscleResourceEnvelope,
) -> str:
    """Serialize a resource-bearing bundle using its explicit adapter schema."""
    return native_visual_resources.dumps_resource_envelope(
        envelope, native_replay_contract_types()
    )


def loads_native_muscle_resource_bundle(
    text: str,
) -> native_visual_resources.NativeMuscleResourceEnvelope:
    """Load a versioned resource envelope without changing shared T01 schema."""
    return native_visual_resources.loads_resource_envelope(
        text, native_replay_contract_types()
    )
