"""Strict direct-model MuJoCo command replay seam for inspected environments.

This module deliberately does not create a Gym environment. A caller supplies
an inspected direct MuJoCo environment factory and an identity binding; the
factory receives the original model path so relative includes and assets keep
their resource root. The seam is useful for testing native command replay, but
it does not certify that any particular MyoSuite SDK constructor is compatible.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, Protocol
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray
from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from src.engines.native_replay_contracts import (
    native_replay_admission_bytes,
    require_no_global_mujoco_callbacks,
    validate_native_replay_bundle,
)

_ACTUATOR_MODEL_FIELDS = (
    "actuator_acc0",
    "actuator_actadr",
    "actuator_actearly",
    "actuator_actlimited",
    "actuator_actnum",
    "actuator_actrange",
    "actuator_armature",
    "actuator_biasprm",
    "actuator_biastype",
    "actuator_cranklength",
    "actuator_ctrllimited",
    "actuator_ctrlrange",
    "actuator_damping",
    "actuator_dampingpoly",
    "actuator_delay",
    "actuator_dynprm",
    "actuator_dyntype",
    "actuator_forcelimited",
    "actuator_forcerange",
    "actuator_gainprm",
    "actuator_gaintype",
    "actuator_gear",
    "actuator_group",
    "actuator_history",
    "actuator_historyadr",
    "actuator_length0",
    "actuator_lengthrange",
    "actuator_plugin",
    "actuator_trnid",
    "actuator_trntype",
    "actuator_user",
)

_CONTACT_GEOMETRY_FIELDS = (
    "geom_bodyid",
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_priority",
    "geom_friction",
    "geom_solmix",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "pair_geom1",
    "pair_geom2",
    "pair_dim",
    "pair_friction",
    "pair_solref",
    "pair_solreffriction",
    "pair_solimp",
    "pair_margin",
    "pair_gap",
)

_CONTACT_OPTION_FIELDS = (
    "cone",
    "disableflags",
    "enableflags",
    "impratio",
    "ls_tolerance",
    "noslip_iterations",
    "noslip_tolerance",
    "sdf_initpoints",
    "sdf_iterations",
    "tolerance",
)

_UNSUPPORTED_MJCF_ELEMENTS = frozenset(
    {"attach", "flexcomp", "plugin", "extension", "hfield", "skin"}
)
_SUPPORTED_FILE_ELEMENTS = frozenset({"include", "mesh", "texture"})
_TEXTURE_FACE_ATTRIBUTES = (
    "fileback",
    "filedown",
    "filefront",
    "fileleft",
    "fileright",
    "fileup",
)


def _require_sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _clock_close(
    actual: float, expected: float, step_size: float, accumulated_steps: int = 0
) -> bool:
    if not np.isfinite(actual) or not np.isfinite(expected):
        return False
    if actual == expected:
        return True
    scale = max(abs(actual), abs(expected), abs(step_size))
    tolerance = (4.0 + 2.0 * accumulated_steps) * float(np.spacing(scale))
    return tolerance < step_size / 2.0 and abs(actual - expected) <= tolerance


def _time_grid_matches_step(times: NDArray[np.float64], step_size: float) -> bool:
    if times.ndim != 1 or times.size < 2 or times[0] != 0.0:
        return False
    if not np.isfinite(times).all() or step_size <= 0.0:
        return False
    return all(
        _clock_close(
            float(interval),
            step_size,
            max(step_size, abs(float(times[index])), abs(float(times[index + 1]))),
        )
        for index, interval in enumerate(np.diff(times))
    )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _safe_relative_reference(value: str, label: str) -> Path:
    """Reject URI and absolute resource forms before native parsing can load them."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    windows_path = PureWindowsPath(value)
    if (
        urlsplit(value).scheme
        or Path(value).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or value.startswith(("//", "\\\\"))
    ):
        raise ValueError(f"{label} must use a contained relative disk path")
    return Path(value)


def _contained_resource_file(path: Path, root: Path) -> Path:
    """Resolve a readable regular file while rejecting symlink escapes."""
    resolved_root = root.resolve(strict=True)
    candidate = path.resolve(strict=False)
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"model source/resource is outside its root: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"model source/resource is missing: {path}") from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"model source/resource is outside its root: {path}")
    return resolved


def _xml_source(path: Path, *, is_entry: bool) -> Any:
    """Parse one source with hardened XML and reject unreviewed loaders."""
    try:
        document = SafeET.fromstring(path.read_bytes())
    except (OSError, SafeET.ParseError, DefusedXmlException) as exc:
        raise ValueError(f"cannot safely parse MJCF source: {path}") from exc
    actual_root = str(document.tag).rsplit("}", 1)[-1]
    allowed_roots = {"mujoco"} if is_entry else {"mujoco", "mujocoinclude"}
    if actual_root not in allowed_roots:
        raise ValueError("MJCF source must use a supported MuJoCo root element")
    for element in document.iter():
        tag = str(element.tag).rsplit("}", 1)[-1]
        if tag in _UNSUPPORTED_MJCF_ELEMENTS:
            raise ValueError(f"unsupported MJCF source loader: {tag}")
        if element.attrib.get("plugin"):
            raise ValueError("plugin-backed MJCF elements are unsupported")
        if tag == "compiler":
            if element.attrib.get("strippath", "false").lower() == "true":
                raise ValueError("MJCF compiler strippath is unsupported")
            for directory_name in ("meshdir", "texturedir"):
                directory = element.attrib.get(directory_name, "")
                if directory:
                    _safe_relative_reference(directory, directory_name)
            if element.attrib.get("assetdir"):
                raise ValueError("MJCF compiler assetdir is unsupported")
        if "file" in element.attrib:
            if tag not in _SUPPORTED_FILE_ELEMENTS:
                raise ValueError(f"unsupported file-bearing MJCF element: {tag}")
            reference = _safe_relative_reference(element.attrib["file"], f"{tag} file")
            suffixes = {
                "include": {".xml"},
                "mesh": {".stl", ".msh"},
                "texture": {".png"},
            }
            if reference.suffix.lower() not in suffixes[tag]:
                raise ValueError(f"unsupported {tag} resource format")
        if "content_type" in element.attrib:
            raise ValueError("MJCF resource content_type overrides are unsupported")
        if any(attribute in element.attrib for attribute in _TEXTURE_FACE_ATTRIBUTES):
            if tag != "texture":
                raise ValueError("file-backed texture faces require a texture element")
            for attribute in _TEXTURE_FACE_ATTRIBUTES:
                if attribute not in element.attrib:
                    continue
                reference = _safe_relative_reference(
                    element.attrib[attribute], f"texture {attribute}"
                )
                if reference.suffix.lower() != ".png":
                    raise ValueError("texture face resources must be PNG")
        if "cubefiles" in element.attrib:
            if tag != "texture":
                raise ValueError("cubefiles are supported only on textures")
            for file_name in element.attrib["cubefiles"].split():
                reference = _safe_relative_reference(file_name, "texture cube file")
                if reference.suffix.lower() != ".png":
                    raise ValueError("texture cube files must be PNG")
    return document


def _included_xml_sources(entry: Path, root: Path) -> set[Path]:
    """Discover disk includes using the tested model-root-first resolution."""
    entry_path = _contained_resource_file(entry, root)
    if entry_path.suffix.lower() != ".xml":
        raise ValueError("bounded direct resource admission supports MJCF XML only")
    model_directory = entry_path.parent
    sources: set[Path] = set()
    pending = [entry_path]
    while pending:
        current = _contained_resource_file(pending.pop(), root)
        if current in sources:
            continue
        sources.add(current)
        document = _xml_source(current, is_entry=current == entry_path)
        for element in document.iter():
            if str(element.tag).rsplit("}", 1)[-1] != "include":
                continue
            reference = _safe_relative_reference(
                element.attrib.get("file", ""), "include file"
            )
            if reference.suffix.lower() != ".xml":
                raise ValueError("MJCF includes must reference XML files")
            model_candidate = model_directory / reference
            selected = (
                model_candidate
                if model_candidate.is_file()
                else current.parent / reference
            )
            pending.append(_contained_resource_file(selected, root))
    return sources


def _native_asset_files(spec: Any, root: Path) -> set[Path]:
    """Resolve the bounded MjSpec mesh/texture surface, including cube faces."""
    files: set[Path] = set()
    model_directory = Path(spec.modelfiledir)
    for assets, directory_name in (
        (spec.meshes, "meshdir"),
        (spec.textures, "texturedir"),
    ):
        for asset in assets:
            compiler = asset.compiler
            directory_value = getattr(compiler, directory_name)
            directory = (
                _safe_relative_reference(directory_value, directory_name)
                if directory_value
                else Path()
            )
            names = [asset.file]
            if directory_name == "texturedir":
                names.extend(asset.cubefiles)
            for name in names:
                if not name:
                    continue
                reference = _safe_relative_reference(name, "native asset path")
                suffix = reference.suffix.lower()
                allowed_suffixes = (
                    {".stl", ".msh"} if directory_name == "meshdir" else {".png"}
                )
                if suffix not in allowed_suffixes:
                    raise ValueError(
                        f"unsupported native asset format for {directory_name}: {suffix}"
                    )
                files.add(
                    _contained_resource_file(
                        model_directory / directory / reference, root
                    )
                )
    return files


def _discover_native_resource_files(entry: Path, root: Path) -> tuple[set[Path], Any]:
    """Audit source loaders, then ask pinned native MjSpec for asset semantics."""
    import mujoco as mj

    if mj.__version__ != "3.8.0":
        raise ValueError("resource discovery is pinned to MuJoCo 3.8.0")
    files = _included_xml_sources(entry, root)
    spec = mj.MjSpec.from_file(str(entry))
    if spec.strippath or spec.hfields or spec.skins or spec.assets:
        raise ValueError("native resource closure excludes strippath/hfield/skin/VFS")
    files.update(_native_asset_files(spec, root))
    return files, spec


def _verified_discovered_resource_hashes(
    discovered: set[Path], root: Path, declared: dict[str, str]
) -> dict[str, str]:
    """Require each native-discovered file to match the supplied manifest."""
    verified: dict[str, str] = {}
    for candidate in discovered:
        path = _contained_resource_file(candidate, root)
        relative_path = path.relative_to(root.resolve(strict=True)).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if declared.get(relative_path) != digest:
            raise ValueError(
                f"discovered native source/resource is undeclared: {relative_path}"
            )
        verified[relative_path] = digest
    return verified


@dataclass(frozen=True, slots=True)
class DeclaredModelResource:
    """A file in the explicit source-model resource closure."""

    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_text(self.relative_path, "model resource path")
        _require_sha256(self.sha256, "model resource sha256")


@dataclass(frozen=True, slots=True)
class DirectModelRegistration:
    """Identity required before a direct native model can consume T01 input."""

    engine_id: str
    model_id: str
    variant_id: str
    model_version: str
    source_model_sha256: str
    loaded_native_model_sha256: str
    provider_id: str
    provider_version: str
    provider_sha256: str
    state_schema_sha256: str
    input_channel_schema_sha256: str
    actuator_law_manifest_sha256: str
    ordered_channel_ids: tuple[str, ...]
    resource_root: Path
    model_path: Path
    resources: tuple[DeclaredModelResource, ...]
    environment_class_id: str
    environment_class_sha256: str
    factory_source_sha256: str
    solver_id: str
    integration_method: str
    contact_policy_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "engine_id",
            "model_id",
            "variant_id",
            "model_version",
            "provider_id",
            "provider_version",
            "environment_class_id",
            "solver_id",
            "integration_method",
            "contact_policy_id",
        ):
            _require_text(getattr(self, field_name), field_name)
        for field_name in (
            "source_model_sha256",
            "loaded_native_model_sha256",
            "provider_sha256",
            "state_schema_sha256",
            "input_channel_schema_sha256",
            "actuator_law_manifest_sha256",
            "environment_class_sha256",
            "factory_source_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        channels = tuple(self.ordered_channel_ids)
        if not channels or any(
            not isinstance(channel, str) or not channel.strip() for channel in channels
        ):
            raise ValueError("ordered channel IDs cannot be empty")
        if len(set(channels)) != len(channels):
            raise ValueError("ordered channel IDs must be unique")
        object.__setattr__(self, "ordered_channel_ids", channels)
        resources = tuple(self.resources)
        if any(not isinstance(item, DeclaredModelResource) for item in resources):
            raise TypeError("resources must contain DeclaredModelResource values")
        object.__setattr__(self, "resources", resources)


class DirectModelFactory(Protocol):
    """Create one direct native environment from its original source path."""

    def __call__(self, model_path: str, /) -> object: ...


@dataclass(frozen=True, slots=True)
class NativeDirectModelReplay:
    """Full-horizon native state and exact applied actuator commands."""

    time_seconds: NDArray[np.float64]
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    actuator_activation: NDArray[np.float64]
    actuator_controls: NDArray[np.float64]
    integration_states: NDArray[np.float64]
    applied_actuator_commands: NDArray[np.float64]
    input_sha256: str
    policy_sha256: str
    resource_closure_sha256: str
    actuator_law_manifest_sha256: str
    compiled_actuator_profile_sha256: str = ""


@dataclass(frozen=True, slots=True)
class _DirectCommandSchedule:
    times: NDArray[np.float64]
    commands: NDArray[np.float64]
    initial_integration: NDArray[np.float64]
    bundle: Any
    resource_digest: str
    law_digest: str
    profile_sha256: str


def resource_closure_sha256(
    resource_root: Path,
    model_path: Path,
    resources: tuple[DeclaredModelResource, ...],
    *,
    expected_loaded_native_model_sha256: str | None = None,
) -> str:
    """Admit bounded native source/assets within an existing declared inventory."""
    import mujoco as mj

    require_no_global_mujoco_callbacks(mj)
    root = resource_root.resolve(strict=True)
    model = _contained_resource_file(model_path, root)
    if not resources:
        raise ValueError("direct model resource closure cannot be empty")
    entries: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for resource in resources:
        rel = Path(resource.relative_path)
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or "." in rel.parts
            or rel.as_posix() != resource.relative_path.replace("\\", "/")
        ):
            raise ValueError("model resource paths must be relative and contained")
        path = _contained_resource_file(root / rel, root)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != resource.sha256:
            raise ValueError(f"model resource digest differs: {resource.relative_path}")
        key = rel.as_posix()
        if key in entries:
            raise ValueError("model resource paths must be unique")
        entries[key] = digest
        paths[key] = path
    model_rel = model.relative_to(root).as_posix()
    if entries.get(model_rel) != hashlib.sha256(model.read_bytes()).hexdigest():
        raise ValueError("resource closure must bind the exact model entry point")
    discovered, spec = _discover_native_resource_files(model, root)
    _verified_discovered_resource_hashes(discovered, root, entries)
    if expected_loaded_native_model_sha256 is not None:
        _require_sha256(
            expected_loaded_native_model_sha256,
            "expected loaded native model sha256",
        )
        if _native_model_sha256(spec.compile()) != expected_loaded_native_model_sha256:
            raise ValueError("native MjSpec compilation differs from registered model")
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != entries[key]
        for key, path in paths.items()
    ):
        raise ValueError("declared model resources changed during native discovery")
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def actuator_law_manifest_sha256(model: Any, channels: tuple[Any, ...]) -> str:
    """Digest ordered channel identities and the compiled native actuator laws."""
    import mujoco as mj

    if len(channels) != int(model.nu) or int(model.nu) <= 0:
        raise ValueError("one explicit T01 channel is required per native actuator")
    rows = []
    for index, channel in enumerate(channels):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, index)
        if not name or channel.target_id != f"actuator:{name}":
            raise ValueError("T01 actuator target differs from compiled actuator order")
        rows.append(
            {
                "channel_id": channel.channel_id,
                "target_id": channel.target_id,
                "unit": channel.unit,
                "coordinate_id": channel.coordinate_id,
                "frame_id": channel.frame_id,
                "native_law": {
                    field: np.asarray(getattr(model, field)[index]).tolist()
                    for field in _ACTUATOR_MODEL_FIELDS
                },
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compiled_actuator_profile_bytes(
    bundle: Any,
    registration: DirectModelRegistration,
    model: Any,
    resource_digest: str,
) -> bytes:
    """Build canonical profile bytes from the exact loaded native interpretation."""
    import mujoco as mj
    from src.engines.native_replay_contracts import native_replay_contract_types

    channels = bundle.input_history.channels
    contracts = native_replay_contract_types()
    channel_rows = []
    for index, channel in enumerate(channels):
        actuator_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, index)
        if actuator_name is None:
            raise ValueError("compiled actuator is missing its native name")
        channel_rows.append(
            {
                "index": index,
                "channel_id": channel.channel_id,
                "target_id": channel.target_id,
                "unit": channel.unit,
                "coordinate_id": channel.coordinate_id,
                "frame_id": channel.frame_id,
                "actuator_name": actuator_name,
                "native_dynamics": mj.mjtDyn(int(model.actuator_dyntype[index])).name,
                "native_transmission": mj.mjtTrn(
                    int(model.actuator_trntype[index])
                ).name,
            }
        )
    profile = {
        "schema_version": contracts.COMPILED_ACTUATOR_PROFILE_SCHEMA_VERSION,
        "engine_id": registration.engine_id,
        "model_id": registration.model_id,
        "variant_id": registration.variant_id,
        "model_version": registration.model_version,
        "source_model_sha256": registration.source_model_sha256,
        "loaded_native_model_sha256": _native_model_sha256(model),
        "provider_id": registration.provider_id,
        "provider_version": registration.provider_version,
        "provider_sha256": registration.provider_sha256,
        "runtime_id": "mujoco",
        "runtime_version": mj.__version__,
        "state_schema_sha256": bundle.state_schema_sha256,
        "initial_state_sha256": bundle.integrity.initial_state_sha256,
        "input_channel_schema_sha256": bundle.input_channel_schema_sha256,
        "actuator_law_manifest_sha256": actuator_law_manifest_sha256(model, channels),
        "resource_closure_sha256": resource_digest,
        "policy_sha256": bundle.policy_sha256,
        "time_grid_sha256": bundle.time_grid_sha256,
        "applied_input_sha256": bundle.applied_input_sha256,
        "ordered_channels": channel_rows,
    }
    return json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")


def native_contact_policy_sha256(model: Any) -> str:
    """Digest compiled geometry contact rules and global native contact options."""
    values = {
        field: np.asarray(getattr(model, field)).tolist()
        for field in _CONTACT_GEOMETRY_FIELDS
    }
    values["options"] = {
        field: getattr(model.opt, field) for field in _CONTACT_OPTION_FIELDS
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def direct_provider_sha256(
    registration: DirectModelRegistration,
    factory: DirectModelFactory,
    environment: object,
    resource_digest: str,
) -> str:
    """Bind provider code, factory/class source, and the declared asset closure."""
    factory_owner: object = factory if inspect.isroutine(factory) else type(factory)
    digest = hashlib.sha256()
    digest.update(native_replay_admission_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(_source_bytes(factory_owner, registration.factory_source_sha256))
    digest.update(
        _source_bytes(type(environment), registration.environment_class_sha256)
    )
    digest.update(resource_digest.encode("ascii"))
    import mujoco as mj

    digest.update(mj.__version__.encode("ascii"))
    package_path = Path(mj.__file__).resolve().parent
    native_libraries = sorted(
        path
        for path in package_path.iterdir()
        if path.is_file()
        and (path.suffix.lower() in {".dll", ".dylib"} or ".so" in path.name.lower())
    )
    if not native_libraries:
        raise ValueError("direct MuJoCo native library is unavailable for hashing")
    for path in native_libraries:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    for module_name in ("mujoco._functions", "mujoco._structs"):
        module = sys.modules.get(module_name)
        module_path = getattr(module, "__file__", None) if module else None
        if module_path is None:
            raise ValueError("direct MuJoCo runtime source is unavailable")
        digest.update(module_name.encode("ascii") + b"\0")
        digest.update(Path(module_path).read_bytes())
    return digest.hexdigest()


def _source_bytes(source_owner: Any, expected_sha256: str) -> bytes:
    source_file = inspect.getsourcefile(source_owner)
    if not source_file:
        raise ValueError("direct model provider source file is unavailable")
    payload = Path(source_file).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("direct model provider source identity differs")
    return payload


def _native_model_sha256(model: Any) -> str:
    import mujoco as mj

    saved = np.zeros(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, buffer=saved)
    return hashlib.sha256(saved.tobytes()).hexdigest()


def _bundle_state(bundle: Any, model: Any) -> dict[str, NDArray[np.float64]]:
    import mujoco as mj

    states = {
        item.component_id: np.asarray(item.values, dtype=np.float64)
        for item in bundle.initial_state
    }
    expected = ["qpos", "qvel"]
    if int(model.na) > 0:
        expected.append("actuator_activation")
    expected.extend(("actuator_internal", "integration"))
    if tuple(states) != tuple(expected):
        raise ValueError("T01 must declare complete direct native integration state")
    shapes = {
        "qpos": (int(model.nq),),
        "qvel": (int(model.nv),),
        "actuator_internal": (int(model.nu),),
        "integration": (mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION),),
    }
    if int(model.na) > 0:
        shapes["actuator_activation"] = (int(model.na),)
    for name, shape in shapes.items():
        if states[name].shape != shape:
            raise ValueError(f"T01 direct state dimension differs for {name}")
    return states


def _validate_identity(
    bundle: Any,
    model: Any,
    registration: DirectModelRegistration,
    factory: DirectModelFactory,
    environment: object,
    resource_digest: str,
    contracts: Any,
) -> tuple[Any, NDArray[np.float64], NDArray[np.float64]]:
    _validate_environment_identity(registration, factory, environment, resource_digest)
    times, commands = _validate_native_execution_identity(bundle, model, registration)
    return bundle, times, commands


def _validate_bundle_identity(
    bundle: Any, registration: DirectModelRegistration, contracts: Any
) -> None:
    if bundle.schema_version != "experiment-replay/1.0.0":
        raise ValueError("unsupported direct-model replay bundle schema")
    model_identity = bundle.model
    expected_identity = (
        registration.engine_id,
        registration.model_id,
        registration.variant_id,
        registration.model_version,
        registration.source_model_sha256,
        registration.provider_id,
        registration.provider_version,
        registration.provider_sha256,
        registration.loaded_native_model_sha256,
    )
    actual_identity = (
        model_identity.engine_id,
        model_identity.model_id,
        model_identity.variant_id,
        model_identity.model_version,
        model_identity.source_model_sha256,
        model_identity.provider_id,
        model_identity.provider_version,
        model_identity.provider_sha256,
        model_identity.loaded_native_model_sha256,
    )
    if actual_identity != expected_identity:
        raise ValueError("T01 direct native model/provider identity differs")
    if bundle.state_schema_sha256 != registration.state_schema_sha256:
        raise ValueError("T01 direct native state-schema identity differs")
    if bundle.input_channel_schema_sha256 != registration.input_channel_schema_sha256:
        raise ValueError("T01 direct input-channel schema identity differs")
    if bundle.model.ordered_input_channel_ids != registration.ordered_channel_ids:
        raise ValueError("T01 direct actuator channel order differs")
    channel_ids = tuple(item.channel_id for item in bundle.input_history.channels)
    if channel_ids != registration.ordered_channel_ids:
        raise ValueError("T01 direct input history order differs")
    if (
        bundle.input_history.input_kind
        is not contracts.ActuationInputKind.ACTUATOR_COMMAND
        or bundle.input_history.interpolation
        is not contracts.InputInterpolation.ZERO_ORDER_HOLD
        or bundle.input_history.timebase_id != "simulation_relative"
    ):
        raise ValueError("direct model replay requires T01 actuator-command ZOH input")
    if bundle.policy.replay_mode is not contracts.ReplayMode.NATIVE_OWN_CONTACT:
        raise ValueError("direct model replay requires its native contact policy")
    if bundle.policy.external_loads_sha256 is not None:
        raise ValueError(
            "direct native-contact replay does not consume external-load payloads"
        )
    if bundle.policy.step_policy != "fixed" or bundle.policy.step_size_seconds is None:
        raise ValueError("direct model replay requires a fixed native step size")
    bundle_times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    if not _time_grid_matches_step(bundle_times, bundle.policy.step_size_seconds):
        raise ValueError("direct T01 clock must match one native step per sample")
    if (
        bundle.policy.observation_access
        or bundle.policy.state_feedback_access
        or bundle.policy.state_reset_allowed
    ):
        raise ValueError(
            "direct model replay forbids observations, feedback, and reset"
        )
    if bundle.blocking_capabilities:
        raise ValueError("direct model replay has blocking required capabilities")


def _validate_environment_identity(
    registration: DirectModelRegistration,
    factory: DirectModelFactory,
    environment: object,
    resource_digest: str,
) -> None:
    if type(environment).__module__ + "." + type(environment).__qualname__ != (
        registration.environment_class_id
    ):
        raise ValueError("direct loader returned an unregistered environment class")
    model_path = registration.model_path.resolve(strict=True)
    module = sys.modules.get(type(environment).__module__)
    module_file = getattr(module, "__file__", None) if module else None
    if module_file is None or hashlib.sha256(
        Path(module_file).read_bytes()
    ).hexdigest() != (registration.environment_class_sha256):
        raise ValueError("direct environment implementation source identity differs")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != (
        registration.source_model_sha256
    ):
        raise ValueError("direct source model bytes differ from frozen identity")
    if direct_provider_sha256(registration, factory, environment, resource_digest) != (
        registration.provider_sha256
    ):
        raise ValueError("direct provider and resource-closure identity differs")


def _validate_native_execution_identity(
    bundle: Any,
    model: Any,
    registration: DirectModelRegistration,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    import mujoco as mj

    actual_model_sha = _native_model_sha256(model)
    if actual_model_sha != registration.loaded_native_model_sha256:
        raise ValueError("loaded direct native model identity differs")
    law_sha = actuator_law_manifest_sha256(model, bundle.input_history.channels)
    if law_sha != registration.actuator_law_manifest_sha256:
        raise ValueError("compiled actuator law/channel manifest differs")
    times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    commands = np.asarray(bundle.input_history.values, dtype=np.float64)
    actual_solver = mj.mjtSolver(model.opt.solver).name
    actual_integrator = mj.mjtIntegrator(model.opt.integrator).name
    if (
        registration.solver_id != actual_solver
        or bundle.policy.solver_id != actual_solver
    ):
        raise ValueError("direct native solver differs from registered solver")
    if bundle.policy.solver_version != mj.__version__:
        raise ValueError("direct native MuJoCo runtime version differs")
    if registration.integration_method != f"{actual_integrator};native-single-step":
        raise ValueError("direct native integration method differs from compiled model")
    if bundle.policy.integration_method != registration.integration_method:
        raise ValueError("direct native integration policy differs")
    native_step = float(model.opt.timestep)
    if not _time_grid_matches_step(times, native_step):
        raise ValueError("direct T01 clock must match one native step per sample")
    if commands.shape != (times.size, int(model.nu)) or not np.isfinite(commands).all():
        raise ValueError(
            "direct T01 command rows differ from the compiled actuator set"
        )
    if (
        bundle.policy.step_policy != "fixed"
        or bundle.policy.step_size_seconds != model.opt.timestep
    ):
        raise ValueError("direct native step size differs from the T01 policy")
    if bundle.policy.contact_policy_id != registration.contact_policy_id:
        raise ValueError("direct native contact policy id differs")
    if bundle.policy.contact_policy_version != "1.0.0":
        raise ValueError("direct native contact policy version differs")
    if bundle.policy.contact_policy_sha256 != native_contact_policy_sha256(model):
        raise ValueError("direct native contact policy digest differs")
    if (
        bundle.policy.initialization_policy_id
        != "mujoco-integration-state-restore-forward"
        or bundle.policy.initialization_policy_version != "1.0.0"
        or bundle.policy.input_player_id != "native-mj-step-direct-actuator-command"
        or bundle.policy.input_player_version != "1.0.0"
    ):
        raise ValueError("direct native initialization/input player policy differs")
    return times, commands


def replay_direct_model_actuator_commands(
    bundle: Any,
    registration: DirectModelRegistration,
    factory: DirectModelFactory,
    *,
    compiled_profile_bytes: bytes | None = None,
) -> NativeDirectModelReplay:
    """Load the exact source path and replay T01 commands through a direct model.

    The factory is an injected, source-bound constructor. This function never
    calls Gym registration, wrapper ``step``, observation, reset, or tracking.
    """
    import mujoco as mj
    from src.engines.native_replay_contracts import native_replay_contract_types

    if registration.engine_id != "mujoco":
        raise ValueError("direct MuJoCo kernel only admits engine_id='mujoco'")
    require_no_global_mujoco_callbacks(mj)
    contracts = native_replay_contract_types()
    bundle = validate_native_replay_bundle(bundle, contracts)
    _validate_bundle_identity(bundle, registration, contracts)
    environment, model, data, resource_digest = _load_registered_direct_model(
        registration, factory, mj
    )
    try:
        require_no_global_mujoco_callbacks(mj)
        loaded_resource_digest = resource_closure_sha256(
            registration.resource_root,
            registration.model_path,
            registration.resources,
        )
        if loaded_resource_digest != resource_digest:
            raise ValueError("model resources changed during native model loading")
        bundle, times, commands = _validate_identity(
            bundle,
            model,
            registration,
            factory,
            environment,
            resource_digest,
            contracts,
        )
        profile_sha256 = ""
        if compiled_profile_bytes is not None:
            if not isinstance(compiled_profile_bytes, bytes):
                raise TypeError("compiled actuator profile must be immutable bytes")
            expected_profile = compiled_actuator_profile_bytes(
                bundle, registration, model, resource_digest
            )
            if compiled_profile_bytes != expected_profile:
                raise ValueError(
                    "compiled actuator profile differs from loaded native identity"
                )
            profile_sha256 = hashlib.sha256(compiled_profile_bytes).hexdigest()
        states = _bundle_state(bundle, model)
        mj.mj_setState(
            model,
            data,
            states["integration"],
            mj.mjtState.mjSTATE_INTEGRATION,
        )
        _verify_restored_state(model, data, states)
        require_no_global_mujoco_callbacks(mj)
        start_time = float(data.time)
        mj.mj_forward(model, data)
        require_no_global_mujoco_callbacks(mj)
        _require_no_native_warnings(data)
        if not np.isclose(data.time, start_time, atol=1e-12, rtol=0):
            raise ValueError("direct initialization forward changed native time")
        schedule = _DirectCommandSchedule(
            times,
            commands,
            states["integration"],
            bundle,
            resource_digest,
            registration.actuator_law_manifest_sha256,
            profile_sha256,
        )
        return _execute_commands(model, data, schedule)
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def _load_registered_direct_model(
    registration: DirectModelRegistration,
    factory: DirectModelFactory,
    mujoco_module: Any,
) -> tuple[Any, Any, Any, str]:
    """Verify registered provider code, resource bytes and native model types."""
    resource_digest = resource_closure_sha256(
        registration.resource_root,
        registration.model_path,
        registration.resources,
        expected_loaded_native_model_sha256=registration.loaded_native_model_sha256,
    )
    model_path = registration.model_path.resolve(strict=True)
    factory_owner: object = factory if inspect.isroutine(factory) else type(factory)
    _source_bytes(factory_owner, registration.factory_source_sha256)
    module_name, separator, _ = registration.environment_class_id.rpartition(".")
    if not separator:
        raise ValueError("direct environment class identity must be fully qualified")
    module = sys.modules.get(module_name)
    module_path = getattr(module, "__file__", None) if module else None
    if (
        module_path is None
        or hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
        != registration.environment_class_sha256
    ):
        raise ValueError("registered direct environment source is not loaded exactly")
    environment = factory(str(model_path))
    try:
        if hasattr(environment, "env"):
            raise ValueError(
                "direct model loader must return the unwrapped environment"
            )
        model = getattr(environment, "model", None)
        data = getattr(environment, "data", None)
        if not isinstance(model, mujoco_module.MjModel) or not isinstance(
            data, mujoco_module.MjData
        ):
            raise ValueError("direct loader must expose native MuJoCo model and data")
        if int(model.nplugin) != 0:
            raise ValueError("direct model replay does not admit native plugins")
    except Exception:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        raise
    return environment, model, data, resource_digest


def _verify_restored_state(
    model: Any, data: Any, states: dict[str, NDArray[np.float64]]
) -> None:
    import mujoco as mj

    _require_normalized_quaternions(model, data.qpos)
    if not np.array_equal(data.qpos, states["qpos"]):
        raise ValueError("direct native qpos restore differs")
    if not np.array_equal(data.qvel, states["qvel"]):
        raise ValueError("direct native qvel restore differs")
    if not np.array_equal(data.ctrl, states["actuator_internal"]):
        raise ValueError("direct native actuator state restore differs")
    if int(model.na) > 0 and not np.array_equal(
        data.act, states["actuator_activation"]
    ):
        raise ValueError("direct native actuator activation restore differs")
    actual = np.empty(
        mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION), dtype=np.float64
    )
    mj.mj_getState(model, data, actual, mj.mjtState.mjSTATE_INTEGRATION)
    if not np.array_equal(actual, states["integration"]):
        raise ValueError("direct full integration-state restore differs")
    if data.time != 0.0 or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("direct initial time and external-load state must be zero")


def _require_normalized_quaternions(model: Any, qpos: NDArray[np.float64]) -> None:
    import mujoco as mj

    normalized = np.asarray(qpos, dtype=np.float64).copy()
    mj.mj_normalizeQuat(model, normalized)
    if not np.allclose(qpos, normalized, atol=1e-12, rtol=0):
        raise ValueError("native quaternion configuration must be normalized")


def _require_no_native_warnings(data: Any) -> None:
    if any(int(warning.number) for warning in data.warning):
        raise ValueError("direct native replay produced a MuJoCo warning")


def _execute_commands(
    model: Any,
    data: Any,
    schedule: _DirectCommandSchedule,
) -> NativeDirectModelReplay:
    import mujoco as mj

    times = schedule.times
    commands = schedule.commands
    initial_integration = schedule.initial_integration
    bundle = schedule.bundle
    resource_digest = schedule.resource_digest
    law_digest = schedule.law_digest
    profile_sha256 = schedule.profile_sha256
    require_no_global_mujoco_callbacks(mj)
    if data.time != times[0]:
        raise ValueError("direct native initial clock differs from frozen grid")
    qpos = [data.qpos.copy()]
    qvel = [data.qvel.copy()]
    activation = [data.act.copy()]
    controls = [data.ctrl.copy()]
    integration = [initial_integration.copy()]
    observed_times = [float(data.time)]
    applied = []
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    limits = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    for index, command in enumerate(commands[:-1]):
        require_no_global_mujoco_callbacks(mj)
        data.ctrl[:] = command
        actual_command = np.asarray(data.ctrl, dtype=np.float64).copy()
        if not np.array_equal(actual_command, command):
            raise ValueError("direct native command readback differs from T01")
        if np.any(actual_command[limited] < limits[limited, 0]) or np.any(
            actual_command[limited] > limits[limited, 1]
        ):
            raise ValueError("T01 command violates native actuator control limits")
        applied.append(actual_command)
        mj.mj_step(model, data)
        require_no_global_mujoco_callbacks(mj)
        _require_no_native_warnings(data)
        _require_normalized_quaternions(model, data.qpos)
        if not np.array_equal(data.ctrl, command):
            raise ValueError("native actuator command changed during integration")
        observed_time = float(data.time)
        if not _clock_close(
            observed_time,
            float(times[index + 1]),
            float(model.opt.timestep),
            index + 1,
        ):
            raise ValueError("observed direct native clock differs from T01")
        mj.mj_forward(model, data)
        require_no_global_mujoco_callbacks(mj)
        _require_no_native_warnings(data)
        state = np.empty(
            mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION), dtype=np.float64
        )
        mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
        if any(
            not np.isfinite(value).all()
            for value in (data.qpos, data.qvel, data.act, data.ctrl, state)
        ):
            raise ValueError("direct native replay produced nonfinite state")
        if data.time != observed_time:
            raise ValueError("direct forward evaluation changed native time")
        observed_times.append(observed_time)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        activation.append(data.act.copy())
        controls.append(data.ctrl.copy())
        integration.append(state)
    return NativeDirectModelReplay(
        np.asarray(observed_times, dtype=np.float64),
        np.asarray(qpos, dtype=np.float64),
        np.asarray(qvel, dtype=np.float64),
        np.asarray(activation, dtype=np.float64),
        np.asarray(controls, dtype=np.float64),
        np.asarray(integration, dtype=np.float64),
        np.asarray(applied, dtype=np.float64),
        bundle.applied_input_sha256,
        bundle.policy_sha256,
        resource_digest,
        law_digest,
        profile_sha256,
    )


__all__ = [
    "DeclaredModelResource",
    "DirectModelFactory",
    "DirectModelRegistration",
    "NativeDirectModelReplay",
    "actuator_law_manifest_sha256",
    "compiled_actuator_profile_bytes",
    "direct_provider_sha256",
    "native_contact_policy_sha256",
    "replay_direct_model_actuator_commands",
    "resource_closure_sha256",
]
