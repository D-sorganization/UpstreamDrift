"""Shared model-source provider resolution for launcher and engine discovery.

This module centralizes provider-aware source resolution so launcher handlers,
model-registry consumers, and engine discovery all rely on one path policy.

It also hosts the **sibling-repo** discovery layer used by UpstreamDrift's
launcher to find the five biomech sibling repos (MuJoCo_Models, Drake_Models,
Pinocchio_Models, OpenSim_Models, Movement_Optimizer). See
``docs/adr/0014-shared-biomech-models.md`` and UpstreamDrift#5184.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def _get_optional_string_attr(model: Any, attr_name: str) -> str | None:
    """Return a stripped string attribute when the model explicitly provides one."""
    value = getattr(model, attr_name, None)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _require_non_empty_string(value: str | None, *, field_name: str) -> str:
    """Return a non-empty string after enforcing the contract boundary."""
    require(
        value is not None and value.strip() != "",
        f"{field_name} must be a non-empty string",
        value,
    )
    assert value is not None
    return value


def _require_module_spec(
    spec: importlib.machinery.ModuleSpec | None,
    *,
    package_name: str,
) -> importlib.machinery.ModuleSpec:
    """Return an importlib spec after enforcing package discovery."""
    require(spec is not None, "installed package provider not found", package_name)
    assert spec is not None
    return spec


@dataclass(frozen=True)
class ResolvedModelSource:
    """Canonical source locations for a provider-backed model."""

    provider_id: str
    source_root: Path
    artifact_path: Path
    working_directory: Path
    python_paths: tuple[Path, ...]


class ModelSourceProvider(Protocol):
    """Interface for resolving model sources from different provider styles."""

    provider_id: str

    def can_resolve(self, model: Any) -> bool:
        """Return True when this provider can resolve the given model."""
        ...

    def resolve(
        self,
        model: Any,
        path_policy: ModelSourcePathPolicy,
        fallback_relative: str | Path | None = None,
    ) -> ResolvedModelSource:
        """Return canonical launch paths for the model."""
        ...


class ModelSourcePathPolicy:
    """Canonical path policy with explicit approved-root boundaries."""

    def __init__(
        self,
        default_root: Path,
        approved_roots: Iterable[Path] = (),
    ) -> None:
        canonical_default = Path(default_root).resolve(strict=False)
        roots = [canonical_default, canonical_default.parent]
        roots.extend(Path(root).resolve(strict=False) for root in approved_roots)

        seen: set[Path] = set()
        deduped_roots: list[Path] = []
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            deduped_roots.append(root)

        self.default_root = canonical_default
        self.approved_roots = tuple(deduped_roots)

    def resolve_source_root(self, declared_root: str | None) -> Path:
        """Resolve the canonical source root for a model."""
        if declared_root is None:
            return self.default_root

        if declared_root in _MODEL_SOURCES:
            return self._canonicalize(
                _MODEL_SOURCES[declared_root](), field_name="source_root"
            )

        root_path = Path(declared_root)
        if not root_path.is_absolute():
            root_path = self.default_root / root_path
        return self._canonicalize(root_path, field_name="source_root")

    def resolve_path(
        self,
        source_root: Path,
        declared_path: str | None,
        *,
        field_name: str,
    ) -> Path:
        """Resolve an artifact, working directory, or python path entry."""
        raw_path = Path(_require_non_empty_string(declared_path, field_name=field_name))
        candidate = raw_path if raw_path.is_absolute() else source_root / raw_path
        return self._canonicalize(candidate, field_name=field_name)

    def resolve_optional_path(
        self,
        source_root: Path,
        declared_path: str | None,
        *,
        fallback_path: str | Path | None = None,
        field_name: str,
    ) -> Path:
        """Resolve an optional path, falling back to source-root-relative data."""
        if declared_path is not None:
            return self.resolve_path(source_root, declared_path, field_name=field_name)
        if fallback_path is None:
            return source_root
        fallback = Path(fallback_path)
        candidate = fallback if fallback.is_absolute() else source_root / fallback
        return self._canonicalize(candidate, field_name=field_name)

    def resolve_python_paths(
        self,
        source_root: Path,
        declared_paths: Iterable[str],
    ) -> tuple[Path, ...]:
        """Resolve and deduplicate extra PYTHONPATH entries."""
        seen: set[Path] = set()
        resolved: list[Path] = []
        for item in declared_paths:
            stripped = item.strip()
            if not stripped:
                continue
            candidate = self.resolve_path(
                source_root,
                stripped,
                field_name="python_paths",
            )
            if candidate in seen:
                continue
            seen.add(candidate)
            resolved.append(candidate)
        return tuple(resolved)

    def _canonicalize(self, candidate: Path, *, field_name: str) -> Path:
        canonical = candidate.resolve(strict=False)
        require(
            any(
                canonical == approved_root or approved_root in canonical.parents
                for approved_root in self.approved_roots
            ),
            f"{field_name} must resolve within approved roots",
            canonical,
        )
        return canonical


class LocalRepoModelSourceProvider:
    """Provider that preserves the current repo-local launcher behaviour."""

    provider_id = "local-repo"

    def can_resolve(self, model: Any) -> bool:
        return _get_optional_string_attr(model, "source_root") is None and (
            _get_optional_string_attr(model, "package_name") is None
        )

    def resolve(
        self,
        model: Any,
        path_policy: ModelSourcePathPolicy,
        fallback_relative: str | Path | None = None,
    ) -> ResolvedModelSource:
        source_root = path_policy.default_root
        return ResolvedModelSource(
            provider_id=self.provider_id,
            source_root=source_root,
            artifact_path=path_policy.resolve_path(
                source_root,
                _get_optional_string_attr(model, "path"),
                field_name="path",
            ),
            working_directory=path_policy.resolve_optional_path(
                source_root,
                _get_optional_string_attr(model, "working_dir"),
                fallback_path=fallback_relative,
                field_name="working_dir",
            ),
            python_paths=path_policy.resolve_python_paths(
                source_root,
                _iter_python_path_values(model),
            ),
        )


class SiblingRepoModelSourceProvider:
    """Provider for sibling/local external repos declared via source_root."""

    provider_id = "sibling-repo"

    def can_resolve(self, model: Any) -> bool:
        return _get_optional_string_attr(model, "source_root") is not None

    def resolve(
        self,
        model: Any,
        path_policy: ModelSourcePathPolicy,
        fallback_relative: str | Path | None = None,
    ) -> ResolvedModelSource:
        source_root = path_policy.resolve_source_root(
            _get_optional_string_attr(model, "source_root")
        )
        return ResolvedModelSource(
            provider_id=self.provider_id,
            source_root=source_root,
            artifact_path=path_policy.resolve_path(
                source_root,
                _get_optional_string_attr(model, "path"),
                field_name="path",
            ),
            working_directory=path_policy.resolve_optional_path(
                source_root,
                _get_optional_string_attr(model, "working_dir"),
                fallback_path=fallback_relative,
                field_name="working_dir",
            ),
            python_paths=path_policy.resolve_python_paths(
                source_root,
                _iter_python_path_values(model),
            ),
        )


class InstalledPackageModelSourceProvider:
    """Provider for models rooted in an installed Python package."""

    provider_id = "installed-package"

    def can_resolve(self, model: Any) -> bool:
        return _get_optional_string_attr(model, "package_name") is not None

    def resolve(
        self,
        model: Any,
        path_policy: ModelSourcePathPolicy,
        fallback_relative: str | Path | None = None,
    ) -> ResolvedModelSource:
        package_name = _require_non_empty_string(
            _get_optional_string_attr(model, "package_name"),
            field_name="package_name",
        )
        spec = _require_module_spec(
            importlib.util.find_spec(package_name),
            package_name=package_name,
        )

        source_root = _resolve_package_root(spec, package_name)
        return ResolvedModelSource(
            provider_id=self.provider_id,
            source_root=source_root,
            artifact_path=path_policy.resolve_path(
                source_root,
                _get_optional_string_attr(model, "path"),
                field_name="path",
            ),
            working_directory=path_policy.resolve_optional_path(
                source_root,
                _get_optional_string_attr(model, "working_dir"),
                fallback_path=fallback_relative,
                field_name="working_dir",
            ),
            python_paths=path_policy.resolve_python_paths(
                source_root,
                _iter_python_path_values(model),
            ),
        )


_PROVIDERS: tuple[ModelSourceProvider, ...] = (
    InstalledPackageModelSourceProvider(),
    SiblingRepoModelSourceProvider(),
    LocalRepoModelSourceProvider(),
)


def resolve_model_source_root(
    model: Any,
    default_root: Path,
    *,
    approved_roots: Iterable[Path] = (),
) -> Path:
    """Resolve only the canonical source root for a model."""
    path_policy = ModelSourcePathPolicy(default_root, approved_roots=approved_roots)
    provider = _select_provider(model)
    if isinstance(provider, InstalledPackageModelSourceProvider):
        package_name = _require_non_empty_string(
            _get_optional_string_attr(model, "package_name"),
            field_name="package_name",
        )
        spec = _require_module_spec(
            importlib.util.find_spec(package_name),
            package_name=package_name,
        )
        return _resolve_package_root(spec, package_name)
    return path_policy.resolve_source_root(
        _get_optional_string_attr(model, "source_root")
    )


def resolve_model_artifact(
    model: Any,
    default_root: Path,
    *,
    approved_roots: Iterable[Path] = (),
) -> Path:
    """Resolve only the canonical artifact path for a model."""
    path_policy = ModelSourcePathPolicy(default_root, approved_roots=approved_roots)
    source_root = resolve_model_source_root(
        model,
        default_root,
        approved_roots=approved_roots,
    )
    return path_policy.resolve_path(
        source_root,
        _get_optional_string_attr(model, "path"),
        field_name="path",
    )


def resolve_model_working_directory(
    model: Any,
    default_root: Path,
    *,
    fallback_relative: str | Path | None = None,
    approved_roots: Iterable[Path] = (),
) -> Path:
    """Resolve only the canonical working directory for a model."""
    path_policy = ModelSourcePathPolicy(default_root, approved_roots=approved_roots)
    source_root = resolve_model_source_root(
        model,
        default_root,
        approved_roots=approved_roots,
    )
    return path_policy.resolve_optional_path(
        source_root,
        _get_optional_string_attr(model, "working_dir"),
        fallback_path=fallback_relative,
        field_name="working_dir",
    )


def resolve_model_python_paths(
    model: Any,
    default_root: Path,
    *,
    approved_roots: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """Resolve only the canonical extra python paths for a model."""
    path_policy = ModelSourcePathPolicy(default_root, approved_roots=approved_roots)
    source_root = resolve_model_source_root(
        model,
        default_root,
        approved_roots=approved_roots,
    )
    return path_policy.resolve_python_paths(
        source_root, _iter_python_path_values(model)
    )


def resolve_model_source(
    model: Any,
    default_root: Path,
    *,
    fallback_relative: str | Path | None = None,
    approved_roots: Iterable[Path] = (),
) -> ResolvedModelSource:
    """Resolve the canonical source information for a model."""
    path_policy = ModelSourcePathPolicy(default_root, approved_roots=approved_roots)
    return _select_provider(model).resolve(
        model,
        path_policy,
        fallback_relative=fallback_relative,
    )


def collect_engine_provider_paths(
    models: Iterable[Any],
    default_root: Path,
    *,
    approved_roots: Iterable[Path] = (),
) -> dict[str, tuple[Path, ...]]:
    """Group engine validation paths by engine type from resolved model sources."""
    grouped: dict[str, list[Path]] = {}
    seen: dict[str, set[Path]] = {}

    for model in models:
        engine_type = _get_optional_string_attr(model, "engine_type")
        if engine_type is None:
            continue

        resolved = resolve_model_source(
            model,
            default_root,
            approved_roots=approved_roots,
        )
        validation_path = (
            resolved.working_directory
            if resolved.working_directory.exists()
            else resolved.source_root
        )

        engine_paths = grouped.setdefault(engine_type, [])
        engine_seen = seen.setdefault(engine_type, set())
        if validation_path in engine_seen:
            continue
        engine_seen.add(validation_path)
        engine_paths.append(validation_path)

    return {engine: tuple(paths) for engine, paths in grouped.items()}


def iter_unique_python_path_strings(
    base_paths: Iterable[str],
    extra_paths: Iterable[str],
) -> tuple[str, ...]:
    """Merge stringified python paths while preserving first-seen order."""
    seen: set[str] = set()
    merged: list[str] = []

    for item in [*base_paths, *extra_paths]:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)

    return tuple(merged)


def _iter_python_path_values(model: Any) -> tuple[str, ...]:
    raw_values = getattr(model, "python_paths", ())
    if not isinstance(raw_values, (list, tuple)):
        return ()
    values: list[str] = []
    for raw_value in raw_values:
        if isinstance(raw_value, str):
            values.append(raw_value)
    return tuple(values)


def _resolve_package_root(
    spec: importlib.machinery.ModuleSpec,
    package_name: str,
) -> Path:
    if spec.submodule_search_locations:
        search_location = next(iter(spec.submodule_search_locations), None)
        resolved_location = _require_non_empty_string(
            search_location,
            field_name="installed package search location",
        )
        return Path(resolved_location).resolve(strict=False)

    origin = _require_non_empty_string(
        spec.origin,
        field_name=f"installed package origin for {package_name}",
    )
    return Path(origin).resolve(strict=False).parent


def _select_provider(model: Any) -> ModelSourceProvider:
    for provider in _PROVIDERS:
        if provider.can_resolve(model):
            return provider
    raise ValueError("No model source provider matched the supplied model")


# ---------------------------------------------------------------------------
# Sibling-repo discovery layer (UpstreamDrift#5184)
#
# The five biomechanics sibling repos publish a ``model_pack.yaml`` /
# ``tool_pack.yaml`` manifest plus a ``<pkg>.model_pack:resolve()`` entry
# point. The functions below resolve the absolute ``models_root`` path for a
# given sibling, walking a four-tier precedence order:
#
#   1. Editable sibling checkout at ``../<RepoName>/`` (detected by the
#      presence of ``pyproject.toml`` — we deliberately do NOT import the
#      sibling package because it may not yet be installed).
#   2. Pip-installed sibling package — ``<pkg>.model_pack:resolve()``.
#   3. Vendored snapshot at ``vendor/biomech-models/<RepoName>/``.
#   4. Environment variable override ``<REPO>_HOME``.
#
# Each provider returns ``None`` if every tier misses, allowing the launcher
# to report missing sources without crashing.
# ---------------------------------------------------------------------------


class SiblingResolutionTier:
    """Identifier strings for the four resolution tiers."""

    EDITABLE = "editable"
    INSTALLED = "installed"
    VENDORED = "vendored"
    ENV = "env"
    MISSING = "missing"


@dataclass(frozen=True)
class SiblingResolution:
    """Result of resolving a sibling biomech repo to a concrete models root."""

    repo_name: str
    package: str
    env_var: str
    tier: str
    models_root: Path | None
    manifest_path: Path | None = None

    @property
    def resolved(self) -> bool:
        """Return True when a concrete models root was discovered."""
        return self.models_root is not None


def _upstreamdrift_repo_root() -> Path:
    """Return the absolute path of the UpstreamDrift checkout root."""
    return Path(__file__).resolve().parents[4]


def _detect_editable_sibling(repo_name: str) -> Path | None:
    """Return the path to an editable sibling checkout if one exists.

    Detection is by checking for ``pyproject.toml`` inside the sibling's
    directory — we deliberately do not import the sibling package, since
    it may not yet expose the published ``model_pack`` entry point.
    """
    candidate = _upstreamdrift_repo_root().parent / repo_name
    if not candidate.is_dir():
        return None
    if not (candidate / "pyproject.toml").is_file():
        return None
    return candidate.resolve()


def _read_models_root_from_manifest(manifest_path: Path) -> Path | None:
    """Return the absolute ``models_root`` declared in a manifest file.

    Returns ``None`` if PyYAML is unavailable, the file is missing, or the
    manifest does not declare a usable ``models_root``.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("PyYAML missing — cannot parse %s", manifest_path)
        return None
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Failed to read manifest %s: %s", manifest_path, exc)
        return None
    if not isinstance(raw, dict):
        return None
    declared = raw.get("models_root")
    if not isinstance(declared, str) or not declared.strip():
        return None
    return (manifest_path.parent / declared).resolve()


def _manifest_path_for(checkout: Path) -> Path | None:
    """Locate the manifest YAML in a sibling checkout (model_pack or tool_pack)."""
    for name in ("model_pack.yaml", "tool_pack.yaml"):
        candidate = checkout / name
        if candidate.is_file():
            return candidate
    return None


def _resolve_via_installed_package(pkg: str) -> tuple[Path, Path | None] | None:
    """Call ``<pkg>.model_pack:resolve()`` on an installed sibling package.

    Returns ``(models_root, manifest_path)`` or ``None`` if the package or
    its ``model_pack`` submodule is unavailable. ``manifest_path`` is best
    effort and may be ``None`` even when ``models_root`` resolves.
    """
    spec = importlib.util.find_spec(pkg)
    if spec is None:
        return None
    submodule_name = f"{pkg}.model_pack"
    if importlib.util.find_spec(submodule_name) is None:
        # Tool-pack-style siblings (Movement-Optimizer) use tool_pack.
        submodule_name = f"{pkg}.tool_pack"
        if importlib.util.find_spec(submodule_name) is None:
            return None
    try:
        module = importlib.import_module(submodule_name)
    except ImportError as exc:
        logger.debug("Could not import %s: %s", submodule_name, exc)
        return None
    resolve_callable = getattr(module, "resolve", None)
    if not callable(resolve_callable):
        return None
    try:
        models_root = Path(resolve_callable()).resolve()
    except Exception as exc:  # noqa: BLE001 — entry-point may raise anything
        logger.debug("%s.resolve() raised %s", submodule_name, exc)
        return None
    manifest_path: Path | None = None
    manifest_callable = getattr(module, "manifest", None)
    if callable(manifest_callable):
        try:
            _ = manifest_callable()
        except Exception:  # noqa: BLE001
            manifest_path = None
    return models_root, manifest_path


def _vendor_snapshot_root(repo_name: str) -> Path:
    """Return the conventional vendor snapshot directory for a sibling."""
    return _upstreamdrift_repo_root() / "vendor" / "biomech-models" / repo_name


def _resolve_sibling(
    repo_name: str,
    pkg: str,
    env_var: str,
) -> SiblingResolution:
    """Walk the four-tier resolution order for one sibling biomech repo.

    Args:
        repo_name: Human-friendly repo directory name (e.g. ``MuJoCo_Models``).
        pkg: Importable Python package name (e.g. ``mujoco_models``).
        env_var: Environment-variable override (e.g. ``MUJOCO_MODELS_HOME``).

    Returns:
        :class:`SiblingResolution` describing which tier (if any) won and
        the absolute ``models_root``.
    """
    # Tier 1: editable checkout
    editable = _detect_editable_sibling(repo_name)
    if editable is not None:
        manifest_path = _manifest_path_for(editable)
        models_root: Path | None = None
        if manifest_path is not None:
            models_root = _read_models_root_from_manifest(manifest_path)
        if models_root is None:
            # Fall back to a conventional models tree if the manifest is
            # absent or unparseable. This keeps editable mode useful while
            # sibling repos are still adding their manifests.
            for candidate_name in ("models", f"src/{pkg}/exercises"):
                candidate = editable / candidate_name
                if candidate.is_dir():
                    models_root = candidate.resolve()
                    break
        if models_root is not None:
            return SiblingResolution(
                repo_name=repo_name,
                package=pkg,
                env_var=env_var,
                tier=SiblingResolutionTier.EDITABLE,
                models_root=models_root,
                manifest_path=manifest_path,
            )

    # Tier 2: installed package
    installed = _resolve_via_installed_package(pkg)
    if installed is not None:
        models_root, manifest_path = installed
        return SiblingResolution(
            repo_name=repo_name,
            package=pkg,
            env_var=env_var,
            tier=SiblingResolutionTier.INSTALLED,
            models_root=models_root,
            manifest_path=manifest_path,
        )

    # Tier 3: vendored snapshot
    vendored = _vendor_snapshot_root(repo_name)
    if vendored.is_dir():
        manifest_path = _manifest_path_for(vendored)
        models_root = None
        if manifest_path is not None:
            models_root = _read_models_root_from_manifest(manifest_path)
        if models_root is None:
            models_root = vendored.resolve()
        return SiblingResolution(
            repo_name=repo_name,
            package=pkg,
            env_var=env_var,
            tier=SiblingResolutionTier.VENDORED,
            models_root=models_root,
            manifest_path=manifest_path,
        )

    # Tier 4: env-var override
    env_value = os.environ.get(env_var)
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.is_dir():
            return SiblingResolution(
                repo_name=repo_name,
                package=pkg,
                env_var=env_var,
                tier=SiblingResolutionTier.ENV,
                models_root=env_path.resolve(),
                manifest_path=_manifest_path_for(env_path),
            )
        logger.warning(
            "Env-var %s points at %s which is not a directory",
            env_var,
            env_path,
        )

    return SiblingResolution(
        repo_name=repo_name,
        package=pkg,
        env_var=env_var,
        tier=SiblingResolutionTier.MISSING,
        models_root=None,
        manifest_path=None,
    )


@dataclass(frozen=True)
class _SiblingSpec:
    """Static descriptor for a registered sibling provider."""

    name: str
    repo_name: str
    package: str
    env_var: str


_SIBLINGS: tuple[_SiblingSpec, ...] = (
    _SiblingSpec(
        "mujoco_models", "MuJoCo_Models", "mujoco_models", "MUJOCO_MODELS_HOME"
    ),
    _SiblingSpec("drake_models", "Drake_Models", "drake_models", "DRAKE_MODELS_HOME"),
    _SiblingSpec(
        "pinocchio_models",
        "Pinocchio_Models",
        "pinocchio_models",
        "PINOCCHIO_MODELS_HOME",
    ),
    _SiblingSpec(
        "opensim_models",
        "OpenSim_Models",
        "opensim_models",
        "OPENSIM_MODELS_HOME",
    ),
    _SiblingSpec(
        "movement_optimizer",
        "Movement_Optimizer",
        "movement_optimizer",
        "MOVEMENT_OPTIMIZER_HOME",
    ),
)

_MODEL_SOURCES: dict[str, Callable[[], Path]] = {}


def register_source(name: str) -> Callable[[Callable[[], Path]], Callable[[], Path]]:
    """Register a named model-source resolver callable.

    The decorator preserves the original callable so direct invocation
    continues to work; the registered lookup table is used by the launcher
    diagnostics and downstream callers that iterate sources by name.
    """

    def _decorator(func: Callable[[], Path]) -> Callable[[], Path]:
        _MODEL_SOURCES[name] = func
        return func

    return _decorator


def get_registered_source(name: str) -> Callable[[], Path]:
    """Return the resolver registered under ``name`` or raise ``KeyError``."""
    return _MODEL_SOURCES[name]


def iter_registered_sources() -> tuple[str, ...]:
    """Return the registered source names in registration order."""
    return tuple(_MODEL_SOURCES)


def resolve_sibling(name: str) -> SiblingResolution:
    """Return the full :class:`SiblingResolution` for one named sibling."""
    for spec in _SIBLINGS:
        if spec.name == name:
            return _resolve_sibling(spec.repo_name, spec.package, spec.env_var)
    raise KeyError(f"Unknown biomech sibling: {name!r}")


def resolve_all_siblings() -> dict[str, SiblingResolution]:
    """Return the resolution for every registered sibling, by name."""
    return {spec.name: resolve_sibling(spec.name) for spec in _SIBLINGS}


def _require_resolved(resolution: SiblingResolution) -> Path:
    """Return the resolved ``models_root`` or raise ``FileNotFoundError``."""
    if resolution.models_root is None:
        raise FileNotFoundError(
            f"Sibling biomech repo {resolution.repo_name!r} could not be resolved "
            f"via editable checkout, installed package, vendored snapshot, "
            f"or env var {resolution.env_var}",
        )
    return resolution.models_root


@register_source("mujoco_models")
def mujoco_models_source() -> Path:
    """Return the absolute ``models_root`` for the MuJoCo_Models sibling."""
    return _require_resolved(resolve_sibling("mujoco_models"))


@register_source("drake_models")
def drake_models_source() -> Path:
    """Return the absolute ``models_root`` for the Drake_Models sibling."""
    return _require_resolved(resolve_sibling("drake_models"))


@register_source("pinocchio_models")
def pinocchio_models_source() -> Path:
    """Return the absolute ``models_root`` for the Pinocchio_Models sibling."""
    return _require_resolved(resolve_sibling("pinocchio_models"))


@register_source("opensim_models")
def opensim_models_source() -> Path:
    """Return the absolute ``models_root`` for the OpenSim_Models sibling."""
    return _require_resolved(resolve_sibling("opensim_models"))


@register_source("movement_optimizer")
def movement_optimizer_source() -> Path:
    """Return the absolute ``models_root`` for the Movement-Optimizer sibling."""
    return _require_resolved(resolve_sibling("movement_optimizer"))
