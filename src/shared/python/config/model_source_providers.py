"""Shared model-source provider resolution for launcher and engine discovery.

This module centralizes provider-aware source resolution so launcher handlers,
model-registry consumers, and engine discovery all rely on one path policy.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.shared.python.core.contracts import require

import os
from collections.abc import Callable

_MODEL_SOURCES: dict[str, Callable[[], Path]] = {}


def register_source(name: str) -> Callable[[Callable[[], Path]], Callable[[], Path]]:
    """Register a global model source provider by name."""

    def decorator(func: Callable[[], Path]) -> Callable[[], Path]:
        _MODEL_SOURCES[name] = func
        return func

    return decorator


def _resolve_sibling(repo_name: str, pkg: str, env_var: str) -> Path:
    """Resolve a sibling biomechanics repo models path."""
    if env_var in os.environ:
        return Path(os.environ[env_var]).resolve()

    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    sibling_checkout = repo_root.parent / repo_name

    if sibling_checkout.exists() and (sibling_checkout / "pyproject.toml").exists():
        spec = importlib.util.spec_from_file_location(
            f"{pkg}.model_pack", sibling_checkout / "src" / pkg / "model_pack.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                if hasattr(module, "resolve"):
                    return Path(module.resolve()).resolve()
            except Exception:
                pass

    try:
        module = importlib.import_module(f"{pkg}.model_pack")
        if hasattr(module, "resolve"):
            return Path(module.resolve()).resolve()
    except ImportError:
        pass

    vendored_path = repo_root / "vendor" / "biomech-models" / repo_name
    if vendored_path.exists():
        return vendored_path.resolve()

    raise RuntimeError(f"Could not resolve sibling model repo: {repo_name}")


@register_source("mujoco_models")
def mujoco_models_source() -> Path:
    return _resolve_sibling("MuJoCo_Models", "mujoco_models", "MUJOCO_MODELS_HOME")


@register_source("drake_models")
def drake_models_source() -> Path:
    return _resolve_sibling("Drake_Models", "drake_models", "DRAKE_MODELS_HOME")


@register_source("pinocchio_models")
def pinocchio_models_source() -> Path:
    return _resolve_sibling(
        "Pinocchio_Models", "pinocchio_models", "PINOCCHIO_MODELS_HOME"
    )


@register_source("opensim_models")
def opensim_models_source() -> Path:
    return _resolve_sibling("OpenSim_Models", "opensim_models", "OPENSIM_MODELS_HOME")


@register_source("movement_optimizer")
def movement_optimizer_source() -> Path:
    return _resolve_sibling(
        "Movement-Optimizer", "movement_optimizer", "MOVEMENT_OPTIMIZER_HOME"
    )


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
