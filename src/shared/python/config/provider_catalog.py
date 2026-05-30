"""Known external provider repositories for launcher migration onboarding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderRepoDefinition:
    """Static onboarding metadata for an external model-provider repository."""

    provider_id: str
    repo_name: str
    provider_kind: str
    engine_type: str | None = None
    manifest_relative_paths: tuple[Path, ...] = ()


KNOWN_EXTERNAL_MODEL_PROVIDERS: tuple[ProviderRepoDefinition, ...] = (
    ProviderRepoDefinition(
        provider_id="mujoco_models",
        repo_name="MuJoCo_Models",
        provider_kind="engine",
        engine_type="mujoco",
    ),
    ProviderRepoDefinition(
        provider_id="drake_models",
        repo_name="Drake_Models",
        provider_kind="engine",
        engine_type="drake",
    ),
    ProviderRepoDefinition(
        provider_id="pinocchio_models",
        repo_name="Pinocchio_Models",
        provider_kind="engine",
        engine_type="pinocchio",
    ),
    ProviderRepoDefinition(
        provider_id="opensim_models",
        repo_name="OpenSim_Models",
        provider_kind="engine",
        engine_type="opensim",
    ),
    ProviderRepoDefinition(
        provider_id="tools",
        repo_name="Tools",
        provider_kind="utility",
        manifest_relative_paths=(
            Path("src") / "pendulum_simulator" / "model_pack.yaml",
        ),
    ),
    ProviderRepoDefinition(
        provider_id="movement_optimizer",
        repo_name="Movement_Optimizer",
        provider_kind="utility",
    ),
)

_DEFAULT_MANIFEST_RELATIVE_PATHS = (
    Path("model_pack.yaml"),
    Path("model_pack.yml"),
    Path(".upstreamdrift") / "model_pack.yaml",
    Path(".upstreamdrift") / "model_pack.yml",
)


def _iter_manifest_paths_for_provider(
    provider_root: Path,
) -> tuple[Path, ...]:
    """Return manifest search paths for one provider root."""
    provider_name = provider_root.name
    extra_paths = tuple(
        provider.manifest_relative_paths
        for provider in KNOWN_EXTERNAL_MODEL_PROVIDERS
        if provider.repo_name == provider_name
    )
    provider_relative_paths = _DEFAULT_MANIFEST_RELATIVE_PATHS + tuple(
        path for paths in extra_paths for path in paths
    )
    return tuple(
        provider_root / relative_path for relative_path in provider_relative_paths
    )


def infer_repo_root_from_config(config_path: str | Path) -> Path:
    """Infer the containing repository root from a config path."""
    resolved_path = Path(config_path).resolve(strict=False)
    config_dir = resolved_path.parent
    if config_dir.name == "config" and config_dir.parent.name == "src":
        return config_dir.parent.parent.resolve(strict=False)
    return config_dir.resolve(strict=False)


def iter_configured_provider_roots(
    config_path: str | Path,
    env_value: str | None,
) -> tuple[Path, ...]:
    """Return explicit env roots plus conventional sibling provider roots."""
    repo_root = infer_repo_root_from_config(config_path)
    workspace_root = repo_root.parent
    discovered: list[Path] = []
    seen: set[Path] = set()

    def _add_root(candidate: Path) -> None:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            return
        seen.add(resolved)
        discovered.append(resolved)

    if env_value:
        for raw_root in env_value.split(os.pathsep):
            root_value = raw_root.strip()
            if not root_value:
                continue
            candidate = Path(root_value)
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            _add_root(candidate)

    for provider in KNOWN_EXTERNAL_MODEL_PROVIDERS:
        _add_root(workspace_root / provider.repo_name)

    return tuple(discovered)


def iter_provider_manifest_specs(
    config_path: str | Path,
    env_value: str | None,
) -> tuple[tuple[Path, Path], ...]:
    """Return provider roots paired with discovered manifest file paths."""
    discovered: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    for provider_root in iter_configured_provider_roots(config_path, env_value):
        for manifest_path in _iter_manifest_paths_for_provider(provider_root):
            if manifest_path in seen or not manifest_path.exists():
                continue
            seen.add(manifest_path)
            discovered.append((provider_root, manifest_path))

    return tuple(discovered)


def iter_known_provider_ids() -> tuple[str, ...]:
    """Return the canonical provider IDs for the onboarded external repos."""
    return tuple(provider.provider_id for provider in KNOWN_EXTERNAL_MODEL_PROVIDERS)


def iter_known_provider_repo_names() -> tuple[str, ...]:
    """Return the expected sibling repo names for the onboarded providers."""
    return tuple(provider.repo_name for provider in KNOWN_EXTERNAL_MODEL_PROVIDERS)


def iter_known_engine_provider_ids() -> tuple[str, ...]:
    """Return provider IDs for engine-backed external repos."""
    return tuple(
        provider.provider_id
        for provider in KNOWN_EXTERNAL_MODEL_PROVIDERS
        if provider.provider_kind == "engine"
    )


def iter_known_utility_provider_ids() -> tuple[str, ...]:
    """Return provider IDs for utility-backed external repos."""
    return tuple(
        provider.provider_id
        for provider in KNOWN_EXTERNAL_MODEL_PROVIDERS
        if provider.provider_kind == "utility"
    )
