"""Unit tests for conventional external provider onboarding metadata."""

from __future__ import annotations

import os
from pathlib import Path

from src.shared.python.config.provider_catalog import (
    infer_repo_root_from_config,
    iter_configured_provider_roots,
    iter_known_engine_provider_ids,
    iter_known_provider_ids,
    iter_known_provider_repo_names,
    iter_known_utility_provider_ids,
    iter_provider_manifest_specs,
)


def test_iter_known_provider_metadata_covers_engine_and_utility_repos() -> None:
    assert iter_known_provider_ids() == (
        "mujoco_models",
        "drake_models",
        "pinocchio_models",
        "opensim_models",
        "tools",
        "movement_optimizer",
    )
    assert iter_known_provider_repo_names() == (
        "MuJoCo_Models",
        "Drake_Models",
        "Pinocchio_Models",
        "OpenSim_Models",
        "Tools",
        "Movement-Optimizer",
    )
    assert iter_known_engine_provider_ids() == (
        "mujoco_models",
        "drake_models",
        "pinocchio_models",
        "opensim_models",
    )
    assert iter_known_utility_provider_ids() == ("tools", "movement_optimizer")


def test_infer_repo_root_from_standard_models_config(tmp_path: Path) -> None:
    config_path = tmp_path / "UpstreamDrift" / "src" / "config" / "models.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("models: []\n", encoding="utf-8")

    repo_root = infer_repo_root_from_config(config_path)

    assert repo_root == config_path.parent.parent.parent


def test_iter_configured_provider_roots_merges_env_and_sibling_defaults(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path
    repo_root = workspace_root / "UpstreamDrift"
    config_path = repo_root / "src" / "config" / "models.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("models: []\n", encoding="utf-8")

    explicit_root = workspace_root / "custom-provider"
    explicit_root.mkdir()

    roots = iter_configured_provider_roots(
        config_path,
        str(explicit_root) + os.pathsep + "Drake_Models",
    )

    assert explicit_root in roots
    assert workspace_root / "MuJoCo_Models" in roots
    assert workspace_root / "Drake_Models" in roots
    assert workspace_root / "Pinocchio_Models" in roots
    assert workspace_root / "OpenSim_Models" in roots
    assert workspace_root / "Tools" in roots
    assert workspace_root / "Movement-Optimizer" in roots


def test_iter_provider_manifest_specs_supports_hidden_manifest_location(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path
    repo_root = workspace_root / "UpstreamDrift"
    config_path = repo_root / "src" / "config" / "models.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("models: []\n", encoding="utf-8")

    provider_root = workspace_root / "MuJoCo_Models"
    hidden_manifest = provider_root / ".upstreamdrift" / "model_pack.yaml"
    hidden_manifest.parent.mkdir(parents=True)
    hidden_manifest.write_text("manifest_version: '1.0.0'\n", encoding="utf-8")

    specs = iter_provider_manifest_specs(config_path, None)

    assert (provider_root, hidden_manifest) in specs


def test_iter_provider_manifest_specs_discovers_tools_pendulum_manifest(
    tmp_path: Path,
) -> None:
    """Tools publishes Pendulum Simulator as a nested provider manifest."""
    workspace_root = tmp_path
    repo_root = workspace_root / "UpstreamDrift"
    config_path = repo_root / "src" / "config" / "models.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("models: []\n", encoding="utf-8")

    tools_root = workspace_root / "Tools"
    nested_manifest = tools_root / "src" / "pendulum_simulator" / "model_pack.yaml"
    nested_manifest.parent.mkdir(parents=True)
    nested_manifest.write_text("manifest_version: '1.0.0'\n", encoding="utf-8")

    specs = iter_provider_manifest_specs(config_path, None)

    assert (tools_root, nested_manifest) in specs
