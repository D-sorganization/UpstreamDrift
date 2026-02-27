"""Launcher DRY/DbC contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.launchers import docker_dialog, docker_manager
from src.launchers.help_dialogs import ContextHelpDock
from src.launchers.launcher_constants import validate_docker_stage
from src.launchers.model_card import DraggableModelCard
from src.launchers.settings_dialog import (
    TAB_CONFIG,
    TAB_DIAGNOSTICS,
    TAB_LAYOUT,
    validate_tab_index,
)
from src.launchers.startup import StartupResults


def test_validate_docker_stage_accepts_known_values() -> None:
    """Known launcher Docker stages should pass validation unchanged."""
    assert validate_docker_stage("all") == "all"
    assert validate_docker_stage("mujoco") == "mujoco"
    assert validate_docker_stage("pinocchio") == "pinocchio"
    assert validate_docker_stage("drake") == "drake"
    assert validate_docker_stage("base") == "base"


def test_validate_docker_stage_rejects_unknown_value() -> None:
    """Unknown Docker stages must raise to prevent invalid build commands."""
    with pytest.raises(ValueError, match="Invalid Docker stage"):
        validate_docker_stage("gpu")


def test_docker_build_thread_rejects_invalid_stage() -> None:
    """Docker build worker should enforce stage precondition at construction."""
    with pytest.raises(ValueError, match="Invalid Docker stage"):
        docker_manager.DockerBuildThread(target_stage="invalid-stage")


def test_docker_dialog_reuses_shared_check_thread() -> None:
    """Dialog should not duplicate Docker check thread implementation."""
    assert docker_dialog.DockerCheckThread is docker_manager.DockerCheckThread


def test_validate_tab_index_accepts_known_values() -> None:
    """Settings dialog tab contract should allow only known tab indexes."""
    assert validate_tab_index(TAB_LAYOUT) == TAB_LAYOUT
    assert validate_tab_index(TAB_CONFIG) == TAB_CONFIG
    assert validate_tab_index(TAB_DIAGNOSTICS) == TAB_DIAGNOSTICS


def test_validate_tab_index_rejects_unknown_value() -> None:
    """Invalid tab indexes should fail fast with actionable errors."""
    with pytest.raises(ValueError, match="Invalid tab index"):
        validate_tab_index(999)


def test_startup_results_from_dict_uses_defaults() -> None:
    """StartupResults.from_dict should provide safe defaults for missing keys."""
    results = StartupResults.from_dict({"docker_available": True})
    assert results.registry is None
    assert results.engine_manager is None
    assert results.available_engines == []
    assert results.ai_available is False
    assert results.docker_available is True
    assert results.startup_time_ms == 0


def test_context_help_doc_mapping_for_known_engines() -> None:
    """Context help should map known engine IDs to the expected docs paths."""
    dock = Mock(spec=ContextHelpDock)
    mujoco_doc = ContextHelpDock._get_doc_file(dock, "mujoco_humanoid")
    drake_doc = ContextHelpDock._get_doc_file(dock, "drake_golf")
    pinocchio_doc = ContextHelpDock._get_doc_file(dock, "pinocchio_golf")
    matlab_doc = ContextHelpDock._get_doc_file(dock, "matlab_models")

    assert mujoco_doc is not None and mujoco_doc.name == "mujoco.md"
    assert drake_doc is not None and drake_doc.name == "drake.md"
    assert pinocchio_doc is not None and pinocchio_doc.name == "pinocchio.md"
    assert matlab_doc is not None and matlab_doc.name == "matlab.md"


def test_model_card_fallback_image_resolution_uses_model_id() -> None:
    """Model card should infer engine artwork from model ID when name not mapped."""
    card = DraggableModelCard.__new__(DraggableModelCard)
    card.model = SimpleNamespace(
        id="custom_mujoco_tool",
        name="Unmapped Name",
        type="custom",
        engine_type="",
    )
    assert card._resolve_image_name() == "mujoco_humanoid.png"
