import pytest
from unittest.mock import MagicMock, patch
from src.launchers.upstream_drift_launcher import LauncherOrchestrator
from src.launchers.ui_components import StartupResults


def test_launcher_orchestrator_initialization():
    orchestrator = LauncherOrchestrator()
    assert orchestrator.registry is None
    assert orchestrator.engine_manager is None
    assert orchestrator.docker_available is False
    assert isinstance(orchestrator.available_models, dict)
    assert isinstance(orchestrator.special_app_lookup, dict)


def test_initialize_from_results():
    orchestrator = LauncherOrchestrator()
    results = MagicMock(spec=StartupResults)
    results.docker_available = True
    results.registry = MagicMock()
    results.engine_manager = MagicMock()

    orchestrator.initialize_from_results(results)

    assert orchestrator.docker_available is True
    assert orchestrator.registry == results.registry
    assert orchestrator.engine_manager == results.engine_manager


@patch("src.launchers.upstream_drift_launcher._lazy_load_model_registry")
def test_init_registry_fallback(mock_lazy_load):
    mock_mr_class = MagicMock()
    mock_lazy_load.return_value = mock_mr_class

    orchestrator = LauncherOrchestrator()
    orchestrator.init_registry(None)

    mock_lazy_load.assert_called_once()
    mock_mr_class.assert_called_once()
    assert orchestrator.registry == mock_mr_class.return_value


@patch("src.launchers.upstream_drift_launcher._lazy_load_engine_manager")
def test_init_engine_manager_fallback(mock_lazy_load):
    mock_em_class = MagicMock()
    mock_lazy_load.return_value = (mock_em_class, MagicMock())

    orchestrator = LauncherOrchestrator()
    orchestrator.init_engine_manager(None)

    mock_lazy_load.assert_called_once()
    mock_em_class.assert_called_once()
    assert orchestrator.engine_manager == mock_em_class.return_value


def test_build_available_models():
    orchestrator = LauncherOrchestrator()
    orchestrator.registry = MagicMock()

    mock_model1 = MagicMock()
    mock_model1.id = "model1"
    mock_model1.type = "special_app"
    mock_model1.name = "Model 1"

    mock_model2 = MagicMock()
    mock_model2.id = "model2"
    mock_model2.type = "standard"
    mock_model2.name = "Model 2"

    orchestrator.registry.get_all_models.return_value = [mock_model1, mock_model2]

    orchestrator.build_available_models()

    assert "model1" in orchestrator.available_models
    assert "model2" in orchestrator.available_models
    assert "model1" in orchestrator.special_app_lookup
    assert "model2" not in orchestrator.special_app_lookup


def test_get_model():
    orchestrator = LauncherOrchestrator()

    # Test ValueError when None
    with pytest.raises(ValueError):
        orchestrator.get_model(None)

    mock_model = MagicMock()
    orchestrator.available_models["test_model"] = mock_model

    # Test from available_models
    assert orchestrator.get_model("test_model") == mock_model

    # Test from registry fallback
    orchestrator.registry = MagicMock()
    orchestrator.registry.get_model.return_value = "registry_model"
    assert orchestrator.get_model("unknown_model") == "registry_model"
    orchestrator.registry.get_model.assert_called_once_with("unknown_model")
