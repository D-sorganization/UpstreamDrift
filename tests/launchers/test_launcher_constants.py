from unittest.mock import patch

import pytest

from src.launchers.launcher_constants import (
    _lazy_imports,
    _lazy_load_engine_manager,
    _lazy_load_model_registry,
    validate_docker_stage,
)


def test_validate_docker_stage():
    """Test Docker stage validation."""
    assert validate_docker_stage("all") == "all"
    assert validate_docker_stage("mujoco") == "mujoco"

    with pytest.raises(ValueError, match="Invalid Docker stage 'invalid'"):
        validate_docker_stage("invalid")


def test_lazy_load_engine_manager():
    """Test lazy loading of engine manager."""
    # Reset lazy imports for testing
    _lazy_imports["EngineManager"] = None
    _lazy_imports["EngineType"] = None

    em, et = _lazy_load_engine_manager()
    assert em is not None
    assert et is not None

    # Second load should use cached
    with patch("importlib.import_module") as mock_import:
        em2, et2 = _lazy_load_engine_manager()
        assert em2 is em
        assert et2 is et
        mock_import.assert_not_called()


def test_lazy_load_model_registry():
    """Test lazy loading of model registry."""
    # Reset lazy imports for testing
    _lazy_imports["ModelRegistry"] = None

    mr = _lazy_load_model_registry()
    assert mr is not None

    # Second load should use cached
    with patch("importlib.import_module") as mock_import:
        mr2 = _lazy_load_model_registry()
        assert mr2 is mr
        mock_import.assert_not_called()
