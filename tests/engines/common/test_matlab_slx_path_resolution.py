"""Tests for MATLAB .slx default path resolution and missing model error handling.

Refs #12984: MATLAB default .slx path resolves to src/src/engines under production suite root.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.engines.loaders import (
    DEFAULT_MATLAB_2D_SLX_RELPATH,
    DEFAULT_MATLAB_3D_SLX_RELPATH,
    _resolve_engine_path,
    load_matlab_2d_engine,
    load_matlab_3d_engine,
)
from src.shared.python.data_io.common_utils import GolfModelingError
from src.shared.python.data_io.path_utils import get_src_root
from src.shared.python.engine_core.interfaces import PhysicsEngine


@pytest.mark.unit
def test_resolve_engine_path_with_src_root() -> None:
    src_root = get_src_root()
    resolved_3d = _resolve_engine_path(src_root, DEFAULT_MATLAB_3D_SLX_RELPATH)
    assert resolved_3d.exists(), f"Path should exist: {resolved_3d}"
    assert "src/src" not in str(resolved_3d).replace("\\", "/")

    resolved_2d = _resolve_engine_path(src_root, DEFAULT_MATLAB_2D_SLX_RELPATH)
    assert resolved_2d.exists(), f"Path should exist: {resolved_2d}"
    assert "src/src" not in str(resolved_2d).replace("\\", "/")


@pytest.mark.unit
def test_resolve_engine_path_with_repo_root() -> None:
    repo_root = get_src_root().parent
    resolved_3d = _resolve_engine_path(repo_root, DEFAULT_MATLAB_3D_SLX_RELPATH)
    assert resolved_3d.exists(), f"Path should exist: {resolved_3d}"
    assert "src/src" not in str(resolved_3d).replace("\\", "/")

    resolved_2d = _resolve_engine_path(repo_root, DEFAULT_MATLAB_2D_SLX_RELPATH)
    assert resolved_2d.exists(), f"Path should exist: {resolved_2d}"
    assert "src/src" not in str(resolved_2d).replace("\\", "/")


@pytest.mark.unit
def test_resolve_engine_path_relative_without_src_prefix() -> None:
    src_root = get_src_root()
    repo_root = src_root.parent
    rel = Path("engines/physics_engines/mujoco/models/simple_pendulum.xml")

    resolved_from_src = _resolve_engine_path(src_root, rel)
    assert resolved_from_src.exists(), (
        f"Should resolve from src_root: {resolved_from_src}"
    )

    resolved_from_repo = _resolve_engine_path(repo_root, rel)
    assert resolved_from_repo.exists(), (
        f"Should resolve from repo_root: {resolved_from_repo}"
    )


@pytest.mark.unit
def test_matlab_3d_loader_loads_default_model_under_production_root() -> None:
    src_root = get_src_root()
    with patch("src.engines.simscape.SimscapeAdapter") as mock_adapter_cls:
        mock_instance = MagicMock(spec=PhysicsEngine)
        mock_adapter_cls.return_value = mock_instance

        engine = load_matlab_3d_engine(src_root)
        assert engine is mock_instance
        mock_instance.load_from_path.assert_called_once()
        loaded_path = Path(mock_instance.load_from_path.call_args[0][0])
        assert loaded_path.exists(), f"Loaded path must exist: {loaded_path}"
        assert "src/src" not in str(loaded_path).replace("\\", "/")


@pytest.mark.unit
def test_matlab_3d_loader_missing_default_model_raises_golf_modeling_error(
    tmp_path: Path,
) -> None:
    with patch("src.engines.simscape.SimscapeAdapter") as mock_adapter_cls:
        mock_instance = MagicMock(spec=PhysicsEngine)
        mock_adapter_cls.return_value = mock_instance

        with pytest.raises(
            GolfModelingError, match="Default MATLAB_3D model not found"
        ):
            load_matlab_3d_engine(tmp_path)
