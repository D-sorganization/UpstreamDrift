"""Tests for shared.python.engine_loaders coverage."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_load_mujoco_success(tmp_path: object) -> None:
    """Test successful loading of MuJoCo engine."""
    from pathlib import Path

    path = Path(str(tmp_path))

    mock_engine_cls = MagicMock()
    mock_engine_instance = MagicMock()
    mock_engine_cls.return_value = mock_engine_instance

    mock_probe_cls = MagicMock()
    mock_probe_instance = mock_probe_cls.return_value
    mock_result = MagicMock()
    mock_result.is_available.return_value = True
    mock_probe_instance.probe.return_value = mock_result

    # Patch at the canonical import location used by src.engines.loaders
    with (
        patch.dict(sys.modules, {"mujoco": MagicMock()}),
        patch(
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.MuJoCoPhysicsEngine",
            mock_engine_cls,
        ),
        patch(
            "src.shared.python.engine_core.engine_probes.MuJoCoProbe",
            mock_probe_cls,
        ),
    ):
        from src.engines.loaders import load_mujoco_engine

        result = load_mujoco_engine(path)

        mock_engine_cls.assert_called_once()
        assert result is mock_engine_instance


def test_load_drake_missing(tmp_path: object) -> None:
    """Test handling of missing Drake engine."""
    from pathlib import Path

    path = Path(str(tmp_path))

    # Ensure pydrake is NOT in sys.modules so import is attempted fresh
    # Back up and then delete any existing pydrake from sys.modules
    pydrake_backup = sys.modules.pop("pydrake", None)
    pydrake_all_backup = sys.modules.pop("pydrake.all", None)

    try:
        # Force ImportError when 'pydrake' is imported
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def side_effect(name, *args, **kwargs):
            if name == "pydrake" or name.startswith("pydrake."):
                raise ImportError(f"No module named {name}")
            return original_import(name, *args, **kwargs)

        # Need to mock other engines to allow import of engine_loaders
        with (
            patch("builtins.__import__", side_effect=side_effect),
            patch.dict(
                sys.modules,
                {
                    "mujoco": MagicMock(),
                    "pinocchio": MagicMock(),
                    "matlab": MagicMock(),
                    "matlab.engine": MagicMock(),
                },
            ),
        ):
            from shared.python.engine_core.engine_loaders import load_drake_engine

            # load_drake_engine catches ImportError and raises GolfModelingError
            with pytest.raises(Exception) as excinfo:
                load_drake_engine(path)

            assert "Drake requirements not met" in str(excinfo.value)
    finally:
        # Restore backed up modules
        if pydrake_backup is not None:
            sys.modules["pydrake"] = pydrake_backup
        if pydrake_all_backup is not None:
            sys.modules["pydrake.all"] = pydrake_all_backup
