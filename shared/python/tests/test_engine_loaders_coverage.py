"""Tests for shared.python.engine_loaders coverage."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock engines before importing loader
with patch.dict(
    sys.modules,
    {
        "mujoco": MagicMock(),
        "pydrake": MagicMock(),
        "pydrake.all": MagicMock(),
        "pinocchio": MagicMock(),
        "matlab": MagicMock(),
        "matlab.engine": MagicMock(),
    },
):
    from shared.python.engine_loaders import (
        load_drake_engine,
        load_mujoco_engine,
    )


def test_load_mujoco_success(tmp_path: object) -> None:
    """Test successful loading of MuJoCo engine."""
    from pathlib import Path
    path = Path(str(tmp_path))

    # Mock internal components needed by load_mujoco_engine
    with patch.dict(
        sys.modules,
        {
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine": MagicMock(),
            "shared.python.engine_probes": MagicMock(),
        },
    ):
        # Mock specific classes
        mock_engine_cls = MagicMock()
        mock_probe_cls = MagicMock()

        sys.modules[
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine"
        ].MuJoCoPhysicsEngine = mock_engine_cls  # type: ignore
        sys.modules["shared.python.engine_probes"].MuJoCoProbe = mock_probe_cls  # type: ignore

        # Setup probe result
        mock_probe_instance = mock_probe_cls.return_value
        mock_result = MagicMock()
        mock_result.is_available.return_value = True
        mock_probe_instance.probe.return_value = mock_result

        load_mujoco_engine(path)

        mock_engine_cls.assert_called_once()


def test_load_drake_missing(tmp_path: object) -> None:
    """Test handling of missing Drake engine."""
    from pathlib import Path
    path = Path(str(tmp_path))
    # Use the exact same exception class used by the module under test

    # We need to ensure pydrake is NOT in sys.modules so import is attempted
    with patch.dict(sys.modules):
        if "pydrake" in sys.modules:
            del sys.modules["pydrake"]

        # Force ImportError when 'pydrake' is imported
        original_import = __import__

        def side_effect(name, *args, **kwargs):
            if name == "pydrake":
                raise ImportError("No module named pydrake")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=side_effect):
            # load_drake_engine catches ImportError and raises GolfModelingError
            # Catch base Exception to avoid class identity issues during patching
            with pytest.raises(Exception) as excinfo:
                load_drake_engine(path)

            assert "Drake requirements not met" in str(excinfo.value)
