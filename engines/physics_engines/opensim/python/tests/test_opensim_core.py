import pytest
import sys
from unittest.mock import MagicMock, patch
from engines.physics_engines.opensim.python.opensim_golf.core import GolfSwingModel

@pytest.fixture
def mock_opensim_env():
    """Context manager to mock opensim environment."""
    # We patch sys.modules to return a MagicMock when 'opensim' is imported
    with patch.dict(sys.modules, {"opensim": MagicMock()}):
        yield

@pytest.fixture
def mock_opensim_missing_env():
    """Context manager to mock missing opensim environment."""
    # We patch sys.modules so 'opensim' appears missing (KeyError or None)
    # But since we want to trigger ImportError, we can just ensure it's not in sys.modules
    # AND we need to ensure that when `import opensim` is called, it fails.
    # Setting it to None in sys.modules causes ModuleNotFoundError in Python 3.
    with patch.dict(sys.modules, {"opensim": None}):
        yield

class TestGolfSwingModel:
    def test_demo_simulation_runs(self):
        """Test that the demo simulation (fallback) runs correctly."""
        # This doesn't require opensim, so it should be fine even if it's missing
        model = GolfSwingModel(model_path=None)
        assert model.use_opensim is False

        result = model.run_simulation()

        assert result.time is not None
        assert len(result.time) > 0
        assert result.states.shape[1] == 4
        assert "ClubHead" in result.marker_positions

    def test_opensim_not_implemented(self, mock_opensim_env):
        """Test that the OpenSim path raises NotImplementedError."""
        # Because we mocked opensim in sys.modules, _try_load_opensim should succeed
        model = GolfSwingModel(model_path="dummy.osim")

        # Verify it thinks it loaded opensim
        assert model.use_opensim is True

        # Now check the NotImplementedError
        with pytest.raises(NotImplementedError, match="OpenSim integration pending environment setup"):
            model._run_opensim_simulation()

    def test_load_opensim_failure_fallback_logic(self, mock_opensim_missing_env):
        """Test that if opensim fails to load (ImportError), it falls back gracefully."""
        # When we initialize with a path, it tries to load opensim.
        # Our fixture sets sys.modules['opensim'] = None, so import should fail.

        model = GolfSwingModel(model_path="test.osim")

        # Should have caught the import error and defaulted to demo
        assert model.use_opensim is False
        assert model._opensim_model is None
