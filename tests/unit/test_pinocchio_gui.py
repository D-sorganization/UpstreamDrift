"""Unit tests for Pinocchio GUI logic."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
    skip_if_unavailable,
)
from src.shared.python.gui_pkg.gui_utils import get_qapp

if PYQT6_AVAILABLE:
    pass


@pytest.fixture(autouse=True, scope="module")
def mock_pinocchio_gui_dependencies() -> Generator[None, None, None]:
    """Fixture to mock pinocchio and meshcat safely for the duration of this module."""
    with patch.dict(
        "sys.modules",
        {
            "pinocchio": MagicMock(),
            "pinocchio.visualize": MagicMock(),
            "meshcat": MagicMock(),
            "meshcat.geometry": MagicMock(),
            "meshcat.visualizer": MagicMock(),
        },
    ):
        yield


@skip_if_unavailable("pyqt6")
class TestPinocchioGUI:
    """Test Pinocchio GUI."""

    @pytest.fixture
    def qapp(self) -> Any:
        """Ensure QApplication exists."""
        app = get_qapp()
        return app

    @pytest.fixture
    def mock_gui(self, qapp) -> Any:
        """Create a mocked PinocchioGUI instance."""
        from contextlib import ExitStack

        # Late import to ensure mocks apply
        from src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui import (
            MESHCAT_AVAILABLE,
            PinocchioGUI,
        )

        with ExitStack() as stack:
            if MESHCAT_AVAILABLE:
                stack.enter_context(
                    patch(
                        "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui.viz.Visualizer"
                    )
                )
                stack.enter_context(
                    patch(
                        "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui.MeshcatVisualizer"
                    )
                )

            mock_urdf = stack.enter_context(
                patch(
                    "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui.get_shared_urdf_path"
                )
            )
            mock_urdf.return_value.exists.return_value = False

            gui = PinocchioGUI()
            return gui

    def test_ensure_analyzer_initialized(self, mock_gui) -> None:
        """Test _ensure_analyzer_initialized method."""
        # 1. Model is None, Analyzer is None -> Should remain None
        mock_gui.model = None
        mock_gui.analyzer = None
        mock_gui._ensure_analyzer_initialized()
        assert mock_gui.analyzer is None

        # 2. Model is set, Analyzer is None -> Should initialize
        mock_gui.model = MagicMock()
        mock_gui.data = MagicMock()

        with patch(
            "src.engines.physics_engines.pinocchio.python.pinocchio_golf.induced_acceleration.InducedAccelerationAnalyzer"
        ) as MockAnalyzer:
            mock_gui._ensure_analyzer_initialized()

            assert mock_gui.analyzer is not None
            MockAnalyzer.assert_called_once_with(mock_gui.model, mock_gui.data)

        # 3. Model is set, Analyzer is set -> Should not re-initialize
        existing_analyzer = MagicMock()
        mock_gui.analyzer = existing_analyzer

        with patch(
            "src.engines.physics_engines.pinocchio.python.pinocchio_golf.induced_acceleration.InducedAccelerationAnalyzer"
        ) as MockAnalyzer:
            mock_gui._ensure_analyzer_initialized()

            assert mock_gui.analyzer is existing_analyzer
            MockAnalyzer.assert_not_called()

    @pytest.mark.unit
    def test_advance_physics_with_commanded_torque(self, mock_gui) -> None:
        """Test _advance_physics passes commanded torque to pin.aba rather than zeros."""
        import numpy as np

        mock_gui.model = MagicMock()
        mock_gui.model.nv = 2
        mock_gui.data = MagicMock()
        mock_gui.q = np.array([0.0, 0.0])
        mock_gui.v = np.array([0.0, 0.0])
        mock_gui.dt = 0.01
        mock_gui.sim_time = 0.0

        with patch(
            "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation.pin"
        ) as mock_pin:
            mock_pin.aba.return_value = np.array([5.0, -2.0])
            mock_pin.integrate.return_value = np.array([0.05, -0.02])

            # 1. Advance with explicit commanded torque
            cmd_tau = np.array([1.5, -0.5])
            mock_gui._advance_physics(tau=cmd_tau)

            np.testing.assert_array_equal(mock_gui.applied_tau, cmd_tau)
            assert mock_pin.aba.call_count == 1
            call_args = mock_pin.aba.call_args[0]
            assert call_args[0] is mock_gui.model
            assert call_args[1] is mock_gui.data
            np.testing.assert_array_equal(call_args[4], cmd_tau)
            np.testing.assert_allclose(mock_gui.v, np.array([0.05, -0.02]))
            assert mock_gui.sim_time == pytest.approx(0.01)

    @pytest.mark.unit
    def test_set_commanded_torque_alters_trajectory(self, mock_gui) -> None:
        """Test that non-zero commanded torque alters resulting state compared to free-fall."""
        import numpy as np

        mock_gui.model = MagicMock()
        mock_gui.model.nv = 2
        mock_gui.data = MagicMock()
        mock_gui.q = np.array([0.0, 0.0])
        mock_gui.v = np.array([0.0, 0.0])
        mock_gui.dt = 0.01
        mock_gui.sim_time = 0.0

        with patch(
            "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation.pin"
        ) as mock_pin:
            # Free fall (tau=0) returns pure gravitational acceleration
            # Actuated (tau!=0) returns acceleration altered by M^-1 * tau
            def fake_aba(model, data, q, v, tau):
                if np.all(tau == 0.0):
                    return np.array([0.0, -9.81])
                return np.array([10.0, 5.0])

            mock_pin.aba.side_effect = fake_aba
            mock_pin.integrate.side_effect = lambda model, q, dq: q + dq

            # Run free-fall step
            mock_gui._advance_physics()
            v_freefall = mock_gui.v.copy()

            # Reset and run with commanded torque
            mock_gui.v = np.array([0.0, 0.0])
            mock_gui.q = np.array([0.0, 0.0])
            mock_gui.set_commanded_torque(np.array([10.0, 20.0]))
            mock_gui._advance_physics()
            v_actuated = mock_gui.v.copy()

            # Non-zero commanded torque MUST produce a different trajectory from free-fall
            assert not np.allclose(v_freefall, v_actuated)
            assert np.allclose(v_freefall, np.array([0.0, -0.0981]))
            assert np.allclose(v_actuated, np.array([0.1, 0.05]))

    @pytest.mark.unit
    def test_record_frame_preserves_applied_torque(self, mock_gui) -> None:
        """Test that _record_frame records the actual applied torque vector."""
        import numpy as np

        mock_gui.model = MagicMock()
        mock_gui.model.nv = 2
        mock_gui.data = MagicMock()
        mock_gui.data.kinetic_energy = 12.5
        mock_gui.data.potential_energy = -5.0
        mock_gui.q = np.array([0.5, -0.2])
        mock_gui.v = np.array([1.0, 2.0])
        mock_gui.applied_tau = np.array([3.0, 4.0])
        mock_gui.sim_time = 0.05
        mock_gui.recorder = MagicMock()
        mock_gui.lbl_rec_status = MagicMock()
        mock_gui._find_club_head_state = MagicMock(
            return_value=(np.zeros(3), np.zeros(3))
        )
        mock_gui._compute_live_analysis = MagicMock(return_value=({}, {}))

        with patch(
            "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation.pin"
        ):
            mock_gui._record_frame()

            mock_gui.recorder.record_frame.assert_called_once()
            call_kwargs = mock_gui.recorder.record_frame.call_args[1]
            np.testing.assert_array_equal(call_kwargs["tau"], np.array([3.0, 4.0]))
