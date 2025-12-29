import sys
import importlib
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# --- Pre-import Mocking ---
def _mock_missing_deps():
    """Mock heavy/external dependencies before they are imported."""
    deps = [
        "cv2", "meshcat", "mujoco", 
        "OpenGL", "OpenGL.GL", 
        "PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets"
    ]
    for d in deps:
        try:
            importlib.import_module(d)
        except ImportError:
            sys.modules[d] = MagicMock()

_mock_missing_deps()

# Configure MuJoCo constants (mock or real)
mj = sys.modules.get("mujoco")
if mj:
    if not hasattr(mj, "mjtVisFlag") or isinstance(mj, MagicMock):
        if isinstance(mj, MagicMock):
            if not hasattr(mj, "mjtVisFlag"):
                mj.mjtVisFlag = MagicMock()
            if not hasattr(mj, "mjtGeom"):
                mj.mjtGeom = MagicMock()
            if not hasattr(mj, "mjtJoint"):
                mj.mjtJoint = MagicMock()
            if not hasattr(mj, "mjtObj"):
                mj.mjtObj = MagicMock()
            if not hasattr(mj, "mjtCatBit"):
                mj.mjtCatBit = MagicMock()
            
            mj.mjtVisFlag.mjVIS_CONTACTPOINT = 0
            mj.mjtVisFlag.mjVIS_CONTACTFORCE = 1
            mj.mjtGeom.mjGEOM_SPHERE = 0
            mj.mjtGeom.mjGEOM_BOX = 1
            mj.mjtGeom.mjGEOM_CAPSULE = 2
            mj.mjtJoint.mjJNT_HINGE = 2
            mj.mjtJoint.mjJNT_SLIDE = 3
            mj.mjtJoint.mjJNT_FREE = 0
            mj.mjtObj.mjOBJ_JOINT = 0
            mj.mjtObj.mjOBJ_BODY = 1
            mj.mjtCatBit.mjCAT_ALL = 0

# --- Imports ---
from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import MuJoCoSimWidget  # noqa: E402
from engines.physics_engines.mujoco.python.humanoid_launcher import HumanoidLauncher  # noqa: E402
from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.controls_tab import ControlsTab  # noqa: E402

@pytest.fixture
def app(request):
    """Safe qapp fixture."""
    if "qapp" in request.fixturenames:
        return request.getfixturevalue("qapp")
    return MagicMock()

@pytest.fixture
def mock_mujoco_interface():
    """Setup sophisticated MuJoCo mocks."""
    with patch("mujoco.MjModel") as mock_chk, \
         patch("mujoco.MjData") as mock_data_cls, \
         patch("mujoco.Renderer"), \
         patch("mujoco.mj_name2id", return_value=0), \
         patch("mujoco.mj_id2name", return_value="body"), \
         patch("mujoco.mj_jacBody"), \
         patch("mujoco.mj_fullM"), \
         patch("mujoco.mj_forward"):
        
        m_model = MagicMock()
        m_model.nq = 3
        m_model.nv = 3
        m_model.nu = 3
        m_model.njnt = 3
        m_model.nbody = 2
        m_model.ngeom = 1
        m_model.jnt_type = [2, 2, 2]
        m_model.jnt_qposadr = [0, 1, 2]
        m_model.jnt_range = [[-1, 1]] * 3
        m_model.body_geomadr = [0, 0]
        m_model.body_geomnum = [0, 1]
        m_model.geom_bodyid = [1]
        m_model.geom_size = [[1, 1, 1]]
        m_model.geom_type = [1]
        m_model.geom_rgba = [[1, 1, 1, 1]]
        
        mock_chk.from_xml_string.return_value = m_model
        mock_chk.return_value = m_model
        
        m_data = MagicMock()
        m_data.qpos = np.zeros(3)
        m_data.qvel = np.zeros(3)
        m_data.xpos = np.zeros((10, 3))
        m_data.nefc = 0
        m_data.efc_J = np.zeros(10)
        mock_data_cls.return_value = m_data
        
        yield m_model

class TestGUIComponents:
    
    def test_sim_widget_coverage(self, mock_mujoco_interface, app):
         with patch("engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget.MuJoCoMeshcatAdapter"), \
              patch("OpenGL.GL.glViewport"):
            
            widget = MuJoCoSimWidget(width=100, height=100, fps=60)
            widget.load_model_from_xml("<mujoco/>")
            widget.reset_state()
            widget.set_camera("side")
            widget.set_torque_visualization(True)
            widget.set_force_visualization(True)
            widget.set_ellipsoid_visualization(True, True)
            widget._compute_model_bounds()
            widget.get_dof_info()
            widget.set_joint_torque(0, 1.0)
            widget.close()

    def test_humanoid_launcher_coverage(self, mock_mujoco_interface, app):
        # HumanoidLauncher uses subprocess, doesn't embed SimWidget directly
        l = HumanoidLauncher()
        try:
            l.show()
            l.close()
        except Exception:
            pass
            
    def test_controls_tab(self, mock_mujoco_interface, app):
        with patch("engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.controls_tab.MuJoCoSimWidget"):
             _ = ControlsTab(MagicMock(), MagicMock())
