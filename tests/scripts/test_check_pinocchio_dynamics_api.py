import importlib.util
import types
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "check_pinocchio_dynamics_api.py"
)
spec = importlib.util.spec_from_file_location(
    "check_pinocchio_dynamics_api", SCRIPT_PATH
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_missing_dynamics_api_reports_wrong_pinocchio_package() -> None:
    wrong_package = types.SimpleNamespace(__version__="0.1")

    assert module.missing_dynamics_api(wrong_package) == module.REQUIRED_DYNAMICS_API


def test_missing_dynamics_api_accepts_complete_robotics_module() -> None:
    robotics_module = types.SimpleNamespace(
        Model=object,
        JointModelFreeFlyer=object,
        SE3=object,
        Inertia=object,
        crba=lambda: None,
        rnea=lambda: None,
        computeCoriolisMatrix=lambda: None,
    )

    assert module.missing_dynamics_api(robotics_module) == ()
