import importlib.util
import types
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("broken", ["missing", "partial"])
def test_required_lane_api_check_fails_on_missing_or_partial_pinocchio(
    monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    def import_module(_name: str) -> types.SimpleNamespace:
        if broken == "missing":
            raise ModuleNotFoundError("pinocchio is absent")
        return types.SimpleNamespace(__version__="0.1")

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    assert module.check_pinocchio_dynamics_api() == 1
