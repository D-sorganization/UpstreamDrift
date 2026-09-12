"""Native imports must not select the legacy generic model writer."""

from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def test_native_import_keeps_export_precision_in_fresh_process() -> None:
    code = """
import sys, pathlib, tomllib, inspect, json, runpy, xml.etree.ElementTree as ET
root = pathlib.Path.cwd()
paths = tomllib.loads((root / 'pyproject.toml').read_text())['tool']['pytest']['ini_options']['pythonpath']
sys.path[:0] = [str(root / path) for path in paths]
from src.engines.physics_engines.mujoco.python.native_model import NativeMujocoModel
from src.shared.python.motion_matching.native_urdf import export_native_urdf, URDFWriter
fixture = runpy.run_path(str(root / 'tests/unit/motion_matching/test_native_urdf_export.py'))
document = fixture['spec'].__wrapped__()
value = 0.12345678901234566
document['bodies'][1]['solids'][0]['com_m'][0] = value
xml, sidecar = export_native_urdf(json.dumps(document).encode())
actual = float(ET.fromstring(xml).find("link[@name='solid_0']/inertial/origin").attrib['xyz'].split()[0])
assert abs(actual - value) < 1e-17, (actual, value)
assert pathlib.Path(inspect.getfile(URDFWriter)).resolve() == root / 'src/shared/python/model_generation/builders/urdf_writer.py'
assert 'src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_engine_attribute_still_resolves_and_caches_in_fresh_process() -> None:
    code = """
import sys, types
import src.engines.physics_engines.mujoco as package
name = 'src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine'
assert name not in sys.modules
sentinel = object()
module = types.ModuleType(name)
module.MuJoCoPhysicsEngine = sentinel
sys.modules[name] = module
assert package.Engine is sentinel
assert package.Engine is sentinel
try:
    package.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
