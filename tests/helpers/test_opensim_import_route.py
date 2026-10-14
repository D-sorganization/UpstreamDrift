"""Every test-side OpenSim import goes through ``require_opensim`` (#12065).

``tests.helpers.opensim_support.require_opensim`` holds the native-runtime
boundary before it imports the SDK.  A test that imports ``opensim`` directly,
or reaches it through ``pytest.importorskip("opensim")``, bypasses that
boundary and can load an ABI-incompatible ``ezc3d.dll`` into a pytest process
that already hosts the Python ``ezc3d`` extension.  This guard scans the test
tree with the AST (so source text inside strings, e.g. child-process scripts,
is not mistaken for an import) and fails on any bypass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_EXEMPT = frozenset(
    {
        _TESTS_ROOT / "helpers" / "opensim_support.py",
        _TESTS_ROOT / "helpers" / "test_opensim_support.py",
        # Explicit exemptions for engine availability and qualification gates
        _TESTS_ROOT / "opensim" / "test_opensim_os0_qualification.py",
        _TESTS_ROOT
        / "shared"
        / "reporting"
        / "swing_report"
        / "test_adapter_parity.py",
        _TESTS_ROOT / "unit" / "test_third_party_integration_batch2.py",
        _TESTS_ROOT / "integration" / "test_urdf_engine_loadability.py",
    }
)


def _is_opensim(name: str | None) -> bool:
    return name is not None and (name == "opensim" or name.startswith("opensim."))


def _is_importorskip(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute):
        return func.attr == "importorskip"
    return isinstance(func, ast.Name) and func.id == "importorskip"


def opensim_bypasses(source: str, filename: str = "<source>") -> list[str]:
    """Return ``"line: statement"`` for each direct OpenSim import in *source*."""
    findings: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            if any(_is_opensim(alias.name) for alias in node.names):
                findings.append(f"{node.lineno}: import opensim")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and _is_opensim(node.module):
                findings.append(f"{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call) and _is_importorskip(node.func):
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and _is_opensim(
                first.value if isinstance(first.value, str) else None
            ):
                findings.append(f"{node.lineno}: importorskip({first.value!r})")
    return findings


@pytest.mark.parametrize(
    "source",
    [
        "import opensim\n",
        "import opensim as osim\n",
        "import numpy, opensim.simulation\n",
        "from opensim import Model\n",
        "def f():\n    import opensim as osim\n",
        "osim = pytest.importorskip('opensim')\n",
        "pytest.importorskip('opensim', reason='x')\n",
        "from pytest import importorskip\nimportorskip('opensim')\n",
    ],
)
def test_scanner_flags_direct_opensim_imports(source: str) -> None:
    assert opensim_bypasses(source)


@pytest.mark.parametrize(
    "source",
    [
        "osim = require_opensim()\n",
        "from src.engines.physics_engines.opensim.python import fk\n",
        "import opensim_models\n",
        "from . import opensim\n",
        "pytest.importorskip('mujoco')\n",
        "CHILD = '''\nimport opensim\n'''\n",
        "spec = importlib.util.find_spec('opensim')\n",
    ],
)
def test_scanner_ignores_routed_and_unrelated_code(source: str) -> None:
    assert opensim_bypasses(source) == []


def test_no_test_file_bypasses_require_opensim() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if path in _EXEMPT:
            continue
        source = path.read_text(encoding="utf-8")
        if "opensim" not in source:  # cheap pre-filter; the AST decides
            continue
        findings = opensim_bypasses(source, str(path))
        if findings:
            offenders[path.relative_to(_TESTS_ROOT).as_posix()] = findings
    assert offenders == {}, (
        "route these OpenSim imports through "
        "tests.helpers.opensim_support.require_opensim():\n"
        + "\n".join(f"  tests/{name}: {hits}" for name, hits in offenders.items())
    )
