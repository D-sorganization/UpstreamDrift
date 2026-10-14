"""Every OpenSim import goes through its single route (#12065).

Test side: ``tests.helpers.opensim_support.require_opensim``.  Production
side: ``src.shared.python.engine_core.opensim_runtime.import_opensim``, which
types the Windows ``ezc3d.dll`` ABI conflict instead of a bare ``ImportError``.

``tests.helpers.opensim_support.require_opensim`` holds the native-runtime
boundary before it imports the SDK.  A test that imports ``opensim`` directly,
or reaches it through ``pytest.importorskip("opensim")``, bypasses that
boundary and can load an ABI-incompatible ``ezc3d.dll`` into a pytest process
that already hosts the Python ``ezc3d`` extension.  This guard scans the test
and src trees with the AST (so source text inside strings, e.g. child-process
scripts, is not mistaken for an import) and fails on any bypass, including
dynamic loads such as ``importlib.import_module("opensim")`` and
``__import__("opensim")`` with a literal name.  Imports under
``if TYPE_CHECKING:`` never execute, so they load no DLL and are ignored, and
``importlib.util.find_spec("opensim")`` only locates the package without
loading it, so availability probes stay allowed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _TESTS_ROOT.parent / "src"
_SRC_EXEMPT = frozenset(
    {_SRC_ROOT / "shared" / "python" / "engine_core" / "opensim_runtime.py"}
)
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
        _TESTS_ROOT / "integration" / "cross_engine" / "test_force_overlay_parity.py",
    }
)


def _is_opensim(name: str | None) -> bool:
    return name is not None and (name == "opensim" or name.startswith("opensim."))


def _is_importorskip(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute):
        return func.attr == "importorskip"
    return isinstance(func, ast.Name) and func.id == "importorskip"


_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})


def _dynamic_importer(func: ast.expr) -> str | None:
    """Name of a dynamic importer call (``import_module``/``__import__``)."""
    name = func.attr if isinstance(func, ast.Attribute) else None
    if isinstance(func, ast.Name):
        name = func.id
    return name if name in _DYNAMIC_IMPORTERS else None


def _literal_opensim(node: ast.Call) -> str | None:
    """The first positional argument when it is a literal OpenSim module name."""
    first = node.args[0] if node.args else None
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value if _is_opensim(first.value) else None
    return None


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def _runtime_nodes(tree: ast.AST) -> list[ast.AST]:
    """Every node except those only reachable under ``if TYPE_CHECKING:``."""
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            skipped.update(id(child) for stmt in node.body for child in ast.walk(stmt))
    return [node for node in ast.walk(tree) if id(node) not in skipped]


def opensim_bypasses(source: str, filename: str = "<source>") -> list[str]:
    """Return ``"line: statement"`` for each direct OpenSim import in *source*."""
    findings: list[str] = []
    for node in _runtime_nodes(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            if any(_is_opensim(alias.name) for alias in node.names):
                findings.append(f"{node.lineno}: import opensim")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and _is_opensim(node.module):
                findings.append(f"{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call):
            module = _literal_opensim(node)
            if module is None:
                continue
            if _is_importorskip(node.func):
                findings.append(f"{node.lineno}: importorskip({module!r})")
            elif (importer := _dynamic_importer(node.func)) is not None:
                findings.append(f"{node.lineno}: {importer}({module!r})")
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
        "if TYPE_CHECKING:\n    pass\nelse:\n    import opensim\n",
        "osim = importlib.import_module('opensim')\n",
        "from importlib import import_module\nosim = import_module('opensim')\n",
        "path = importlib.import_module('opensim._simulation').__file__\n",
        "osim = __import__('opensim')\n",
        "ok = hasattr(__import__('opensim'), 'Visualizer')\n",
        "def f():\n    return import_module('x') or import_module('opensim')\n",
    ],
)
def test_scanner_flags_direct_opensim_imports(source: str) -> None:
    assert opensim_bypasses(source)


@pytest.mark.parametrize(
    "source",
    [
        "osim = require_opensim()\n",
        "osim = import_opensim()\n",
        "from src.engines.physics_engines.opensim.python import fk\n",
        "import opensim_models\n",
        "from . import opensim\n",
        "pytest.importorskip('mujoco')\n",
        "CHILD = '''\nimport opensim\n'''\n",
        "spec = importlib.util.find_spec('opensim')\n",
        "if TYPE_CHECKING:\n    import opensim as osim\n",
        "if typing.TYPE_CHECKING:\n    from opensim import Model\n",
        "osim = import_opensim('opensim._simulation')\n",
        "mod = importlib.import_module('opensim_models')\n",
        "mod = importlib.import_module(dotted)\n",
        "np = __import__('numpy')\n",
        "if importlib.util.find_spec('opensim') is not None:\n    pass\n",
        "if TYPE_CHECKING:\n    osim = importlib.import_module('opensim')\n",
        "lib = ctypes.CDLL('opensim')\n",
    ],
)
def test_scanner_ignores_routed_and_unrelated_code(source: str) -> None:
    assert opensim_bypasses(source) == []


def _offenders(root: Path, exempt: frozenset[Path]) -> dict[str, list[str]]:
    """Map ``<tree>/<relative path>`` to the bypasses found under *root*."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        if path in exempt:
            continue
        source = path.read_text(encoding="utf-8")
        if "opensim" not in source:  # cheap pre-filter; the AST decides
            continue
        findings = opensim_bypasses(source, str(path))
        if findings:
            offenders[path.relative_to(root.parent).as_posix()] = findings
    return offenders


def _report(route: str, offenders: dict[str, list[str]]) -> str:
    lines = [f"  {name}: {hits}" for name, hits in offenders.items()]
    return "\n".join([f"route these OpenSim imports through {route}:", *lines])


def test_no_test_file_bypasses_require_opensim() -> None:
    offenders = _offenders(_TESTS_ROOT, _EXEMPT)
    assert offenders == {}, _report(
        "tests.helpers.opensim_support.require_opensim()", offenders
    )


def test_no_src_file_bypasses_import_opensim() -> None:
    offenders = _offenders(_SRC_ROOT, _SRC_EXEMPT)
    assert offenders == {}, _report(
        "src.shared.python.engine_core.opensim_runtime.import_opensim()", offenders
    )
