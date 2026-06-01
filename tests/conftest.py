"""Shared fixtures and utilities for the Golf Modeling Suite test suite.

This module centralizes common setup logic to improve test orthogonality
and adherence to the DRY principle.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prioritize local packages over installed site-packages to prevent package shadowing (e.g. tools)
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Resolve tools package shadowing by explicitly adding root tools directory to its __path__
# Resolve tools package shadowing by explicitly adding all local tools directories to its __path__
try:
    import tools

    root_path = Path(__file__).resolve().parent.parent
    local_dirs = [
        str(root_path / "tools"),
        str(root_path / "src" / "tools"),
        str(root_path / "src" / "shared" / "python" / "tools"),
        str(root_path / "tests" / "ui" / "tools"),
    ]

    if hasattr(tools, "__path__"):
        for d in local_dirs:
            if d not in tools.__path__:
                tools.__path__.insert(0, d)
except ImportError:
    pass

import os


# ---------------------------------------------------------------------------
# Fleet Testing Standards §5: thread-safety + headless env vars.
# Must be set BEFORE any heavy import (numpy, matplotlib, Qt, etc.) so that
# C-extension thread pools and matplotlib/Qt backends pick them up.
# See: docs/FLEET_TESTING_STANDARDS.md in the Repository_Management repo.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib
import importlib.util
import sys
import warnings
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# On Windows, missing PyQt6 DLLs can cause a fatal crash.
# Mock them immediately before any imports happen.
try:
    import PyQt6.QtCore as _pyqt6_qtcore

    _has_pyqt6 = _pyqt6_qtcore is not None
except ImportError:
    _has_pyqt6 = False

if not _has_pyqt6 and "PyQt6" not in sys.modules:
    mock_core = MagicMock()
    mock_core.QLibraryInfo.version.return_value.toString.return_value = "6.6.0"
    mock_core.QLibraryInfo.version.return_value.segments.return_value = (6, 6, 0)
    mock_core.PYQT_VERSION_STR = "6.6.0"
    mock_core.PYQT_VERSION = 0x060600
    mock_core.__version__ = "6.6.0"
    mock_core.qVersion.return_value = "6.6.0"

    pyqt_mock = MagicMock()
    pyqt_mock.QtCore = mock_core
    pyqt_mock.QtGui = MagicMock()

    class DummyWidget:
        def __init__(self, *args, **kwargs):
            self.__dict__["_mocks"] = {}

        def __getattr__(self, name):
            if name not in self.__dict__["_mocks"]:
                mock = MagicMock()
                if name == "font":
                    font_mock = MagicMock()
                    font_mock.families.return_value = ["Outfit"]
                    mock.return_value = font_mock
                elif name == "actions":
                    mock.return_value = [MagicMock()] * 4
                elif name == "menuBar":
                    mock.return_value = DummyWidget()
                elif name == "findChildren":

                    def mock_findChildren(*args, **kwargs):
                        btns = []
                        for n in [
                            "Home",
                            "Engines",
                            "Biomechanics",
                            "Settings",
                            "Documentation",
                        ]:
                            b = MagicMock()
                            b.accessibleName.return_value = n
                            b.icon.return_value.isNull.return_value = False
                            btns.append(b)
                        return btns

                    mock.side_effect = mock_findChildren
                self.__dict__["_mocks"][name] = mock
            return self.__dict__["_mocks"][name]

        @classmethod
        def instance(cls):
            return MagicMock()

    DummyWidget.Shape = MagicMock()
    DummyWidget.ToolButtonPopupMode = MagicMock()
    DummyWidget.setTabOrder = MagicMock()
    DummyWidget.DockWidgetFeature = MagicMock()

    qt_widgets = MagicMock()
    qt_widgets.QWidget = DummyWidget
    qt_widgets.QApplication = DummyWidget
    qt_widgets.QLabel = DummyWidget
    qt_widgets.QComboBox = DummyWidget
    qt_widgets.QToolBar = DummyWidget
    qt_widgets.QDockWidget = DummyWidget
    qt_widgets.QSplitter = DummyWidget
    qt_widgets.QScrollArea = DummyWidget
    qt_widgets.QToolButton = DummyWidget
    qt_widgets.QDialog = DummyWidget
    qt_widgets.QVBoxLayout = DummyWidget
    qt_widgets.QHBoxLayout = DummyWidget
    qt_widgets.QGridLayout = DummyWidget
    qt_widgets.QFrame = DummyWidget
    qt_widgets.QPushButton = DummyWidget
    qt_widgets.QDoubleSpinBox = DummyWidget
    qt_widgets.QSlider = DummyWidget
    qt_widgets.QGroupBox = DummyWidget
    qt_widgets.QMainWindow = DummyWidget
    qt_widgets.QSplitter = DummyWidget
    qt_widgets.QMenuBar = DummyWidget
    qt_widgets.QMenu = DummyWidget
    pyqt_mock.QtWidgets = qt_widgets
    pyqt_mock.QtWebEngineWidgets = MagicMock()
    sys.modules["PyQt6"] = pyqt_mock
    sys.modules["PyQt6.QtCore"] = mock_core
    sys.modules["PyQt6.QtGui"] = pyqt_mock.QtGui
    sys.modules["PyQt6.QtWidgets"] = pyqt_mock.QtWidgets
    sys.modules["PyQt6.QtWebEngineWidgets"] = pyqt_mock.QtWebEngineWidgets


@pytest.fixture(autouse=True)
def _no_real_network_in_unit_lane(request, monkeypatch):
    """Block real outbound HTTP from unit tests by default.

    Fleet Testing Standards §5: unit-marked tests must not make real
    network calls. Tests that need the network must be marked
    ``requires_network`` (and typically ``slow``).
    """
    if "unit" not in request.keywords:
        return

    def _refuse(*_a, **_kw):
        raise RuntimeError(
            "Unit test made a real network call. Mock with `responses` "
            "or `pytest-httpx`, or mark the test "
            "`@pytest.mark.requires_network`."
        )

    for module in ("httpx", "requests", "urllib.request"):
        try:
            mod = __import__(module, fromlist=["*"])
        except ImportError:
            continue
        for attr in ("get", "post", "put", "delete", "request"):
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, _refuse, raising=False)


@dataclass(frozen=True)
class OptionalCollectionRule:
    """Collection rule for test stacks that require optional modules."""

    path_suffixes: tuple[str, ...]
    modules: tuple[str, ...] = ()
    symbols: tuple[tuple[str, str], ...] = ()


_PROCESS_CALCULATOR_ANCHOR = "sidekick.process_calculators.acid_gas_dewpoint_calculator"
_PROCESS_CALCULATOR_TESTS = (
    "tests/unit/process_calculators",
    "tests/unit/injury/test_injury_risk.py",
    "tests/unit/injury/test_joint_stress.py",
    "tests/unit/injury/test_spinal_load_analysis.py",
    "tests/unit/sidekick/test_acid_gas_dewpoint.py",
    "tests/unit/sidekick/test_analysis_utils.py",
    "tests/unit/sidekick/test_baghouse_calculator.py",
    "tests/unit/sidekick/test_electrode_and_thermal.py",
    "tests/unit/sidekick/test_financial_calculator.py",
    "tests/unit/sidekick/test_flare_calculator.py",
    "tests/unit/sidekick/test_gas_properties.py",
    "tests/unit/sidekick/test_pipe_database.py",
    "tests/unit/sidekick/test_pressure_drop_interface.py",
    "tests/unit/sidekick/test_process_constants.py",
    "tests/unit/sidekick/test_syngas_compression.py",
    "tests/unit/sidekick/test_ui_modules_importable.py",
    "tests/unit/sidekick/test_wgs_reactor_calculator.py",
)
_CALC_BACKEND_TESTS = (
    "tests/security/test_rate_limiting.py",
    "tests/unit/calc_backend",
    "tests/unit/test_calc_backend_protocols.py",
    "tests/unit/api/test_acid_gas_dewpoint_mocked.py",
    "tests/unit/api/test_baghouse_mocked.py",
    "tests/unit/api/test_financial_mocked.py",
    "tests/unit/api/test_flare_mocked.py",
    "tests/unit/api/test_flow_rate_api.py",
    "tests/unit/api/test_ode_solver.py",
    "tests/unit/api/test_pressure_drop.py",
    "tests/unit/api/test_scrubber_mocked.py",
    "tests/unit/api/test_syngas_water_mocked.py",
    "tests/unit/api/test_thermal_profile.py",
    "tests/unit/api/test_wgs_reactor_mocked.py",
)
_OPTIONAL_COLLECTION_RULES = (
    OptionalCollectionRule(
        path_suffixes=_PROCESS_CALCULATOR_TESTS,
        modules=(_PROCESS_CALCULATOR_ANCHOR,),
    ),
    OptionalCollectionRule(
        path_suffixes=_CALC_BACKEND_TESTS,
        modules=("src.shared.python.calc_backend.contracts.acid_gas_dewpoint",),
    ),
    OptionalCollectionRule(
        path_suffixes=(
            "tests/unit/signal_toolkit",
            "tests/unit/shared_python/test_signal_toolkit_calculus.py",
            "tests/unit/shared_python/test_signal_toolkit_core.py",
            "tests/unit/shared_python/test_signal_toolkit_filters.py",
            "tests/unit/shared_python/test_signal_toolkit_fitting.py",
            "tests/unit/shared_python/test_signal_toolkit_limits.py",
            "tests/unit/shared_python/test_signal_toolkit_noise.py",
            "tests/unit/shared_python/test_signal_toolkit_series.py",
            "tests/unit/dbc/test_dbc_runtime_calculus.py",
            "tests/unit/dbc/test_dbc_runtime_signal_toolkit.py",
        ),
        modules=("src.shared.python.signal_toolkit.core",),
    ),
    OptionalCollectionRule(
        path_suffixes=(
            "tests/unit/data_io/test_data_processor.py",
            "tests/unit/data_io/test_dataset_generator.py",
            "tests/unit/test_dataset_generator.py",
        ),
        symbols=(
            ("src.shared.python.data_processing.processor", "DatasetInfo"),
            ("src.shared.python.data_io.dataset_generator", "SimulationSample"),
        ),
    ),
    OptionalCollectionRule(
        path_suffixes=("tests/unit/test_c3d_export_features.py",),
        modules=("c3d_reader",),
    ),
    OptionalCollectionRule(
        path_suffixes=("tests/unit/test_setup_golf_suite.py",),
        modules=("setup_golf_suite",),
    ),
    OptionalCollectionRule(
        path_suffixes=("tests/unit/test_start_api_server.py",),
        modules=("start_api_server",),
    ),
)
_OPTIONAL_COLLECTION_WARNED_PATHS: set[str] = set()


def _normalized_collection_path(path: object) -> str:
    return Path(str(path)).as_posix().lower()


def _matches_collection_suffix(path_text: str, suffix: str) -> bool:
    normalized_suffix = suffix.lower().strip("/")
    return (
        path_text == normalized_suffix
        or path_text.startswith(f"{normalized_suffix}/")
        or path_text.endswith(f"/{normalized_suffix}")
        or f"/{normalized_suffix}/" in path_text
    )


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _symbol_available(module_name: str, symbol_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return False
    return hasattr(module, symbol_name)


def _rule_requirement_missing(rule: OptionalCollectionRule) -> bool:
    missing_module = any(not _module_available(module) for module in rule.modules)
    missing_symbol = any(
        not _symbol_available(module, symbol) for module, symbol in rule.symbols
    )
    return missing_module or missing_symbol


def _rule_missing_requirements(
    rule: OptionalCollectionRule,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing_modules = tuple(
        module for module in rule.modules if not _module_available(module)
    )
    missing_symbols = tuple(
        f"{module}.{symbol}"
        for module, symbol in rule.symbols
        if not _symbol_available(module, symbol)
    )
    return missing_modules, missing_symbols


def _should_ignore_optional_collection_path(path: object) -> bool:
    path_text = _normalized_collection_path(path)
    for rule in _OPTIONAL_COLLECTION_RULES:
        if any(
            _matches_collection_suffix(path_text, suffix)
            for suffix in rule.path_suffixes
        ):
            return _rule_requirement_missing(rule)
    return False


def _warn_optional_collection_skip(path: object) -> None:
    path_text = _normalized_collection_path(path)
    if path_text in _OPTIONAL_COLLECTION_WARNED_PATHS:
        return

    for rule in _OPTIONAL_COLLECTION_RULES:
        if not any(
            _matches_collection_suffix(path_text, suffix)
            for suffix in rule.path_suffixes
        ):
            continue

        missing_modules, missing_symbols = _rule_missing_requirements(rule)
        missing_parts = [*missing_modules, *missing_symbols]
        if not missing_parts:
            return

        warnings.warn(
            pytest.PytestWarning(
                "Skipping optional test collection for "
                f"{path_text} because required optional imports are missing: "
                + ", ".join(missing_parts)
            ),
            stacklevel=2,
        )
        _OPTIONAL_COLLECTION_WARNED_PATHS.add(path_text)
        return


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Do not collect tests for optional stacks that are absent in this checkout."""
    should_ignore = _should_ignore_optional_collection_path(collection_path)
    if should_ignore:
        _warn_optional_collection_skip(collection_path)
    return should_ignore


_BIOMECH_SIBLINGS_DIRECT = (
    "MuJoCo_Models",
    "Drake_Models",
    "Pinocchio_Models",
    "OpenSim_Models",
    "Movement-Optimizer",
)


def _default_biomech_mode() -> str:
    """Return ``editable`` if any sibling checkout exists, else ``vendored``."""
    repo_root = Path(__file__).resolve().parent.parent
    workspace_root = repo_root.parent
    for repo_name in _BIOMECH_SIBLINGS_DIRECT:
        if (workspace_root / repo_name).is_dir():
            return "editable"
    return "vendored"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options for Tools vendoring resolution."""
    parser.addoption(
        "--tools-mode",
        action="store",
        default="local",
        choices=["local", "vendored"],
        help="Tools resolution mode: 'local' (src/shared/python) or 'vendored' (vendor/ud-tools/src/shared/python)",
    )
    parser.addoption(
        "--biomech-mode",
        action="store",
        default=None,
        choices=["editable", "vendored", "env"],
        help=(
            "Biomech sibling-repo resolution mode. Defaults to 'editable' if "
            "any sibling checkout exists at ../<RepoName>/, else 'vendored'."
        ),
    )


@pytest.fixture(scope="session")
def biomech_mode(request: pytest.FixtureRequest) -> str:
    """Expose the active ``--biomech-mode`` value to tests."""
    explicit = request.config.getoption("--biomech-mode")
    if explicit is not None:
        return str(explicit)
    return _default_biomech_mode()


def pytest_configure(config: pytest.Config) -> None:
    """Dynamically adjust system path based on selected Tools mode."""
    mode = config.getoption("--tools-mode")
    root_dir = Path(__file__).resolve().parent.parent
    local_path = str((root_dir / "src/shared/python").resolve())
    vendored_path = str((root_dir / "vendor/ud-tools/src/shared/python").resolve())

    # Only process if directories actually exist
    if not os.path.exists(local_path) or not os.path.exists(vendored_path):
        return

    # Clean existing occurrences to enforce determinism (case-insensitive on Windows)
    clean_path = []
    for p in sys.path:
        try:
            resolved_p = str(Path(p).resolve()).lower()
            if resolved_p not in (local_path.lower(), vendored_path.lower()):
                clean_path.append(p)
        except Exception as e:  # noqa: BLE001, F841
            clean_path.append(p)
    sys.path = clean_path

    if mode == "vendored":
        # Force vendored tools to have precedence
        sys.path.insert(0, vendored_path)
        sys.path.append(local_path)
    else:
        # Force local shared codebase to have precedence
        sys.path.insert(0, local_path)
        sys.path.append(vendored_path)

    # Prevent dual-loading of shared contracts and training modules under different path aliases.
    # With both '.' and 'src/shared/python' in sys.path, contracts/training can be
    # imported as both 'src.shared.python.contracts'/'src.shared.python.training' and
    # 'contracts'/'training', creating two distinct class objects that break pytest and type checks.
    # Pre-load via the canonical path and alias all alternate module names.
    try:
        import importlib

        canonical_name = "src.shared.python.contracts"
        canonical_mod = importlib.import_module(canonical_name)
        # Always override — even if already present — to ensure a single class identity.
        # xdist workers may have loaded 'contracts' via the short sys.path entry before
        # pytest_configure runs, creating a stale second module instance.
        for alias in ("contracts", "shared.python.contracts"):
            sys.modules[alias] = canonical_mod

        # Alias training and all of its submodules recursively
        training_dir = root_dir / "src/shared/python/training"
        if training_dir.exists():
            canonical_tr_name = "src.shared.python.training"
            canonical_tr_mod = importlib.import_module(canonical_tr_name)
            sys.modules["training"] = canonical_tr_mod
            sys.modules["shared.python.training"] = canonical_tr_mod

            for path in training_dir.rglob("*.py"):
                if path.name == "__init__.py":
                    if path.parent == training_dir:
                        continue
                    rel = path.parent.relative_to(training_dir)
                else:
                    rel = path.with_suffix("").relative_to(training_dir)

                sub_path = str(rel).replace(os.path.sep, ".")
                if sub_path:
                    sub_name = f"src.shared.python.training.{sub_path}"
                    mod = importlib.import_module(sub_name)
                    sys.modules[f"training.{sub_path}"] = mod
                    sys.modules[f"shared.python.training.{sub_path}"] = mod
    except Exception as e:  # noqa: BLE001, F841
        pass  # Don't block test collection if this fails


# Engine module prefixes whose sys.modules entries must be isolated between
# tests.  Pinocchio's C extension (pinocchio_pywrap_default) is corrupted by
# PinocchioProbe.probe(); Drake gets replaced with MagicMock objects by tests
# that mock pydrake, causing downstream TypeError comparisons. Drake engine
# modules also get polluted when imported with different paths (src.engines.*
# vs engines.*), breaking test_drake_wrapper.py.
_PROTECTED_PREFIXES = (
    "pinocchio",
    "pydrake",
    "src.engines",
)


def _matches_protected(name: str) -> bool:
    """Return True if *name* is a protected engine module."""
    for prefix in _PROTECTED_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False


@pytest.fixture(autouse=True)
def _protect_engine_modules() -> Generator[None, None, None]:
    """Prevent engine module state corruption from leaking between tests.

    Several tests instantiate ``EngineManager`` or ``UpstreamDriftLauncher`` which
    trigger engine probes that import pinocchio/drake.  The probes can corrupt
    C extension module state or leave MagicMock objects in ``sys.modules``.
    Subsequent tests then fail with ``NameError`` or ``TypeError``.

    This fixture snapshots all engine-related ``sys.modules`` entries before
    each test and restores them afterward so that corruption cannot leak
    across test boundaries.
    """
    protected_keys = {k for k in sys.modules if _matches_protected(k)}
    saved = {k: sys.modules[k] for k in protected_keys}
    yield
    # Remove any engine modules added or mutated during the test
    for k in list(sys.modules):  # list() needed: mutating dict during iteration
        if _matches_protected(k):
            if k in saved:
                sys.modules[k] = saved[k]
            else:
                del sys.modules[k]
    # Restore any that were removed during the test
    for k, v in saved.items():
        if k not in sys.modules:
            sys.modules[k] = v


@pytest.fixture
def pendulum_urdf(tmp_path: Path) -> str:
    """Create a standardized simple pendulum URDF for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="pendulum">
  <link name="world"/>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
</robot>
"""
    urdf_path = tmp_path / "pendulum.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)


@pytest.fixture
def clean_pendulum_dynamics() -> Callable[..., Any]:
    """Fixture to provide standardized DoublePendulumDynamics setup for unit tests."""
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DoublePendulumDynamics,
        DoublePendulumParameters,
        LowerSegmentProperties,
        SegmentProperties,
    )

    def _create(m1_kg: float = 1.0, l1_m: float = 1.0) -> Any:
        assert m1_kg is not None, "m1_kg must be provided"
        assert l1_m is not None, "l1_m must be provided"
        assert m1_kg > 0.0, "m1_kg must be positive"
        assert l1_m > 0.0, "l1_m must be positive"
        upper_segment = SegmentProperties(
            length_m=l1_m,
            mass_kg=m1_kg,
            center_of_mass_ratio=1.0,
            inertia_about_com=0.0,
        )
        # Quasi-massless link 2
        epsilon_kg = 1e-10
        lower_segment = LowerSegmentProperties(
            length_m=1.0,
            shaft_mass_kg=epsilon_kg,
            clubhead_mass_kg=epsilon_kg,
            shaft_com_ratio=0.5,
        )
        params = DoublePendulumParameters(
            upper_segment=upper_segment,
            lower_segment=lower_segment,
            plane_inclination_deg=0.0,
            damping_shoulder=0.0,
            damping_wrist=0.0,
            gravity_enabled=True,
            constrained_to_plane=True,
        )
        return DoublePendulumDynamics(parameters=params)

    return _create


# Mock classes that need to be defined before importing the engine
class MockPhysicsEngine:
    pass


@pytest.fixture
def mock_drake_dependencies() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Fixture to mock pydrake and interfaces safely.

    This fixture mocks pydrake modules to allow testing Drake integration
    without having Drake installed.
    """
    mock_pydrake = MagicMock()
    mock_interfaces = MagicMock(PhysicsEngine=MockPhysicsEngine)

    with patch.dict(
        "sys.modules",
        {
            "pydrake": mock_pydrake,
            "pydrake.geometry": MagicMock(),
            "pydrake.math": MagicMock(),
            "pydrake.multibody": MagicMock(),
            "pydrake.multibody.plant": MagicMock(),
            "pydrake.multibody.parsing": MagicMock(),
            "pydrake.multibody.tree": MagicMock(),
            "pydrake.systems": MagicMock(),
            "pydrake.systems.framework": MagicMock(),
            "pydrake.systems.analysis": MagicMock(),
            "pydrake.all": MagicMock(),
            "shared.python.interfaces": mock_interfaces,
        },
    ):
        yield mock_pydrake, mock_interfaces


@pytest.fixture(scope="module")
def mock_mujoco_dependencies() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Fixture to mock mujoco and interfaces safely.

    This fixture mocks mujoco modules to allow testing MuJoCo integration
    without having MuJoCo installed.
    """
    mock_mujoco = MagicMock()
    mock_interfaces = MagicMock(PhysicsEngine=MockPhysicsEngine)

    # Create common MuJoCo structure mocks
    # These are needed for attribute access in many tests
    mock_model = MagicMock()
    mock_model.nv = 2
    mock_model.nu = 2
    mock_model.nq = 2
    mock_model.nbody = 2

    mock_data = MagicMock()
    mock_data.qpos = MagicMock()
    mock_data.qvel = MagicMock()
    mock_data.qacc = MagicMock()
    mock_data.ctrl = MagicMock()

    mock_mujoco.MjModel.return_value = mock_model
    mock_mujoco.MjData.return_value = mock_data

    with patch.dict(
        "sys.modules",
        {
            "mujoco": mock_mujoco,
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.interfaces": mock_interfaces,
        },
    ):
        yield mock_mujoco, mock_interfaces
