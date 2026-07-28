"""Shared fixtures and utilities for the Golf Modeling Suite test suite.

This module centralizes common setup logic to improve test orthogonality
and adherence to the DRY principle.
"""

from __future__ import annotations

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
# ---------------------------------------------------------------------------
# Patch broken transitive imports before any test module is collected.
# Other agents are refactoring src.shared.python.data_io and
# src.shared.python.config, which temporarily removes symbols that
# __init__.py still re-exports.  Inject stub packages so deeper imports
# (LauncherManifest, ModelHandlerRegistry, etc.) can succeed.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
import importlib as _early_importlib
from pathlib import Path as _Path

_tools_path = str(
    (
        _Path(__file__).resolve().parents[1]
        / "vendor"
        / "ud-tools"
        / "src"
        / "shared"
        / "python"
    ).resolve()
)
if _tools_path not in _sys.path:
    _sys.path.insert(0, _tools_path)

for _pkg in ("chat", "sidekick", "ai"):
    _pkg_mod = _sys.modules.get(_pkg)
    _v_path = str(_Path(_tools_path) / _pkg)
    if (
        _pkg_mod is not None
        and hasattr(_pkg_mod, "__path__")
        and _v_path not in _pkg_mod.__path__
    ):
        _pkg_mod.__path__.append(_v_path)


def _ensure_importable_package(module_name: str, package_path: str) -> None:
    try:
        _early_importlib.import_module(module_name)
    except (AttributeError, ImportError):
        module = _types.ModuleType(module_name)
        module.__path__ = [package_path]
        module.__package__ = module_name
        _sys.modules[module_name] = module


_data_io_name = "src.shared.python.data_io"
_ensure_importable_package(_data_io_name, "src/shared/python/data_io")

_config_name = "src.shared.python.config"
_ensure_importable_package(_config_name, "src/shared/python/config")

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
    _pyqt6_qtcore = importlib.import_module("PyQt6.QtCore")
    _pyqt6_qtgui = importlib.import_module("PyQt6.QtGui")
    _pyqt6_qtwidgets = importlib.import_module("PyQt6.QtWidgets")
    _has_pyqt6 = all(
        module is not None for module in (_pyqt6_qtcore, _pyqt6_qtgui, _pyqt6_qtwidgets)
    )
except (AttributeError, ImportError):
    _has_pyqt6 = False

if not _has_pyqt6:
    for module_name in tuple(sys.modules):
        if module_name == "PyQt6" or module_name.startswith("PyQt6."):
            sys.modules.pop(module_name, None)

    class DummySignal:
        def __init__(self, *args, **kwargs):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args, **kwargs):
            for callback in list(self._callbacks):
                callback(*args, **kwargs)

    class DummyQObject:
        def __init__(self, *args, **kwargs):
            super().__init__()

    class DummyQSettings:
        _values: dict[tuple[str, str, str], object] = {}

        def __init__(self, organization: str = "", application: str = ""):
            self._organization = organization
            self._application = application

        def value(self, key: str, defaultValue=None, type=None):
            value = self._values.get(
                (self._organization, self._application, key), defaultValue
            )
            if type is not None and value is not None:
                try:
                    return type(value)
                except (TypeError, ValueError):
                    return defaultValue
            return value

        def setValue(self, key: str, value):
            self._values[(self._organization, self._application, key)] = value

    class DummyQStandardPaths:
        class StandardLocation:
            AppConfigLocation = 0

        @staticmethod
        def writableLocation(_location):
            return str(Path.cwd())

    mock_core = MagicMock()
    mock_core.QObject = DummyQObject
    mock_core.QSettings = DummyQSettings
    mock_core.QStandardPaths = DummyQStandardPaths
    mock_core.pyqtSignal = DummySignal
    mock_core.QLibraryInfo.version.return_value.toString.return_value = "6.6.0"
    mock_core.QLibraryInfo.version.return_value.segments.return_value = (6, 6, 0)
    mock_core.PYQT_VERSION_STR = "6.6.0"
    mock_core.PYQT_VERSION = 0x060600
    mock_core.__version__ = "6.6.0"
    mock_core.qVersion.return_value = "6.6.0"

    pyqt_mock = MagicMock()
    pyqt_mock.__ud_fake__ = True
    pyqt_mock.QtCore = mock_core
    pyqt_mock.QtGui = MagicMock()

    class DummyWidget:
        def __init__(self, *args, **kwargs):
            self.__dict__["_mocks"] = {}

        def __getattr__(self, name):
            if name not in self.__dict__["_mocks"]:
                self.__dict__["_mocks"][name] = MagicMock()
            return self.__dict__["_mocks"][name]

        @classmethod
        def instance(cls):
            return MagicMock()

    DummyWidget.Shape = MagicMock()  # type: ignore[attr-defined]
    DummyWidget.ToolButtonPopupMode = MagicMock()  # type: ignore[attr-defined]
    DummyWidget.setTabOrder = MagicMock()  # type: ignore[attr-defined]
    DummyWidget.DockWidgetFeature = MagicMock()  # type: ignore[attr-defined]

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
    "tests/unit/calc_backend",
    "tests/unit/test_calc_backend_protocols.py",
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
    # NOTE (#8006): rules for `c3d_reader`, `setup_golf_suite` and
    # `start_api_server` were removed here. None of those names was importable in
    # ANY supported configuration, so the rules were not tolerating an optional
    # stack -- they were permanently deleting 20 tests with no skip entry and no
    # CI signal. Do not add a rule for a module that cannot be imported by some
    # documented extra; `tests/unit/test_optional_collection_rules.py` enforces
    # this.
)
_OPTIONAL_COLLECTION_WARNED_PATHS: set[str] = set()
_FAKE_PYQT6_GUI_TESTS = (
    "tests/launchers",
    "tests/shared/wave7_python_core/test_theme_typography.py",
    "tests/imports/test_gui_import_boundaries.py",
    "tests/unit/launcher",
    "tests/unit/launchers",
    "tests/unit/shared_python/test_advanced_analysis_features.py",
    "tests/unit/shared_python/test_dashboard_advanced_analysis.py",
    "tests/unit/shared_python/test_launcher_integration.py",
    "tests/unit/shared_python/test_openpose_gui_coverage.py",
    "tests/unit/shared_python/test_simulation_gui_base.py",
    "tests/unit/theme",
    "tests/unit/tools/starting_pose_matcher",
    "tests/unit/ui",
    "tests/ui",
)


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


def _fake_pyqt6_active() -> bool:
    pyqt6_module = sys.modules.get("PyQt6")
    return not _has_pyqt6 and bool(getattr(pyqt6_module, "__ud_fake__", False))


def _is_fake_pyqt6_gui_path(path: object) -> bool:
    path_text = _normalized_collection_path(path)
    return any(
        _matches_collection_suffix(path_text, suffix)
        for suffix in _FAKE_PYQT6_GUI_TESTS
    )


def _skip_fake_pyqt6_gui_items(items: list[pytest.Item]) -> None:
    if not _fake_pyqt6_active():
        return

    skip_marker = pytest.mark.skip(
        reason="real PyQt6 is unavailable; GUI tests must not pass against stubs"
    )
    for item in items:
        item_path = getattr(item, "path", getattr(item, "fspath", ""))
        if _is_fake_pyqt6_gui_path(item_path):
            item.add_marker(skip_marker)


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
    explicit_tools = os.environ.get("TOOLS_REPO_PATH")
    tools_root = Path(explicit_tools or root_dir / "vendor/ud-tools").resolve()
    parent_paths = [
        str((tools_root / "src/shared/python").resolve()),
        str((tools_root / "src").resolve()),
        str((tools_root / "src/python/src").resolve()),
    ]
    vendored_path = str((root_dir / "vendor/ud-tools/src/shared/python").resolve())

    # Only process if directories actually exist
    if not os.path.exists(local_path) or not all(
        os.path.exists(path) for path in parent_paths
    ):
        return

    # Clean existing occurrences to enforce determinism (case-insensitive on Windows)
    controlled_paths = {
        local_path.casefold(),
        vendored_path.casefold(),
        *(path.casefold() for path in parent_paths),
    }
    clean_path = []
    for p in sys.path:
        try:
            resolved_p = str(Path(p).resolve()).casefold()
            if resolved_p not in controlled_paths:
                clean_path.append(p)
        except Exception as e:  # noqa: BLE001, F841
            clean_path.append(p)
    sys.path = clean_path

    parent_mode = explicit_tools is not None or mode == "vendored"
    if parent_mode:
        for path in reversed(parent_paths):
            sys.path.insert(0, path)
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

        canonical_name = (
            "shared.python.contracts" if parent_mode else "src.shared.python.contracts"
        )
        if parent_mode:
            sys.modules.pop("shared.python.contracts", None)
        canonical_mod = importlib.import_module(canonical_name)
        # Always override — even if already present — to ensure a single class identity.
        # xdist workers may have loaded 'contracts' via the short sys.path entry before
        # pytest_configure runs, creating a stale second module instance.
        contract_aliases = (
            ("contracts",) if parent_mode else ("contracts", "shared.python.contracts")
        )
        for alias in contract_aliases:
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


@pytest.fixture
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


# ---------------------------------------------------------------------------
# Suite-marker enforcement (issue #7158, defect 2)
# ---------------------------------------------------------------------------
#
# Goal: every test should declare which suite it belongs to via one of the
# recognized "suite" markers below.  Unmarked tests run by default and nothing
# requires a suite marker, which lets tests drift out of every CI lane's
# selection expression.
#
# This hook runs in REPORT-ONLY mode first (the ratchet pattern used elsewhere
# in this repo): it counts collected tests that carry none of the suite markers
# and surfaces the count. CI can reject net-new unmarked tests with
# ``UD_RATCHET_SUITE_MARKERS=1`` and the committed baseline; once the baseline
# reaches zero, ``UD_ENFORCE_SUITE_MARKERS=1`` makes any missing marker a
# collection error.
#
# Acceptance-criteria mapping: "Add a pytest_collection_modifyitems hook ...
# that fails collection when a test has none of {unit, integration, e2e,
# slow, ...} — start in report-only mode with a baseline count, then flip to
# enforcing."
#
# The classification logic lives in tests/support/suite_markers.py so it can be
# unit-tested in isolation (see tests/unit/test_suite_marker_enforcement_7158).
from tests.support.suite_markers import (  # noqa: E402
    SUITE_MARKERS,
    find_unmarked,
    find_unmarked_baseline_drift,
    load_baseline_nodeids,
    suite_marker_ratchet_enabled,
    suite_markers_enforced,
)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Report (or, when enforced, fail on) tests lacking a suite marker.

    Postcondition: stores the unmarked count on ``config`` as
    ``_ud_unmarked_suite_count`` for the terminal summary; raises
    ``pytest.UsageError`` only when enforcement is enabled.
    """
    _skip_fake_pyqt6_gui_items(items)

    unmarked = find_unmarked(items)  # type: ignore[arg-type]
    config._ud_unmarked_suite_count = len(unmarked)  # type: ignore[attr-defined]
    config._ud_unmarked_suite_nodeids = [  # type: ignore[attr-defined]
        item.nodeid for item in unmarked
    ]

    if unmarked and suite_markers_enforced():
        listing = "\n".join(f"  - {item.nodeid}" for item in unmarked)
        raise pytest.UsageError(
            f"{len(unmarked)} test(s) carry none of the required suite markers "
            f"{sorted(SUITE_MARKERS)}:\n{listing}"
        )

    if unmarked and suite_marker_ratchet_enabled():
        baseline = load_baseline_nodeids()
        drift = find_unmarked_baseline_drift(unmarked, baseline)
        config._ud_unmarked_suite_drift_count = len(drift)  # type: ignore[attr-defined]
        if drift:
            listing = "\n".join(f"  - {item.nodeid}" for item in drift)
            raise pytest.UsageError(
                f"{len(drift)} net-new test(s) carry none of the required suite "
                f"markers {sorted(SUITE_MARKERS)} and are absent from "
                "scripts/config/suite_marker_baseline.json:\n"
                f"{listing}"
            )


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    """Surface the unmarked-suite-marker count in the terminal summary."""
    count = getattr(config, "_ud_unmarked_suite_count", None)
    if not count:
        return
    if suite_markers_enforced():
        mode = "ENFORCED"
    elif suite_marker_ratchet_enabled():
        drift_count = getattr(config, "_ud_unmarked_suite_drift_count", 0)
        mode = f"ratchet, drift={drift_count}"
    else:
        mode = "report-only"
    terminalreporter.write_line(
        f"[suite-markers:{mode}] {count} collected test(s) carry no suite marker "
        f"(one of {sorted(SUITE_MARKERS)}); see issue #7158.",
        yellow=True,
    )


# ---------------------------------------------------------------------------
# Rust-wheel parity enforcement (issue #7601)
#
# Several parity / binding suites guard themselves with
# ``pytest.mark.skipif(not is_rust_available())`` or
# ``pytest.importorskip("upstream_...")`` so a clean checkout (no maturin
# build) stays green. The downside is that the main Test lane silently SKIPS
# those tests instead of exercising the Rust kernels, so a parity regression
# can land unnoticed.
#
# The dedicated ``rust-wheel-parity`` CI job builds + installs all six PyO3
# wheels and runs the parity suite with ``CI_RUST_WHEELS_EXPECTED=1``. In that
# lane a missing wheel (and therefore a skipped parity test) is a hard error:
# the wheel is *expected* to be present, so a skip means the build/install
# regressed. This hook converts such skips into failures.
#
# Note: this lives in the test layer on purpose. The Python facades
# (``rust_kernel`` et al.) keep their graceful pure-Python fallback untouched.
# ---------------------------------------------------------------------------

# Python extension modules produced by the maturin wheel build. A skip whose
# reason names one of these (or the generic "Rust kernel not available"
# message) is treated as a missing-wheel regression when wheels are expected.
_RUST_WHEEL_MODULES: tuple[str, ...] = (
    "upstream_physics",
    "upstream_mocap_preproc",
    "upstream_mocap_io",
    "upstream_muscle",
    "upstream_motion_matching",
    "ai_backend",
)


def _rust_wheels_expected() -> bool:
    return os.environ.get("CI_RUST_WHEELS_EXPECTED") == "1"


def _skip_reason_is_missing_rust_wheel(reason: str) -> bool:
    """Return True if a skip reason indicates an unavailable Rust wheel."""
    lowered = reason.lower()
    if any(module in lowered for module in _RUST_WHEEL_MODULES):
        return True
    # ``is_rust_available()`` skipif messages and importorskip phrasing.
    return "rust kernel not available" in lowered or (
        "rust" in lowered and "not available" in lowered
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    """Fail (instead of skip) Rust-parity tests when wheels are expected.

    Precondition: only active when ``CI_RUST_WHEELS_EXPECTED=1``; otherwise the
    normal skip behaviour is preserved so clean checkouts stay green.
    Postcondition: a skip caused by a missing Rust wheel becomes a failure so
    the wheel-installing CI lane cannot pass while silently skipping parity.
    """
    outcome = yield
    if not _rust_wheels_expected():
        return

    report = outcome.get_result()
    if not report.skipped:
        return

    longrepr = report.longrepr
    reason = ""
    # Skipped longrepr is typically a (path, lineno, "Skipped: <reason>") tuple.
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    else:
        reason = str(longrepr)

    if _skip_reason_is_missing_rust_wheel(reason):
        report.outcome = "failed"
        report.longrepr = (
            f"CI_RUST_WHEELS_EXPECTED=1 but {item.nodeid} skipped because a Rust "
            f"wheel is unavailable ({reason!r}). In the rust-wheel-parity lane "
            "all six PyO3 wheels must be importable so parity actually runs. "
            "This skip indicates the maturin build or wheel install regressed. "
            "See issue #7601."
        )
