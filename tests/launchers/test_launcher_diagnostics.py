"""Tests for launcher_diagnostics."""

import json  # noqa: E402
from unittest.mock import MagicMock, mock_open, patch  # noqa: E402

import yaml  # noqa: E402

from src.launchers.launcher_diagnostics import (  # noqa: E402
    DiagnosticResult,
    LauncherDiagnostics,
    reset_layout_config,
    run_cli_diagnostics,
)


def test_diagnostic_result_to_dict():
    result = DiagnosticResult(
        name="test", status="pass", message="ok", details={"a": 1}, duration_ms=10.123
    )
    d = result.to_dict()
    assert d["name"] == "test"
    assert d["status"] == "pass"
    assert d["message"] == "ok"
    assert d["details"] == {"a": 1}
    assert d["duration_ms"] == 10.12


@patch.object(LauncherDiagnostics, "check_python_environment")
@patch.object(LauncherDiagnostics, "check_models_yaml")
@patch.object(LauncherDiagnostics, "check_model_registry")
@patch.object(LauncherDiagnostics, "check_layout_config")
@patch.object(LauncherDiagnostics, "check_asset_files")
@patch.object(LauncherDiagnostics, "check_pyqt6_availability")
@patch.object(LauncherDiagnostics, "check_engine_availability")
def test_run_all_checks(
    mock_engine, mock_qt, mock_assets, mock_layout, mock_registry, mock_yaml, mock_env
):
    diag = LauncherDiagnostics()

    # Mock some results
    res1 = DiagnosticResult("test1", "pass", "msg1")
    res2 = DiagnosticResult("test2", "fail", "msg2")
    res3 = DiagnosticResult("test3", "warning", "msg3")

    def add_results(*args, **kwargs):
        diag.results.extend([res1, res2, res3])

    mock_env.side_effect = add_results

    report = diag.run_all_checks()

    assert report["summary"]["total_checks"] == 3
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 1
    assert report["summary"]["warnings"] == 1
    assert report["summary"]["status"] == "degraded"
    assert len(report["checks"]) == 3
    assert "recommendations" in report


def test_check_python_environment():
    diag = LauncherDiagnostics()
    res = diag.check_python_environment()
    assert res.name == "python_environment"
    assert res.status == "pass"
    assert "platform" in res.details


@patch("pathlib.Path.exists")
def test_check_models_yaml_missing(mock_exists):
    mock_exists.return_value = False
    diag = LauncherDiagnostics()
    res = diag.check_models_yaml()

    assert res.name == "models_yaml"
    assert res.status == "fail"
    assert "not found" in res.message


@patch("pathlib.Path.exists", return_value=True)
def test_check_models_yaml_valid(mock_exists):
    diag = LauncherDiagnostics()

    valid_data = {
        "models": [{"id": id} for id in LauncherDiagnostics.EXPECTED_TILE_IDS]
    }

    with patch("builtins.open", mock_open(read_data=yaml.dump(valid_data))):
        res = diag.check_models_yaml()

    assert res.name == "models_yaml"
    assert res.status == "pass"


@patch("pathlib.Path.exists", return_value=True)
def test_check_models_yaml_empty(mock_exists):
    diag = LauncherDiagnostics()
    with patch("builtins.open", mock_open(read_data="")):
        res = diag.check_models_yaml()

    assert res.name == "models_yaml"
    assert res.status == "fail"
    assert "empty" in res.message


@patch("pathlib.Path.exists", return_value=True)
def test_check_models_yaml_missing_models_key(mock_exists):
    diag = LauncherDiagnostics()
    with patch("builtins.open", mock_open(read_data="foo: bar")):
        res = diag.check_models_yaml()

    assert res.name == "models_yaml"
    assert res.status == "fail"
    assert "missing 'models'" in res.message


@patch("pathlib.Path.exists", return_value=True)
def test_check_models_yaml_incomplete(mock_exists):
    diag = LauncherDiagnostics()

    valid_data = {"models": [{"id": "mujoco_unified"}]}

    with patch("builtins.open", mock_open(read_data=yaml.dump(valid_data))):
        res = diag.check_models_yaml()

    assert res.name == "models_yaml"
    assert res.status == "fail"
    assert "Missing" in res.message


@patch("src.shared.python.config.model_registry.ModelRegistry")
def test_check_model_registry_success(mock_registry_class):
    diag = LauncherDiagnostics()
    mock_registry = MagicMock()
    mock_registry_class.return_value = mock_registry

    # Mock getting all expected models
    mock_models = [
        MagicMock(id=id, name="Test") for id in LauncherDiagnostics.EXPECTED_TILE_IDS
    ]
    mock_registry.get_all_models.return_value = mock_models

    res = diag.check_model_registry()
    assert res.status == "pass"


@patch("src.shared.python.config.model_registry.ModelRegistry")
def test_check_model_registry_missing(mock_registry_class):
    diag = LauncherDiagnostics()
    mock_registry = MagicMock()
    mock_registry_class.return_value = mock_registry

    # Mock returning fewer models
    mock_models = [MagicMock(id="mujoco_unified", name="Test")]
    mock_registry.get_all_models.return_value = mock_models

    res = diag.check_model_registry()
    assert res.status == "fail"
    assert "missing" in res.message


@patch("pathlib.Path.exists")
def test_check_layout_config_missing(mock_exists):
    # Mock first exists (CONFIG_DIR) and second/third (LAYOUT_CONFIG_FILE)
    mock_exists.side_effect = [True, False, False]
    diag = LauncherDiagnostics()
    res = diag.check_layout_config()
    assert res.status == "pass"
    assert "No saved layout" in res.message


@patch("pathlib.Path.exists", return_value=True)
def test_check_layout_config_json_error(mock_exists):
    diag = LauncherDiagnostics()
    with patch("builtins.open", mock_open(read_data="{bad json")):
        res = diag.check_layout_config()
        assert res.status == "warning"


@patch("pathlib.Path.exists", return_value=True)
def test_check_layout_config_success(mock_exists):
    diag = LauncherDiagnostics()
    layout_data = {"model_order": LauncherDiagnostics.EXPECTED_TILE_IDS}
    with patch("builtins.open", mock_open(read_data=json.dumps(layout_data))):
        res = diag.check_layout_config()
        assert res.status == "pass"


@patch("pathlib.Path.exists", return_value=True)
def test_check_layout_config_incomplete(mock_exists):
    diag = LauncherDiagnostics()
    layout_data = {"model_order": ["mujoco_unified"]}
    with patch("builtins.open", mock_open(read_data=json.dumps(layout_data))):
        res = diag.check_layout_config()
        assert res.status == "warning"
        assert "missing" in res.message


@patch("pathlib.Path.exists")
def test_check_asset_files_dir_missing(mock_exists):
    mock_exists.return_value = False
    diag = LauncherDiagnostics()
    res = diag.check_asset_files()
    assert res.status == "fail"


@patch("pathlib.Path.exists", autospec=True)
@patch("pathlib.Path.iterdir")
def test_check_asset_files_success(mock_iterdir, mock_exists):
    def exists_side_effect(self):
        # Only some assets exist
        return "mujoco" not in str(self)

    mock_exists.side_effect = exists_side_effect

    # Mock iterdir to return some files
    mock_file = MagicMock()
    mock_file.is_file.return_value = True
    mock_file.name = "drake.png"
    mock_iterdir.return_value = [mock_file]

    diag = LauncherDiagnostics()
    res = diag.check_asset_files()
    # It will warn because mujoco is missing
    assert res.status == "warning"


@patch("src.shared.python.engine_core.engine_manager.EngineManager")
def test_check_engine_availability_success(mock_manager_class):
    diag = LauncherDiagnostics()
    mock_mgr = MagicMock()
    mock_manager_class.return_value = mock_mgr

    # Mock some engines installed
    import enum

    class MockStatus(enum.Enum):
        AVAILABLE = "available"

    class MockType(enum.Enum):
        MUJOCO = "mujoco"

    mock_probe_result = MagicMock()
    mock_probe_result.status = MockStatus.AVAILABLE
    mock_probe_result.is_available.return_value = True
    mock_probe_result.version = "1.0"
    mock_probe_result.missing_dependencies = []
    mock_probe_result.diagnostic_message = "ok"

    mock_probe = MagicMock()
    mock_probe.probe.return_value = mock_probe_result

    mock_mgr.get_available_engines.return_value = [MockType.MUJOCO]
    mock_mgr.engine_status = {MockType.MUJOCO: MockStatus.AVAILABLE}
    mock_mgr.engine_paths = {MockType.MUJOCO: "/fake"}
    mock_mgr.probes = {MockType.MUJOCO: mock_probe}

    res = diag.check_engine_availability()
    assert res.status == "pass"


@patch("pathlib.Path.exists")
@patch("pathlib.Path.rename")
def test_reset_layout_config(mock_rename, mock_exists):
    mock_exists.return_value = True
    assert reset_layout_config() is True
    mock_rename.assert_called_once()


@patch.object(LauncherDiagnostics, "run_all_checks")
def test_run_cli_diagnostics(mock_run):
    mock_run.return_value = {
        "summary": {"status": "healthy", "passed": 1, "failed": 0, "warnings": 0},
        "checks": [{"name": "check1", "status": "pass", "message": "msg"}],
        "recommendations": ["Do this"],
    }
    run_cli_diagnostics()


@patch("pathlib.Path.exists", return_value=True)
def test_check_models_yaml_yaml_error(mock_exists):
    diag = LauncherDiagnostics()
    with patch("builtins.open", mock_open(read_data="[ : invalid yaml")):
        res = diag.check_models_yaml()
    assert res.status == "fail"
    assert "YAML parsing error" in res.message


@patch(
    "src.shared.python.config.model_registry.ModelRegistry",
    side_effect=ImportError("mock"),
)
def test_check_model_registry_import_error(mock_registry_class):
    diag = LauncherDiagnostics()
    res = diag.check_model_registry()
    assert res.status == "fail"
    assert "Failed to import" in res.message


@patch(
    "src.shared.python.config.model_registry.ModelRegistry",
    side_effect=RuntimeError("mock"),
)
def test_check_model_registry_runtime_error(mock_registry_class):
    diag = LauncherDiagnostics()
    res = diag.check_model_registry()
    assert res.status == "fail"
    assert "ModelRegistry error" in res.message


@patch("pathlib.Path.exists", return_value=True)
def test_check_layout_config_os_error(mock_exists):
    diag = LauncherDiagnostics()
    with patch("builtins.open", side_effect=OSError("denied")):
        res = diag.check_layout_config()
    assert res.status == "warning"
    assert "Error reading layout" in res.message


def test_check_pyqt6_availability_success():
    diag = LauncherDiagnostics()
    with (
        patch("PyQt6.QtCore.PYQT_VERSION_STR", "6.0", create=True),
        patch("PyQt6.QtCore.QT_VERSION_STR", "6.0", create=True),
        patch("PyQt6.QtWidgets.QApplication", create=True),
    ):
        res = diag.check_pyqt6_availability()
    assert res.status == "pass"
    assert "PyQt6 available" in res.message


def test_check_pyqt6_availability_import_error():
    diag = LauncherDiagnostics()
    with patch.dict("sys.modules", {"PyQt6.QtCore": None, "PyQt6.QtWidgets": None}):
        res = diag.check_pyqt6_availability()
    assert res.status == "fail"


@patch("src.shared.python.engine_core.engine_manager.EngineManager")
def test_check_engine_availability_probe_error(mock_manager_class):
    diag = LauncherDiagnostics()
    mock_mgr = MagicMock()
    mock_manager_class.return_value = mock_mgr

    import enum

    class MockStatus(enum.Enum):
        AVAILABLE = "available"

    class MockType(enum.Enum):
        MUJOCO = "mujoco"

    mock_probe = MagicMock()
    mock_probe.probe.side_effect = RuntimeError("probe failed")

    mock_mgr.get_available_engines.return_value = []
    mock_mgr.engine_status = {MockType.MUJOCO: MockStatus.AVAILABLE}
    mock_mgr.engine_paths = {MockType.MUJOCO: "/fake"}
    mock_mgr.probes = {MockType.MUJOCO: mock_probe}

    res = diag.check_engine_availability()
    assert res.status == "warning"


@patch("src.shared.python.engine_core.engine_manager.EngineManager")
def test_check_engine_availability_no_probe(mock_manager_class):
    diag = LauncherDiagnostics()
    mock_mgr = MagicMock()
    mock_manager_class.return_value = mock_mgr

    import enum

    from src.shared.python.engine_core.engine_registry import EngineStatus

    class MockType(enum.Enum):
        MUJOCO = "mujoco"

    # No probe
    mock_mgr.get_available_engines.return_value = [MockType.MUJOCO]
    mock_mgr.engine_status = {MockType.MUJOCO: EngineStatus.AVAILABLE}
    mock_mgr.engine_paths = {MockType.MUJOCO: "/fake"}
    mock_mgr.probes = {}

    res = diag.check_engine_availability()
    assert res.status == "pass"


@patch(
    "src.shared.python.engine_core.engine_manager.EngineManager",
    side_effect=ImportError("mock"),
)
def test_check_engine_availability_import_error(mock_manager_class):
    diag = LauncherDiagnostics()
    res = diag.check_engine_availability()
    assert res.status == "warning"


@patch(
    "src.shared.python.engine_core.engine_manager.EngineManager",
    side_effect=RuntimeError("mock"),
)
def test_check_engine_availability_runtime_error(mock_manager_class):
    diag = LauncherDiagnostics()
    res = diag.check_engine_availability()
    assert res.status == "warning"


def test_generate_recommendations():
    diag = LauncherDiagnostics()
    res1 = DiagnosticResult("models_yaml", "fail", "msg", {})
    res2 = DiagnosticResult("model_registry", "fail", "msg", {})
    res3 = DiagnosticResult("pyqt6_availability", "fail", "msg", {})
    res4 = DiagnosticResult("asset_files", "fail", "msg", {})
    res5 = DiagnosticResult(
        "layout_config", "warning", "msg", {"missing_from_saved": ["drake"]}
    )
    res6 = DiagnosticResult("asset_files", "warning", "msg", {})

    diag.results.extend([res1, res2, res3, res4, res5, res6])
    recs = diag._generate_recommendations()
    assert len(recs) == 6

    # Test healthy branch
    diag.results.clear()
    diag.results.append(DiagnosticResult("test", "pass", "ok"))
    recs = diag._generate_recommendations()
    assert len(recs) == 1
    assert "operational" in recs[0]


@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.rename", side_effect=OSError("mock"))
def test_reset_layout_config_error(mock_rename, mock_exists):
    from src.launchers.launcher_diagnostics import reset_layout_config

    assert reset_layout_config() is False


@patch.object(LauncherDiagnostics, "run_all_checks")
def test_run_cli_diagnostics_failures_and_warnings(mock_run):
    mock_run.return_value = {
        "summary": {"status": "degraded", "passed": 0, "failed": 1, "warnings": 1},
        "checks": [
            {
                "name": "fail_check",
                "status": "fail",
                "message": "fail msg",
                "details": {"missing_expected_ids": ["x"]},
            },
            {
                "name": "warn_check",
                "status": "warning",
                "message": "warn msg",
                "details": {"missing_from_saved": ["y"]},
            },
        ],
        "recommendations": ["Do this", "Do that"],
    }
    run_cli_diagnostics()
