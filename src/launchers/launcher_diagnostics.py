# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Diagnostic utilities for UpstreamDrift GUI Launcher.

This module provides comprehensive diagnostic tools for troubleshooting
launcher issues including:
- Model registry verification
- Tile loading diagnostics
- Asset file verification
- Layout configuration validation
- Engine availability checking
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass

# Constants — use centralized root discovery (issue #2354)
REPOS_ROOT = get_repo_root()
ASSETS_DIR = Path(__file__).parent / "assets"
CONFIG_DIR = Path.home() / ".golf_modeling_suite"
LAYOUT_CONFIG_FILE = CONFIG_DIR / "launcher_layout.json"


def _read_manifest_schema(manifest_path: Path | None) -> str | None:
    """Return the ``schema`` field of a biomech manifest, or ``None``.

    Best-effort: silently returns ``None`` when PyYAML is missing, the file
    cannot be read, or the field is absent. Used by the launcher diagnostics
    to report each sibling's published manifest version.
    """
    if manifest_path is None or not manifest_path.is_file():
        return None
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("schema")
    if isinstance(value, str) and value.strip():
        return value
    return None


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""

    name: str
    status: str  # "pass", "fail", "warning"
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2),
        }


class LauncherDiagnostics:
    """Diagnostic utilities for the UpstreamDrift Launcher."""

    # Expected tile model IDs
    EXPECTED_TILE_IDS = [
        "mujoco_unified",
        "drake_golf",
        "pinocchio_golf",
        "opensim_golf",
        "myosim_suite",
        "putting_green",
        "simscape_2d",
        "simscape_3d",
        "dataset_generator",
        "matlab_analysis",
        "c3d_viewer",
        "openpose_analysis",
        "mediapipe_analysis",
        "model_explorer",
        "video_analyzer",
        "data_explorer",
        "project_map",
    ]

    # Expected tile names
    EXPECTED_TILE_NAMES = {
        "mujoco_unified": "MuJoCo",
        "drake_golf": "Drake",
        "pinocchio_golf": "Pinocchio",
        "opensim_golf": "OpenSim",
        "myosim_suite": "MyoSuite",
        "putting_green": "Putting Green",
        "simscape_2d": "Simscape 2D",
        "simscape_3d": "Simscape 3D",
        "dataset_generator": "Dataset Generator",
        "matlab_analysis": "Analysis GUI",
        "c3d_viewer": "C3D Viewer",
        "openpose_analysis": "OpenPose",
        "mediapipe_analysis": "MediaPipe",
        "model_explorer": "Model Explorer",
        "video_analyzer": "Video Analyzer",
        "data_explorer": "Data Explorer",
        "project_map": "Project Map",
    }

    def __init__(self) -> None:
        """Initialize diagnostics."""
        self.results: list[DiagnosticResult] = []
        self._start_time = time.time()

    @staticmethod
    def _push_to_ring_buffer(results: list[DiagnosticResult]) -> None:
        """Emit diagnostic results into the app-state ring buffer (issue #5474).

                Called at the end of :meth:
        un_all_checks after all checks complete.
                Uses a lazy import so the ring buffer is only required at call-time.
                Best-effort: any import or record error is silently swallowed.
        """
        try:
            from src.shared.python.ai.chat_context import record_event

            for r in results:
                record_event(
                    "diagnostic",
                    {
                        "check": r.name,
                        "status": r.status,
                        "message": r.message,
                        "duration_ms": r.duration_ms,
                    },
                )
        except Exception:  # noqa: BLE001
            pass

    def _record_result(self, result: DiagnosticResult) -> None:
        """Append *result* to ``self.results`` and push it to the ring buffer.

        Best-effort ring-buffer write: any failure from `record_event` is
        silently swallowed so a buffer outage never breaks the diagnostic run.
        """
        self.results.append(result)
        try:
            from src.shared.python.ai.chat_context import record_event

            record_event(
                "diagnostic",
                {
                    "check": result.name,
                    "status": result.status,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                },
            )
        except Exception:  # noqa: BLE001
            pass

    def run_all_checks(self) -> dict[str, Any]:
        """Run all diagnostic checks and return comprehensive report.

        Returns:
            Dictionary containing all diagnostic results and summary
        """
        self.results = []

        # Core checks
        self.check_python_environment()
        self.check_models_yaml()
        self.check_model_registry()
        self.check_launcher_provider_compatibility()
        self.check_layout_config()
        self.check_asset_files()
        self.check_pyqt6_availability()
        self.check_engine_availability()
        self.check_biomech_siblings()
        self.check_tools_sidebar()

        # Feed all results into the app-state ring buffer so the Sidekick
        # chat assistant can include them via get_chat_context() (issue #5474).
        self._push_to_ring_buffer(self.results)
        # Calculate summary
        passed = sum(1 for r in self.results if r.status == "pass")
        failed = sum(1 for r in self.results if r.status == "fail")
        warnings = sum(1 for r in self.results if r.status == "warning")

        return {
            "summary": {
                "total_checks": len(self.results),
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "status": "healthy" if failed == 0 else "degraded",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "expected_tiles": len(self.EXPECTED_TILE_IDS),
            },
            "checks": [r.to_dict() for r in self.results],
            "recommendations": self._generate_recommendations(),
        }

    def check_python_environment(self) -> DiagnosticResult:
        """Check Python environment configuration."""
        start = time.time()
        details: dict[str, Any] = {
            "python_version": sys.version,
            "platform": sys.platform,
            "repos_root": str(REPOS_ROOT),
            "repos_root_exists": REPOS_ROOT.exists(),
            "assets_dir": str(ASSETS_DIR),
            "assets_dir_exists": ASSETS_DIR.exists(),
        }

        result = DiagnosticResult(
            name="python_environment",
            status="pass",
            message="Python environment configured correctly",
            details=details,
            duration_ms=(time.time() - start) * 1000,
        )
        self._record_result(result)
        return result

    def _validate_models_yaml_content(
        self, data: Any, details: dict[str, Any]
    ) -> DiagnosticResult | None:
        if details is None:
            raise ValueError("details must be provided")
        details["raw_content_preview"] = str(data)[:500] if data else "empty"

        if not data:
            return DiagnosticResult(
                name="models_yaml",
                status="fail",
                message="models.yaml is empty",
                details=details,
                duration_ms=0,
            )

        if "models" not in data:
            return DiagnosticResult(
                name="models_yaml",
                status="fail",
                message="models.yaml missing 'models' key",
                details=details,
                duration_ms=0,
            )
        return None

    def _check_models_yaml_completeness(
        self, models: list, details: dict[str, Any]
    ) -> DiagnosticResult:
        if models is None:
            raise ValueError("models must be provided")
        details["model_count"] = len(models)
        details["model_ids"] = [m.get("id", "unknown") for m in models]

        found_ids = set(details["model_ids"])
        expected_ids = set(self.EXPECTED_TILE_IDS)
        missing_ids = expected_ids - found_ids
        extra_ids = found_ids - expected_ids

        details["missing_expected_ids"] = list(missing_ids)
        details["extra_ids"] = list(extra_ids)

        if missing_ids:
            return DiagnosticResult(
                name="models_yaml",
                status="fail",
                message=f"Missing {len(missing_ids)} expected models: {missing_ids}",
                details=details,
                duration_ms=0,
            )
        return DiagnosticResult(
            name="models_yaml",
            status="pass",
            message=f"models.yaml valid with {len(models)} models",
            details=details,
            duration_ms=0,
        )

    def check_models_yaml(self) -> DiagnosticResult:
        """Check models.yaml configuration file."""
        start = time.time()

        models_yaml_path = REPOS_ROOT / "src" / "config" / "models.yaml"
        details: dict[str, Any] = {
            "path": str(models_yaml_path),
            "exists": models_yaml_path.exists(),
        }

        if not models_yaml_path.exists():
            result = DiagnosticResult(
                name="models_yaml",
                status="fail",
                message=f"models.yaml not found at {models_yaml_path}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
            self.results.append(result)
            return result

        try:
            import yaml

            with open(models_yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            early_result = self._validate_models_yaml_content(data, details)
            if early_result is not None:
                early_result.duration_ms = (time.time() - start) * 1000
                self.results.append(early_result)
                return early_result

            result = self._check_models_yaml_completeness(data["models"], details)

        except yaml.YAMLError as e:
            details["yaml_error"] = str(e)
            result = DiagnosticResult(
                name="models_yaml",
                status="fail",
                message=f"YAML parsing error: {e}",
                details=details,
                duration_ms=0,
            )
        except ImportError as e:
            details["error"] = str(e)
            result = DiagnosticResult(
                name="models_yaml",
                status="fail",
                message=f"Error reading models.yaml: {e}",
                details=details,
                duration_ms=0,
            )

        result.duration_ms = (time.time() - start) * 1000
        self.results.append(result)
        return result

    def check_model_registry(self) -> DiagnosticResult:
        """Check ModelRegistry loading."""
        start = time.time()
        details: dict[str, Any] = {}

        try:
            from src.shared.python.config.model_registry import ModelRegistry

            registry_path = REPOS_ROOT / "src" / "config" / "models.yaml"
            registry = ModelRegistry(registry_path)

            all_models = registry.get_all_models()
            details["registry_loaded"] = True
            details["model_count"] = len(all_models)
            details["loaded_model_ids"] = [m.id for m in all_models]
            details["loaded_model_names"] = [m.name for m in all_models]

            # Check for expected models
            loaded_ids = set(details["loaded_model_ids"])
            expected_ids = set(self.EXPECTED_TILE_IDS)
            missing_ids = expected_ids - loaded_ids

            details["missing_from_registry"] = list(missing_ids)

            if missing_ids:
                result = DiagnosticResult(
                    name="model_registry",
                    status="fail",
                    message=f"Registry missing {len(missing_ids)} models: {missing_ids}",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                result = DiagnosticResult(
                    name="model_registry",
                    status="pass",
                    message=f"ModelRegistry loaded {len(all_models)} models successfully",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )

        except ImportError as e:
            details["import_error"] = str(e)
            result = DiagnosticResult(
                name="model_registry",
                status="fail",
                message=f"Failed to import ModelRegistry: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
        except (RuntimeError, TypeError, AttributeError) as e:
            details["error"] = str(e)
            result = DiagnosticResult(
                name="model_registry",
                status="fail",
                message=f"ModelRegistry error: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )

        self.results.append(result)
        return result

    def check_launcher_provider_compatibility(self) -> DiagnosticResult:
        """Check that launcher model entries resolve cleanly as local/provider sources."""
        start = time.time()
        details: dict[str, Any] = {}

        try:
            from src.launchers.launcher_provider_compatibility import (
                evaluate_launcher_model_compatibility,
            )
            from src.shared.python.config.model_registry import ModelRegistry

            registry_path = REPOS_ROOT / "src" / "config" / "models.yaml"
            registry = ModelRegistry(registry_path)
            results = evaluate_launcher_model_compatibility(
                registry.get_all_models(), REPOS_ROOT
            )

            details["model_count"] = len(results)
            details["compatible_model_ids"] = [
                result.model_id for result in results if result.is_compatible
            ]
            details["incompatible_models"] = [
                {
                    "model_id": result.model_id,
                    "provider": result.provider,
                    "issues": list(result.issues),
                }
                for result in results
                if not result.is_compatible
            ]

            if details["incompatible_models"]:
                result = DiagnosticResult(
                    name="launcher_provider_compatibility",
                    status="warning",
                    message=(
                        "Launcher provider compatibility found "
                        f"{len(details['incompatible_models'])} incompatible models"
                    ),
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                result = DiagnosticResult(
                    name="launcher_provider_compatibility",
                    status="pass",
                    message=(
                        "Launcher provider compatibility validated "
                        f"{len(results)} models"
                    ),
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )

        except ImportError as e:
            details["import_error"] = str(e)
            result = DiagnosticResult(
                name="launcher_provider_compatibility",
                status="warning",
                message=f"Launcher provider compatibility unavailable: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            details["error"] = str(e)
            result = DiagnosticResult(
                name="launcher_provider_compatibility",
                status="warning",
                message=f"Launcher provider compatibility error: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )

        self.results.append(result)
        return result

    def check_layout_config(self) -> DiagnosticResult:
        """Check saved layout configuration."""
        start = time.time()
        details: dict[str, Any] = {
            "config_dir": str(CONFIG_DIR),
            "config_dir_exists": CONFIG_DIR.exists(),
            "layout_file": str(LAYOUT_CONFIG_FILE),
            "layout_file_exists": LAYOUT_CONFIG_FILE.exists(),
        }

        if not LAYOUT_CONFIG_FILE.exists():
            result = DiagnosticResult(
                name="layout_config",
                status="pass",
                message="No saved layout (will use defaults with 17 tiles)",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
            self.results.append(result)
            return result

        try:
            with open(LAYOUT_CONFIG_FILE, encoding="utf-8") as f:
                layout_data = json.load(f)

            details["layout_content"] = layout_data
            saved_order = layout_data.get("model_order", [])
            details["saved_model_order"] = saved_order
            details["saved_model_count"] = len(saved_order)

            # Check if saved layout has all expected tiles
            saved_ids = set(saved_order)
            expected_ids = set(self.EXPECTED_TILE_IDS)
            missing_from_saved = expected_ids - saved_ids

            details["missing_from_saved"] = list(missing_from_saved)

            if missing_from_saved:
                result = DiagnosticResult(
                    name="layout_config",
                    status="warning",
                    message=f"Saved layout missing {len(missing_from_saved)} tiles - this may cause only {len(saved_order)} tiles to show",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                result = DiagnosticResult(
                    name="layout_config",
                    status="pass",
                    message=f"Saved layout has {len(saved_order)} tiles",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )

        except json.JSONDecodeError as e:
            details["json_error"] = str(e)
            result = DiagnosticResult(
                name="layout_config",
                status="warning",
                message=f"Invalid layout JSON - will use defaults: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            details["error"] = str(e)
            result = DiagnosticResult(
                name="layout_config",
                status="warning",
                message=f"Error reading layout config: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )

        self.results.append(result)
        return result

    def check_asset_files(self) -> DiagnosticResult:
        """Check that required asset files exist."""
        start = time.time()

        expected_assets = {
            "mujoco_humanoid.png": "MuJoCo tile",
            "drake.png": "Drake tile",
            "pinocchio.png": "Pinocchio tile",
            "opensim.png": "OpenSim tile",
            "myosim.png": "MyoSuite tile",
            "matlab_logo.png": "MATLAB tile",
            "c3d_icon.png": "Motion Capture tile",
            "urdf_icon.png": "Model Explorer tile",
            "golf_robot_icon.png": "Application icon",
        }

        details: dict[str, Any] = {
            "assets_dir": str(ASSETS_DIR),
            "assets_dir_exists": ASSETS_DIR.exists(),
        }

        if not ASSETS_DIR.exists():
            result = DiagnosticResult(
                name="asset_files",
                status="fail",
                message=f"Assets directory not found: {ASSETS_DIR}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
            self.results.append(result)
            return result

        missing_assets = []
        found_assets = []
        for asset_name, description in expected_assets.items():
            asset_path = ASSETS_DIR / asset_name
            if asset_path.exists():
                found_assets.append(asset_name)
            else:
                missing_assets.append(f"{asset_name} ({description})")

        details["found_assets"] = found_assets
        details["missing_assets"] = missing_assets
        details["found_count"] = len(found_assets)
        details["missing_count"] = len(missing_assets)

        all_files = [f.name for f in ASSETS_DIR.iterdir() if f.is_file()]
        details["all_asset_files"] = sorted(all_files)
        details["total_asset_files"] = len(all_files)

        if missing_assets:
            result = DiagnosticResult(
                name="asset_files",
                status="warning",
                message=f"Missing {len(missing_assets)} asset files",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
        else:
            result = DiagnosticResult(
                name="asset_files",
                status="pass",
                message=f"All {len(expected_assets)} required assets found",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )

        self.results.append(result)
        return result

    def check_pyqt6_availability(self) -> DiagnosticResult:
        """Check PyQt6 availability."""
        start = time.time()
        details: dict[str, Any] = {}

        try:
            from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
            from PyQt6.QtWidgets import (  # noqa: F401 - needed for availability check
                QApplication,
            )

            details["pyqt6_available"] = True
            details["qt_version"] = QT_VERSION_STR
            details["pyqt_version"] = PYQT_VERSION_STR

            result = DiagnosticResult(
                name="pyqt6_availability",
                status="pass",
                message=f"PyQt6 available (Qt {QT_VERSION_STR}, PyQt {PYQT_VERSION_STR})",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
        except ImportError as e:
            details["pyqt6_available"] = False
            details["import_error"] = str(e)
            result = DiagnosticResult(
                name="pyqt6_availability",
                status="fail",
                message=f"PyQt6 not available: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )

        self.results.append(result)
        return result

    def check_engine_availability(self) -> DiagnosticResult:
        """Check physics engine availability with per-engine probe details."""
        start = time.time()
        details: dict[str, Any] = {}

        try:
            from src.shared.python.engine_core.engine_manager import EngineManager
            from src.shared.python.engine_core.engine_registry import EngineStatus

            manager = EngineManager()
            available = manager.get_available_engines()

            details["engine_manager_available"] = True
            details["available_engines"] = [e.value for e in available]
            details["engine_count"] = len(available)

            # Per-engine status with probe results
            engines_detail: list[dict[str, Any]] = []
            for engine_type, status in manager.engine_status.items():
                engine_info: dict[str, Any] = {
                    "name": engine_type.value,
                    "directory_status": status.value,
                    "path": str(manager.engine_paths.get(engine_type, "N/A")),
                }

                # Run probe if available
                probe = manager.probes.get(engine_type)
                if probe:
                    try:
                        probe_result = probe.probe()
                        engine_info["probe_status"] = probe_result.status.value
                        engine_info["version"] = probe_result.version
                        engine_info["missing_deps"] = probe_result.missing_dependencies
                        engine_info["diagnostic"] = probe_result.diagnostic_message
                        engine_info["installed"] = probe_result.is_available()
                    except (RuntimeError, ValueError, OSError) as e:
                        engine_info["probe_status"] = "error"
                        engine_info["diagnostic"] = str(e)
                        engine_info["installed"] = False
                else:
                    engine_info["probe_status"] = "no_probe"
                    engine_info["installed"] = status == EngineStatus.AVAILABLE

                engines_detail.append(engine_info)

            details["engines"] = engines_detail
            installed_count = sum(1 for e in engines_detail if e["installed"])
            total_count = len(engines_detail)

            if installed_count > 0:
                result = DiagnosticResult(
                    name="engine_availability",
                    status="pass",
                    message=f"{installed_count}/{total_count} engines installed",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                result = DiagnosticResult(
                    name="engine_availability",
                    status="warning",
                    message="No physics engines detected",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )

        except ImportError as e:
            details["engine_manager_available"] = False
            details["import_error"] = str(e)
            result = DiagnosticResult(
                name="engine_availability",
                status="warning",
                message=f"EngineManager not available: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )
        except (RuntimeError, TypeError, AttributeError) as e:
            details["error"] = str(e)
            result = DiagnosticResult(
                name="engine_availability",
                status="warning",
                message=f"Engine check error: {e}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )

        self.results.append(result)
        return result

    def check_biomech_siblings(self) -> DiagnosticResult:
        """Report the resolution tier for each of the five biomech sibling repos.

        See ``docs/adr/0014-shared-biomech-models.md`` (UpstreamDrift#5184).
        """
        start = time.time()
        details: dict[str, Any] = {}

        try:
            from src.shared.python.config.model_source_providers import (
                resolve_all_siblings,
            )

            resolutions = resolve_all_siblings()
            siblings_detail: list[dict[str, Any]] = []
            resolved_count = 0
            for name, resolution in resolutions.items():
                manifest_schema = _read_manifest_schema(resolution.manifest_path)
                if resolution.resolved:
                    resolved_count += 1
                siblings_detail.append(
                    {
                        "name": name,
                        "repo_name": resolution.repo_name,
                        "package": resolution.package,
                        "env_var": resolution.env_var,
                        "tier": resolution.tier,
                        "models_root": (
                            str(resolution.models_root)
                            if resolution.models_root is not None
                            else None
                        ),
                        "manifest_path": (
                            str(resolution.manifest_path)
                            if resolution.manifest_path is not None
                            else None
                        ),
                        "manifest_schema": manifest_schema,
                    }
                )

            details["siblings"] = siblings_detail
            details["resolved"] = resolved_count
            details["total"] = len(siblings_detail)

            if resolved_count == len(siblings_detail):
                status = "pass"
                message = f"All {resolved_count} biomech siblings resolved"
            elif resolved_count == 0:
                status = "warning"
                message = (
                    "No biomech siblings resolved - install editable checkouts "
                    "or run scripts/update_biomech_vendor.py"
                )
            else:
                status = "warning"
                message = (
                    f"{resolved_count}/{len(siblings_detail)} biomech siblings resolved"
                )
        except ImportError as exc:
            details["import_error"] = str(exc)
            status = "warning"
            message = f"Biomech sibling resolver unavailable: {exc}"
        except (OSError, RuntimeError, ValueError) as exc:
            details["error"] = str(exc)
            status = "warning"
            message = f"Biomech sibling check error: {exc}"

        result = DiagnosticResult(
            name="biomech_siblings",
            status=status,
            message=message,
            details=details,
            duration_ms=(time.time() - start) * 1000,
        )
        self.results.append(result)
        return result

    def check_tools_sidebar(self) -> DiagnosticResult:
        """Report whether the optional shared Tools sidebar is importable.

        The sidebar widget itself ships from the sibling
        ``D-sorganization/Tools`` repository. When that repo is not installed
        next to UpstreamDrift the launcher continues to work; this check
        surfaces whether the Sidekick design tokens are actually reaching a
        PyQt sidebar or being dropped on the floor.
        """
        start = time.time()
        details: dict[str, Any] = {}

        try:
            from src.shared.python.gui_launcher.tools_sidebar_integration import (
                _resolved_sidebar_module_name,
                is_tools_sidebar_available,
            )

            available = is_tools_sidebar_available()
            module_name = _resolved_sidebar_module_name() if available else None
            details["available"] = available
            details["module_name"] = module_name

            if available:
                result = DiagnosticResult(
                    name="tools_sidebar",
                    status="pass",
                    message=f"Tools sidebar available ({module_name})",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                result = DiagnosticResult(
                    name="tools_sidebar",
                    status="info",
                    message=(
                        "Tools sidebar not installed (expected) - launcher runs "
                        "without the optional PyQt sidebar (Sidekick tokens "
                        "still apply to the React/Tauri shell)"
                    ),
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )
        except ImportError as exc:
            details["import_error"] = str(exc)
            result = DiagnosticResult(
                name="tools_sidebar",
                status="warning",
                message=f"Tools sidebar probe unavailable: {exc}",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )

        self.results.append(result)
        return result

    def _generate_recommendations(self) -> list[str]:  # noqa: C901
        """Generate recommendations based on diagnostic results."""
        recommendations = []

        for result in self.results:
            if result.status == "fail":
                if result.name == "models_yaml":
                    recommendations.append(
                        "CRITICAL: Ensure src/config/models.yaml exists and contains all 17 model definitions"
                    )
                elif result.name == "model_registry":
                    recommendations.append(
                        "Check ModelRegistry initialization - verify YAML parsing is working"
                    )
                elif result.name == "pyqt6_availability":
                    recommendations.append("Install PyQt6 with: pip install PyQt6")
                elif result.name == "asset_files":
                    recommendations.append(
                        "Restore missing asset files in src/launchers/assets/"
                    )
                elif result.name == "launcher_provider_compatibility":
                    recommendations.append(
                        "Fix model provider metadata so launcher entries resolve valid source roots, artifacts, and working directories"
                    )

            elif result.status == "warning":
                if result.name == "layout_config":
                    details = result.details
                    if details.get("missing_from_saved"):
                        recommendations.append(
                            f"LIKELY CAUSE: Saved layout is missing tiles. Delete {LAYOUT_CONFIG_FILE} to reset to defaults with all 17 tiles"
                        )
                elif result.name == "asset_files":
                    recommendations.append("Some tile icons may not display correctly")
                elif result.name == "launcher_provider_compatibility":
                    recommendations.append(
                        "Review incompatible provider-backed models before enabling shared external packs in the launcher"
                    )

        if not recommendations:
            recommendations.append("All systems operational - no issues detected")

        return recommendations


def reset_layout_config() -> bool:
    """Reset the launcher layout configuration to defaults.

    Returns:
        True if reset was successful, False otherwise
    """
    try:
        if LAYOUT_CONFIG_FILE.exists():
            # Backup existing config
            backup_path = LAYOUT_CONFIG_FILE.with_suffix(".json.bak")
            LAYOUT_CONFIG_FILE.rename(backup_path)
            logger.info("Backed up existing config to %s", backup_path)

        logger.info("Layout config reset - launcher will use defaults (17 tiles)")
        return True
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Failed to reset layout config: %s", e)
        return False


def run_cli_diagnostics() -> None:
    """Run diagnostics and print results to console."""
    logger.info("=" * 60)
    logger.info("UpstreamDrift - Launcher Diagnostics")
    logger.info("=" * 60)
    logger.debug("")

    diag = LauncherDiagnostics()
    results = diag.run_all_checks()

    # Print summary
    summary = results["summary"]
    status_icon = "✅" if summary["status"] == "healthy" else "⚠️"
    logger.info(f"{status_icon} Status: {summary['status'].upper()}")
    logger.info(f"   Passed: {summary['passed']}")
    logger.error(f"   Failed: {summary['failed']}")
    logger.warning(f"   Warnings: {summary['warnings']}")
    logger.debug("")

    # Print each check
    for check in results["checks"]:
        if check["status"] == "pass":
            icon = "✅"
        elif check["status"] == "fail":
            icon = "❌"
        else:
            icon = "⚠️"

        logger.info(f"{icon} {check['name']}: {check['message']}")

        # Print key details for failures/warnings
        if check["status"] in ("fail", "warning"):
            details = check.get("details", {})
            for key in [
                "missing_from_saved",
                "missing_expected_ids",
                "missing_from_registry",
            ]:
                if key in details and details[key]:
                    logger.info(f"     {key}: {details[key]}")

    logger.debug("")
    logger.info("Recommendations:")
    for rec in results["recommendations"]:
        logger.info(f"  \u2192 {rec}")

    logger.debug("")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Golf Modeling Suite Launcher Diagnostics"
    )
    parser.add_argument(
        "--reset-layout", action="store_true", help="Reset layout config to defaults"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    if args.reset_layout:
        reset_layout_config()
    elif args.json:
        diag = LauncherDiagnostics()
        results = diag.run_all_checks()
        logger.info(json.dumps(results, indent=2))
    else:
        run_cli_diagnostics()
