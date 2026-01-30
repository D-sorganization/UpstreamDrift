"""
Diagnostic utilities for Golf Modeling Suite API.

This module provides comprehensive diagnostic tools for troubleshooting
API and web application issues including:
- Static file serving verification
- Route registration checks
- CORS configuration validation
- WebSocket connectivity testing
- Performance metrics collection
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


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


class APIDiagnostics:
    """Diagnostic utilities for the Golf Modeling Suite API."""

    def __init__(self, app: Any = None) -> None:
        """Initialize diagnostics with optional FastAPI app reference.

        Args:
            app: FastAPI application instance for route inspection
        """
        self.app = app
        self.results: list[DiagnosticResult] = []
        self._start_time = time.time()

    def run_all_checks(self) -> dict[str, Any]:
        """Run all diagnostic checks and return comprehensive report.

        Returns:
            Dictionary containing all diagnostic results and summary
        """
        self.results = []

        # Core checks
        self.check_python_environment()
        self.check_static_files()
        self.check_ui_build()
        self.check_api_routes()
        self.check_cors_config()
        self.check_dependencies()
        self.check_engine_manager()

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
            "executable": sys.executable,
            "path_count": len(sys.path),
        }

        # Check for required environment variables
        env_vars = {
            "GOLF_SUITE_MODE": os.environ.get("GOLF_SUITE_MODE", "not set"),
            "GOLF_AUTH_DISABLED": os.environ.get("GOLF_AUTH_DISABLED", "not set"),
            "GOLF_PORT": os.environ.get("GOLF_PORT", "not set"),
        }
        details["environment_variables"] = env_vars

        result = DiagnosticResult(
            name="python_environment",
            status="pass",
            message="Python environment configured correctly",
            details=details,
            duration_ms=(time.time() - start) * 1000,
        )
        self.results.append(result)
        return result

    def check_static_files(self) -> DiagnosticResult:
        """Check static file serving configuration."""
        start = time.time()

        # Determine UI path relative to this file
        api_dir = Path(__file__).parent
        project_root = api_dir.parent.parent
        ui_dist_path = project_root / "ui" / "dist"

        details: dict[str, Any] = {
            "ui_dist_path": str(ui_dist_path),
            "ui_dist_exists": ui_dist_path.exists(),
            "project_root": str(project_root),
        }

        if ui_dist_path.exists():
            # Check for critical files
            index_html = ui_dist_path / "index.html"
            assets_dir = ui_dist_path / "assets"

            details["index_html_exists"] = index_html.exists()
            details["assets_dir_exists"] = assets_dir.exists()

            if assets_dir.exists():
                js_files = list(assets_dir.glob("*.js"))
                css_files = list(assets_dir.glob("*.css"))
                details["js_files_count"] = len(js_files)
                details["css_files_count"] = len(css_files)
                details["js_files"] = [f.name for f in js_files[:5]]  # First 5
                details["css_files"] = [f.name for f in css_files[:5]]

            if index_html.exists() and assets_dir.exists():
                result = DiagnosticResult(
                    name="static_files",
                    status="pass",
                    message="Static files are properly configured",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                result = DiagnosticResult(
                    name="static_files",
                    status="fail",
                    message="Static files incomplete - missing index.html or assets",
                    details=details,
                    duration_ms=(time.time() - start) * 1000,
                )
        else:
            result = DiagnosticResult(
                name="static_files",
                status="fail",
                message="UI build directory not found - run 'npm run build' in ui/ directory",
                details=details,
                duration_ms=(time.time() - start) * 1000,
            )

        self.results.append(result)
        return result

    def check_ui_build(self) -> DiagnosticResult:
        """Check if UI has been built and is up to date."""
        start = time.time()

        api_dir = Path(__file__).parent
        project_root = api_dir.parent.parent
        ui_dir = project_root / "ui"
        ui_dist = ui_dir / "dist"
        package_json = ui_dir / "package.json"
        node_modules = ui_dir / "node_modules"

        details: dict[str, Any] = {
            "ui_dir": str(ui_dir),
            "ui_dir_exists": ui_dir.exists(),
            "package_json_exists": package_json.exists(),
            "node_modules_exists": node_modules.exists(),
            "dist_exists": ui_dist.exists(),
        }

        # Check for source files
        if ui_dir.exists():
            src_dir = ui_dir / "src"
            details["src_dir_exists"] = src_dir.exists()
            if src_dir.exists():
                tsx_files = list(src_dir.rglob("*.tsx"))
                ts_files = list(src_dir.rglob("*.ts"))
                details["tsx_file_count"] = len(tsx_files)
                details["ts_file_count"] = len(ts_files)

        # Determine status
        if not ui_dir.exists():
            status = "fail"
            message = "UI directory not found"
        elif not package_json.exists():
            status = "fail"
            message = "package.json not found - UI project not initialized"
        elif not node_modules.exists():
            status = "warning"
            message = "node_modules missing - run 'npm install' in ui/ directory"
        elif not ui_dist.exists():
            status = "fail"
            message = "UI not built - run 'npm run build' in ui/ directory"
        else:
            status = "pass"
            message = "UI build is present"

        result = DiagnosticResult(
            name="ui_build",
            status=status,
            message=message,
            details=details,
            duration_ms=(time.time() - start) * 1000,
        )
        self.results.append(result)
        return result

    def check_api_routes(self) -> DiagnosticResult:
        """Check API route registration."""
        start = time.time()

        details: dict[str, Any] = {"routes": []}

        if self.app:
            for route in self.app.routes:
                route_info = {
                    "path": getattr(route, "path", "unknown"),
                    "methods": list(getattr(route, "methods", [])),
                    "name": getattr(route, "name", "unnamed"),
                }
                details["routes"].append(route_info)

            details["total_routes"] = len(details["routes"])

            # Check for essential routes
            essential_routes = ["/api/health", "/api/engines", "/api/docs"]
            route_paths = [r["path"] for r in details["routes"]]

            missing_routes = []
            for er in essential_routes:
                # Check if any route matches (accounting for prefix patterns)
                found = any(
                    er in rp or rp.endswith(er.replace("/api", ""))
                    for rp in route_paths
                )
                if not found:
                    missing_routes.append(er)

            details["missing_essential_routes"] = missing_routes

            if missing_routes:
                status = "warning"
                message = f"Some essential routes may be missing: {missing_routes}"
            else:
                status = "pass"
                message = f"API routes registered: {details['total_routes']} routes"
        else:
            status = "warning"
            message = "No FastAPI app reference - cannot inspect routes"

        result = DiagnosticResult(
            name="api_routes",
            status=status,
            message=message,
            details=details,
            duration_ms=(time.time() - start) * 1000,
        )
        self.results.append(result)
        return result

    def check_cors_config(self) -> DiagnosticResult:
        """Check CORS configuration."""
        start = time.time()

        expected_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ]

        details: dict[str, Any] = {
            "expected_origins": expected_origins,
            "cors_middleware_present": False,
        }

        if self.app:
            # Check for CORS middleware
            for middleware in getattr(self.app, "user_middleware", []):
                middleware_class = getattr(middleware, "cls", None)
                if middleware_class and "CORS" in str(middleware_class):
                    details["cors_middleware_present"] = True
                    details["middleware_options"] = getattr(middleware, "options", {})
                    break

        status = "pass" if details["cors_middleware_present"] else "warning"
        message = (
            "CORS middleware configured"
            if details["cors_middleware_present"]
            else "CORS middleware not detected - cross-origin requests may fail"
        )

        result = DiagnosticResult(
            name="cors_config",
            status=status,
            message=message,
            details=details,
            duration_ms=(time.time() - start) * 1000,
        )
        self.results.append(result)
        return result

    def check_dependencies(self) -> DiagnosticResult:
        """Check critical Python dependencies."""
        start = time.time()

        dependencies = {
            "fastapi": False,
            "uvicorn": False,
            "pydantic": False,
            "numpy": False,
            "scipy": False,
        }

        versions: dict[str, str] = {}

        for dep in dependencies:
            try:
                module = __import__(dep)
                dependencies[dep] = True
                versions[dep] = getattr(module, "__version__", "unknown")
            except ImportError:
                pass

        details: dict[str, Any] = {
            "dependencies": dependencies,
            "versions": versions,
            "all_present": all(dependencies.values()),
        }

        missing = [k for k, v in dependencies.items() if not v]

        if missing:
            status = "fail"
            message = f"Missing dependencies: {missing}"
        else:
            status = "pass"
            message = "All critical dependencies available"

        result = DiagnosticResult(
            name="dependencies",
            status=status,
            message=message,
            details=details,
            duration_ms=(time.time() - start) * 1000,
        )
        self.results.append(result)
        return result

    def check_engine_manager(self) -> DiagnosticResult:
        """Check engine manager availability."""
        start = time.time()

        details: dict[str, Any] = {}

        try:
            from src.shared.python.engine_manager import EngineManager

            manager = EngineManager()
            engines = manager.get_available_engines()

            details["engine_manager_available"] = True
            details["available_engines"] = [e.value for e in engines]
            details["engine_count"] = len(engines)

            if engines:
                status = "pass"
                message = f"Engine manager ready with {len(engines)} engines"
            else:
                status = "warning"
                message = "Engine manager available but no engines detected"

        except ImportError as e:
            details["engine_manager_available"] = False
            details["import_error"] = str(e)
            status = "fail"
            message = f"Engine manager import failed: {e}"
        except Exception as e:
            details["engine_manager_available"] = False
            details["error"] = str(e)
            status = "warning"
            message = f"Engine manager initialization issue: {e}"

        result = DiagnosticResult(
            name="engine_manager",
            status=status,
            message=message,
            details=details,
            duration_ms=(time.time() - start) * 1000,
        )
        self.results.append(result)
        return result

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []

        for result in self.results:
            if result.status == "fail":
                if result.name == "static_files":
                    recommendations.append(
                        "CRITICAL: Build the UI by running 'cd ui && npm install && npm run build'"
                    )
                elif result.name == "ui_build":
                    if "node_modules" in result.message:
                        recommendations.append(
                            "Run 'npm install' in the ui/ directory to install dependencies"
                        )
                    else:
                        recommendations.append(
                            "Run 'npm run build' in the ui/ directory to build the web app"
                        )
                elif result.name == "dependencies":
                    recommendations.append(
                        "Install missing dependencies with 'pip install -r requirements.txt'"
                    )

            elif result.status == "warning":
                if result.name == "cors_config":
                    recommendations.append(
                        "Verify CORS middleware is properly configured for your deployment"
                    )
                elif result.name == "engine_manager":
                    recommendations.append(
                        "Check physics engine installations (MuJoCo, Drake, etc.)"
                    )

        if not recommendations:
            recommendations.append("All systems operational - no issues detected")

        return recommendations


def get_diagnostic_endpoint_html(results: dict[str, Any]) -> str:
    """Generate HTML diagnostic report for browser viewing.

    Args:
        results: Diagnostic results dictionary

    Returns:
        HTML string for browser display
    """
    summary = results.get("summary", {})
    checks = results.get("checks", [])
    recommendations = results.get("recommendations", [])

    status_color = "#28a745" if summary.get("status") == "healthy" else "#dc3545"

    checks_html = ""
    for check in checks:
        status = check.get("status", "unknown")
        if status == "pass":
            icon = "✅"
            color = "#28a745"
        elif status == "fail":
            icon = "❌"
            color = "#dc3545"
        else:
            icon = "⚠️"
            color = "#ffc107"

        checks_html += f"""
        <div style="border: 1px solid {color}; border-radius: 8px; padding: 12px; margin: 8px 0; background: #1e1e1e;">
            <h4 style="margin: 0 0 8px 0; color: {color};">{icon} {check.get('name', 'Unknown')}</h4>
            <p style="margin: 0 0 8px 0; color: #ccc;">{check.get('message', '')}</p>
            <details style="color: #888;">
                <summary style="cursor: pointer;">Details ({check.get('duration_ms', 0):.2f}ms)</summary>
                <pre style="background: #0d0d0d; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 12px;">{_format_details(check.get('details', {}))}</pre>
            </details>
        </div>
        """

    recs_html = "".join(
        f"<li style='color: #ffd700; margin: 4px 0;'>{r}</li>" for r in recommendations
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Golf Modeling Suite - API Diagnostics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0d0d0d;
                color: #f0f0f0;
                margin: 0;
                padding: 20px;
            }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            h1 {{ color: #0a84ff; border-bottom: 2px solid #0a84ff; padding-bottom: 10px; }}
            .summary {{
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
            }}
            .stat {{
                text-align: center;
                padding: 10px;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
            }}
            .stat-label {{ color: #888; font-size: 0.9em; }}
            .status-badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-weight: bold;
                background: {status_color};
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏌️ Golf Modeling Suite - API Diagnostics</h1>

            <div class="summary">
                <div class="stat">
                    <div class="stat-value" style="color: {status_color};">{summary.get('status', 'unknown').upper()}</div>
                    <div class="stat-label">System Status</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #28a745;">{summary.get('passed', 0)}</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #dc3545;">{summary.get('failed', 0)}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #ffc107;">{summary.get('warnings', 0)}</div>
                    <div class="stat-label">Warnings</div>
                </div>
            </div>

            <h2>📋 Diagnostic Checks</h2>
            {checks_html}

            <h2>💡 Recommendations</h2>
            <ul style="background: #1a1a2e; padding: 20px 40px; border-radius: 8px;">
                {recs_html}
            </ul>

            <p style="color: #666; text-align: center; margin-top: 30px;">
                Generated at {summary.get('timestamp', 'unknown')} |
                <a href="/api/diagnostics" style="color: #0a84ff;">JSON View</a> |
                <a href="/api/docs" style="color: #0a84ff;">API Docs</a>
            </p>
        </div>
    </body>
    </html>
    """


def _format_details(details: dict[str, Any], indent: int = 2) -> str:
    """Format details dictionary for display."""
    import json

    try:
        return json.dumps(details, indent=indent, default=str)
    except Exception:
        return str(details)
