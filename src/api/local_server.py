"""
Local-first API server for Golf Modeling Suite.

Runs entirely on localhost with NO authentication required.
This is the default mode - free, offline, no accounts needed.

Diagnostic Features:
- /api/diagnostics - JSON diagnostic report
- /api/diagnostics/html - Browser-friendly diagnostic page
- /api/debug/routes - List all registered routes
- /api/debug/static - Check static file configuration
"""

from __future__ import annotations

import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

# Fix MIME types for JavaScript modules on Windows
# Windows registry often has incorrect/missing MIME types for .js files
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/x-icon", ".ico")

# Ensure we're running in local mode
os.environ.setdefault("GOLF_SUITE_MODE", "local")
os.environ.setdefault("GOLF_AUTH_DISABLED", "true")

# NOTE: These imports are placed after env setup intentionally
# The environment variables must be set before FastAPI initialization
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.api.diagnostics import (  # noqa: E402
    APIDiagnostics,
    get_diagnostic_endpoint_html,
)
from src.api.routes import (  # noqa: E402
    analysis,
    chat_ws,
    engines,
    export,
    simulation,
    simulation_ws,
)
from src.api.services.chat_service import ChatService  # noqa: E402
from src.shared.python.engine_manager import EngineManager  # noqa: E402
from src.shared.python.logging_config import get_logger  # noqa: E402

logger = logging.getLogger(__name__)

logger = get_logger(__name__)


# Track startup metrics for diagnostics
_startup_metrics: dict[str, Any] = {
    "startup_time": None,
    "static_files_mounted": False,
    "ui_path": None,
    "engines_loaded": [],
    "errors": [],
}


def _resolve_ui_dist_path() -> Path:
    """Resolve the UI build path for static file serving."""
    env_override = os.environ.get("GOLF_UI_DIST")
    if env_override:
        return Path(env_override)
    return Path(__file__).parent.parent.parent / "ui" / "dist"


def create_local_app() -> FastAPI:
    """Create FastAPI app configured for local use."""

    app = FastAPI(
        title="Golf Modeling Suite",
        description="Local physics simulation for golf biomechanics",
        version="2.0.0",
        docs_url="/api/docs",  # Swagger UI available locally
        redoc_url="/api/redoc",
    )

    # CORS: Allow local origins only
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Vite dev server
            "http://localhost:5173",  # Vite default
            "http://localhost:8080",  # Production UI
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8001",
            "http://127.0.0.1:8080",
            "http://localhost:8001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize services (lazy loading)
    # Using the existing EngineManager we saw in src.shared.python
    engine_manager = EngineManager()

    # Store in app state for dependency injection
    app.state.engine_manager = engine_manager
    app.state.chat_service = ChatService()

    # Register routes (no auth required in local mode)
    # Note: Routers already define their own paths (e.g., /engines), so prefix is just /api
    app.include_router(engines.router, prefix="/api", tags=["Engines"])
    app.include_router(simulation.router, prefix="/api", tags=["Simulation"])
    app.include_router(
        simulation_ws.router, prefix="/api", tags=["Simulation WebSocket"]
    )
    app.include_router(chat_ws.router, prefix="/api", tags=["Chat"])
    app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
    app.include_router(export.router, prefix="/api", tags=["Export"])

    # Store engine manager for diagnostics
    _startup_metrics["engines_loaded"] = [
        e.value for e in engine_manager.get_available_engines()
    ]

    # Launcher manifest endpoint - serves tile configuration for React frontend
    @app.get("/api/launcher/manifest")
    async def get_launcher_manifest() -> dict[str, Any]:
        """Return the launcher manifest (tile configuration) for the web UI."""
        import json

        manifest_path = (
            Path(__file__).parent.parent / "config" / "launcher_manifest.json"
        )
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return {"version": "1.0.0", "tiles": []}

    # Launcher logos endpoint - serves tile logo images for React frontend
    @app.get("/api/launcher/logos/{logo_name:path}")
    async def get_launcher_logo(logo_name: str) -> Any:
        """Serve logo images from assets/logos directory."""
        from fastapi.responses import FileResponse

        logos_dir = Path(__file__).parent.parent.parent / "assets" / "logos"
        logo_path = logos_dir / logo_name
        if logo_path.exists() and logo_path.is_file():
            return FileResponse(str(logo_path))
        # Also check launcher assets directory
        launcher_logos = Path(__file__).parent.parent / "launchers" / "assets"
        alt_path = launcher_logos / logo_name
        if alt_path.exists() and alt_path.is_file():
            return FileResponse(str(alt_path))
        return JSONResponse(
            status_code=404, content={"detail": f"Logo not found: {logo_name}"}
        )

    # ── Launcher: launch engines/tools as subprocesses ────────────────
    # Initialize launcher service (lazy-loads ProcessManager and
    # ModelHandlerRegistry from src.launchers on first use, breaking the
    # api -> launchers module-level dependency).
    from src.api.services.launcher_service import LauncherService

    _repo_root = Path(__file__).parent.parent.parent
    _launcher_service = LauncherService(repo_root=_repo_root)
    app.state.process_manager = _launcher_service.process_manager

    @app.post("/api/launcher/launch/{tile_id}")
    async def launch_tile(tile_id: str) -> dict[str, Any]:
        """Launch an engine or tool by tile ID.

        Looks up the tile in the launcher manifest and uses the model
        handler registry to spawn it as a subprocess.
        """
        import json as _json

        logger.info("[launch] Received launch request for tile_id=%s", tile_id)

        # Load manifest to find tile config
        manifest_path = (
            Path(__file__).parent.parent / "config" / "launcher_manifest.json"
        )
        if not manifest_path.exists():
            logger.error("[launch] Manifest not found at %s", manifest_path)
            return JSONResponse(
                status_code=404,
                content={"detail": "Launcher manifest not found"},
            )

        with open(manifest_path, encoding="utf-8") as f:
            manifest = _json.load(f)

        tile = None
        for t in manifest.get("tiles", []):
            if t.get("id") == tile_id:
                tile = t
                break

        if tile is None:
            logger.warning("[launch] Tile not found: %s", tile_id)
            return JSONResponse(
                status_code=404,
                content={"detail": f"Tile not found: {tile_id}"},
            )

        model_type = tile.get("type", "")
        repo_path = Path(__file__).parent.parent.parent
        logger.info(
            "[launch] Resolved tile: name=%s type=%s path=%s",
            tile.get("name"),
            model_type,
            tile.get("path"),
        )

        # Use a simple namespace so handlers can read .id, .type, .path, etc.
        class _TileModel:
            """Minimal model object compatible with handler.launch()."""

            def __init__(self, data: dict) -> None:
                for k, v in data.items():
                    setattr(self, k, v)

        model = _TileModel(tile)

        handler = _launcher_service.get_handler(model_type)
        if handler is None:
            logger.error(
                "[launch] No handler for type=%s (tile=%s)", model_type, tile_id
            )
            return JSONResponse(
                status_code=400,
                content={"detail": f"No handler for type: {model_type}"},
            )

        logger.info(
            "[launch] Using handler %s for tile %s",
            type(handler).__name__,
            tile_id,
        )
        success = handler.launch(model, repo_path, _launcher_service.process_manager)
        if success:
            logger.info(
                "[launch] Successfully launched tile %s (type=%s)", tile_id, model_type
            )
            return {"status": "launched", "tile_id": tile_id, "name": tile.get("name")}
        else:
            logger.error(
                "[launch] Handler returned failure for tile %s (type=%s)",
                tile_id,
                model_type,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": f"Failed to launch {tile.get('name', tile_id)}"},
            )

    @app.get("/api/launcher/processes")
    async def list_running_processes() -> dict[str, Any]:
        """List currently running engine/tool processes."""
        return {"processes": _launcher_service.get_running_processes()}

    @app.post("/api/launcher/stop/{name}")
    async def stop_process(name: str) -> dict[str, Any]:
        """Stop a running engine/tool process by name."""
        if not _launcher_service.stop_process(name):
            logger.warning("[stop] Process not found: %s", name)
            return JSONResponse(
                status_code=404, content={"detail": f"Process not found: {name}"}
            )
        return {"status": "stopped", "name": name}

    # Health check
    @app.get("/api/health")
    async def health_check() -> dict[str, Any]:
        return {
            "status": "healthy",
            "mode": "local",
            "auth_required": False,
            "engines": [e.value for e in engine_manager.get_available_engines()],
            "ui_available": _startup_metrics.get("static_files_mounted", False),
        }

    # Diagnostic endpoints
    @app.get("/api/diagnostics")
    async def get_diagnostics() -> dict[str, Any]:
        """Return comprehensive diagnostic information as JSON."""
        diagnostics = APIDiagnostics(app)
        results = diagnostics.run_all_checks()
        results["startup_metrics"] = _startup_metrics
        return results

    @app.get("/api/diagnostics/html", response_class=HTMLResponse)
    async def get_diagnostics_html() -> HTMLResponse:
        """Return diagnostic information as an HTML page."""
        diagnostics = APIDiagnostics(app)
        results = diagnostics.run_all_checks()
        results["startup_metrics"] = _startup_metrics
        html_content = get_diagnostic_endpoint_html(results)
        return HTMLResponse(content=html_content)

    @app.get("/api/debug/routes")
    async def debug_routes() -> dict[str, Any]:
        """List all registered API routes for debugging."""
        routes = []
        for route in app.routes:
            route_info = {
                "path": getattr(route, "path", "unknown"),
                "methods": list(getattr(route, "methods", [])),
                "name": getattr(route, "name", "unnamed"),
            }
            routes.append(route_info)
        return {
            "total_routes": len(routes),
            "routes": sorted(routes, key=lambda x: x["path"]),
        }

    @app.get("/api/debug/static")
    async def debug_static() -> dict[str, Any]:
        """Check static file configuration."""
        ui_path = Path(__file__).parent.parent.parent / "ui" / "dist"
        details: dict[str, Any] = {
            "ui_path": str(ui_path),
            "ui_exists": ui_path.exists(),
            "startup_metrics": _startup_metrics,
        }

        if ui_path.exists():
            details["index_html"] = (ui_path / "index.html").exists()
            details["assets_dir"] = (ui_path / "assets").exists()
            if (ui_path / "assets").exists():
                js_files = list((ui_path / "assets").glob("*.js"))
                css_files = list((ui_path / "assets").glob("*.css"))
                details["js_files"] = [f.name for f in js_files]
                details["css_files"] = [f.name for f in css_files]

        return details

    # Serve static UI files in production
    ui_path = _resolve_ui_dist_path()
    _startup_metrics["ui_path"] = str(ui_path)

    if ui_path.exists():
        logger.info(f"UI build found at {ui_path}, mounting static files")
        _startup_metrics["static_files_mounted"] = True

        # Mount logos directory so web UI can load tile logos
        logos_path = Path(__file__).parent.parent.parent / "assets" / "logos"
        if logos_path.exists():
            app.mount(
                "/logos",
                StaticFiles(directory=str(logos_path)),
                name="logos",
            )
            logger.info(f"Mounted /logos from {logos_path}")

        # Check if assets directory exists before mounting
        assets_path = ui_path / "assets"
        if assets_path.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_path)),
                name="static_assets",
            )
            logger.info(f"Mounted /assets from {assets_path}")
        else:
            logger.warning(f"Assets directory not found at {assets_path}")
            _startup_metrics["errors"].append(
                f"Assets directory missing: {assets_path}"
            )

        # SPA catch-all: serve index.html for all non-API routes
        index_html = ui_path / "index.html"
        if index_html.exists():
            from fastapi.responses import FileResponse

            @app.get("/{full_path:path}")
            async def serve_spa(request: Request, full_path: str) -> Any:
                """Serve the SPA index.html for all non-API routes."""
                if full_path.startswith("api/"):
                    return JSONResponse(
                        status_code=404,
                        content={"detail": "API route not found", "path": full_path},
                    )
                # Check if it's a static file first
                static_file = ui_path / full_path
                if full_path and static_file.exists() and static_file.is_file():
                    return FileResponse(str(static_file))
                return FileResponse(str(index_html))

    else:
        warning = f"UI build not found at {ui_path}. Run npm install && npm run build."
        logger.warning(warning)
        _startup_metrics["errors"].append(warning)

        # Provide a helpful error page when UI is not built
        @app.get("/{full_path:path}")
        async def serve_error_page(request: Request, full_path: str) -> HTMLResponse:
            """Serve a helpful error page when UI is not built."""
            if full_path.startswith("api/"):
                return JSONResponse(
                    status_code=404,
                    content={"detail": "API route not found", "path": full_path},
                )

            error_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Golf Modeling Suite - Setup Required</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        color: #f0f0f0;
                        margin: 0;
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .container {
                        text-align: center;
                        padding: 40px;
                        background: rgba(0,0,0,0.3);
                        border-radius: 16px;
                        max-width: 600px;
                    }
                    h1 { color: #0a84ff; margin-bottom: 10px; }
                    .emoji { font-size: 4em; margin-bottom: 20px; }
                    .code {
                        background: #0d0d0d;
                        padding: 15px 20px;
                        border-radius: 8px;
                        font-family: monospace;
                        margin: 20px 0;
                        text-align: left;
                    }
                    a {
                        color: #0a84ff;
                        text-decoration: none;
                    }
                    a:hover { text-decoration: underline; }
                    .btn {
                        display: inline-block;
                        background: #0a84ff;
                        color: white;
                        padding: 12px 24px;
                        border-radius: 8px;
                        margin-top: 20px;
                        text-decoration: none;
                    }
                    .btn:hover {
                        background: #0066cc;
                        text-decoration: none;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="emoji">🏌️‍♂️</div>
                    <h1>Golf Modeling Suite</h1>
                    <h2>Web UI Setup Required</h2>
                    <p>The web interface has not been built yet. Run these commands:</p>
                    <div class="code">
                        cd ui<br>
                        npm install<br>
                        npm run build
                    </div>
                    <p>Then restart the server.</p>
                    <p style="color: #888; margin-top: 30px;">
                        <strong>API is working!</strong> Check:
                    </p>
                    <p>
                        <a href="/api/health">/api/health</a> |
                        <a href="/api/docs">/api/docs</a> |
                        <a href="/api/diagnostics/html">/api/diagnostics/html</a>
                    </p>
                    <a href="/api/diagnostics/html" class="btn">Run Diagnostics</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=503)

    _startup_metrics["startup_time"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    return app


def print_logo_animated() -> None:
    """Print the Upstream Drift logo with scroll animation."""
    import sys
    import time

    # ANSI escape codes
    ORANGE = "\033[38;5;208m"
    RESET = "\033[0m"

    logo = [
        r"██╗   ██╗██████╗ ███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗",
        r"██║   ██║██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║",
        r"██║   ██║██████╔╝███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║",
        r"██║   ██║██╔═══╝ ╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║",
        r"╚██████╔╝██║     ███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║",
        r" ╚═════╝ ╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝",
        r"",
        r"██████╗ ██████╗ ██╗███████╗████████╗",
        r"██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝",
        r"██║  ██║██████╔╝██║█████╗     ██║   ",
        r"██║  ██║██╔══██╗██║██╔══╝     ██║   ",
        r"██████╔╝██║  ██║██║██║        ██║   ",
        r"╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   ",
    ]

    logger.info("")
    try:
        for line in logo:
            logger.info("    %s%s%s", ORANGE, line, RESET)
            sys.stdout.flush()
            time.sleep(0.03)  # Scroll effect
    except UnicodeEncodeError:
        logger.info("    %sUPSTREAM DRIFT%s", ORANGE, RESET)
    logger.info("")


def print_matrix_status(message: str, indent: int = 4) -> None:
    """Print status message in matrix green style."""
    GREEN = "\033[38;5;46m"  # Bright matrix green
    RESET = "\033[0m"
    logger.info("%s%s>%s %s%s%s", " " * indent, GREEN, RESET, GREEN, message, RESET)


def print_server_info(host: str, port: int) -> None:
    """Print server info box."""
    CYAN = "\033[38;5;51m"
    RESET = "\033[0m"

    try:
        logger.info(f"""
{CYAN}    ┌─────────────────────────────────────────────────────────┐
    │              Golf Modeling Suite - Local Server         │
    ├─────────────────────────────────────────────────────────┤
    │  Running at: http://{host}:{port:<5}                       │
    │  API Docs:   http://{host}:{port}/api/docs               │
    │                                                         │
    │  Mode: LOCAL (no auth required)                         │
    │  Press Ctrl+C to stop.                                  │
    └─────────────────────────────────────────────────────────┘{RESET}
    """)
    except UnicodeEncodeError:
        logger.info("\n    Golf Modeling Suite - Local Server")
        logger.info("    Running at: http://%s:%s", host, port)
        logger.info("    API Docs:   http://%s:%s/api/docs", host, port)
        logger.info("    Mode: LOCAL (no auth required)")
        logger.info("    Press Ctrl+C to stop.\n")


def main() -> None:
    """Launch local server with auto-open browser."""
    import time
    import webbrowser
    from threading import Timer

    import uvicorn

    DIM = "\033[2m"
    RESET = "\033[0m"

    logger.info("\n%sInitializing Golf Modeling Suite...%s\n", DIM, RESET)

    app = create_local_app()

    host = "127.0.0.1"
    port = int(os.environ.get("GOLF_PORT", 8000))

    # Print startup info in matrix green
    print_matrix_status("Loading physics engine manager...")
    time.sleep(0.1)
    print_matrix_status("Registering API routes...")
    time.sleep(0.1)
    print_matrix_status("Configuring static file server...")
    time.sleep(0.1)
    print_matrix_status(f"Server ready on port {port}")
    logger.info("")

    # Open browser after server starts
    def open_browser() -> None:
        if os.environ.get("GOLF_NO_BROWSER") != "true":
            webbrowser.open(f"http://{host}:{port}")

    Timer(1.5, open_browser).start()

    # Print server info
    print_server_info(host, port)

    # Logo last - stays visible at bottom of terminal
    print_logo_animated()

    # Start server (this blocks)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
