"""
Local-first API server for Golf Modeling Suite.

Runs entirely on localhost with NO authentication required.
This is the default mode - free, offline, no accounts needed.
"""

import mimetypes
import os
from pathlib import Path

# Fix MIME types for JavaScript modules on Windows
# Windows registry often has incorrect/missing MIME types for .js files
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

# Ensure we're running in local mode
os.environ.setdefault("GOLF_SUITE_MODE", "local")
os.environ.setdefault("GOLF_AUTH_DISABLED", "true")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import analysis, engines, export, simulation, simulation_ws

# Note: EngineManager and Services will need to be properly imported from shared logic
# Assuming these exist based on the plan, or we bridge them to existing managers
from src.shared.python.engine_manager import EngineManager

# We might need to wrap these or use existing services if they don't match exactly
# For now we'll stub the missing ones or assume they'll be refactored


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
            "http://127.0.0.1:8080",
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
    # app.state.simulation_service = simulation_service # TODO: Implement service wrapper
    # app.state.analysis_service = analysis_service   # TODO: Implement service wrapper

    # Register routes (no auth required in local mode)
    # Note: Routers already define their own paths (e.g., /engines), so prefix is just /api
    app.include_router(engines.router, prefix="/api", tags=["Engines"])
    app.include_router(simulation.router, prefix="/api", tags=["Simulation"])
    app.include_router(
        simulation_ws.router, prefix="/api", tags=["Simulation WebSocket"]
    )
    app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
    app.include_router(export.router, prefix="/api", tags=["Export"])

    # Health check
    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy",
            "mode": "local",
            "auth_required": False,
            "engines": [
                e.value for e in engine_manager.get_available_engines()
            ],  # Adapted to existing method
        }

    # Serve static UI files in production
    ui_path = Path(__file__).parent.parent.parent / "ui" / "dist"
    if ui_path.exists():
        app.mount("/", StaticFiles(directory=str(ui_path), html=True), name="ui")

    return app


def print_logo_animated():
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

    print()
    for line in logo:
        print(f"    {ORANGE}{line}{RESET}")
        sys.stdout.flush()
        time.sleep(0.03)  # Scroll effect
    print()


def print_matrix_status(message: str, indent: int = 4):
    """Print status message in matrix green style."""
    GREEN = "\033[38;5;46m"  # Bright matrix green
    RESET = "\033[0m"
    print(f"{' ' * indent}{GREEN}>{RESET} {GREEN}{message}{RESET}")


def print_server_info(host: str, port: int):
    """Print server info box."""
    CYAN = "\033[38;5;51m"
    RESET = "\033[0m"

    print(f"""
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


def main():
    """Launch local server with auto-open browser."""
    import time
    import webbrowser
    from threading import Timer

    DIM = "\033[2m"
    RESET = "\033[0m"

    print(f"\n{DIM}Initializing Golf Modeling Suite...{RESET}\n")

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
    print()

    # Open browser after server starts
    def open_browser():
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
