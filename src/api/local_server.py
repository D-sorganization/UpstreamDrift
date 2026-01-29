"""
Local-first API server for Golf Modeling Suite.

Runs entirely on localhost with NO authentication required.
This is the default mode - free, offline, no accounts needed.
"""

import os
import sys
from pathlib import Path

# Ensure we're running in local mode
os.environ.setdefault("GOLF_SUITE_MODE", "local")
os.environ.setdefault("GOLF_AUTH_DISABLED", "true")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import engines, simulation, analysis, export, simulation_ws
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
            "http://localhost:3000",      # Vite dev server
            "http://localhost:5173",      # Vite default
            "http://localhost:8080",      # Production UI
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
    # Note: We need to ensure these routers are compatible with the new structure
    app.include_router(engines.router, prefix="/api/engines", tags=["Engines"])
    app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
    app.include_router(simulation_ws.router, tags=["Simulation"])  # WebSocket routes don't usually use prefix/tags the same way, but good for docs
    app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
    app.include_router(export.router, prefix="/api/export", tags=["Export"])

    # Health check
    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy",
            "mode": "local",
            "auth_required": False,
            "engines": [e.value for e in engine_manager.get_available_engines()], # Adapted to existing method
        }

    # Serve static UI files in production
    ui_path = Path(__file__).parent.parent.parent / "ui" / "dist"
    if ui_path.exists():
        app.mount("/", StaticFiles(directory=str(ui_path), html=True), name="ui")

    return app


def main():
    """Launch local server with auto-open browser."""
    import webbrowser
    from threading import Timer

    app = create_local_app()

    host = "127.0.0.1"
    port = int(os.environ.get("GOLF_PORT", 8000))

    # Open browser after server starts
    def open_browser():
        if os.environ.get("GOLF_NO_BROWSER") != "true":
            webbrowser.open(f"http://{host}:{port}")

    Timer(1.5, open_browser).start()

    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           Golf Modeling Suite - Local Server                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║   Running at: http://{host}:{port:<5}                            ║
    ║   API Docs:   http://{host}:{port}/api/docs                    ║
    ║                                                               ║
    ║   Mode: LOCAL (no authentication required)                    ║
    ║   All features available. No account needed.                  ║
    ║                                                               ║
    ║   Press Ctrl+C to stop.                                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
