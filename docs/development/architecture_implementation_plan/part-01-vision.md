# Golf Modeling Suite - Modern Architecture Implementation Plan (Part 1/2)

Part of the [Golf Modeling Suite - Modern Architecture Implementation Plan](../ARCHITECTURE_IMPLEMENTATION_PLAN.md) index.

## Vision

Transform the Golf Modeling Suite from a collection of separate desktop applications into a **modern, unified, local-first application** that is:

- **Free and open source** - No API keys, subscriptions, or cloud requirements for core functionality
- **Local-first** - Runs entirely on user's machine by default
- **Modern UX** - Single cohesive interface, web-based UI that works everywhere
- **Easy to install** - One-command installation, minimal dependencies exposed to user
- **Shareable** - Easy to package, distribute, and collaborate on
- **Cloud-optional** - Cloud features available for those who want them, never required

---

## Architecture Principles

### 1. Local-First, Always Free

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER'S MACHINE                               │
│                                                                  │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │   Web UI    │ ───▶ │  Local API  │ ───▶ │  Engines    │    │
│   │  (Browser)  │      │ (FastAPI)   │      │ (MuJoCo,etc)│    │
│   └─────────────┘      └─────────────┘      └─────────────┘    │
│                                                                  │
│   Everything runs locally. No internet required.                 │
│   No API keys. No accounts. No subscriptions.                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2. API as Internal Backbone (Not External Gate)

- The API is an **internal architecture pattern**, not a paywall
- Local mode: API runs on `localhost:8000` with **no authentication**
- Cloud mode (optional): API runs remotely with authentication
- Same codebase, different deployment context

### 3. Progressive Enhancement

- Core features work offline, locally, free
- Cloud adds convenience (sharing, collaboration, remote compute)
- Never lock features behind cloud/payment

---

## Phase 1: Local-First API Consolidation

**Duration: 2-3 weeks**
**Goal: Make the API the internal backbone while keeping everything local and free**

### 1.1 Create Local Development Server

Create a zero-config local server that requires no API keys.

**File: `src/api/local_server.py`**

```python
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

from golf_suite.api.routes import engines, simulation, analysis, export
from golf_suite.api.services import EngineManager, SimulationService, AnalysisService


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
    engine_manager = EngineManager()
    simulation_service = SimulationService(engine_manager)
    analysis_service = AnalysisService()

    # Store in app state for dependency injection
    app.state.engine_manager = engine_manager
    app.state.simulation_service = simulation_service
    app.state.analysis_service = analysis_service

    # Register routes (no auth required in local mode)
    app.include_router(engines.router, prefix="/api/engines", tags=["Engines"])
    app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
    app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
    app.include_router(export.router, prefix="/api/export", tags=["Export"])

    # Health check
    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy",
            "mode": "local",
            "auth_required": False,
            "engines": engine_manager.list_available(),
        }

    # Serve static UI files in production
    ui_path = Path(__file__).parent.parent.parent / "ui" / "dist"
    if ui_path.exists():
        app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")

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
```

### 1.2 Conditional Authentication Middleware

Modify existing auth to be optional based on deployment mode.

**File: `src/api/auth/middleware.py`**

```python
"""Authentication middleware that respects local mode."""

import os
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer

# Check deployment mode
def is_local_mode() -> bool:
    """Check if running in local mode (no auth required)."""
    return (
        os.environ.get("GOLF_SUITE_MODE", "local") == "local" or
        os.environ.get("GOLF_AUTH_DISABLED", "false").lower() == "true"
    )


class OptionalAuth(HTTPBearer):
    """Bearer auth that's optional in local mode."""

    async def __call__(self, request: Request):
        if is_local_mode():
            # Local mode: no auth required, return mock user
            return LocalUser()

        # Cloud mode: require real authentication
        return await super().__call__(request)


class LocalUser:
    """Mock user for local mode - full access, no restrictions."""
    id: str = "local-user"
    email: str = "local@localhost"
    role: str = "ADMIN"  # Full access locally
    quota_remaining: int = float('inf')

    def has_permission(self, permission: str) -> bool:
        return True  # Everything allowed locally
```

### 1.3 Refactor Engine Routes for Direct Access

**File: `src/api/routes/engines.py`** (enhanced)

```python
"""Engine management routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..dependencies import get_engine_manager
from ..auth.middleware import OptionalAuth

router = APIRouter()


class EngineStatus(BaseModel):
    name: str
    available: bool
    loaded: bool
    version: Optional[str]
    capabilities: list[str]


class EngineListResponse(BaseModel):
    engines: list[EngineStatus]
    mode: str  # "local" or "cloud"


@router.get("/", response_model=EngineListResponse)
async def list_engines(
    engine_manager=Depends(get_engine_manager),
    _user=Depends(OptionalAuth(auto_error=False)),  # Optional auth
):
    """List all available physics engines and their status."""
    engines = []
    for engine_type in engine_manager.supported_engines:
        status = engine_manager.get_engine_status(engine_type)
        engines.append(EngineStatus(
            name=engine_type,
            available=status.available,
            loaded=status.loaded,
            version=status.version,
            capabilities=status.capabilities,
        ))

    return EngineListResponse(
        engines=engines,
        mode="local" if is_local_mode() else "cloud",
    )


@router.post("/{engine_type}/load")
async def load_engine(
    engine_type: str,
    model_path: Optional[str] = None,
    engine_manager=Depends(get_engine_manager),
):
    """Load a physics engine with optional model."""
    try:
        engine = engine_manager.load_engine(engine_type, model_path)
        return {
            "status": "loaded",
            "engine": engine_type,
            "model": model_path,
            "state": engine.get_state() if hasattr(engine, 'get_state') else None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{engine_type}/unload")
async def unload_engine(
    engine_type: str,
    engine_manager=Depends(get_engine_manager),
):
    """Unload a physics engine to free resources."""
    engine_manager.unload_engine(engine_type)
    return {"status": "unloaded", "engine": engine_type}
```

### 1.4 WebSocket Support for Real-Time Simulation

**File: `src/api/routes/simulation_ws.py`**

```python
"""WebSocket routes for real-time simulation streaming."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import json

router = APIRouter()


class SimulationFrame(BaseModel):
    """Single frame of simulation data."""
    time: float
    state: dict
    analysis: dict | None = None


@router.websocket("/ws/simulate/{engine_type}")
async def simulation_stream(
    websocket: WebSocket,
    engine_type: str,
):
    """
    Stream simulation in real-time over WebSocket.

    Client sends: {"action": "start", "config": {...}}
    Server sends: {"frame": 0, "time": 0.0, "state": {...}, ...}

    No authentication required in local mode.
    """
    await websocket.accept()

    engine_manager = websocket.app.state.engine_manager

    try:
        # Wait for start command
        start_msg = await websocket.receive_json()

        if start_msg.get("action") != "start":
            await websocket.send_json({"error": "Expected 'start' action"})
            return

        config = start_msg.get("config", {})

        # Load engine
        engine = engine_manager.load_engine(engine_type, config.get("model"))

        # Set initial state if provided
        if "initial_state" in config:
            engine.set_state(config["initial_state"])

        # Simulation parameters
        duration = config.get("duration", 3.0)
        timestep = config.get("timestep", 0.002)

        await websocket.send_json({"status": "running", "duration": duration})

        # Run simulation, streaming frames
        time = 0.0
        frame = 0

        while time < duration:
            # Check for client commands (pause, stop, etc.)
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.001
                )
                if msg.get("action") == "stop":
                    break
                if msg.get("action") == "pause":
                    await websocket.send_json({"status": "paused"})
                    # Wait for resume
                    while True:
                        msg = await websocket.receive_json()
                        if msg.get("action") == "resume":
                            break
                        if msg.get("action") == "stop":
                            raise StopIteration
            except asyncio.TimeoutError:
                pass  # No message, continue simulation

            # Step simulation
            engine.step(timestep)
            time += timestep
            frame += 1

            # Send frame data (throttle to 60fps for UI)
            if frame % max(1, int(1 / (60 * timestep))) == 0:
                state = engine.get_state()

                frame_data = {
                    "frame": frame,
                    "time": round(time, 4),
                    "state": state,
                }

                # Include analysis if requested
                if config.get("live_analysis"):
                    frame_data["analysis"] = {
                        "joint_angles": engine.get_joint_angles() if hasattr(engine, 'get_joint_angles') else None,
                        "velocities": engine.get_velocities() if hasattr(engine, 'get_velocities') else None,
                    }

                await websocket.send_json(frame_data)

        # Send completion
        await websocket.send_json({
            "status": "complete",
            "total_frames": frame,
            "total_time": round(time, 4),
        })

    except WebSocketDisconnect:
        pass  # Client disconnected
    except StopIteration:
        await websocket.send_json({"status": "stopped"})
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()
```

### 1.5 Update Main Entry Point

**File: `launch_golf_suite.py`** (modified)

```python
#!/usr/bin/env python3
"""
Golf Modeling Suite - Unified Launcher

Usage:
    golf-suite              # Launch web UI (default, recommended)
    golf-suite --classic    # Launch classic PyQt6 launcher
    golf-suite --api-only   # Launch API server only (for development)
    golf-suite --engine X   # Launch specific engine directly
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Golf Modeling Suite - Biomechanical Golf Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    golf-suite                    Launch web UI (opens in browser)
    golf-suite --classic          Launch classic desktop UI
    golf-suite --api-only         Start API server without UI
    golf-suite --engine mujoco    Launch MuJoCo engine directly
        """,
    )

    parser.add_argument(
        "--classic",
        action="store_true",
        help="Use classic PyQt6 desktop launcher instead of web UI",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start API server only (no UI)",
    )
    parser.add_argument(
        "--engine",
        choices=["mujoco", "drake", "pinocchio", "opensim", "myosuite"],
        help="Launch a specific engine directly",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for local server (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open browser",
    )

    args = parser.parse_args()

    if args.engine:
        # Direct engine launch (legacy support)
        launch_engine_directly(args.engine)
    elif args.classic:
        # Classic PyQt6 launcher
        from src.launchers.upstream_drift_launcher import main as classic_main
        classic_main()
    elif args.api_only:
        # API server only
        os.environ["GOLF_NO_BROWSER"] = "true"
        from src.api.local_server import main as api_main
        api_main()
    else:
        # Default: Web UI (recommended)
        os.environ["GOLF_PORT"] = str(args.port)
        if args.no_browser:
            os.environ["GOLF_NO_BROWSER"] = "true"
        from src.api.local_server import main as server_main
        server_main()


def launch_engine_directly(engine: str):
    """Launch a specific engine GUI directly (legacy mode)."""
    engine_launchers = {
        "mujoco": "src.engines.physics_engines.mujoco.python.humanoid_launcher",
        "drake": "src.engines.physics_engines.drake.python.drake_gui_app",
        "pinocchio": "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui",
    }

    if engine not in engine_launchers:
        print(f"Direct launch not available for {engine}. Use web UI instead.")
        sys.exit(1)

    import importlib
    module = importlib.import_module(engine_launchers[engine])
    module.main()


if __name__ == "__main__":
    main()
```

---

## Phase 2: Modern Web UI

**Duration: 4-6 weeks**
**Goal: Build a beautiful, responsive web UI that replaces scattered desktop windows**

### 2.1 Technology Stack

| Component      | Choice                       | Rationale                              |
| -------------- | ---------------------------- | -------------------------------------- |
| **Framework**  | React 18 + TypeScript        | Industry standard, huge ecosystem      |
| **Build Tool** | Vite                         | Fast dev server, optimized builds      |
| **Styling**    | Tailwind CSS                 | Rapid development, consistent design   |
| **State**      | Zustand                      | Simple, performant, no boilerplate     |
| **3D Viz**     | Three.js + React Three Fiber | Powerful, can embed or replace Meshcat |
| **Charts**     | Recharts or Visx             | React-native charting                  |
| **Forms**      | React Hook Form + Zod        | Type-safe form validation              |
| **API Client** | TanStack Query               | Caching, real-time updates             |

### 2.2 Project Structure

```
ui/
├── src/
│   ├── main.tsx                 # Entry point
│   ├── App.tsx                  # Root component
│   ├── api/
│   │   ├── client.ts            # API client (fetch + WebSocket)
│   │   ├── hooks.ts             # React Query hooks
│   │   └── types.ts             # TypeScript types from OpenAPI
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx      # Navigation sidebar
│   │   │   ├── Header.tsx       # Top bar with status
│   │   │   └── MainLayout.tsx   # Page wrapper
│   │   ├── simulation/
│   │   │   ├── EngineSelector.tsx
│   │   │   ├── SimulationControls.tsx
│   │   │   ├── ParameterPanel.tsx
│   │   │   └── SimulationProgress.tsx
│   │   ├── visualization/
│   │   │   ├── Scene3D.tsx      # Three.js 3D view
│   │   │   ├── MeshcatEmbed.tsx # Embedded Meshcat (fallback)
│   │   │   ├── JointVisualizer.tsx
│   │   │   └── TrajectoryPlot.tsx
│   │   ├── analysis/
│   │   │   ├── LivePlot.tsx
│   │   │   ├── BiomechanicsPanel.tsx
│   │   │   └── ComparisonView.tsx
│   │   └── common/
│   │       ├── Button.tsx
│   │       ├── Card.tsx
│   │       ├── Slider.tsx
│   │       └── ...
│   ├── pages/
│   │   ├── Dashboard.tsx        # Home / overview
│   │   ├── Simulation.tsx       # Main simulation page
│   │   ├── Analysis.tsx         # Post-hoc analysis
│   │   ├── Models.tsx           # Model browser
│   │   └── Settings.tsx         # Local settings
│   ├── stores/
│   │   ├── simulationStore.ts   # Simulation state
│   │   ├── engineStore.ts       # Engine status
│   │   └── settingsStore.ts     # User preferences
│   └── styles/
│       └── globals.css          # Tailwind imports
├── public/
│   ├── models/                  # 3D model assets
│   └── favicon.ico
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

### 2.3 Core Components

**Main Simulation Page Layout:**

```tsx
// ui/src/pages/Simulation.tsx

import { useState } from "react";
import { useSimulation } from "../api/hooks";
import { EngineSelector } from "../components/simulation/EngineSelector";
import { SimulationControls } from "../components/simulation/SimulationControls";
import { ParameterPanel } from "../components/simulation/ParameterPanel";
import { Scene3D } from "../components/visualization/Scene3D";
import { LivePlot } from "../components/analysis/LivePlot";

export function SimulationPage() {
  const [selectedEngine, setSelectedEngine] = useState<string>("mujoco");
  const { isRunning, currentFrame, start, stop, pause } =
    useSimulation(selectedEngine);

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Left Sidebar - Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto">
        <h2 className="text-xl font-bold text-white mb-4">Simulation</h2>

        <EngineSelector
          value={selectedEngine}
          onChange={setSelectedEngine}
          disabled={isRunning}
        />

        <ParameterPanel engine={selectedEngine} />

        <SimulationControls
          isRunning={isRunning}
          onStart={start}
          onStop={stop}
          onPause={pause}
        />
      </aside>

      {/* Main Content - 3D View + Analysis */}
      <main className="flex-1 flex flex-col">
        {/* 3D Visualization */}
        <div className="flex-1 relative">
          <Scene3D engine={selectedEngine} frame={currentFrame} />

          {/* Overlay: Status */}
          <div className="absolute top-4 left-4 bg-black/50 px-3 py-1 rounded">
            <span className="text-green-400">
              {isRunning ? `Frame ${currentFrame?.frame}` : "Ready"}
            </span>
          </div>
        </div>

        {/* Bottom: Live Analysis */}
        <div className="h-64 bg-gray-800 border-t border-gray-700">
          <LivePlot data={currentFrame?.analysis} />
        </div>
      </main>

      {/* Right Sidebar - Live Data */}
      <aside className="w-72 bg-gray-800 border-l border-gray-700 p-4">
        <h3 className="text-lg font-semibold text-white mb-3">Live Data</h3>
        <JointAnglesDisplay frame={currentFrame} />
        <VelocitiesDisplay frame={currentFrame} />
      </aside>
    </div>
  );
}
```

**WebSocket Hook for Real-Time Simulation:**

```tsx
// ui/src/api/hooks.ts

import { useState, useCallback, useRef, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";

interface SimulationFrame {
  frame: number;
  time: number;
  state: Record<string, number[]>;
  analysis?: {
    joint_angles?: number[];
    velocities?: number[];
  };
}

interface SimulationConfig {
  model?: string;
  duration?: number;
  timestep?: number;
  live_analysis?: boolean;
  initial_state?: Record<string, number[]>;
}

export function useSimulation(engineType: string) {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<SimulationFrame | null>(
    null,
  );
  const [frames, setFrames] = useState<SimulationFrame[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const start = useCallback(
    (config: SimulationConfig = {}) => {
      // Connect to local WebSocket (no auth needed)
      const ws = new WebSocket(
        `ws://localhost:8000/api/ws/simulate/${engineType}`,
      );
      wsRef.current = ws;

      ws.onopen = () => {
        setIsRunning(true);
        setFrames([]);
        ws.send(
          JSON.stringify({
            action: "start",
            config: {
              duration: 3.0,
              timestep: 0.002,
              live_analysis: true,
              ...config,
            },
          }),
        );
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === "complete") {
          setIsRunning(false);
          return;
        }

        if (data.frame !== undefined) {
          setCurrentFrame(data);
          setFrames((prev) => [...prev, data]);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setIsRunning(false);
      };

      ws.onclose = () => {
        setIsRunning(false);
      };
    },
    [engineType],
  );

  const stop = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: "stop" }));
  }, []);

  const pause = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: "pause" }));
    setIsPaused(true);
  }, []);

  const resume = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: "resume" }));
    setIsPaused(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return {
    isRunning,
    isPaused,
    currentFrame,
    frames,
    start,
    stop,
    pause,
    resume,
  };
}
```

**Engine Selector Component:**

```tsx
// ui/src/components/simulation/EngineSelector.tsx

import { useQuery } from "@tanstack/react-query";
import { fetchEngines } from "../../api/client";

interface Props {
  value: string;
  onChange: (engine: string) => void;
  disabled?: boolean;
}

export function EngineSelector({ value, onChange, disabled }: Props) {
  const { data: engines, isLoading } = useQuery({
    queryKey: ["engines"],
    queryFn: fetchEngines,
  });

  if (isLoading) {
    return <div className="animate-pulse h-10 bg-gray-700 rounded" />;
  }

  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-gray-300 mb-2">
        Physics Engine
      </label>
      <div className="grid grid-cols-1 gap-2">
        {engines?.map((engine) => (
          <button
            key={engine.name}
            onClick={() => onChange(engine.name)}
            disabled={disabled || !engine.available}
            className={`
              p-3 rounded-lg border text-left transition-all
              ${
                value === engine.name
                  ? "border-blue-500 bg-blue-500/20 text-white"
                  : "border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500"
              }
              ${!engine.available && "opacity-50 cursor-not-allowed"}
              ${disabled && "cursor-not-allowed"}
            `}
          >
            <div className="font-medium">{engine.name}</div>
            <div className="text-xs text-gray-400">
              {engine.available
                ? engine.loaded
                  ? "● Loaded"
                  : "○ Available"
                : "✗ Not installed"}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
```

### 2.4 3D Visualization with Three.js

```tsx
// ui/src/components/visualization/Scene3D.tsx

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid, Environment } from "@react-three/drei";
import { useRef, useMemo } from "react";
import * as THREE from "three";

interface Props {
  engine: string;
  frame: SimulationFrame | null;
}

export function Scene3D({ engine, frame }: Props) {
  return (
    <Canvas camera={{ position: [3, 2, 3], fov: 50 }} className="bg-gray-900">
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={10}
      />

      <Grid
        infiniteGrid
        cellSize={0.5}
        cellThickness={0.5}
        sectionSize={2}
        sectionThickness={1}
        fadeDistance={30}
      />

      <GolferModel frame={frame} />
      <ClubTrajectory frame={frame} />

      <Environment preset="studio" />
    </Canvas>
  );
}

function GolferModel({ frame }: { frame: SimulationFrame | null }) {
  const groupRef = useRef<THREE.Group>(null);

  // Update pose from simulation frame
  useFrame(() => {
    if (!frame?.state || !groupRef.current) return;

    // Apply joint angles to skeleton
    // This would map simulation state to Three.js bone rotations
  });

  return (
    <group ref={groupRef}>
      {/* Simplified golfer visualization */}
      {/* In production, load GLTF model and apply skeleton */}
      <mesh position={[0, 1, 0]}>
        <capsuleGeometry args={[0.15, 0.6, 8, 16]} />
        <meshStandardMaterial color="#4a90d9" />
      </mesh>
      {/* Add more body parts... */}
    </group>
  );
}

function ClubTrajectory({ frame }: { frame: SimulationFrame | null }) {
  // Render club head trajectory trail
  const points = useMemo(() => {
    // Build trail from recent frames
    return [];
  }, [frame]);

  if (points.length < 2) return null;

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length}
          array={new Float32Array(points.flat())}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color="#ffcc00" linewidth={2} />
    </line>
  );
}
```

### 2.5 Build & Bundle Configuration

**File: `ui/vite.config.ts`**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },

  server: {
    port: 3000,
    proxy: {
      // Proxy API requests to local backend during development
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
    },
  },

  build: {
    outDir: "dist",
    sourcemap: false,
    minify: "terser",
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three", "@react-three/fiber", "@react-three/drei"],
          react: ["react", "react-dom"],
          charts: ["recharts"],
        },
      },
    },
  },
});
```

**File: `ui/package.json`**

```json
{
  "name": "golf-suite-ui",
  "version": "2.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@tanstack/react-query": "^5.8.0",
    "zustand": "^4.4.7",
    "three": "^0.159.0",
    "@react-three/fiber": "^8.15.0",
    "@react-three/drei": "^9.88.0",
    "recharts": "^2.10.0",
    "react-hook-form": "^7.48.0",
    "zod": "^3.22.4",
    "@hookform/resolvers": "^3.3.2",
    "clsx": "^2.0.0",
    "lucide-react": "^0.294.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.38",
    "@types/react-dom": "^18.2.17",
    "@types/three": "^0.159.0",
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.16",
    "eslint": "^8.54.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "postcss": "^8.4.31",
    "tailwindcss": "^3.3.5",
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
```

---
