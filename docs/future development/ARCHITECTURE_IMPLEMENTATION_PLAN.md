# Golf Modeling Suite - Modern Architecture Implementation Plan

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
        from src.launchers.golf_launcher import main as classic_main
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

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Framework** | React 18 + TypeScript | Industry standard, huge ecosystem |
| **Build Tool** | Vite | Fast dev server, optimized builds |
| **Styling** | Tailwind CSS | Rapid development, consistent design |
| **State** | Zustand | Simple, performant, no boilerplate |
| **3D Viz** | Three.js + React Three Fiber | Powerful, can embed or replace Meshcat |
| **Charts** | Recharts or Visx | React-native charting |
| **Forms** | React Hook Form + Zod | Type-safe form validation |
| **API Client** | TanStack Query | Caching, real-time updates |

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

import { useState } from 'react';
import { useSimulation } from '../api/hooks';
import { EngineSelector } from '../components/simulation/EngineSelector';
import { SimulationControls } from '../components/simulation/SimulationControls';
import { ParameterPanel } from '../components/simulation/ParameterPanel';
import { Scene3D } from '../components/visualization/Scene3D';
import { LivePlot } from '../components/analysis/LivePlot';

export function SimulationPage() {
  const [selectedEngine, setSelectedEngine] = useState<string>('mujoco');
  const {
    isRunning,
    currentFrame,
    start,
    stop,
    pause
  } = useSimulation(selectedEngine);

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
          <Scene3D
            engine={selectedEngine}
            frame={currentFrame}
          />

          {/* Overlay: Status */}
          <div className="absolute top-4 left-4 bg-black/50 px-3 py-1 rounded">
            <span className="text-green-400">
              {isRunning ? `Frame ${currentFrame?.frame}` : 'Ready'}
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

import { useState, useCallback, useRef, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';

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
  const [currentFrame, setCurrentFrame] = useState<SimulationFrame | null>(null);
  const [frames, setFrames] = useState<SimulationFrame[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const start = useCallback((config: SimulationConfig = {}) => {
    // Connect to local WebSocket (no auth needed)
    const ws = new WebSocket(`ws://localhost:8000/api/ws/simulate/${engineType}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsRunning(true);
      setFrames([]);
      ws.send(JSON.stringify({
        action: 'start',
        config: {
          duration: 3.0,
          timestep: 0.002,
          live_analysis: true,
          ...config,
        },
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.status === 'complete') {
        setIsRunning(false);
        return;
      }

      if (data.frame !== undefined) {
        setCurrentFrame(data);
        setFrames(prev => [...prev, data]);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsRunning(false);
    };

    ws.onclose = () => {
      setIsRunning(false);
    };
  }, [engineType]);

  const stop = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'stop' }));
  }, []);

  const pause = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'pause' }));
    setIsPaused(true);
  }, []);

  const resume = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'resume' }));
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

import { useQuery } from '@tanstack/react-query';
import { fetchEngines } from '../../api/client';

interface Props {
  value: string;
  onChange: (engine: string) => void;
  disabled?: boolean;
}

export function EngineSelector({ value, onChange, disabled }: Props) {
  const { data: engines, isLoading } = useQuery({
    queryKey: ['engines'],
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
              ${value === engine.name
                ? 'border-blue-500 bg-blue-500/20 text-white'
                : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
              }
              ${!engine.available && 'opacity-50 cursor-not-allowed'}
              ${disabled && 'cursor-not-allowed'}
            `}
          >
            <div className="font-medium">{engine.name}</div>
            <div className="text-xs text-gray-400">
              {engine.available ? (
                engine.loaded ? '● Loaded' : '○ Available'
              ) : (
                '✗ Not installed'
              )}
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

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import { useRef, useMemo } from 'react';
import * as THREE from 'three';

interface Props {
  engine: string;
  frame: SimulationFrame | null;
}

export function Scene3D({ engine, frame }: Props) {
  return (
    <Canvas
      camera={{ position: [3, 2, 3], fov: 50 }}
      className="bg-gray-900"
    >
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
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },

  server: {
    port: 3000,
    proxy: {
      // Proxy API requests to local backend during development
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/api/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          'three': ['three', '@react-three/fiber', '@react-three/drei'],
          'react': ['react', 'react-dom'],
          'charts': ['recharts'],
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

## Phase 3: Unified Distribution

**Duration: 2-3 weeks**
**Goal: One-command install that works everywhere**

### 3.1 Distribution Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTION CHANNELS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PRIMARY (Recommended for most users):                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  pipx install golf-suite                                │   │
│   │  → Downloads pre-built wheel                            │   │
│   │  → Installs in isolated environment                     │   │
│   │  → Adds 'golf-suite' command to PATH                    │   │
│   │  → Works on Windows, Mac, Linux                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ALTERNATIVE (For Docker users):                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  docker run -p 8000:8000 golfsuite/app                  │   │
│   │  → All engines pre-installed                            │   │
│   │  → Opens browser to localhost:8000                      │   │
│   │  → Zero dependency conflicts                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   DEVELOPER (Full source):                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  git clone ... && pip install -e ".[all]"               │   │
│   │  → Full development environment                         │   │
│   │  → All optional dependencies                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 PyPI Package with Bundled UI

**File: `pyproject.toml`** (updated build section)

```toml
[project]
name = "golf-suite"
version = "2.0.0"
description = "Biomechanical golf simulation and analysis suite"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
keywords = ["golf", "biomechanics", "simulation", "physics", "mujoco"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Physics",
]

dependencies = [
    # Core (always installed)
    "numpy>=1.26.4,<3.0.0",
    "scipy>=1.13.1",
    "fastapi>=0.126.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.5.0",
    "httpx>=0.27.0",

    # Default physics engine (lightweight, cross-platform)
    "mujoco>=3.3.0,<4.0.0",
]

[project.optional-dependencies]
# Additional physics engines
drake = ["drake>=1.22.0"]
pinocchio = ["pin>=2.6.0", "meshcat>=0.3.0"]
all-engines = ["golf-suite[drake,pinocchio]"]

# Analysis extras
analysis = ["opencv-python>=4.8.0", "scikit-learn>=1.3.0"]

# Development
dev = ["pytest>=8.0.0", "ruff>=0.1.0", "mypy>=1.8.0"]

# Everything
all = ["golf-suite[all-engines,analysis,dev]"]

[project.scripts]
golf-suite = "golf_suite.cli:main"

[project.urls]
Homepage = "https://github.com/D-sorganization/Golf_Modeling_Suite"
Documentation = "https://golf-suite.readthedocs.io"
Repository = "https://github.com/D-sorganization/Golf_Modeling_Suite"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
# Include pre-built UI in the wheel
include = [
    "src/golf_suite/**/*.py",
    "src/golf_suite/ui/dist/**/*",  # Bundled web UI
    "src/config/*.yaml",
]

[tool.hatch.build.hooks.custom]
# Build UI before packaging
path = "build_hooks.py"
```

**File: `build_hooks.py`**

```python
"""Custom build hooks to bundle UI into Python package."""

import subprocess
import shutil
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class UIBuildHook(BuildHookInterface):
    """Build the React UI and include it in the wheel."""

    def initialize(self, version, build_data):
        ui_dir = Path(self.root) / "ui"
        dist_dir = ui_dir / "dist"
        target_dir = Path(self.root) / "src" / "golf_suite" / "ui" / "dist"

        # Check if we need to build
        if not dist_dir.exists() or self.config.get("force_ui_build"):
            print("Building UI...")

            # Install dependencies
            subprocess.run(
                ["npm", "ci"],
                cwd=ui_dir,
                check=True
            )

            # Build production bundle
            subprocess.run(
                ["npm", "run", "build"],
                cwd=ui_dir,
                check=True
            )

        # Copy to package location
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(dist_dir, target_dir)

        print(f"UI bundled to {target_dir}")
```

### 3.3 Docker Distribution

**File: `Dockerfile.unified`**

```dockerfile
# Golf Modeling Suite - Unified Container
# Includes all engines, web UI, ready to run

# Stage 1: Build UI
FROM node:20-slim AS ui-builder

WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm ci

COPY ui/ ./
RUN npm run build


# Stage 2: Python runtime with all engines
FROM mambaorg/micromamba:1.5-jammy AS runtime

# Create non-root user
ARG MAMBA_USER=golfer
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000

USER root

# System dependencies for OpenGL and visualization
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libegl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

# Create conda environment with all engines
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Activate environment by default
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=golf-suite

WORKDIR /app

# Copy application code
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ ./src/
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml ./

# Copy pre-built UI
COPY --from=ui-builder --chown=$MAMBA_USER:$MAMBA_USER /app/ui/dist ./src/golf_suite/ui/dist

# Install package
RUN micromamba run -n $ENV_NAME pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run server
CMD ["micromamba", "run", "-n", "golf-suite", "golf-suite", "--no-browser"]
```

**File: `docker-compose.yml`**

```yaml
version: '3.8'

services:
  golf-suite:
    build:
      context: .
      dockerfile: Dockerfile.unified
    ports:
      - "8000:8000"
    volumes:
      # Persist user data
      - golf-data:/app/data
      # Optional: mount local models
      - ./models:/app/models:ro
    environment:
      - GOLF_SUITE_MODE=local
      - GOLF_AUTH_DISABLED=true
    restart: unless-stopped

volumes:
  golf-data:
```

### 3.4 One-Line Install Script

**File: `install.sh`**

```bash
#!/bin/bash
# Golf Modeling Suite - Quick Install Script
# Usage: curl -fsSL https://golf-suite.io/install.sh | bash

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Golf Modeling Suite - Installation Script             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected: $OS $ARCH"

# Check for Python 3.11+
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
        echo "✓ Python $PY_VERSION found"
    else
        echo "✗ Python 3.11+ required (found $PY_VERSION)"
        echo "  Please install Python 3.11 or newer"
        exit 1
    fi
else
    echo "✗ Python 3 not found"
    echo "  Please install Python 3.11 or newer"
    exit 1
fi

# Check for pipx (recommended) or pip
if command -v pipx &> /dev/null; then
    echo "✓ pipx found - using isolated installation"
    INSTALL_CMD="pipx install golf-suite"
elif command -v pip3 &> /dev/null; then
    echo "⚠ pipx not found - using pip (consider installing pipx)"
    INSTALL_CMD="pip3 install --user golf-suite"
else
    echo "✗ Neither pipx nor pip found"
    exit 1
fi

echo
echo "Installing Golf Modeling Suite..."
echo "  Command: $INSTALL_CMD"
echo

$INSTALL_CMD

echo
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Installation Complete!                     ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║                                                               ║"
echo "║   To start:   golf-suite                                      ║"
echo "║   Help:       golf-suite --help                               ║"
echo "║                                                               ║"
echo "║   The app will open in your browser at localhost:8000         ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
```

### 3.5 GitHub Releases with Pre-built Binaries

**File: `.github/workflows/release.yml`**

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-ui:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: ui/package-lock.json

      - name: Build UI
        run: |
          cd ui
          npm ci
          npm run build

      - name: Upload UI artifact
        uses: actions/upload-artifact@v4
        with:
          name: ui-dist
          path: ui/dist

  build-wheel:
    needs: build-ui
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download UI
        uses: actions/download-artifact@v4
        with:
          name: ui-dist
          path: src/golf_suite/ui/dist

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Build wheel
        run: |
          pip install build
          python -m build

      - name: Upload wheel
        uses: actions/upload-artifact@v4
        with:
          name: wheel
          path: dist/*.whl

  build-docker:
    needs: build-ui
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download UI
        uses: actions/download-artifact@v4
        with:
          name: ui-dist
          path: ui/dist

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.unified
          push: true
          tags: |
            golfsuite/app:${{ github.ref_name }}
            golfsuite/app:latest

  publish-pypi:
    needs: build-wheel
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - name: Download wheel
        uses: actions/download-artifact@v4
        with:
          name: wheel
          path: dist

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}

  create-release:
    needs: [build-wheel, build-docker]
    runs-on: ubuntu-latest
    steps:
      - name: Download wheel
        uses: actions/download-artifact@v4
        with:
          name: wheel
          path: dist

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*.whl
          generate_release_notes: true
```

---

## Phase 4: Optional Cloud Features

**Duration: 4-6 weeks (optional, can defer)**
**Goal: Add cloud conveniences without making them required**

### 4.1 Cloud Architecture (Optional Add-on)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUD FEATURES (OPTIONAL)                     │
│                                                                  │
│   Users who WANT cloud features can opt-in:                     │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  golf-suite cloud login                                 │   │
│   │  → Enables: Sharing, Sync, Remote Compute               │   │
│   │  → Free tier available                                  │   │
│   │  → All local features still work without login          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Cloud Services:                                               │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │  Share   │  │   Sync   │  │  Remote  │  │  Collab  │      │
│   │ Results  │  │ Settings │  │ Compute  │  │  Review  │      │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                                                  │
│   Pricing (example):                                            │
│   • Free: Share up to 10 simulations, 1GB storage              │
│   • Pro ($9/mo): Unlimited sharing, 50GB, priority compute     │
│   • Team ($29/mo): Collaboration, shared workspaces            │
│                                                                  │
│   IMPORTANT: Core simulation ALWAYS works locally, free.        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Cloud-Local Hybrid API

**File: `src/api/cloud_client.py`**

```python
"""
Optional cloud client for Golf Modeling Suite.

Cloud features are opt-in. The app works fully offline without this.
"""

import os
import httpx
from pathlib import Path
from typing import Optional

CLOUD_API_URL = "https://api.golf-suite.io"


class CloudClient:
    """Client for optional cloud features."""

    def __init__(self):
        self.token: Optional[str] = None
        self._load_cached_token()

    def _load_cached_token(self):
        """Load token from local cache if user previously logged in."""
        token_file = Path.home() / ".golf-suite" / "cloud_token"
        if token_file.exists():
            self.token = token_file.read_text().strip()

    @property
    def is_logged_in(self) -> bool:
        return self.token is not None

    async def login(self, email: str, password: str) -> bool:
        """
        Log in to cloud services (optional).

        This enables sharing, sync, and remote compute features.
        The app works fully without logging in.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CLOUD_API_URL}/auth/login",
                json={"email": email, "password": password},
            )

            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self._save_token()
                return True
            return False

    def _save_token(self):
        """Cache token locally for convenience."""
        token_dir = Path.home() / ".golf-suite"
        token_dir.mkdir(exist_ok=True)
        (token_dir / "cloud_token").write_text(self.token)

    def logout(self):
        """Log out and clear cached credentials."""
        self.token = None
        token_file = Path.home() / ".golf-suite" / "cloud_token"
        if token_file.exists():
            token_file.unlink()

    async def share_simulation(self, simulation_data: dict) -> Optional[str]:
        """
        Share a simulation result and get a shareable link.

        Returns None if not logged in (user must opt-in to cloud).
        """
        if not self.is_logged_in:
            return None

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CLOUD_API_URL}/simulations/share",
                headers={"Authorization": f"Bearer {self.token}"},
                json=simulation_data,
            )

            if response.status_code == 200:
                return response.json()["share_url"]
            return None

    async def sync_settings(self, settings: dict) -> bool:
        """Sync local settings to cloud (if logged in)."""
        if not self.is_logged_in:
            return False

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{CLOUD_API_URL}/users/settings",
                headers={"Authorization": f"Bearer {self.token}"},
                json=settings,
            )
            return response.status_code == 200


# Global instance (lazy initialized)
_cloud_client: Optional[CloudClient] = None


def get_cloud_client() -> CloudClient:
    """Get the cloud client instance."""
    global _cloud_client
    if _cloud_client is None:
        _cloud_client = CloudClient()
    return _cloud_client
```

### 4.3 UI Integration for Cloud Features

```tsx
// ui/src/components/cloud/CloudStatus.tsx

import { useCloud } from '../../stores/cloudStore';

export function CloudStatus() {
  const { isLoggedIn, user, login, logout } = useCloud();

  if (!isLoggedIn) {
    return (
      <div className="p-4 bg-gray-800 rounded-lg">
        <p className="text-gray-400 text-sm mb-2">
          Running locally. Cloud features available:
        </p>
        <ul className="text-gray-500 text-xs mb-3">
          <li>• Share simulations with a link</li>
          <li>• Sync settings across devices</li>
          <li>• Collaborate with team members</li>
        </ul>
        <button
          onClick={() => setShowLoginModal(true)}
          className="text-blue-400 text-sm hover:text-blue-300"
        >
          Sign in for cloud features →
        </button>
        <p className="text-gray-600 text-xs mt-2">
          (All core features work without signing in)
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 bg-gray-800 rounded-lg">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-white text-sm">{user.email}</p>
          <p className="text-gray-400 text-xs">Cloud connected</p>
        </div>
        <button
          onClick={logout}
          className="text-gray-400 text-xs hover:text-gray-300"
        >
          Sign out
        </button>
      </div>
    </div>
  );
}
```

---

## Implementation Timeline

```
Week 1-2:   Phase 1.1-1.3  - Local API server, auth bypass, route refactoring
Week 3:     Phase 1.4-1.5  - WebSocket support, entry point updates
Week 4-5:   Phase 2.1-2.3  - UI scaffold, core components
Week 6-7:   Phase 2.4      - 3D visualization, live charts
Week 8:     Phase 2.5      - Polish, testing, responsive design
Week 9-10:  Phase 3.1-3.3  - PyPI packaging, Docker image
Week 11:    Phase 3.4-3.5  - Install scripts, GitHub releases
Week 12+:   Phase 4        - Optional cloud features (can defer)
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| **Time to first simulation** | < 5 minutes from install |
| **Install success rate** | > 95% on supported platforms |
| **No internet required** | 100% of core features |
| **Bundle size (UI)** | < 5 MB gzipped |
| **Docker image size** | < 2 GB |
| **Startup time** | < 3 seconds to UI |

---

## Migration Path for Existing Users

1. **Existing PyQt6 users**: `--classic` flag launches familiar UI
2. **Existing API users**: API endpoints remain compatible
3. **Docker users**: New unified image, old images still work
4. **Scripts/automation**: CLI interface unchanged

---

## Summary

This plan delivers:

- **Free, local-first**: No API keys, accounts, or internet needed
- **Modern UX**: Web-based UI, runs in any browser
- **Easy install**: `pipx install golf-suite` or `docker run`
- **Future-proof**: Clean API separation enables any frontend
- **Cloud-optional**: Sharing/sync available but never required
- **Reversible**: Classic UI preserved, gradual migration
