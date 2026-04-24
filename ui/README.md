# UpstreamDrift UI

Web-based interface for the **Golf Modeling Suite** physics simulator. Built with React, TypeScript, and Vite, this UI provides simulation control, parameter tuning, 3D visualization, and advanced analysis tools for golf ball flight and club-head dynamics.

## What This Is

The UpstreamDrift UI is a modern web interface that connects to the backend Python/Rust physics engine via REST API. It enables:

- **Simulation Launcher** – Configure and execute physics simulations (driver, iron, putter)
- **Physics Parameter Editor** – Adjust ball properties, launch conditions, environmental factors
- **3D Visualization** – Real-time trajectory rendering using Three.js
- **Results Analysis** – Trajectory plots, metrics, landing dispersion patterns
- **Advanced Tools** – Model explorer, motion capture analysis, video playback, data analysis

## Quick Start

### Prerequisites

- **Node.js 18+** and npm/pnpm
- Backend API running at `http://localhost:8001` (or set via `VITE_API_URL`)

### Installation & Development

```bash
# Install dependencies
npm install

# Start dev server (http://localhost:5180 with HMR)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run tests
npm test
npm run test:run
npm run test:coverage

# Type-check
npm run type-check

# Lint
npm run lint
```

### Tauri Desktop Build

The UI can also run as a native desktop app via Tauri:

```bash
npm run tauri:dev       # Dev mode with Tauri window
npm run tauri:build     # Build distributable executable
```

## Directory Structure

```text
src/
├── api/                  # Backend client hooks (useSimulation, useEngineManager, etc.)
├── components/
│   ├── analysis/         # Charts, live plots, results panels
│   ├── simulation/        # Simulation controls, launch condition editors
│   ├── ui/              # Toast, diagnostics, help panels, reusable widgets
│   └── visualization/    # 3D rendering, trajectory display
├── pages/               # Route pages (Dashboard, Simulation, tools)
│   ├── Dashboard.tsx    # Landing page with quick links
│   ├── Simulation.tsx   # Main simulation control interface
│   ├── ModelExplorer.tsx # URDF model browser
│   ├── PuttingGreen.tsx # Specialized putter analysis
│   ├── VideoAnalyzer.tsx # Video-based ball tracking
│   ├── DataExplorer.tsx # Historical results and dataset analysis
│   └── MotionCapture.tsx # Mocap-based swing analysis
├── stores/              # Zustand state (UI state, simulation config)
├── integration/         # Test utilities
└── test/               # Test setup and utilities
```

## Key Features

### Simulation Engine Selection

- **MuJoCo** – Fast, stable physics with contact dynamics
- **Drake** – Symbolic manipulation and analysis
- **Pinocchio** – Rigid body dynamics and optimization
- **OpenSim** – Biomechanics-focused

### Physics Parameters

- Ball mass, diameter, aerodynamic coefficients
- Launch angle, velocity, spin rate, azimuth
- Environmental conditions: gravity, air density, wind
- Adjustable time step and simulation duration

### Visualization

- 3D trajectory with position/velocity vectors
- Live-updating plots (height, speed, distance over time)
- Landing zone heatmaps (dispersion patterns)
- Model morphology browser with URDF loading

### Analysis

- Carry distance, flight time, max height calculations
- Trajectory statistics (mean, std dev, percentiles)
- Comparative simulations (sensitivity analysis)
- CSV export for external analysis

## Backend API Integration

The UI proxies requests to the backend during development (see `vite.config.ts`):

```text
Development:  localhost:5180 → localhost:8001 (via /api proxy)
Production:   Direct API calls (configure VITE_API_URL)
WebSocket:    /api/ws for live simulation streaming
```

Key API endpoints used:

- `POST /api/simulate` – Run a simulation
- `GET /api/engines` – List available physics engines
- `GET /api/models` – Available URDF models
- `WS /api/ws` – Real-time simulation results

## Environment Variables

```bash
# .env or .env.local (not committed)
VITE_API_URL=http://localhost:8001    # Backend base URL
VITE_DEBUG=false                       # Enable debug logging
```

## Testing

Uses Vitest + React Testing Library:

```bash
# Run all tests in watch mode
npm test

# Run once (CI mode)
npm run test:run

# Coverage report
npm run test:coverage
```

Test files are colocated (e.g., `Component.test.tsx` next to `Component.tsx`).

## Code Quality

- **Formatter:** Prettier (configured in `package.json`)
- **Linter:** ESLint with React + TypeScript rules
- **Type Safety:** TypeScript strict mode
- **Styling:** Tailwind CSS + CSS modules

Run locally:

```bash
npm run lint
npm run type-check
```

## Technology Stack

| Layer       | Technology                        |
| ----------- | --------------------------------- |
| Framework   | React 19, React Router 7          |
| Language    | TypeScript 5.9                    |
| Build       | Vite 7.2                          |
| Styling     | Tailwind CSS 3.4, PostCSS         |
| 3D Graphics | Three.js, React Three Fiber       |
| Charts      | Recharts 3.7                      |
| State       | Zustand 5                         |
| HTTP        | React Query 5                     |
| Desktop     | Tauri 2                           |
| Testing     | Vitest 1.3, React Testing Library |
| Linting     | ESLint 9                          |

## Common Development Tasks

### Add a New Simulation Tool Page

1. Create `src/pages/MyTool.tsx` (export default component as `MyToolPage`)
2. Add route in `src/App.tsx`: `<Route path="/tools/my-tool" element={<MyToolPage />} />`
3. Link from Dashboard: `import { Link } from 'react-router-dom'`

### Use Backend API

```typescript
import { useSimulation } from '@/api/useSimulation';

function MyComponent() {
  const { simulate, result, isLoading, error } = useSimulation();

  const handleRun = async () => {
    await simulate({
      engineName: 'mujoco',
      ballProperties: { /* ... */ },
      launchConditions: { /* ... */ },
    });
  };

  return (
    <div>
      <button onClick={handleRun} disabled={isLoading}>Run</button>
      {result && <p>Carry: {result.carry_distance} m</p>}
      {error && <p>Error: {error}</p>}
    </div>
  );
}
```

### Debugging

Enable debug logging:

```bash
# Terminal
VITE_DEBUG=true npm run dev

# Browser console
localStorage.setItem('DEBUG', '*')
```

Diagnostics panel: Press `Ctrl+Shift+?` (or see Help panel).

## Troubleshooting

| Issue                    | Solution                                                    |
| ------------------------ | ----------------------------------------------------------- |
| API connection refused   | Ensure backend runs on 8001; check `vite.config.ts` proxy   |
| WebSocket failures       | Backend may not support WS; verify `/api/ws` endpoint       |
| 3D viz not rendering     | Check WebGL support; try Chrome/Firefox; ensure URDF loads  |
| Slow dev rebuild         | Try `npm run type-check` separately; check disk I/O         |
| Test failures on Windows | Use `npm run test:run` (not watch); check line endings (LF) |

## Contributing

- Follow UpstreamDrift's coding standards (DRY, Design by Contract, TDD)
- Keep components under 300 lines; split if needed
- Add tests for new features (aim for >80% coverage)
- No `console.log` in production code (use logging service)
- File PR against `staging` branch

See `CLAUDE.md` in the root repo for full project guidelines.
