# Golf Modeling Suite - Docker Setup

## Quick Start

The Golf Modeling Suite runs in a Docker container with all physics engines pre-installed (MuJoCo, Drake, Pinocchio, OpenSim).

### Prerequisites
- Docker installed and running
- Docker Compose (usually included with Docker Desktop)

### Running the Suite

**Option 1: Full Stack (Recommended)**
```bash
# Build and start everything
docker-compose up --build

# Access:
# - Frontend UI: http://localhost:5180
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

**Option 2: Backend Only** (use local frontend for development)
```bash
# Start just the backend
docker-compose up backend

# Then in another terminal, run frontend locally:
cd ui && npm run dev
```

**Option 3: Development Mode** (live code changes)
```bash
# Start with volume mounts for live editing:
docker-compose up

# Edit code locally - changes are reflected immediately
# Backend has --reload flag enabled
```

### Installed Physics Engines

The Docker image includes:

| Engine | Version | Status | Capabilities |
|--------|---------|--------|--------------|
| **MuJoCo** | 3.2.3+ | ✅ Ready | Contact-rich simulation, muscle/tendon models |
| **Drake** | Latest | ✅ Ready | Optimization, trajectory planning, multibody dynamics |
| **Pinocchio** | Latest | ✅ Ready | Rigid body kinematics/dynamics, fast algorithms |
| **OpenSim** | 4.x | | ⚙️  Planned | Musculoskeletal simulation, biomechanics |
| **MyoSuite** | Latest | ⚙️ Planned | Muscle-tendon control, neural activation |

### Managing the Environment

```bash
# Stop all services
docker-compose down

# Rebuild after Dockerfile changes
docker-compose build

# View logs
docker-compose logs -f backend

# Shell into container
docker-compose exec backend bash

# Test physics engines manually
docker-compose exec backend python -c "import mujoco; print(mujoco.__version__)"
docker-compose exec backend python -c "import pydrake; print('Drake OK')"
docker-compose exec backend python -c "import pinocchio; print(pinocchio.__version__)"
```

### Troubleshooting

**Port already in use:**
```bash
# Kill existing processes
pkill -f uvicorn
pkill -f "npm.*dev"

# Or change ports in docker-compose.yml
```

**Build failures:**
```bash
# Clear Docker cache and rebuild
docker system prune -a
docker-compose build --no-cache
```

**Permission issues:**
```bash
# Fix file ownership (Linux)
docker-compose exec backend chown -R $(id -u):$(id -g) /workspace
```

**GPU support (for visualization):**
```yaml
# Add to backend service in docker-compose.yml:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Architecture

```
┌─────────────────────────────────────┐
│   Browser (localhost:5180)          │
│   ┌──────────┐                      │
│   │    UI    │  React + Three.js    │
│   └────┬─────┘                      │
│        │ API calls                  │
└────────┼─────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Docker Container (localhost:8000)  │
│  ┌────────────────────────────────┐ │
│  │  FastAPI Backend               │ │
│  │  ├─ Engine Manager             │ │
│  │  ├─ Simulation Service         │ │
│  │  └─ Analysis Service           │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │  Physics Engines               │ │
│  │  ├─ MuJoCo ✅                  │ │
│  │  ├─ Drake ✅                   │ │
│  │  ├─ Pinocchio ✅               │ │
│  │  ├─ OpenSim (planned)          │ │
│  │  └─ MyoSuite (planned)         │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │  Environments                  │ │
│  │  ├─ Putting Green              │ │
│  │  ├─ Golf Course (planned)      │ │
│  │  └─ Driving Range (planned)    │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Development Workflow

1. **Make code changes** locally in your editor
2. **Changes auto-reload** in Docker container (backend has `--reload`)
3. **Frontend** rebuilds via Vite HMR
4. **Test** at http://localhost:5180

### Production Deployment

```bash
# Build production image
docker build -t golf-suite:v1.0 -f Dockerfile .

# Run production container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/workspace/data \
  --name golf-suite-prod \
  golf-suite:v1.0 \
  python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Next Steps

See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the roadmap to:
1. Register Putting Green engine properly
2. Create Golf Course environment
3. Full biomechanical simulation pipeline
4. Add MyoSuite and advanced features
