# Docker Development Environment

The Golf Modeling Suite ships a complete Docker environment with **all physics
engines pre-installed** (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite).

---

## Quick Start

```bash
# Clone and start everything
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
docker-compose up --build
```

| Service   | URL                        |
|-----------|----------------------------|
| Frontend  | http://localhost:5180       |
| Backend   | http://localhost:8001       |
| API Docs  | http://localhost:8001/docs  |

---

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────┐
│  Frontend (Node 20)     │────▶│  Backend (Python 3.11)│
│  localhost:5180          │     │  localhost:8001        │
│  Vite + React + TS      │     │  FastAPI + Uvicorn     │
└─────────────────────────┘     │                        │
                                │  Engines:              │
                                │  ├─ MuJoCo  ≥3.2.3    │
                                │  ├─ Drake              │
                                │  ├─ Pinocchio          │
                                │  ├─ OpenSim            │
                                │  ├─ MyoSuite           │
                                │  └─ Putting Green      │
                                └──────────────────────────┘
```

---

## Detailed Usage

### Development (recommended)

Live code editing with hot-reload on both frontend and backend:

```bash
docker-compose up
```

Source files are volume-mounted, so edits are reflected immediately.

### Build Only (no compose)

```bash
# Build image
docker build -t golf-suite:latest .

# Run container
docker run -p 8001:8001 -v $(pwd):/workspace golf-suite:latest \
    python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001
```

### Hybrid Mode (Docker backend, local frontend)

```bash
# Start only the backend in Docker
docker-compose up backend

# In another terminal, run the frontend locally
cd ui && npm install && npm run dev
```

Set `VITE_API_URL=http://localhost:8001` in the frontend `.env` for this mode.

### Production Build

```bash
docker build -t golf-suite:prod --target runtime .
docker run -d -p 8001:8001 --name golf-suite-prod golf-suite:prod \
    python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001
```

---

## Platform Notes

### Linux

Works out of the box. For GPU passthrough (NVIDIA):

```bash
docker-compose up  # CPU rendering via EGL (default)

# For GPU rendering, add to docker-compose.yml:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - capabilities: [gpu]
```

### macOS

Docker Desktop ≥ 4.x required. Performance is good with the `delegated` volume
mount strategy (already configured).

```bash
# Install Docker Desktop from https://docker.com
docker-compose up
```

### Windows

Use Docker Desktop with WSL 2 backend:

1. Install [Docker Desktop](https://docker.com) with WSL 2 enabled
2. Clone the repo inside WSL 2 for best I/O performance
3. Run `docker-compose up` from WSL 2 terminal

---

## Verifying Engine Installation

After `docker-compose up`, verify engines:

```bash
# Check engine probes via API
curl http://localhost:8001/api/engines/mujoco/probe
curl http://localhost:8001/api/engines/drake/probe
curl http://localhost:8001/api/engines/pinocchio/probe
curl http://localhost:8001/api/engines/putting_green/probe

# Or use the /engines endpoint:
curl http://localhost:8001/engines | python3 -m json.tool

# Run tests inside the container
docker exec golf-suite-backend pytest tests/api/ -v
```

---

## Environment Variables

| Variable        | Default           | Description                   |
|-----------------|-------------------|-------------------------------|
| `API_HOST`      | `0.0.0.0`         | Backend bind address          |
| `API_PORT`      | `8001`            | Backend port                  |
| `ENVIRONMENT`   | `development`     | `development` or `production` |
| `MUJOCO_GL`     | `egl`             | MuJoCo rendering backend      |
| `DATABASE_URL`  | `sqlite:///...`   | Database connection string     |
| `VITE_API_URL`  | `http://backend:8001` | Frontend → backend URL    |

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs backend

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up
```

### Port conflicts

```bash
# Check what's using port 8001
lsof -i :8001

# Use a different port
API_PORT=9001 docker-compose up
```

### Engine import errors

```bash
# Shell into the container
docker exec -it golf-suite-backend bash

# Test imports manually
python3 -c "import mujoco; print(mujoco.__version__)"
python3 -c "import pydrake; print('Drake OK')"
python3 -c "import pinocchio; print(pinocchio.__version__)"
```

---

## File Structure

```
├── Dockerfile              # Multi-stage build (builder + runtime)
├── Dockerfile.unified      # Alternative single-stage build
├── docker-compose.yml      # Full-stack orchestration
├── .dockerignore           # Build context exclusions
└── docs/
    └── DOCKER_SETUP.md     # This file
```
