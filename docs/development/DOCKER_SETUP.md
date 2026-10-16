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

| Service  | URL                        |
| -------- | -------------------------- |
| Frontend | http://localhost:5180      |
| Backend  | http://localhost:8001      |
| API Docs | http://localhost:8001/docs |

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

## Model Explorer in Docker (ControlTower)

The `explorer-*` compose services run the PyQt6 Model Explorer and the MP4
exporter inside one image built from `Dockerfile.modular` (`PROFILE=full`:
MuJoCo, Pinocchio, Drake, OpenSim, MyoSuite). On Linux the engines that are
stubs on native Windows (Pinocchio, Drake) are real pip wheels, so this is the
preferred place to exercise all of them.

All three services share the `explorer-base` build stage; they differ only in
default environment (`QT_QPA_PLATFORM`, `MUJOCO_GL`, `DISPLAY`). ControlTower
runs a Docker daemon inside the `ControlTower-Runner` WSL2 distro (not Docker
Desktop), so run `docker compose` **from that distro**: bind-mount paths are
daemon-side Linux paths.

```bash
wsl -d ControlTower-Runner
cd /mnt/c/Users/diete/Repositories/UpstreamDrift   # or any checkout
```

### One Command

```bash
scripts/docker/explorer.sh headless   # MP4 -> ./output/explorer/explorer_headless.mp4
scripts/docker/explorer.sh gui        # noVNC at http://localhost:6080/vnc.html
scripts/docker/explorer.sh gui-wslg   # native window on the Windows desktop
scripts/docker/explorer.sh engines    # engine status table (JSON)
```

The image build verifies the pinned Tools tree against two build args. The
wrapper computes them (`TOOLS_GITLINK_SHA` from the `vendor/ud-tools` gitlink,
`TOOLS_SOURCE_SHA256` from `scripts/packaging/pinned_tools_provenance.py`) and
then runs the `docker compose` commands below. If you call `docker compose`
directly, export both variables first or the build stops with
`Pinned Tools container provenance is invalid`. A worktree created by Windows
git has a `.git` pointer WSL git cannot follow; export `TOOLS_GITLINK_SHA`
yourself in that case.

### Headless Export (No Display)

```bash
docker compose --profile explorer-headless run --rm explorer-headless
# -> ./output/explorer/explorer_headless.mp4  (summary JSON on stdout)
```

Defaults: the built-in 25-DOF canonical humanoid, a 2 s passive simulation,
1280x720 at 30 fps. Override by appending arguments:

```bash
docker compose --profile explorer-headless run --rm explorer-headless \
  export --model docs/development/full_body_models/evidence/ground_support/anthro_driver/mjx_package.xml \
  --out /out/anthro_driver.mp4 --duration 3 --fps 30
# replay a results file instead of simulating (npz with "qpos", npy or csv):
#   --results /out/my_run.npz
```

The exporter re-opens the written MP4 and fails unless the decoded frame count
equals the rendered frame count. Other one-shot modes:

```bash
docker compose --profile explorer-headless run --rm explorer-headless engines
docker compose --profile explorer-headless run --rm explorer-headless probe-gl
docker compose --profile explorer-headless run --rm explorer-headless \
  app-screenshot --out /out/explorer.png
```

`engines` prints a JSON status table (ok / missing / broken) with a tiny
functional smoke per engine; `app-screenshot` builds the real Qt window under
`QT_QPA_PLATFORM=offscreen` and saves a PNG of it.

### GUI in the Browser (noVNC, Works From Any Machine)

```bash
docker compose --profile explorer-gui up explorer-gui
# open http://localhost:6080/vnc.html?autoconnect=1&resize=scale
```

Xvfb, openbox, x11vnc and noVNC start inside the container and the Model
Explorer opens in that virtual desktop. The port binds to `127.0.0.1` only
because the VNC session has no password. Closing the app window stops the
container.

### GUI as a Native Windows Window (WSLg)

```bash
docker compose --profile explorer-gui-wslg up explorer-gui-wslg
```

This mounts `/tmp/.X11-unix` and `/mnt/wslg` from the WSL distro, so the app
appears on the Windows desktop like any WSLg program. It needs WSLg (Windows 11
or recent Windows 10) and a daemon running inside WSL2.

### Environment

| Variable                   | Default                    | Meaning                                      |
| -------------------------- | -------------------------- | -------------------------------------------- |
| `MUJOCO_GL`                | `egl`                      | `egl` or `osmesa`; anything else is rejected |
| `QT_QPA_PLATFORM`          | `offscreen` / `xcb`        | set per service; `xcb` for the GUI services  |
| `DISPLAY`                  | `:99` (noVNC), `:0` (WSLg) | X display the GUI renders to                 |
| `UD_EXPLORER_DISPLAY_MODE` | `vnc`                      | `host` uses the mounted host X11 socket      |
| `VNC_GEOMETRY`             | `1600x900x24`              | virtual screen size for noVNC                |
| `EXPLORER_PROFILE`         | `full`                     | build profile from `docker/profiles.yaml`    |
| `UD_GUI_APP`               | model explorer launcher    | python module the `gui` mode runs            |

### Verified in the Container

Verified on ControlTower (2026-10-16) with `explorer.sh engines`: MuJoCo 3.8.0,
Pinocchio 4.1.0, Drake 1.58.0, OpenSim 4.6, MyoSuite 3.0.0 and PyQt6 (Qt 6.11.0)
all report `ok`. The headless export writes a 60-frame 1280x720 MP4 whose
decoded frame count matches, and the noVNC and WSLg services both show the
"Model Explorer" window. The export raises a clear error instead of hanging
when MuJoCo resets an unstable passive simulation (for example the anthro
driver model); pass `--results` to replay a trajectory for such models.

### Image Retention and Health Check

Every explorer image carries the label `keep=true` (Dockerfile `LABEL` plus
compose `labels:`). The host prune jobs (`runner-cleanup --disk-guard` from
Runner_Dashboard and the host-only `docker-cache-prune`) skip images with that
label (`--filter label!=keep`). Four images are required, and the compose
services build them on demand, so a pruned image is rebuilt rather than pulled:

| Image                              | Profile             |
| ---------------------------------- | ------------------- |
| `upstream-drift:explorer-headless` | `explorer-headless` |
| `upstream-drift:explorer-gui`      | `explorer-gui`      |
| `ud-sim:latest`                    | `explorer-images`   |
| `upstream-drift:full`              | `explorer-images`   |

`scripts/docker/explorer.sh` runs `docker image inspect` on the image it is
about to use and rebuilds it when missing. `scripts/docker/explorer.sh ensure`
checks all four (use it before `ct-run.sh` batches), and warns when an image
lacks the `keep` label.

### Troubleshooting

- `pull access denied` for `ud-sim` or `upstream-drift:full`: the image was
  pruned. Run `scripts/docker/explorer.sh ensure` to rebuild it.
- `ERROR: MUJOCO_GL ...` or `no display ...`: the entrypoint refuses to start
  without a render backend or display. Use the compose service rather than a
  bare `docker run`, or pass `-e MUJOCO_GL=egl -e QT_QPA_PLATFORM=offscreen`.
- `GL probe failed` with `egl`: retry with `MUJOCO_GL=osmesa docker compose ...`
  (pure software, slower). On a GPU host add the overlay in
  `docker-compose.gpu.yml`.
- `host X11 socket ... is not mounted`: the daemon is not in the WSL distro that
  owns WSLg. Use the noVNC service instead.
- Windows Git Bash rewrites `/path` arguments to `C:/Program Files/Git/...`.
  Run compose from WSL, or prefix the command with `MSYS_NO_PATHCONV=1`.
- Disk: the image is about 5 GB. Remove only this image with
  `docker image rm upstream-drift:explorer-headless upstream-drift:explorer-gui`
  and never run `docker system prune` on a shared runner host.
- A first build downloads the profile wheels (several GB); later builds reuse
  the layer cache unless `requirements.lock` or `src/` changes.

---

## Environment Variables

| Variable       | Default               | Description                   |
| -------------- | --------------------- | ----------------------------- |
| `API_HOST`     | `0.0.0.0`             | Backend bind address          |
| `API_PORT`     | `8001`                | Backend port                  |
| `ENVIRONMENT`  | `development`         | `development` or `production` |
| `MUJOCO_GL`    | `egl`                 | MuJoCo rendering backend      |
| `DATABASE_URL` | `sqlite:///...`       | Database connection string    |
| `VITE_API_URL` | `http://backend:8001` | Frontend → backend URL        |

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
