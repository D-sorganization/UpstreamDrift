# UpstreamDrift Deployment and Installable Artifacts

This directory documents the deployment targets and installable artifacts across UpstreamDrift, including container images, desktop application bundles, Windows installers, and compiled Rust extension wheels.

## Container Image and Docker Compose

UpstreamDrift provides containerized environments for headless simulation, API hosting, and batch physics evaluation.

- **Dockerfile**: Located at `Dockerfile` at the repository root.
- **Docker Compose Stack**: Located at `docker/docker-compose.yml` (and root `docker-compose.yml`) for multi-container orchestration and profile-based setups.
- **Published Image**: Available from the GitHub Container Registry:
  ```text
  ghcr.io/d-sorganization/upstreamdrift
  ```
- **Running Container**:
  ```bash
  docker compose -f docker/docker-compose.yml up
  ```

## Desktop Bundles and Windows Installer

UpstreamDrift ships desktop graphical applications via Tauri and standalone Windows installers:

- **Tauri Desktop Bundle**: Source and configuration live in `ui/src-tauri`. Builds native desktop bundles that wrap the frontend and interface with the simulation backend.
- **Windows Installer Builder**: Implemented in `installer/windows/build_installer.py`. Packages the standalone application, required dependencies, and desktop shortcuts into an installable Windows package.
- **Building the Installer**:
  ```bash
  python installer/windows/build_installer.py
  ```

## Rust Extension Wheel Bundle

Performance-critical kernels are implemented in Rust under `rust_core/` and packaged as Maturin extension wheels:

1. `upstream-physics`
2. `upstream-mocap-preproc`
3. `upstream-mocap-io`
4. `upstream-muscle`
5. `upstream-motion-matching`
6. `upstream-pinocchio-id`
7. `upstream-urdf`
8. `upstream-mesh`
9. `upstream-realtime`
10. `ai-backend`

### Installing Rust Extensions

You can install all 10 Rust extension wheels simultaneously using the `rust` optional extra or pre-built wheels:

- **Via Pip Extra**:
  ```bash
  pip install -e ".[rust]"
  ```
- **Via Pre-built Wheels**:
  ```bash
  pip install target/wheels/*.whl
  ```
