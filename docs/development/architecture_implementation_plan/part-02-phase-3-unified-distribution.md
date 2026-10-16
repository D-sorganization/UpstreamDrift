# Golf Modeling Suite - Modern Architecture Implementation Plan (Part 2/2)

Part of the [Golf Modeling Suite - Modern Architecture Implementation Plan](../ARCHITECTURE_IMPLEMENTATION_PLAN.md) index.

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
Homepage = "https://github.com/dieterolson/UpstreamDrift"
Documentation = "https://golf-suite.readthedocs.io"
Repository = "https://github.com/dieterolson/UpstreamDrift"

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
version: "3.8"

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
      - "v*"

jobs:
  build-ui:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
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
          python-version: "3.11"

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

import { useCloud } from "../../stores/cloudStore";

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

| Metric                       | Target                       |
| ---------------------------- | ---------------------------- |
| **Time to first simulation** | < 5 minutes from install     |
| **Install success rate**     | > 95% on supported platforms |
| **No internet required**     | 100% of core features        |
| **Bundle size (UI)**         | < 5 MB gzipped               |
| **Docker image size**        | < 2 GB                       |
| **Startup time**             | < 3 seconds to UI            |

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
