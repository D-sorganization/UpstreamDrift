# MuJoCo Golf Swing Model - Docker Development Environment

# ==========================================
# Stage: Base System Dependencies
# ==========================================
FROM ubuntu:22.04 AS base

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Never write to the pip HTTP cache during image builds.
#
# Two reasons:
#   1. The cache adds ~500-700 MB to the final image for files only
#      useful during the build itself (HTTP responses pip would re-use
#      across invocations on a long-lived workstation, not at runtime).
#   2. On WSL2 + containerd's overlayfs snapshotter, the millions of
#      tiny files pip writes under /root/.cache/pip cause `lchown ...
#      no such file or directory` failures during `exporting to image`:
#      the snapshot finalizer GCs files between layer commit and unpack,
#      and overlayfs then can't chown them. Disabling the cache sidesteps
#      that bug entirely. Builds also get slightly faster overall because
#      writing the cache is pure overhead in CI/single-shot contexts.
#
# PIP_DISABLE_PIP_VERSION_CHECK=1 suppresses the per-invocation network
# ping pip otherwise makes to PyPI to check for newer pip versions; it's
# noise in build logs and pointless in pinned-image contexts.
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
#
# Uses --no-install-recommends to skip apt's "Recommends" graph: the
# packages listed below are the explicit, authoritative dependency set
# for the engine images (matches the convention used by the repo root
# Dockerfile and resolves Trivy DS-0029). Anything that turns out to be
# implicitly required should be added to the explicit list rather than
# being pulled in via Recommends — that keeps the image self-documenting
# and the build deterministic across base-image revisions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    patchelf \
    x11-apps \
    libegl1 \
    libopengl0 \
    libdbus-1-3 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash --no-log-init developer

# Create Python venv for MuJoCo environment
RUN python3.11 -m venv /opt/mujoco-env && \
    /opt/mujoco-env/bin/pip install --upgrade pip setuptools wheel

# Activate virtual environment by default
ENV PATH="/opt/mujoco-env/bin:$PATH"
ENV VIRTUAL_ENV="/opt/mujoco-env"

# Install common Python dependencies
RUN pip install \
    numpy==1.26.4 \
    pandas==2.2.3 \
    "matplotlib>=3.8.0,<=3.9.1" \
    scipy==1.13.1 \
    PyQt6>=6.6.0 \
    opencv-python>=4.8.0 \
    imageio[ffmpeg]>=2.31.0 \
    scikit-learn>=1.3.0 \
    h5py>=3.9.0 \
    pillow>=10.0.0 \
    pytest==8.2.1 \
    pytest-cov==5.0.0 \
    pytest-mock==3.14.0 \
    ruff==0.5.0 \
    black==24.4.2 \
    mypy==1.10.0 \
    pre-commit==4.0.0 \
    pyyaml==6.0.1 \
    "pathlib2>=2.3.6,<=2.3.7" \
    types-PyYAML==6.0.12.12 \
    "defusedxml>=0.7.1" \
    structlog>=24.0.0 \
    colorama>=0.4.6 \
    simpleeval>=0.9.13 \
    sympy>=1.12 \
    mediapipe>=0.10.0 \
    gymnasium>=0.29.0 \
    stable-baselines3>=2.0.0 \
    trimesh>=4.0.0 \
    robot_descriptions>=1.12.0 \
    ezc3d>=1.4.0 \
    fastapi>=0.126.0 \
    "uvicorn[standard]>=0.24.0" \
    pydantic>=2.5.0 \
    httpx>=0.25.0

# ==========================================
# Stage: MuJoCo
# ==========================================
FROM base AS mujoco
# Install MuJoCo and DeepMind Robotics Tools
RUN pip install \
    mujoco>=3.0.0 \
    dm_control>=1.0.14 \
    jax[cuda12_pip]
# Optional: Additional MuJoCo playground utilities. Disabled by default because
# they significantly increase image build time and size and are not required for
# most workflows. To enable these tools, remove the '#' from the RUN command below.
# RUN pip install git+https://github.com/google-deepmind/mujoco_playground.git@v0.1.0

# ==========================================
# Stage: Pinocchio
# ==========================================
FROM base AS pinocchio
# Install Pinocchio and related advanced robotics/visualization tools
RUN pip install \
    pin>=2.6.0 \
    pin-pink \
    meshcat \
    qpsolvers

# ==========================================
# Stage: Drake
# ==========================================
FROM base AS drake
# Install Drake using pip (official method as of Drake 1.0+)
# See https://drake.mit.edu/python_bindings.html
# Pin version to ensure reproducible builds
RUN pip install drake>=1.0.0 meshcat>=0.3.0

# ==========================================
# Stage: All (Default)
# ==========================================
FROM base AS all
# Combine all dependencies from MuJoCo and Pinocchio stages
RUN pip install \
    mujoco>=3.0.0 \
    dm_control>=1.0.14 \
    jax[cuda12_pip] \
    pin>=2.6.0 \
    pin-pink \
    meshcat>=0.3.0 \
    qpsolvers \
    drake>=1.0.0

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER developer

# Default command (can be overridden)
CMD ["/bin/bash"]
