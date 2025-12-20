# MuJoCo Golf Swing Model - Docker Development Environment

# ==========================================
# Stage: Base System Dependencies
# ==========================================
FROM ubuntu:22.04 AS base

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

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
    imageio>=2.31.0 \
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
    "defusedxml>=0.7.1"

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
# TODO: Drake installation is currently a placeholder.
# Users should add specific installation commands here (e.g. pip install drake)
# See https://drake.mit.edu/python_bindings.html
RUN echo "Drake stage is a placeholder"

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
    meshcat \
    qpsolvers

# Set working directory
WORKDIR /workspace

# Default command (can be overridden)
CMD ["/bin/bash"]
