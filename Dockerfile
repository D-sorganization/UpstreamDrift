# Comprehensive Dockerfile for Golf Modeling Suite
# Unifies Robotics (MuJoCo, Drake, Pinocchio) and Biomechanics (OpenSim, MyoSim)

FROM mambaorg/micromamba:1.5.8

USER root

# System dependencies
# - GL libraries for MuJoCo/Visualization
# - Build tools for extensions
# - FFmpeg for video processing (OpenPose inputs)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    patchelf \
    ffmpeg \
    xvfb \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

USER micromamba

# Create comprehensive environment
# Channels:
# - conda-forge: Main scientific stack & Pinocchio
# - opensim-org: OpenSim binaries
RUN micromamba install -y -n base -c conda-forge -c opensim-org \
    python=3.11 \
    numpy \
    scipy>=1.13.1 \
    matplotlib \
    pandas \
    sympy \
    pyqt>=6.6.0 \
    mujoco>=3.2.3 \
    pinocchio \
    casadi>=3.6.3 \
    opensim>=4.4.0 \
    opencv \
    pyyaml \
    defusedxml \
    h5py \
    scikit-learn \
    pillow \
    && micromamba clean --all --yes

# Activate environment for subsequent commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install Pip-only packages
# - Drake (often better via pip/wheel for specific versions)
# - MyoSim (if pip available, else source install logic below)
# - Dev tools
RUN pip install --no-cache-dir \
    drake>=1.22.0 \
    myosim>=2.4.2 \
    pre-commit \
    ruff \
    black \
    mypy \
    pytest \
    pytest-cov \
    pip-audit \
    shimmy \
    gymnasium

# OpenPose Installation Note:
# Full OpenPose build requires CUDA + CuDNN and extensive compilation time.
# For this container, we rely on 'opencv' for basic pose estimation fallback
# unless a pre-built 'pyopenpose' wheel is mounted or built in a separate stage.

WORKDIR /workspace

# Set Python Path
ENV PYTHONPATH="/workspace:/workspace/shared/python:/workspace/engines"

# Default command
CMD ["/bin/bash"]
