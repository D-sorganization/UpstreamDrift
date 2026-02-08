# Drake Golf Model - Docker Development Environment
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    liboctomap-dev \
    libassimp-dev \
    libspdlog-dev \
    libfmt-dev \
    libccd-dev \
    libfcl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash developer

# Install Miniforge for conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniforge3-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment for visualization (Gepetto Viewer)
RUN conda create -y -n drake-viz python=3.10 \
    -c conda-forge gepetto-viewer gepetto-viewer-corba pinocchio \
    && conda clean -afy

# Create Python venv for modeling stack
RUN python3.10 -m venv /opt/modeling && \
    /opt/modeling/bin/pip install --upgrade pip setuptools wheel

# Install modeling dependencies
RUN /opt/modeling/bin/pip install \
    numpy==2.1.0 \
    pandas==2.2.3 \
    matplotlib==3.9.1 \
    scipy==1.14.0 \
    pin>=2.6.0 \
    pin-pink \
    drake \
    qpsolvers \
    osqp \
    scs \
    quadprog \
    pytest==8.2.1 \
    pytest-cov==5.0.0 \
    pytest-mock==3.14.0 \
    ruff==0.5.0 \
    black==24.4.2 \
    mypy==1.10.0 \
    pre-commit==4.0.0 \
    pyyaml==6.0.1 \
    pathlib2==2.3.7 \
    types-PyYAML==6.0.12.12 \
    "defusedxml>=0.7.1" \
    structlog>=24.0.0 \
    colorama>=0.4.6 \
    simpleeval>=0.9.13 \
    sympy>=1.12 \
    PyQt6>=6.6.0 \
    opencv-python>=4.8.0 \
    mediapipe>=0.10.0 \
    "imageio[ffmpeg]>=2.31.0" \
    gymnasium>=0.29.0 \
    trimesh>=4.0.0 \
    robot_descriptions>=1.12.0 \
    ezc3d>=1.4.0 \
    fastapi>=0.126.0 \
    "uvicorn[standard]>=0.24.0" \
    pydantic>=2.5.0 \
    httpx>=0.25.0

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER developer

# Default command (can be overridden)
CMD ["/bin/bash"]


