# Unified Robotics Development Environment
# Supports: Drake, Pinocchio, MuJoCo
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
# Includes tools for building, math libraries, and GUI support
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    # Mathematics & Physics
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    liboctomap-dev \
    libassimp-dev \
    libccd-dev \
    libfcl-dev \
    # Logging
    libspdlog-dev \
    libfmt-dev \
    # GUI & Rendering
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    x11-apps \
    # DeepMind Control Suite Rendering
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    ffmpeg \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# 2. Miniforge (Conda)
# Useful for proprietary or complex robotics binaries (like gepetto-viewer)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniforge3-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

# 3. Dedicated Conda Environment for Visualization
# Gepetto Viewer is complicated to build from source, easier via Conda
RUN conda create -y -n robot-viz -c conda-forge python=3.10 gepetto-viewer gepetto-viewer-corba pinocchio \
    && conda clean -afy

# 4. Main Python Environment (VirtualEnv)
# We use a venv to avoid conflicts with system python or base conda.
# This will be the default environment for your code.
ENV VIRTUAL_ENV=/opt/robotics_env
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 5. Application Dependencies
WORKDIR /setup
COPY requirements.txt .
COPY setup_env.sh .

COPY inspect_geoms.py .
COPY example_dynamic_stance.py .
COPY dump_names.py .

COPY measure_height.py .
COPY src /workspace/src

# Run the setup script to install pip dependencies
RUN sed -i 's/\r$//' setup_env.sh && chmod +x setup_env.sh && ./setup_env.sh

# 6. Workspace Setup
WORKDIR /workspace

# 7. Environment Variables
# Add workspace to python path so local modules are importable
ENV MUJOCO_GL="osmesa"
ENV PYOPENGL_PLATFORM="osmesa"
ENV PYTHONPATH="/workspace"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 8. Interactive Shell
CMD ["/bin/bash"]
