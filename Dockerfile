# Comprehensive Dockerfile for Golf Modeling Suite
# Unifies Robotics (MuJoCo, Drake, Pinocchio) and Biomechanics (OpenSim, MyoSim)

FROM continuumio/miniconda3:latest

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

# Create comprehensive environment
# Install core scientific packages via conda
RUN conda install -y -c conda-forge \
    python=3.11 \
    numpy \
    scipy \
    matplotlib \
    pandas \
    sympy \
    pyqt \
    opencv \
    pyyaml \
    defusedxml \
    h5py \
    scikit-learn \
    pillow \
    && conda clean --all --yes

# Install physics engines via pip for better compatibility
RUN pip install --no-cache-dir \
    mujoco>=3.2.3 \
    drake \
    meshcat \
    casadi \
    && echo "Physics engines installed successfully"

# Set up Python path for shared modules
ENV PYTHONPATH="/workspace:/workspace/shared/python:/workspace/engines"

# Create workspace directory structure
RUN mkdir -p /workspace/shared/python /workspace/engines

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
# Full OpenPose build requires CUDA + CuDNN and extensive compilation time.
# For this container, we rely on 'opencv' for basic pose estimation fallback
# unless a pre-built 'pyopenpose' wheel is mounted or built in a separate stage.

WORKDIR /workspace

# Set Python Path
ENV PYTHONPATH="/workspace:/workspace/shared/python:/workspace/engines"

# Default command
CMD ["/bin/bash"]
