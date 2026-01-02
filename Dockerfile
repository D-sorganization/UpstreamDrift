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
    pkg-config \
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    liboctomap-dev \
    libassimp-dev \
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
    pyqt6 \
    opencv \
    pyyaml \
    defusedxml \
    h5py \
    scikit-learn \
    pillow \
    && conda clean --all --yes

# Install Pinocchio ecosystem via conda-forge (recommended for better compatibility)
RUN conda install -y -c conda-forge \
    pinocchio \
    crocoddyl \
    && conda clean --all --yes

# Install physics engines and additional packages via pip
RUN pip install --no-cache-dir \
    mujoco>=3.2.3 \
    drake \
    meshcat \
    casadi \
    pin-pink \
    qpsolvers \
    osqp \
    ezc3d>=1.4.0 \
    && echo "Physics engines and robotics packages installed successfully"

# Verify ezc3d installation for C3D file reading
RUN python -c "import ezc3d; print(f'ezc3d version: {ezc3d.__version__}')" || \
    (echo "ezc3d installation failed, attempting alternative installation..." && \
     conda install -y -c conda-forge ezc3d && \
     conda clean --all --yes)

# Set up Python path for shared modules
ENV PYTHONPATH="/workspace:/workspace/shared/python:/workspace/engines"

# Create workspace directory structure
RUN mkdir -p /workspace/shared/python /workspace/engines

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
