# Comprehensive Dockerfile for Golf Modeling Suite
# Unifies Robotics (MuJoCo, Drake, Pinocchio) and Biomechanics (OpenSim, MyoSim)

# Stage 1: Builder stage with full development tools
FROM continuumio/miniconda3:24.11.1-0 AS builder

# System dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
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
    ezc3d \
    && conda clean --all --yes

# Install Pinocchio ecosystem via conda-forge (recommended for better compatibility)
RUN conda install -y -c conda-forge \
    pinocchio \
    crocoddyl \
    && conda clean --all --yes

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies from requirements.txt
# Filter out comments, WSL/Linux notes, and optional packages
RUN grep -v '^#' /tmp/requirements.txt | grep -v '^$' | grep -v 'robot_descriptions' > /tmp/filtered_requirements.txt && \
    pip install --no-cache-dir -r /tmp/filtered_requirements.txt

# Install additional physics engines and API server dependencies
RUN pip install --no-cache-dir \
    mujoco>=3.2.3 \
    drake \
    meshcat \
    casadi \
    pin-pink \
    qpsolvers \
    osqp \
    myosuite \
    opensim \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    slowapi \
    pydantic \
    python-multipart \
    sqlalchemy \
    email-validator \
    bcrypt \
    python-jose[cryptography] \
    passlib \
    PyJWT \
    aiofiles \
    python-dateutil \
    websockets \
    && echo "Physics engines and API dependencies installed successfully"


# Stage 2: Runtime stage with minimal footprint
FROM continuumio/miniconda3:24.11.1-0 AS runtime

# Runtime system dependencies only
# - GL libraries for MuJoCo/Visualization
# - FFmpeg for video processing (OpenPose inputs)
# - curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    patchelf \
    ffmpeg \
    xvfb \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
ARG USER_NAME=golfer
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME}

# Copy conda environment from builder
COPY --from=builder /opt/conda /opt/conda

# Set up Python path for shared modules
ENV PYTHONPATH="/workspace:/workspace/shared/python:/workspace/engines"
ENV PATH="/opt/conda/bin:$PATH"

# Create workspace directory structure with proper ownership
RUN mkdir -p /workspace/shared/python /workspace/engines && \
    chown -R ${USER_NAME}:${USER_NAME} /workspace

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER ${USER_NAME}

# Expose default port (if running web server)
EXPOSE 8000

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["/bin/bash"]
