# Comprehensive Dockerfile for Golf Modeling Suite
# Unifies Robotics (MuJoCo, Drake, Pinocchio) and Biomechanics (OpenSim, MyoSim)

# Stage 1: Builder stage with full development tools
FROM continuumio/miniconda3:24.11.1-0 AS builder

ENV DEBIAN_FRONTEND=noninteractive

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
    python=3.12 \
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
# Filter out comments, WSL/Linux notes, and blank lines
RUN grep -v '^#' /tmp/requirements.txt | grep -v '^$' > /tmp/filtered_requirements.txt && \
    pip install --no-cache-dir -r /tmp/filtered_requirements.txt

# Install additional physics engines and API server dependencies
# Note: opensim is excluded because it is not reliably pip-installable;
#       install it via conda or from source if needed.
RUN pip install --no-cache-dir \
    mujoco>=3.2.3 \
    drake \
    meshcat \
    pin-pink \
    qpsolvers \
    osqp \
    myosuite \
    gymnasium>=0.29.0 \
    stable-baselines3>=2.0.0 \
    mediapipe>=0.10.0 \
    "imageio[ffmpeg]>=2.31.0" \
    trimesh>=4.0.0 \
    robot_descriptions>=1.12.0 \
    fastapi>=0.126.0 \
    "uvicorn[standard]>=0.24.0" \
    slowapi>=0.1.9 \
    "pydantic[email]>=2.5.0" \
    python-multipart \
    sqlalchemy>=2.0.0 \
    bcrypt>=4.1.0 \
    "PyJWT>=2.10.1" \
    "cryptography>=44.0.1" \
    httpx>=0.25.0 \
    aiofiles \
    python-dateutil \
    websockets \
    simpleeval>=0.9.13 \
    structlog>=24.1.0 \
    colorama>=0.4.6 \
    && echo "Physics engines and API dependencies installed successfully"


# Stage 2: Runtime stage with minimal footprint
FROM continuumio/miniconda3:24.11.1-0 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system dependencies only
# - GL libraries for MuJoCo/Visualization
# - X11/XCB libraries for PyQt6
# - FFmpeg for video processing (OpenPose inputs)
# - curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    libegl1 \
    libglib2.0-0 \
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
    libdbus-1-3 \
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
# /workspace is the project root (src/ lives here), enabling "from src.xxx" imports
ENV PYTHONPATH="/workspace"
ENV PATH="/opt/conda/bin:$PATH"

# Create workspace directory structure with proper ownership
RUN mkdir -p /workspace && \
    chown -R ${USER_NAME}:${USER_NAME} /workspace

# Set working directory
WORKDIR /workspace

# Copy application source code and configuration
COPY --chown=${USER_NAME}:${USER_NAME} src/ ./src/
COPY --chown=${USER_NAME}:${USER_NAME} pyproject.toml ./
COPY --chown=${USER_NAME}:${USER_NAME} setup.py ./
COPY --chown=${USER_NAME}:${USER_NAME} launch_golf_suite.py ./
COPY --chown=${USER_NAME}:${USER_NAME} start_api_server.py ./
COPY --chown=${USER_NAME}:${USER_NAME} conftest.py ./
COPY --chown=${USER_NAME}:${USER_NAME} build_hooks.py ./
COPY --chown=${USER_NAME}:${USER_NAME} .env.example ./.env.example

# Switch to non-root user
USER ${USER_NAME}

# Expose default port (if running web server)
EXPOSE 8000

# Health check for container monitoring
# The core routes register /health on the FastAPI app (src/api/routes/core.py)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/bin/bash"]
