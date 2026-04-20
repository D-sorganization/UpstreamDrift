# Comprehensive Dockerfile for Golf Modeling Suite
# Unifies Robotics (MuJoCo, Drake, Pinocchio) and Biomechanics (OpenSim, MyoSim)

# Stage 1: Builder stage with full development tools
FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

# Fix Debian OpenSSL vulnerability and upgrade system packages (e.g. 3.5.5-1~deb13u2)
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to fix vulnerability (pip >= 25.3)
RUN pip install --no-cache-dir pip>=25.3 wheel setuptools

# Copy requirements file
COPY requirements.lock /tmp/requirements.txt

# Install Python dependencies from requirements.txt
RUN grep -v '^#' /tmp/requirements.txt | grep -v '^$' > /tmp/filtered_requirements.txt && \
    pip install --no-cache-dir -r /tmp/filtered_requirements.txt || true

# Install additional physics engines and API server dependencies
# We explicitly include runtime packages needed by API import paths: pandas, matplotlib, sympy, and defusedxml
RUN pip install --no-cache-dir \
    mujoco>=3.2.3 \
    drake \
    meshcat \
    pin-pink \
    qpsolvers \
    osqp \
    myosuite \
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
    simpleeval>=1.0.5 \
    structlog>=24.1.0 \
    colorama>=0.4.6 \
    pandas \
    matplotlib \
    sympy \
    defusedxml \
    numpy \
    scipy \
    && echo "Physics engines and API dependencies installed successfully"


# Stage 2: Runtime stage with minimal footprint (target budget ~4,000 MB)
FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade openssl to fix Debian vulnerabilities
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
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

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up Python path for shared modules
ENV PYTHONPATH="/workspace"
ENV PATH="/opt/venv/bin:$PATH"

# Create workspace directory structure with proper ownership
RUN mkdir -p /workspace && \
    chown -R ${USER_NAME}:${USER_NAME} /workspace

# Set working directory
WORKDIR /workspace

# Copy application source code and configuration
COPY --chown=${USER_NAME}:${USER_NAME} src/ ./src/
COPY --chown=${USER_NAME}:${USER_NAME} pyproject.toml ./
COPY --chown=${USER_NAME}:${USER_NAME} launch_golf_suite.py ./
COPY --chown=${USER_NAME}:${USER_NAME} start_api_server.py ./
COPY --chown=${USER_NAME}:${USER_NAME} conftest.py ./
COPY --chown=${USER_NAME}:${USER_NAME} build_hooks.py ./
COPY --chown=${USER_NAME}:${USER_NAME} .env.example ./.env.example

# Switch to non-root user
USER ${USER_NAME}

# Expose default port (if running web server)
EXPOSE 8001

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command — starts the FastAPI server on port 8001.
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8001"]


# Stage 3: Training stage for advanced ML workflows
FROM runtime AS training

USER root

# Install heavy ML dependencies specifically for training workloads
RUN pip install --no-cache-dir \
    gymnasium>=0.29.0 \
    stable-baselines3>=2.0.0 \
    "tensorboard>=2.14.0" \
    "ray[rllib]>=2.9.0" \
    && echo "Training dependencies installed successfully"

USER ${USER_NAME}

CMD ["/bin/bash"]
