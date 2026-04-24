# Comprehensive Dockerfile for Golf Modeling Suite
# Unifies Robotics (MuJoCo, Drake, Pinocchio) and Biomechanics (OpenSim, MyoSim)

# Stage 1: Builder stage with full development tools
# Digest pinned to continuumio/miniconda3:24.11.1-0 (all-platform manifest).
# To rotate: run `docker manifest inspect continuumio/miniconda3:<new-tag>` and
# update both the tag and the digest here; review conda/Python release notes.
FROM continuumio/miniconda3:24.11.1-0@sha256:6a66425f001f739d4778dd732e020afeb06175f49478fafc3ec673658d61550b AS builder

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
    pyqt6 \
    opencv \
    pyyaml \
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
COPY requirements.lock /tmp/requirements.txt

# Install Python dependencies from requirements.txt
# Filter out comments, WSL/Linux notes, and blank lines
RUN grep -v '^#' /tmp/requirements.txt | grep -v '^$' > /tmp/filtered_requirements.txt && \
    pip install --no-cache-dir -r /tmp/filtered_requirements.txt

# Install additional physics engines and API server dependencies
# We explicitly include runtime packages needed by API import paths: pandas, matplotlib, sympy, and defusedxml
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    mujoco>=3.2.3 \
    drake \
    meshcat \
    pin-pink \
    qpsolvers \
    osqp \
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
# Same digest as builder — keep both in sync when rotating.
FROM continuumio/miniconda3:24.11.1-0@sha256:6a66425f001f739d4778dd732e020afeb06175f49478fafc3ec673658d61550b AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Upgrade openssl to fix Debian vulnerabilities
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6 \
    ffmpeg \
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
# The core routes register /health on the FastAPI app (src/api/routes/core.py)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command — starts the FastAPI server on port 8001.
# Override with `docker run ... /bin/bash` for an interactive shell.
#
# Hardening flags (salvaged from stale PR #2723, issue #2786):
# - --workers 1: single worker keeps HEALTHCHECK and in-process state
#   (rate limiter, engine registry) consistent; scale horizontally instead.
# - --proxy-headers: honor X-Forwarded-For / X-Forwarded-Proto when the
#   container sits behind a reverse proxy, so access logs and client IP
#   rate limiting reflect the real client.
# - --forwarded-allow-ips=*: accept proxy headers from any upstream inside
#   the container network; operators should pin this at the proxy layer.
# - --access-log: keep structured request logs on stdout for observability.
CMD ["python3", "-m", "uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--access-log"]


# Stage 3: Training stage for advanced ML workflows
FROM runtime AS training

USER root

# Install CUDA toolkit via conda for GPU training support
RUN conda install -y -c pytorch -c nvidia -c conda-forge \
    cuda-toolkit \
    cudnn \
    pytorch \
    pytorch-cuda=12.4 \
    && conda clean --all --yes

# Install heavy ML dependencies specifically for training workloads
RUN pip install --no-cache-dir \
    gymnasium>=0.29.0 \
    stable-baselines3>=2.0.0 \
    "tensorboard>=2.14.0" \
    "ray[rllib]>=2.9.0" \
    && echo "Training dependencies installed successfully"

USER ${USER_NAME}

CMD ["/bin/bash"]
