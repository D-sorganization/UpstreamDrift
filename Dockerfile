# Stage 1: Builder — install all Python dependencies into an isolated venv
# Base image pinned by digest for reproducible builds
FROM python:3.12-slim@sha256:4386a385d81dba9f72ed72a6fe4237755d7f5440c84b417650f38336bbc43117 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Build tools for packages that compile C extensions (cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Core API + physics stack from lockfile
COPY requirements.lock /tmp/requirements.lock
RUN pip install --upgrade pip==26.1 && \
    pip install -r /tmp/requirements.lock

# Auth and server extensions - pinned versions (to be added to requirements.lock)
# These should be consolidated into requirements.lock via pip-compile
RUN pip install \
    slowapi==0.1.9 \
    "pydantic[email]==2.12.5" \
    python-multipart==0.0.27 \
    sqlalchemy==2.0.44 \
    bcrypt==4.3.0 \
    "PyJWT==2.12.0" \
    "cryptography==46.0.7" \
    aiofiles==24.1.0 \
    python-dateutil==2.9.0.post0 \
    structlog==25.5.0 \
    colorama==0.4.6

# Shared-code runtime deps imported at module top-level by
# src/shared/python (pandas, matplotlib, sympy) and API routes that parse
# XML (defusedxml). These used to come from the conda base; keep them
# explicit for the slim build so the API import chain resolves.
# Pinned versions for reproducible builds
RUN pip install \
    "pandas==2.3.3" \
    "matplotlib==3.10.8" \
    "sympy==1.14.0" \
    "defusedxml==0.7.1"

# Pinocchio via pip (binary wheels available since 2024 — no conda needed)
# Pinned versions for reproducible builds
RUN pip install \
    pin==3.3.1 \
    pin-pink==2.0.0 \
    qpsolvers==4.7.0 \
    osqp==1.0.5 \
    meshcat==0.3.2 \
    "robot_descriptions==1.14.0" \
    "imageio[ffmpeg]==2.37.0" \
    "trimesh==4.9.0"


# Stage 2: Runtime — slim production image for the API server
# Base image pinned by digest for reproducible builds
FROM python:3.12-slim@sha256:4386a385d81dba9f72ed72a6fe4237755d7f5440c84b417650f38336bbc43117 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Keep the base interpreter's bundled pip aligned with the venv so image
# scanners do not report the runtime layer's global site-packages as stale.
RUN python -m pip install --upgrade --no-cache-dir pip==26.1

# MuJoCo headless rendering + health check
# X11/XCB/PyQt6 libs removed — not needed in a headless API server
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libosmesa6 \
    libglew2.2 \
    libegl1 \
    libglib2.0-0t64 \
    patchelf \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

ARG USER_NAME=golfer
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME}

# Copy only the venv — no conda overhead
COPY --from=builder /opt/venv /opt/venv

# /workspace is the project root; "from src.xxx" imports resolve here
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/workspace"

RUN mkdir -p /workspace && chown -R ${USER_NAME}:${USER_NAME} /workspace

WORKDIR /workspace

# src/engines/Simscape_Multibody_Models/ (MATLAB) excluded via .dockerignore
COPY --chown=${USER_NAME}:${USER_NAME} src/ ./src/
COPY --chown=${USER_NAME}:${USER_NAME} pyproject.toml ./
COPY --chown=${USER_NAME}:${USER_NAME} launch_golf_suite.py ./
COPY --chown=${USER_NAME}:${USER_NAME} scripts/ci/start_api_server.py ./
COPY --chown=${USER_NAME}:${USER_NAME} .env.example ./.env.example

USER ${USER_NAME}

EXPOSE 8001

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
# - --forwarded-allow-ips: explicitly set trusted proxy IPs; defaults to
#   localhost only for security. Set FORWARDED_ALLOW_IPS env var in production
#   to specify trusted proxy IPs (e.g., your load balancer's internal IP).
# - --access-log: keep structured request logs on stdout for observability.
CMD ["python3", "-m", "uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "${FORWARDED_ALLOW_IPS:-127.0.0.1}", \
     "--access-log"]


# Stage 3: Training — adds PyTorch + RL stack for GPU training workflows
FROM runtime AS training

USER root

# PyTorch cu124 wheels bundle CUDA runtime libs; host driver provides libcuda via nvidia-container-toolkit
# Pinned versions for reproducible builds
RUN /opt/venv/bin/pip install --no-cache-dir \
    "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu124

RUN /opt/venv/bin/pip install --no-cache-dir \
    "gymnasium==1.1.1" \
    "stable-baselines3==2.7.0" \
    "tensorboard==2.20.0" \
    "ray[rllib]==2.51.0"

USER ${USER_NAME}

CMD ["/bin/bash"]
