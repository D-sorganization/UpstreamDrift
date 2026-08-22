# See docker/README.md and ADR-0021 (docs/adr/0021-container-strategy.md) for container policy.
# Stage 1: Builder — install all Python dependencies into an isolated venv
# Base image pinned by digest for reproducible builds
FROM python:3.12-slim@sha256:c2d8472b831337ab296a8ce652e1ba786e9e3034fc445dc58b50a7f5251f0003 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Build tools for packages that compile C extensions (cryptography, etc.)
# `curl` + `pkg-config`/`libssl-dev` are also needed to install the Rust
# toolchain and build the PyO3 crates below (issue #7600).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---------------------------------------------------------------------------
# Rust extension wheels (issue #7600 — ship Rust to installs/deploys).
#
# The release wheel is pure-Python (hatchling); the PyO3 crates under
# rust_core/ were never built into the image, so deployed containers fell back
# to the slow Python paths. Build them here in the builder stage with maturin
# and stage the wheels into /wheels for pip to install — mirroring the
# build-then-install pattern from Gasification_Model PR #4323. No external
# wheel index is used; everything is compiled from source in this layer.
#
# Pin the toolchain via rustup; keep it in /opt/cargo so it never leaks into
# the runtime stage (only the built wheels are pip-installed into the venv).
# ---------------------------------------------------------------------------
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH="/opt/cargo/bin:/opt/venv/bin:$PATH"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain 1.94.0 && \
    rustc --version && cargo --version

RUN pip install --upgrade pip==26.2.1 && pip install maturin==1.13.3

# Cargo needs the git CLI to fetch the tools-core git dependency reliably.
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

# Build each PyO3 crate's wheel into /wheels. Keep this list in lock-step with
# the maturin build loop in .github/workflows/ci-standard.yml.
COPY Cargo.toml /tmp/rust/Cargo.toml
COPY rust_core /tmp/rust/rust_core
RUN mkdir -p /wheels && \
    cd /tmp/rust && \
    for crate in \
        rust_core/upstream-physics \
        rust_core/upstream-mocap-preproc \
        rust_core/upstream-mocap-io \
        rust_core/upstream-muscle \
        rust_core/upstream-motion-matching \
        rust_core/ai_backend; do \
        maturin build --release --features python \
            -m "$crate/Cargo.toml" --out /wheels; \
    done && \
    ls -1 /wheels

# Core API + physics stack from lockfile
COPY requirements.lock /tmp/requirements.lock
RUN pip install --upgrade pip==26.2.1 && \
    pip install -r /tmp/requirements.lock

# Auth and server extensions - pinned versions (to be added to requirements.lock)
# These should be consolidated into requirements.lock via pip-compile
RUN pip install \
    slowapi==0.1.9 \
    "pydantic[email]==2.12.5" \
    python-multipart==0.0.31 \
    aiofiles==24.1.0 \
    python-dateutil==2.9.0.post0 \
    structlog==25.5.0 \
    colorama==0.4.6

# Feature dependencies for the slim build. This block no longer patches the API
# import chain (#8032): defusedxml and pandas are now core dependencies resolved
# from requirements.lock, and matplotlib is imported under TYPE_CHECKING only.
# matplotlib and sympy still back real container features (plotting, symbolic
# controls), and pandas stays pinned here to hold the container on the 2.x line.
# Pinned versions for reproducible builds
RUN pip install \
    "pandas==2.3.3" \
    "matplotlib==3.10.8" \
    "sympy==1.14.0"

# Pinocchio via pip (binary wheels available since 2024 — no conda needed)
# Pinned versions for reproducible builds
RUN pip install \
    pin==3.3.1 \
    pin-pink==2.0.0 \
    qpsolvers==4.7.0 \
    osqp==1.0.5 \
    meshcat==0.3.2 \
    tornado==6.5.7 \
    "robot_descriptions==1.14.0" \
    "imageio[ffmpeg]==2.37.0" \
    "trimesh==4.9.0"

# Install the Rust extension wheels built above into the venv (issue #7600).
# Their pure-Python deps (e.g. numpy) are already present from the lockfile, so
# --no-deps keeps the resolver from pulling unpinned transitives.
RUN pip install --no-deps /wheels/*.whl && \
    python -c "import upstream_physics, ai_backend; print('Rust wheels installed:', upstream_physics.__name__, ai_backend.__name__)"

# Reassert scanner-fixed transitive tooling after every dependency layer.
# Trivy gates GHSA-6v7p-g79w-8964 and CVE-2025-47273 at the final image.
RUN pip install --upgrade --no-cache-dir \
    "msgpack==1.2.1" \
    "setuptools==78.1.1"

# Audit the *resolved* environment inside the image (issue #7159 D2). The
# Dockerfile pins ~36 lines that can drift from requirements.lock, so a manual
# edit could otherwise bake a CVE into the runtime image without the CI lane
# ever auditing it. We reuse the SAME waiver policy as CI
# (scripts/config/pip_audit_waivers.json + scripts/ci/check_pip_audit_waivers.py)
# so there is one source of truth (DRY). Runs in the builder stage only, so
# pip-audit does not ship in the runtime image. Air-gapped builds may pass
# --build-arg SKIP_AUDIT=true; the default is enforced.
ARG SKIP_AUDIT=false
COPY scripts/config/pip_audit_waivers.json /tmp/pip_audit_waivers.json
COPY scripts/ci/check_pip_audit_waivers.py /tmp/check_pip_audit_waivers.py
RUN set -eu; \
    if [ "$SKIP_AUDIT" = "true" ]; then \
        echo "SKIP_AUDIT=true — skipping in-image pip-audit (air-gapped build)"; \
    else \
        pip install --no-cache-dir pip-audit==2.10.0; \
        waiver_flags="$(python /tmp/check_pip_audit_waivers.py \
            --waiver-file /tmp/pip_audit_waivers.json)"; \
        # shellcheck disable=SC2086 - intentional word-splitting of flags
        python -m pip_audit $waiver_flags; \
        pip uninstall -y pip-audit; \
    fi


# Stage 2: Runtime — slim production image for the API server
# Base image pinned by digest for reproducible builds
FROM python:3.12-slim@sha256:c2d8472b831337ab296a8ce652e1ba786e9e3034fc445dc58b50a7f5251f0003 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Keep the base interpreter's bundled pip aligned with the venv so image
# scanners do not report the runtime layer's global site-packages as stale.
RUN python -m pip install --upgrade --no-cache-dir pip==26.2.1 && \
    python -m pip install --upgrade --no-cache-dir setuptools==78.1.1

# MuJoCo headless rendering + health check
# X11/XCB/PyQt6 libs removed — not needed in a headless API server
RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
    libc-bin \
    libc6 \
    libcap2 \
    libsystemd0 \
    libudev1 \
    libgl1 \
    libosmesa6 \
    libglew2.2 \
    libegl1 \
    libglib2.0-0t64 \
    patchelf \
    sed \
    ffmpeg \
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
    PYTHONPATH="/workspace" \
    # Headless MuJoCo rendering default (issue #7161 D3): set in the image so a
    # bare `docker run` matches the compose default (MUJOCO_GL=osmesa) instead
    # of only being defined in docker-compose.yml (K — reproducibility).
    MUJOCO_GL="osmesa"

RUN mkdir -p /workspace && chown -R ${USER_NAME}:${USER_NAME} /workspace

WORKDIR /workspace

# src/engines/Simscape_Multibody_Models/ (MATLAB) excluded via .dockerignore
COPY --chown=${USER_NAME}:${USER_NAME} src/ ./src/
COPY --chown=${USER_NAME}:${USER_NAME} pyproject.toml ./
COPY --chown=${USER_NAME}:${USER_NAME} launch_golf_suite.py ./
COPY --chown=${USER_NAME}:${USER_NAME} scripts/ci/start_api_server.py ./
COPY --chown=${USER_NAME}:${USER_NAME} .env.example ./.env.example
COPY --chown=${USER_NAME}:${USER_NAME} docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER ${USER_NAME}

EXPOSE 8001

# The core routes register /health on the FastAPI app (src/api/routes/core.py).
# Use a Python one-liner (issue #7161 D3) so the healthcheck does not depend on
# curl being present from an apt layer — python is always in the venv.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health', timeout=5).status==200 else 1)" || exit 1

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
#
# The flags above are passed by docker/entrypoint.sh. The entrypoint runs under
# /bin/sh so FORWARDED_ALLOW_IPS is expanded at runtime (issue #7129) — an
# exec-form CMD cannot perform shell parameter expansion, so the variable would
# otherwise be passed to uvicorn as a literal string. The wrapper still exec's
# uvicorn so it receives signals as PID 1.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


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

# The training image is an interactive workbench, not the API server. Reset the
# inherited API entrypoint (issue #7129) so `CMD ["/bin/bash"]` launches a shell
# instead of being passed as arguments to the uvicorn entrypoint wrapper.
ENTRYPOINT []
CMD ["/bin/bash"]
