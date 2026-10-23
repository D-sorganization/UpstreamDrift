# See docker/README.md and ADR-0021 (docs/adr/0021-container-strategy.md) for container policy.
# Security-scan image (anthropics/oss-scanner contract, issue #13658).
#
# The scanner builds this file with the repository as the build context. The
# network is available during the build and removed afterwards, so everything
# the audit needs to build, import and test the code is fetched here.
#
# Scope: the core Python package (src/), the pinned Tools tree it imports
# (vendor/ud-tools), the dev/test dependency lockfile, and the two Rust crates
# that parse or integrate untrusted input (upstream-mocap-io, upstream-physics).
# The heavy optional engines (Drake, Pinocchio, OpenSim, MyoSuite, MATLAB/
# Simscape, PyTorch/RL) are left out on purpose; see threat_model.md.
#
# Sources of truth reused here: requirements-dev.lock and the unit-gate install
# steps in .github/workflows/ci-standard.yml, the Rust toolchain pin and maturin
# version from the root Dockerfile, and the Tools pin in requirements-tools.txt.
# The build context filter is .oss-scanner/Dockerfile.dockerignore (BuildKit
# uses it in preference to the root .dockerignore, which drops tests/).
# No secrets and no token build arguments are used.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SKIP_UI_BUILD=1 \
    QT_QPA_PLATFORM=offscreen \
    QT_API=pyqt6 \
    MPLBACKEND=Agg \
    MUJOCO_GL=egl \
    SDL_VIDEODRIVER=dummy

# Compilers for C extensions and the PyO3 crates, plus the headless GL/Qt
# runtime libraries that mujoco and PyQt6 load at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git pkg-config libssl-dev \
      libgl1 libegl1 libosmesa6 libglib2.0-0 libxkbcommon0 libdbus-1-3 \
      libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain pinned like the root Dockerfile. RUSTUP_TOOLCHAIN overrides
# rust-toolchain.toml (channel "stable" plus a wasm target) so cargo never
# tries to download another toolchain once the network is gone.
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    RUSTUP_TOOLCHAIN=1.94.0 \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    PATH="/opt/cargo/bin:/opt/venv/bin:$PATH"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain 1.94.0 --no-modify-path \
    && rustc --version && cargo --version

RUN python -m venv /opt/venv \
    && pip install --upgrade pip==26.2.1 \
    && pip install maturin==1.13.3

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# Pinned Tools tree. The scanner's checkout may not initialise submodules, so
# fetch the exact revision recorded in requirements-tools.txt (kept equal to
# the vendor/ud-tools gitlink and Cargo.toml). Tools is a public repository.
RUN set -eux; \
    if [ ! -f vendor/ud-tools/src/contracts.py ]; then \
      tools_sha="$(sed -n 's/.*Tools\.git@\([0-9a-f]\{40\}\).*/\1/p' requirements-tools.txt)"; \
      test -n "$tools_sha"; \
      rm -rf vendor/ud-tools; \
      git init -q vendor/ud-tools; \
      git -C vendor/ud-tools fetch -q --depth 1 https://github.com/D-sorganization/Tools.git "$tools_sha"; \
      git -C vendor/ud-tools checkout -q FETCH_HEAD; \
      rm -rf vendor/ud-tools/.git; \
    fi; \
    test -f vendor/ud-tools/src/contracts.py

# Python dependencies exactly as the CI unit gate installs them.
RUN pip install --force-reinstall --no-deps "sortedcontainers>=2.4.0" "hypothesis>=6.0.0" \
    && pip install --ignore-installed -r requirements-dev.lock \
    && (pip uninstall -y pytest-vcr pytest-recording || true) \
    && pip install --force-reinstall "numpy==2.2.6" "scipy==1.15.3"

# Editable install of the package itself (as CI does). If the build hook's
# pinned-Tools provenance check rejects this context, the PYTHONPATH below
# still mirrors the pytest `pythonpath` and the runtime image.
RUN pip install --no-deps -e ".[gui-test]" \
    || echo "WARNING: editable install failed; relying on PYTHONPATH"
ENV PYTHONPATH="/src/vendor/ud-tools/src/shared/python:/src/vendor/ud-tools/src:/src/vendor/ud-tools/src/python/src:/src:/src/src:/src/src/shared/python:/src/src/engines/physics_engines/mujoco/python"

# Rust: the C3D/TRC/BVH parser crate and the physics kernel, built and
# installed with debug info so crashes symbolise. A failure here fails the
# build, as it would in CI.
ENV CARGO_PROFILE_RELEASE_DEBUG=true
RUN set -eux; \
    for crate in rust_core/upstream-mocap-io rust_core/upstream-physics; do \
      maturin build --release --features python -m "$crate/Cargo.toml" --out /wheels; \
    done; \
    pip install --force-reinstall --no-deps /wheels/*.whl; \
    python -c "import upstream_mocap_io, upstream_physics; print('Rust wheels ok')"

# Warm the cargo test build for the parser crate so it can be re-run offline.
RUN cargo test -p upstream-mocap-io --no-run \
    || echo "WARNING: cargo test build for upstream-mocap-io failed"

# Cheap tests so you know the image works; failures never abort the build.
RUN cargo test --offline -p upstream-mocap-io \
    || echo "WARNING: cargo test reported failures"
RUN python -m pytest tests/api tests/motion_capture \
      -m "not slow and not live_simulation and not requires_gl and not requires_drake and not requires_opensim and not requires_pinocchio and not requires_myosuite and not requires_network and not requires_gpu and not requires_mocap_fixtures" \
      -n 4 --timeout=120 -p no:cacheprovider --no-cov -q --tb=line \
    || echo "WARNING: pytest reported failures"

# The audit runs as an unprivileged uid; let it read and write the checkout
# and the venv (for scratch builds and reproductions).
RUN chmod -R a+rwX /src /opt/venv /opt/cargo
