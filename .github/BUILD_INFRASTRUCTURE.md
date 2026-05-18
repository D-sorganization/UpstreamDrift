# Build Infrastructure Guide

**Last Updated**: 2026-04-25  
**Maintainers**: D-sorganization/UpstreamDrift  
**Scope**: Docker builds, Python packaging, Rust extensions, CI/CD workflows

## Quick Start

### Docker Images

```bash
# Build runtime (headless API server)
docker build --target runtime -t upstream-drift:runtime .

# Build training (GPU-optimized RL)
docker build --target training -t upstream-drift:training .

# Run API server
docker run -p 8001:8001 upstream-drift:runtime

# Run training session (requires NVIDIA GPU)
docker run --gpus all -it upstream-drift:training /bin/bash
```

### Python Development

```bash
python3 -m ruff check .                              # Lint (zero violations)
python3 -m ruff format .                             # Auto-format
python3 -m pytest -n auto --timeout=60               # Run tests
python3 scripts/check_file_size_budget.py            # Enforce 1200 lines/file
```

### Rust Extensions

```bash
maturin develop                     # Build and install locally
maturin build --release             # Build wheel for distribution
```

## Docker Multi-Stage Build

### Stage 1: Builder

**Purpose**: Compile Python dependencies into isolated venv

```dockerfile
FROM python:3.12-slim AS builder
# - Installs build-essential, git
# - Creates /opt/venv
# - Installs all Python dependencies (requirements.lock + extras)
# - Creates /opt/venv with all compiled C extensions
```

**Optimization**:

- Caching: Dependencies only rebuild if requirements.lock changes
- Isolation: Build tools not included in final image
- Size: Builder layer discarded in final runtime image

### Stage 2: Runtime

**Purpose**: Minimal production API server image

```dockerfile
FROM python:3.12-slim AS runtime
# - Copies /opt/venv from builder (no recompilation)
# - Adds runtime dependencies: libgl1, libosmesa6, ffmpeg, curl
# - Configures non-root user (golfer:1000)
# - Sets up health check on /health endpoint
# - Runs uvicorn on port 8001 (single worker)
```

**Security**:

- Non-root user: Containers run as golfer:1000
- Slim base: python:3.12-slim reduces attack surface
- No build tools: gcc, git excluded from runtime
- Health check: Validates /health endpoint (30s interval)

**Size Budget**: 4 GB maximum (enforced by docker-size-gates.yml)

### Stage 3: Training

**Purpose**: GPU-optimized RL training environment

```dockerfile
FROM runtime AS training
# - Inherits runtime image
# - Adds PyTorch cu124 (CUDA 12.4 wheel-bundled runtime)
# - Adds Gymnasium + stable-baselines3 (RL control policies)
# - Adds Ray[rllib] (distributed training)
# - Adds TensorBoard (metrics visualization)
```

**Deployment**:

- Requires: NVIDIA container runtime with `--gpus all`
- CUDA: Host driver provides libcuda via nvidia-container-toolkit

## CI/CD Workflow Hierarchy

### Authoritative Core Workflows

#### ci-standard.yml

```
Scope: All PRs to main, all pushes to main, weekly schedule
Runner: Self-hosted d-sorg-fleet (fails closed if unavailable)
Timeout: 20 min

Jobs:
  pick-runner — Route to self-hosted or fail closed
  quality-gate — Ruff, mypy, bandit, pip-audit, size budgets (5 min)
  tests — pytest across Python 3.10/3.11/3.12 with 30% coverage (15 min)
  shared-tools-consumer-contracts — Tools.git interop (5 min)
```

**Blocking Checks**:

- Ruff lint + format (zero violations)
- Type hints: mypy strict on src/api/, baseline on src/
- Security: bandit (medium+), pip-audit (all CVEs)
- Size budget: max 1200 lines/file, module baselines enforced
- Coverage: minimum 30% across src/

#### ci-optional-stack.yml

```
Trigger: Workflow dispatch (manual opt-in)
Purpose: Heavy integration tests for optional physics engines
Engines tested: Drake, Pinocchio, OpenSim, MyoSuite
Timeout: 40 min
Rationale: Full engine matrix too expensive to run on every PR
```

#### nightly-cross-engine.yml

```
Trigger: Scheduled nightly
Purpose: Full cross-validation of all physics engines
Outputs: Engine interoperability metrics and performance reports
Frequency: Daily at scheduled time (catches drift independent of PRs)
```

#### docker-size-gates.yml

```
Trigger: Push to main branch only
Purpose: Validate Docker image size and health
Checks:
  1. Build runtime image (buildx + GHA cache)
  2. Enforce < 4 GB size limit
  3. Run health check: curl /health
  4. Smoke test: import core physics stacks (numpy, scipy, mujoco, pinocchio)
Timeout: 10 min
```

### Helper Workflows (Jules Automation)

These MUST NOT duplicate core CI checks. They are AI-driven assistants:

```
Assessment Loop:
  Jules-Assessment-Generator     → Generate code quality reports
  Jules-Assessment-AutoFix       → Auto-fix findings (dry-run capable)
  Jules-Assessment-Remediator    → Remediate structural issues

Code Quality:
  Jules-Code-Quality-Fixer       → Fix quality violations
  Jules-Code-Quality-Reviewer    → Suggest improvements

PR Handling:
  Jules-PR-AutoFix               → Fix CI failures in PRs
  Jules-PR-Compiler              → Compile PR metadata
  Jules-PR-Cleanup               → Archive stale PRs

Repair & Response:
  Jules-Auto-Repair              → Respond to core CI failures
  Jules-Hotfix-Creator           → Create emergency fix branches
```

### Archived Workflows

Disabled in `.github/workflows/archived/`:

- `auto-remediate-issues.yml.disabled` — Superseded by Jules-Assessment-Remediator
- `assessment-auto-fix.yml.disabled` — Superseded by Jules-Assessment-AutoFix

## Dependency Management

### Python Dependencies

**Strategy**: Lockfile-based with optional extras

```
requirements.lock           # Core + physics engines (production)
requirements-dev.lock       # dev + testing + linting tools
pyproject.toml [dependencies]
  [optional-dependencies]
    dev      — pytest, ruff, mypy, bandit
    gui-test — PyQt6, pytest-qt, PySide6
```

**Security Auditing**:

```bash
# Run via ci-standard.yml quality-gate
pip-audit -r requirements-dev.lock \
  --ignore-file .github/security/pip-audit-ignore.yml

# Waivers tracked with expiry dates (automatic failure if stale)
```

### Rust Dependencies

**Workspace Configuration** (Cargo.toml):

```toml
[workspace]
members = ["rust_core/upstream-physics"]
resolver = "2"

[workspace.dependencies]
# Pinned git revision — enables self-contained builds
tools-core = { git = "https://github.com/D-sorganization/Tools.git",
               rev = "ea2690362481379b94135894f9dfac2b70d1bc65" }
pyo3 = { version = "0.24", features = ["extension-module"] }
nalgebra = "0.33"
```

**Build Command**: `maturin develop` (installs extension into active venv)

## Caching Strategy

### GitHub Actions Cache (GHA)

```yaml
# docker-size-gates.yml & ci-optional-stack.yml
uses: docker/setup-buildx-action@4d04d5d9486b7bd6fa91e7baf45bbb4f8b9deedd
  cache-from: type=gha
  cache-to: type=gha,mode=max
```

- **5 GB limit** per repository
- Caches intermediate Docker layers across runs
- Resets automatically monthly by GitHub

### Python venv Caching

- Builder stage compiles dependencies into `/opt/venv`
- Runtime stage COPYs pre-built venv (no recompilation)
- Only rebuilds if `requirements.lock` or `pyproject.toml` changes

### Cargo Cache

- `target/` directory caches compiled Rust artifacts
- Not persisted across CI runs (local development only)
- Future: Integrate cargo-cache action for CI persistence

## Performance Metrics

### CI Execution Times

- **Quality Gate**: ~5 min (ruff, mypy, bandit, pip-audit)
- **Core Tests**: ~15 min (pytest parallel, 3 Python versions)
- **Tools Consumer Contracts**: ~5 min (interoperability)
- **Total Core CI**: ~25 min (parallelized jobs)
- **Docker Build**: ~10 min (buildx + cache)

### Image Sizes

- **Runtime**: ~1.5 GB (slim base + venv + headless libs)
- **Training**: ~4-5 GB (runtime + PyTorch cu124)
- **Compressed (DockerHub)**: 500-800 MB (layer compression)

## Security Hardening

### Container Security

1. **Non-Root User**: Images run as `golfer:1000`, not root
2. **Slim Base**: `python:3.12-slim` minimal attack surface
3. **No Build Tools**: gcc, git, build-essential excluded from runtime
4. **Health Check**: Validates /health endpoint before marking ready
5. **Read-Only FS**: Production containers can run with `--read-only` flag

### Static Analysis

```bash
# Bandit (SAST)
bandit -r . -x ./tests,./archive -ll -ii  # Block medium+ severity

# pip-audit (dependency CVE check)
pip-audit -r requirements-dev.lock         # Check against known CVEs

# mypy (type safety)
mypy src/api --strict                      # Enforce type hints on API layer
mypy src --follow-imports=skip             # Gradual adoption elsewhere
```

### Code Quality

```bash
# Ruff linting (zero violations)
ruff check .

# Ruff formatting (consistent with pre-commit)
ruff format --check .

# File size budget (1200 lines max per file)
python scripts/check_file_size_budget.py
```

## Troubleshooting

### Docker Build Fails

```bash
# Check buildx setup
docker buildx create --use

# Rebuild with verbose progress
docker buildx build --progress=plain --target runtime .

# View layer sizes
docker history upstream-drift:runtime

# Clear cache and retry
docker buildx prune -a
```

### Pytest Coverage Too Low

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html .coverage/

# View report
python -m http.server 8000 -d htmlcov/
# Open http://localhost:8000
```

### Rust Extension Build Fails

```bash
# Ensure Rust installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Rebuild with verbose output
maturin develop --release --verbose

# Check tools-core revision
grep "rev = " Cargo.toml
```

### Self-Hosted Runner Issues

```bash
# Check runner status
gh api /orgs/D-sorganization/actions/runners \
  --jq '.runners[] | select(.labels[].name == "d-sorg-fleet")'

# If offline, ci-standard.yml fails closed (safe default)
```

## Related Documentation

- `.github/INFRASTRUCTURE_CONSOLIDATION.md` — Consolidation roadmap
- `Dockerfile` — Multi-stage definitions
- `Cargo.toml` — Rust workspace config
- `.github/workflows/ci-standard.yml` — Authoritative CI
- `CLAUDE.md` — Contributor policy
- `SPEC.md` — Architecture specification
