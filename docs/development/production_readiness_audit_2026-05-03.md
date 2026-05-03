# Production Readiness Audit (Managerial Review)

Date: 2026-05-03  
Scope: Whole-repo architecture, operability, quality gates, and productization fitness.

## Executive Verdict

**Not yet production-grade as a coherent software product.** The repository contains substantial valuable technical assets, but current composition resembles a **research mega-monorepo** rather than a tightly governed production application.

Top blockers:

1. **Architecture/spec drift**: canonical spec maps to paths that do not exist.
2. **Monolithic code concentration**: 494 Python files exceed the repository’s own 400-line limit.
3. **CI contract fragmentation**: multiple overlapping workflows and unclear single source of truth.
4. **Packaging/runtime ambiguity**: mixed interfaces (PyQt, FastAPI, Tauri, Rust) without one authoritative deployment path.
5. **Governance mismatch**: standards in AGENTS/SPEC exceed what appears enforceable in current codebase.

## Key Findings

### 1) Spec-to-code mismatch (critical)

`SPEC.md` documents an architecture rooted in modules like `src/api/main.py` and `src/config/configuration.py`, but key paths are absent in current tree. This undermines onboarding, trust, and release confidence.

### 2) Monolith and maintainability risk (critical)

A repository scan found **494 Python files over 400 lines**, conflicting with stated engineering policy and increasing defect surface, review fatigue, and change coupling.

### 3) Workflow sprawl and CI discoverability debt (high)

`.github/workflows/` currently includes many automation and bot workflows plus archived entries. While automation is useful, there is no obvious “single operational CI contract” for humans; this raises release risk and confusion about what blocks merges.

### 4) Product boundary ambiguity (high)

The project claims to be one “suite,” but mixes:
- research sandboxes,
- production-adjacent launchers,
- multiple engine adapters with differing maturity,
- large legacy/vendor-like directories,
- and multiple UI/runtime surfaces.

Without stronger boundarying, it is difficult to guarantee SLOs, support expectations, and secure deployment.

### 5) Quality policy inconsistency (medium-high)

User/developer guidance references strict black/isort/ruff/mypy and quality scripts; however, repository config includes selective exclusions and reduced strictness in places. This is understandable during migration, but not consistent with “professional-grade” claims unless transparently tiered.

## Recommended GitHub Issues to Create

> These are written to be copy-pasted into `gh issue create`.

### Issue 1 — SPEC drift remediation
**Title:** `docs(spec): eliminate architecture drift between SPEC.md and real module map`

**Body:**
- Problem: `SPEC.md` references non-existent core paths (e.g., `src/api/main.py`, `src/config/configuration.py`, flat engine adapter files).
- Why it matters: breaks architecture trust, invalidates onboarding/review assumptions, and weakens release readiness.
- Acceptance criteria:
  1. Update `SPEC.md` module map to existing paths OR implement missing paths with compatibility shims.
  2. Add CI check that validates listed critical paths exist.
  3. Add “spec ownership” section naming update cadence and owners.

### Issue 2 — Monolith decomposition program
**Title:** `refactor(maintainability): reduce oversized Python modules above 400 LOC threshold`

**Body:**
- Problem: 494 Python files exceed 400 lines.
- Why it matters: high cognitive load, high regression risk, difficult reviews.
- Acceptance criteria:
  1. Generate ranked inventory (LOC + churn + defect density).
  2. Refactor top 25 files into cohesive modules with stable API facades.
  3. Add CI warning gate for new/expanded files >400 LOC and blocker at >800 LOC.

### Issue 3 — CI contract simplification
**Title:** `ci(governance): establish single blocking CI workflow and classify all other workflows`

**Body:**
- Problem: workflow sprawl obscures blocking status.
- Why it matters: unclear merge quality bar and ownership.
- Acceptance criteria:
  1. Publish CI taxonomy (blocking, advisory, scheduled, bot-maintenance).
  2. Add README + CONTRIBUTING table mapping each workflow to purpose/owner/blocking behavior.
  3. Enforce branch protection against exactly defined blocking checks.

### Issue 4 — Product boundary and support tiers hardening
**Title:** `architecture(product): define production core vs experimental/research boundaries`

**Body:**
- Problem: production and research surfaces are mixed in a single runtime/developer experience.
- Why it matters: unclear support contract, release blast radius too large.
- Acceptance criteria:
  1. Create boundary doc: `core`, `extended`, `experimental`, `archived`.
  2. Separate import/runtime paths so core can run without experimental dependencies.
  3. Add CI matrix validating core independently from extended/experimental.

### Issue 5 — Release engineering and deployment contract
**Title:** `release: define supported deployment topology and reproducible release artifacts`

**Body:**
- Problem: multiple entrypoints (CLI, launcher, API, Tauri, Rust bindings) with ambiguous primary release unit.
- Why it matters: hard to operate or support in production.
- Acceptance criteria:
  1. Pick canonical production artifact(s) (e.g., Python package + API container + optional desktop).
  2. Add version compatibility matrix (Python, engines, OS, GPU stack).
  3. Add smoke tests per shipped artifact.

### Issue 6 — Observability and operations baseline
**Title:** `ops(observability): define minimum telemetry and failure handling baseline`

**Body:**
- Problem: broad feature claims but unclear uniform SRE baseline.
- Why it matters: poor production diagnosability.
- Acceptance criteria:
  1. Standard structured logging schema across core modules.
  2. Health/readiness checks for API and launcher-critical subsystems.
  3. Error budget + alert mapping for production endpoints.

### Issue 7 — Security posture tightening for mixed-surface repo
**Title:** `security: enforce hardened dependency and artifact policy across core + optional stacks`

**Body:**
- Problem: many optional integrations increase supply-chain and runtime attack surface.
- Why it matters: high likelihood of latent vulnerabilities and inconsistent patch cadence.
- Acceptance criteria:
  1. Security policy by tier (core required deps vs optional engines).
  2. Separate SBOMs for core and full-stack installs.
  3. Automated vuln triage SLA labels (critical/high/medium).

## Managerial Bottom Line

- **Does the big picture make conceptual sense?** Yes: unified biomechanical suite across engines is strategically coherent.
- **Is it set up like modern professional software today?** Not fully; execution and governance maturity lag behind ambition.
- **Can it become production-grade?** Yes, if architecture boundaries, CI governance, and module decomposition are tackled as a deliberate program over multiple releases.
