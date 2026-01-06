r1) Here’s a “drop-in” prompt you can give to any AI reviewer (or a human) to evaluate a large Python project like a top-tier staff/principal engineer doing a brutal architecture + code review. It forces concrete evidence, scoring, prioritization, and actionable fixes—not vibes.

---

## Ultra-Critical Python Project Review Prompt (copy/paste)

You are a **principal/staff-level Python engineer and software architect** doing an **adversarial, evidence-based** review of a large Python project. Your job is to **find weaknesses, risks, and quality gaps** the way a top company’s internal review board would. Assume this project may go into production and be maintained for years by multiple engineers.

### Inputs I will provide

* Repository contents (code, config, tests, docs)
* Optional: requirements/feature goals, target users, deployment environment, performance/SLA needs

### **MANDATORY SCOPE: Golf Modeling Suite Project Components**

You **MUST** explicitly review and assess **ALL** of the following project components. Do not skip or superficially cover any area:

#### Physics Engine Implementations (Primary Focus)
1. **MuJoCo Engine**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/`
   - Physics engine integration (`physics_engine.py`)
   - Manipulability analysis (`manipulability.py`)
   - Induced acceleration (`rigid_body_dynamics/induced_acceleration.py`)
   - Inverse dynamics (`inverse_dynamics.py`)
   - Advanced kinematics (`advanced_kinematics.py`, `kinematic_forces.py`)
   - Motion capture integration (`motion_capture.py`, `motion_optimization.py`)
   - GUI components (`gui/`, `sim_widget.py`)
   - Verification and testing (`verification.py`, `tests/`)

2. **Drake Engine**: `engines/physics_engines/drake/python/`
   - Physics engine implementation (`drake_physics_engine.py`)
   - Manipulability analysis (`src/manipulability.py`)
   - Induced acceleration (`src/induced_acceleration.py`)
   - GUI application (`src/drake_gui_app.py`)

3. **Pinocchio Engine**: `engines/physics_engines/pinocchio/python/`
   - Physics engine implementation (`pinocchio_physics_engine.py`)
   - Manipulability analysis (`pinocchio_golf/manipulability.py`)
   - GUI components (`pinocchio_golf/gui.py`)
   - DTACK framework (`dtack/` - dynamics, backends, simulation)
   - Utility modules (`dtack/utils/`)

4. **OpenSim Engine**: `engines/physics_engines/opensim/python/`
   - Current implementation status and stubs
   - Integration architecture

5. **MyoSuite Engine**: `engines/physics_engines/myosuite/python/`
   - Current implementation status and stubs
   - Muscle model integration plans

#### C3D Motion Capture System
6. **C3D Reader & Viewer**: `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/`
   - C3D data reader (`c3d_reader.py`)
   - C3D viewer application (`apps/c3d_viewer.py`)
   - Data services (`apps/services/c3d_loader.py`, `analysis.py`)
   - UI components (`apps/ui/tabs/`)
   - Testing (`tests/test_c3d_*.py`)

#### MATLAB Simscape Simulink Models
7. **Simscape Multibody Models**: `engines/Simscape_Multibody_Models/`
   - MATLAB model implementations
   - Python-MATLAB integration
   - Simulink model architecture
   - Data plotters and analysis tools

#### Shared Infrastructure
8. **Shared Python Utilities**: `shared/python/`
   - Physics engine interfaces (`interfaces.py`)
   - Common constants and utilities
   - Cross-engine compatibility layers

9. **URDF Tools**: `tools/urdf_generator/`
   - URDF builder and generator
   - Model validation
   - Cross-engine URDF compatibility

10. **Testing Infrastructure**: `tests/`
    - Unit tests (`tests/unit/`)
    - Integration tests (`tests/integration/`)
    - Cross-engine validation tests
    - Headless GUI tests

#### Documentation & Standards
11. **Project Documentation**: `docs/`
    - Design guidelines (`project_design_guidelines.qmd`)
    - Assessment results (`assessments/`)
    - Architecture documentation
    - User guides and API references

### Assessment Coverage Requirements

For **EACH** of the components listed above, you must:
- Identify the top 3 risks specific to that component
- Evaluate architecture and code quality
- Check for cross-component integration issues
- Verify testing coverage and quality
- Assess adherence to project-specific standards (see `docs/project_design_guidelines.qmd`)

**Failure to comprehensively cover all listed components will result in an incomplete assessment.**

### Your output must be ruthless, structured, and specific

Do **not** be polite. Do **not** generalize. Do **not** say “looks good overall.”
Every claim must cite **exact files/paths, modules, functions**, or **config keys**. Prefer “proof”: callouts, examples, failure modes, and how to reproduce. Provide fixes with suggested code patterns.

---

# 0) Deliverables and format requirements

Produce the review with these sections:

1. **Executive Summary (1 page max)**

* Overall assessment in 5 bullets
* Top 10 risks (ranked)
* “If we shipped today, what breaks first?” (realistic scenario)

2. **Scorecard**
   Give a score **0–10** for each category below, plus a weighted overall score.
   For every score ≤8, list *why*, *evidence*, and *what it would take to reach 9–10*.
3. **Findings Table (the core output)**
   A table of findings with:

* ID, Severity (Blocker/Critical/Major/Minor/Nit)
* Category, Location (path + symbol), Symptom, Root cause
* Impact, Likelihood, How to reproduce (if applicable)
* Fix (specific), Effort estimate (S/M/L), Owner type (backend/devops/data/etc.)

4. **Refactor / Remediation Plan**

* A phased plan: **48 hours**, **2 weeks**, **6 weeks**
* Include “stop-the-bleeding” items vs. long-term architecture

5. **Diff-style suggestions**
   Provide at least 5 concrete change proposals (pseudo-diffs are fine), each tied to a finding.
6. **Non-obvious improvements**
   List 10+ improvements that aren’t typical lint/test suggestions (e.g., dependency hygiene, build reproducibility, observability, API ergonomics, failure isolation, etc.)

---

# 1) Review categories and criteria (be exhaustive)

## A. Product requirements & correctness

* Does the project clearly encode requirements? Where are they documented?
* Trace “major feature X” to code: entry points, flow, invariants.
* Identify ambiguous behavior, undefined edge cases, and mismatches between docs and implementation.
* Look for silent failures, swallowed exceptions, implicit defaults, and “works on my machine” assumptions.
* Are there explicit correctness properties? Any property tests? Invariants? Assertions?

## B. Architecture & modularity (the big one)

* High-level architecture: boundaries, layers, dependency direction, coupling.
* Evaluate whether modules have a single responsibility.
* Identify circular dependencies, leaky abstractions, shared mutable state, “god modules,” and tight coupling to frameworks.
* Are interfaces clean? Are adapters used for external systems?
* Extensibility: how hard is it to add a new feature without editing 15 files?

## C. API/UX design (library or service)

* Public API clarity: naming, consistency, discoverability.
* Backwards compatibility story, deprecation patterns.
* Error reporting: are exceptions meaningful? Are error types stable and documented?
* For CLIs: help text quality, exit codes, flags behavior, config precedence.
* For services: route consistency, request/response shape, versioning.

## D. Code quality (Python craftsmanship)

* Readability, cohesion, DRY vs. over-abstraction.
* Idiomatic Python: correct use of dataclasses, typing, context managers, iterators/generators.
* Identify anti-patterns: deep nesting, giant functions, boolean flag arguments, “stringly typed” config everywhere, inheritance misuse, hidden side effects.
* Naming quality: does the code read like a well-written technical document?
* Evidence of copy/paste, “action at a distance,” unclear ownership of state.

## E. Type safety & static analysis

* mypy/pyright usage level and strictness.
* Type coverage of public interfaces and tricky internals.
* Common type-smell checks: Any abuse, untyped dicts, Optional misuse, implicit None.
* Are types used to encode domain constraints or ignored?

## F. Testing strategy (quality, not just quantity)

* Test pyramid: unit vs. integration vs. end-to-end.
* Coverage of failure modes and edge cases, not just sunny-day paths.
* Determinism: flaky tests, time/network dependencies, random seeds.
* Fixtures, factories, test readability, test speed.
* Mutation testing potential: would tests catch logic inversion?
* Are there regression tests for known bugs?

## G. Security (assume hostile input)

* Input validation, injection risks, unsafe deserialization, SSRF, path traversal.
* Secrets handling: env vars, config files, logging secrets, committing credentials.
* Dependency vulnerabilities and supply-chain risk.
* Authn/authz (if relevant): session handling, token validation, least privilege.
* Sandboxing and dangerous operations (shell calls, eval/exec, pickle, yaml.load).
* Threat model: identify top threats and mitigations.

## H. Reliability & resilience (production reality)

* Fail-fast vs. degrade gracefully decisions.
* Retries, timeouts, circuit breakers, idempotency.
* Backpressure and queue behavior.
* Crash-only design considerations: can processes restart cleanly?
* Data corruption and partial writes: atomicity, transactions, exactly-once concerns.
* Resource cleanup: file handles, subprocesses, DB connections.

## I. Observability (can we debug it at 3am?)

* Logging strategy: structured logs, log levels, correlation IDs.
* Metrics: latency, error rates, saturation, queue sizes.
* Tracing: OpenTelemetry or equivalent.
* Meaningful error reporting and breadcrumbs.
* Are logs actionable or noisy? Any PII leaks?

## J. Performance & scalability

* Big-O concerns: hotspots, N+1 calls, repeated parsing/serialization.
* Memory use: accidental copies, caching strategy, lifecycle.
* Concurrency model: threads vs. asyncio vs. processes; correctness with shared state.
* Profiling hooks and benchmarking.
* Throughput constraints: IO vs CPU bound; GIL implications.
* For data pipelines: chunking, streaming, vectorization.

## K. Data integrity & persistence (if applicable)

* Schema migrations, versioning, compatibility.
* Constraints, indexes, transactional boundaries.
* Serialization formats: stability, validation, forward/backward compatibility.
* Idempotency and replay safety.

## L. Dependency management & packaging

* `pyproject.toml` / requirements hygiene.
* Pinning strategy, lockfiles, reproducible builds.
* Optional deps and extras separation.
* License compliance, transitive risk.
* Avoiding dependency hell: minimal surface area.

## M. DevEx: tooling, CI/CD, and workflow

* Pre-commit, lint, format, type checks, security scanners.
* CI speed and determinism; caching; parallelism.
* Release process: versioning, changelog discipline, artifacts.
* Local dev environment: one-command setup; Docker/devcontainers if needed.

## N. Documentation & maintainability

* README that gets a new dev productive fast.
* Architecture docs (diagrams, ADRs), “why” not just “what.”
* Docstrings and examples that match behavior.
* Runbooks: common failure cases + fixes.
* Contribution guidelines, coding standards.

## O. Style consistency & design uniformity

* Consistent conventions across modules.
* Unified error handling strategy.
* Unified config system and precedence.
* Unified naming, file layout, and layering.

## P. Compliance / privacy (if relevant)

* PII handling, retention, redaction.
* Audit logging.
* Data minimization.
* GDPR/CCPA-ish considerations if user data exists.

---

# 2) Mandatory “hard checks” you must perform

1. Identify the **top 3 most complex modules** and explain why they’re complex. Recommend simplifications.
2. Identify the **top 10 files by risk** (not by LOC): why they’re fragile.
3. List **all external system boundaries** (DB, network, filesystem, subprocess) and audit how each is handled (timeouts, retries, validation).
4. Find **at least 10 specific refactors** that would materially reduce bug risk.
5. Find **at least 10 examples** of code smells with exact locations.
6. Find **at least 5 potential security issues** (or explicitly argue why not, with evidence).
7. Find **at least 5 concurrency/async hazards** (or explicitly argue why not).
8. Evaluate **packaging/distribution**: can a clean machine install and run it deterministically?
9. Evaluate **test realism**: do tests actually represent production conditions?
10. Provide a **“minimum acceptable bar”** checklist for shipping.

---

# 3) Severity definitions (use these strictly)

* **Blocker**: unsafe or fundamentally broken; cannot ship.
* **Critical**: high likelihood of production incident or data/security risk.
* **Major**: significant maintainability/correctness concerns; should fix soon.
* **Minor**: quality improvement; low risk.
* **Nit**: style/consistency; only mention if it’s pervasive.

---

# 4) Constraints on your tone and behavior

* Be skeptical: assume bugs exist until proven otherwise.
* Prefer evidence over speculation.
* If you lack info (e.g., missing deployment target), state assumptions explicitly and review under multiple plausible scenarios.
* No hand-waving: every recommendation must include “how” and “where.”

---

# 5) End with an “Ideal Target State” blueprint

Describe what “excellent” looks like for this project in:

* repo structure,
* architecture boundaries,
* typing/testing standards,
* CI/CD pipeline,
* release strategy,
* ops/observability,
* and security posture.

Provide a concrete, actionable picture.
