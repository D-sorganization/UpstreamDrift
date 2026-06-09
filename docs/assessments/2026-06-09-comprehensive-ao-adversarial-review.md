# Comprehensive A-O Adversarial Review — 2026-06-09

**Assessor:** Claude (fleet agent, session branch `claude/zealous-noether-sdyvbm`)
**Scope:** Full repository — `src/api`, `src/engines` + `pose_interchange`, `src/shared/python`
(realtime, chat, ai, security, launchers), `tests/` + pytest config, build/deploy/CI
(Dockerfiles, compose, workflows, install, alembic), `rust_core` + PyO3 bindings,
`ui/` (TypeScript/React/Tauri), top-level launch scripts.
**Method:** Seven parallel adversarial review passes with file:line evidence required for
every finding; high-severity claims independently re-verified against source before filing.
Findings that already overlapped open issues #7125–#7133 were excluded by construction.

## Outcome

- **~70 raw findings** produced; **5 rejected as false positives** during verification
  (see below); the remainder consolidated into **20 GitHub issues**, each with evidence,
  severity, A-O category, a TDD-first fix plan, and DbC/DRY/LOD acceptance criteria.

## Issues filed, by A-O category

### A — Architecture & Implementation
- **#7144 (BLOCKER)** Pinocchio reference adapter contradicts canonical-v2 velocity layout —
  cross-engine v/a silently transposed (linear↔angular). Verified against
  `docs/conventions/canonical-v2.md` before filing.
- **#7148 (CRITICAL)** realtime transport: unsynchronized singleton init, no shutdown
  lifecycle, subscriber-thread leak on unsubscribe timeout.

### B — Code Quality & Hygiene
- **#7142 (MAJOR)** `LocalUser` stub missing User-model fields → AttributeError in local mode.
- **#7145 (MAJOR)** pose_interchange DbC gaps: asymmetric quaternion-layout heuristic (Drake),
  `JointSlot.length` ignored, unhelpful layout error.
- **#7147 (MAJOR)** upstream-pinocchio-id PyO3 boundary: shape-mismatch panics escape
  `except Exception` as `PanicException`, callback errors lose tracebacks, error paths untested.
- **#7165 (MAJOR)** Frontend trusts backend JSON shapes blindly (no runtime validation;
  NaN tile sort).

### F — Installation & Deployment
- **#7161 (MAJOR)** Docker build chain: pip pin skew (26.1 vs 25.3), heavy-test deps swallowed
  by `|| echo`, compose/healthcheck hardening.
- **#7163 (MAJOR)** Backend port conventions split 8000 vs 8001; Vite dev proxy wired to the
  Docker port. Verified.

### G — Testing & Validation
- **#7153 (CRITICAL)** Cross-engine contact "tests" are `assert True` documentation stubs.
- **#7155 (CRITICAL)** Test isolation: autouse `_protect_engine_modules` masks `sys.modules`
  corruption; module-scoped mujoco mock creates order-dependent visibility.
- **#7156 (CRITICAL)** `time.sleep()`-synchronized realtime/soak/chat tests flaky by design.
- **#7157 (MAJOR)** Assertion-quality ratchet: `.called`-only checks, unverified mock returns,
  `assert result is not None`.
- **#7158 (MAJOR)** pytest gating: blanket ImportError module-skips hide bugs; no marker
  discipline; over-broad coverage omits.

### H — Error Handling & Debugging
- **#7140 (MAJOR)** Unbounded pagination `limit`; precondition lambdas don't bind real args.
- **#7143 (MAJOR)** Quota incremented/committed before route handler runs.
- **#7146 (MAJOR)** Finite-difference helpers silently return zeros for short trajectories —
  undocumented contract (Python ×2 + Rust).
- **#7149 (MAJOR)** file pub-sub: publish failures invisible, rotation stat outside lock,
  Qt watcher re-arm races.
- **#7150 (MAJOR)** Chat WebSocket router: no disconnect cleanup — session leak.
- **#7151 (MAJOR)** launcher_process_manager `_stream_output`: pipe leak, blocked reader
  threads never reclaimed.
- **#7166 (MAJOR)** Frontend: `setSpeed` unhandled rejections, ChatPanel bypasses
  `getApiBase()`, unmount race.
- **#7167 (MAJOR)** API server DbC bundle: triple route registration unverified, late CORS
  validation, dead `return None`, silent name truncation.
- **#7168 (MINOR)** Launch scripts raw tracebacks; alembic env hard-import blocks offline mode.

### I — Security & Input Validation
- **#7139 (CRITICAL)** API key auth accepts expired keys — `expires_at` never checked; cached
  keys not re-validated. Verified by grep before filing.
- **#7152 (MAJOR)** `is_relative_to` ValueError unhandled in secure_subprocess; MCP npm package
  names unvalidated; credentials module logs env-var names.
- **#7159 (CRITICAL)** CI: `npm audit` failures swallowed by `|| echo`; Docker builds never run
  pip-audit. Verified.
- **#7164 (MAJOR)** Tauri v2 app declares no capabilities files — IPC permission surface
  implicit and unpinned.

### K — Reproducibility & Provenance
- **#7160 (MAJOR)** Python version matrix incoherent: pyproject ≥3.10, lock built on 3.12,
  Docker 3.12, CI 3.11, install.sh ≥3.11.

### O — CI/CD
- **#7162 (MAJOR)** ci-optional-stack runs pytest on files that don't exist
  (`tests/test_pinocchio_ecosystem.py`, `tests/test_pinocchio_recorder.py`) under
  `continue-on-error` — Pinocchio optional-stack coverage silently zero. Verified.

## False positives rejected during verification

1. **JWT `exp` datetime "not JSON-serializable"** — PyJWT natively converts `datetime` for
   registered claims; `src/api/auth/security.py:187` is correct as written.
2. **Single-frame `IndexError` in `inverse_dyn_pinocchio._finite_difference`** — the
   `if len(times) >= 2:` guard prevents it; Python and Rust agree (both return zeros).
   Reframed as the silent-contract issue #7146.
3. **Drake adapter "wrong rotation frame" for angular velocity** — the canonical-v2 table
   explicitly specifies CC-28 maps body-local angular velocity through `R_FM`; the code does
   exactly that.
4. **`BaseAgentAdapter` ellipsis-body abstract methods "incomplete"** — idiomatic
   `@abstractmethod` on an ABC; instantiation without override is already impossible.
5. **Tauri capabilities claimed as a functional BLOCKER for app commands** — downgraded to a
   MAJOR security-hygiene issue (#7164) pending runtime verification, since Tauri v2 app-command
   defaults differ from plugin-command gating.

## Categories with no new findings

C (Documentation), D (UX), E (Performance), J (Extensibility), L (Maintainability),
M (Education), N (Visualization) had no *new* discrete defects beyond what open issues
#7131–#7133 already track at the systemic level; per-issue DbC/LOD/DRY acceptance criteria
carry the L-category enforcement into every fix.
