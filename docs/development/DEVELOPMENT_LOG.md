# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-08 by claude

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#9762 · bioptim Optimal-Control Backend and the Swing-Dynamics Fixes

- **State:** in_review
- **Owner:** claude
- **Issue:** #9762 (epic); prerequisites #9755, #9756, #9757, #9758, #9759, #9760, #9761
- **Branch:** `claude/fixes-epic-implementation-x2bu36`
- **PR:** not created
- **Paths:** `src/shared/python/optimization/ocp/`,
  `src/shared/python/optimization/casadi_backend.py`,
  `src/shared/python/optimization/model_provider.py`,
  `src/shared/python/optimization/backend_registry.py`,
  `src/shared/python/motion_pipeline/model_bridge.py`,
  `src/shared/python/estimation/`, `benchmarks/bioptim_parity.py`,
  `docs/adr/0050-optimizer-backend-registry-and-bioptim.md`,
  `docs/estimation/bioptim_parity.md`, `docs/issues/EPIC_BIOPTIM_OCP_INTEGRATION.md`,
  `.github/workflows/ci-optional-stack.yml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`edc302294`)
- **Summary:** Adopts `pyomeca/bioptim` as an opt-in optimal-control layer
  driven by UpstreamDrift's own CasADi dynamics through bioptim's custom-model
  protocol (no biorbd, no conda), and fixes the defects that made the existing
  dynamic backends unphysical. The URDF bridge and model provider now emit
  anthropometric link inertials so torque limits mean something (#9755); the
  CasADi backend gains mass-matrix, forward-dynamics and RK4 kernels, a
  `dynamics_defect` diagnostic and a real multiple-shooting transcription, and
  its finite-difference path is deprecated (#9756); the MAP estimators count and
  can refuse non-finite residuals (#9757) and gate free parameters on
  identifiability (#9758); the optional-stack lane gained CasADi, Crocoddyl and
  bioptim legs (#9759); `backend_registry` plus ADR-0050 assign each of six
  backends a problem class (#9760). Phases 0-3 of the epic are implemented and
  tested: compat shims, `SymbolicSwingModel` validated against Pinocchio,
  `SwingBioModel`, the clubhead-speed OCP with a parity benchmark, and the
  keypoint-tracking OCP. Two structural findings are recorded rather than
  hidden: maximising terminal speed is concave and does not converge in any
  backend once the dynamics are enforced (so the OCP defaults to a convex
  target-speed objective), and the six-marker set cannot observe the full
  seven-DOF chain (hip and trunk rotation are an exact null direction), so every
  tracking solve reports what it could not see.
- **Next step:** Open the pull request for
  `claude/fixes-epic-implementation-x2bu36` and record CI.

### DL-#9494 · Resolve the CLAUDE.md `--no-verify` Contradiction by Fixing the Windows Hook Environment

- **State:** in_review
- **Owner:** claude
- **Issue:** #9494
- **Branch:** `claude/issue-9494-precommit-env`
- **PR:** #9744
- **Paths:** `CLAUDE.md`, `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`, `SPEC.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`dbc6727aa`)
- **Summary:** CLAUDE.md forbade `git commit --no-verify` while agents on
  Windows reported every pre-commit invocation failing (hook virtualenvs
  targeting Python 3.11, absent from the workstation). Investigation found no
  interpreter pin left in `.pre-commit-config.yaml` (`default_language_version:
python: python`, 3.11 pin removed by #1792/#2720), and on Python 3.13.3 every
  commit-stage hook plus pre-push `mypy`/`bandit` passes after a from-scratch
  environment build. Option (a) of the issue is therefore satisfied; CLAUDE.md
  now documents the resolved environment and states that the `--no-verify`
  prohibition stands on Windows with no blanket exception.
- **Next step:** Record CI on PR #9744; on merge, confirm the protected-main
  sync lands the resolved hook environment note.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.
