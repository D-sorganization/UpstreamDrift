# BunkerShot3D pro-grade upgrade — state ledger

**Read this first on resumption.** Updated 2026-08-13.

Epic: https://github.com/D-sorganization/UpstreamDrift/issues/8607

## Where everything lives

| Thing                                    | Path                                                             |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Integration branch                       | `feat/bunkershot-pro-epic` (pushed)                              |
| Integration worktree                     | `C:\Users\diete\Repositories\UpstreamDrift-worktrees\bunker-pro` |
| Agent worktrees                          | `D:\bunker-worktrees\{geometry,sand,schema,backends,study}`      |
| Defect audit (B1-B31)                    | `_review/bunkershot-pro-2026-08-13/baseline-findings.md`         |
| Research digest                          | `_review/bunkershot-pro-2026-08-13/research-digest.md`           |
| **Research addendum (wins on conflict)** | `_review/bunkershot-pro-2026-08-13/research-digest-addendum.md`  |
| ADR                                      | `docs/adr/0032-bunkershot3d-club-design-architecture.md`         |

**Agent worktrees are on D: deliberately — C: is at 100% (about 600 MB free of 953 GB).**
Do not create new worktrees on C:. See "Blockers" below.

## Environment gotchas that cost time if rediscovered

- **Push with the venv on PATH** or the pre-push pytest hook runs the wrong interpreter
  (user-site Python 3.12 with a NumPy 1.x/2.4.6 binary mismatch) and fails on unrelated code:
  `export PATH="/c/Users/diete/Repositories/UpstreamDrift/.venv/Scripts:$PATH"` before `git push`.
- Prettier reformats Markdown on pre-push; `git add` the result and amend, do not fight it.
- Installed: numpy 2.2.6, scipy 1.17.1, pydantic 2.13.4, h5py, hypothesis, structlog, matplotlib,
  pandas, mujoco 3.9.0. **NOT installed and not to be added**: trimesh, pint, icontract, SALib,
  scikit-learn, torch, zarr, pytest-regressions.
- **No NVIDIA GPU on this machine** (Intel Iris Xe). Anything needing Newton/Warp/CUDA is
  optional and CI-skippable, per ADR-0032.
- Bash heredocs with apostrophes in the body break the shell wrapper here — use the Write tool
  or a Python driver for anything long (this bit twice).

## Status

### Done and committed on `feat/bunkershot-pro-epic`

- `b4dc4757a` — canonicalised the package import root (`src.bunkershot3d` -> `bunkershot3d`),
  removed a `sys.path` mutation at import time, widened the MuJoCo guard to catch `OSError`.
  Added `tests/unit/repo_hygiene/test_bunkershot3d_canonical_import.py`.
  **tests/bunkershot3d went from 138 passed / 2 failed to 157 passed / 0 failed.**
- ADR-0032 written and committed.
- Epic #8607 filed with 11 child issues #8608-#8618, all linked, with research findings
  attached as comments on #8609, #8610, #8611, #8613, #8614, #8616.

### Wave 1 — ALL FIVE MERGED into `feat/bunkershot-pro-epic` (pushed)

| Issue | Workstream                        | Commit      |
| ----- | --------------------------------- | ----------- |
| #8617 | W10 result schema v2 + provenance | `b445a128c` |
| #8610 | W3 sand state model               | `5b1783e49` |
| #8615 | W8 study layer (DOE/Sobol/GP)     | `8e4115f25` |
| #8612 | W5 backend correctness            | `1aa1e84ac` |
| #8609 | W2 parametric wedge geometry      | `b60c21519` |

Integrated suite: **1309 passed, 4 skipped** (baseline was 157). No merge conflicts —
the disjoint file ownership held.

Follow-ups the merges created:

- **Drivers do not yet pass a run manifest.** `BunkerShotResultWriter` takes `manifest=` /
  `set_manifest()`; one line per driver. Result files carry `schema_version` but no provenance.
- **`src/bunkershot3d/__init__.py` does not re-export `sand`, `study` or the new geometry
  API** — consumers must import the subpackage directly. One-line follow-up.
- **SPEC.md**: `spec-check` blocks PRs touching `src/**` without a SPEC.md update. It fires on
  pull_request only, so epic-branch merges are unaffected. Do **one** coordinated update when
  the epic PRs to `main`.
- **`reference_swing.csv` still does not exist** (B33). It is now a loud
  `TrajectoryUnavailableError` rather than a silent 5 m/s substitution, but the F0 solver
  work (#8611) will want a real reference trajectory.
- **Rocker radii need a decision** — see the CORRECTION box in `research-digest.md` §2. The
  patent's heel/toe/centre radii are local curvature, not blade-spanning arcs; the
  blade-scale values used by the reference geometry are inferred, not published.
- **Lofted head CG sits ~5 mm above the leading edge** vs the patent's 9.65–17.02 mm band.
  The parametric blade has no hosel or toe taper yet; the test asserts a plausibility band
  and says so rather than pretending a match.

### Wave 2 — not started (depends on wave 1)

| Issue | Workstream                                      | Depends on                                          |
| ----- | ----------------------------------------------- | --------------------------------------------------- |
| #8611 | W4 **DRFT solver** — the default F0 tier        | #8609 geometry, #8610 sand                          |
| #8613 | W6 the ball + `SwingBallFlightPipeline` handoff | #8611                                               |
| #8614 | W7 designer metrics                             | #8611, #8617                                        |
| #8616 | W9 V&V suite + credibility statement            | #8611                                               |
| #8618 | W11 designer workbench GUI                      | most of the above                                   |
| #8608 | W1 foundations (value objects, units)           | partly absorbed by W2/W3 — re-scope before starting |

**#8611 is the critical path.** Its implementation data (the 20-term 3D-RFT polynomial table,
the material-scaling cubic, the two DRFT corrections) is in the research addendum §3.

## How to resume

1. `cd C:\Users\diete\Repositories\UpstreamDrift-worktrees\bunker-pro && git fetch origin`
2. Check which wave-1 branches have commits: `git log --oneline feat/bunker-<name> ^feat/bunkershot-pro-epic`
3. For each finished branch: review the diff, run
   `python -m pytest tests/bunkershot3d -p no:randomly --timeout=180`, then merge into
   `feat/bunkershot-pro-epic`. Merge in dependency order: schema, sand, geometry, backends, study.
4. Re-run the full bunker suite on the integrated branch before starting wave 2.
5. Launch wave 2 agents the same way — isolated worktree on D:, disjoint file ownership,
   TDD, no push/PR from the agent.
6. Tick the epic checklist as each workstream lands.

## Non-negotiables carried from the ADR

- Every result carries a **fidelity tier and a validity verdict**; out-of-envelope queries refuse
  rather than return a plausible number. This matters more than usual because **we are ~60x
  outside RFT's stated Fr < 0.4 envelope** and ~20x beyond any published validation.
- No new hard dependencies.
- Borrowed constants must be labelled borrowed, never presented as measurements
  (this package already failed that way once — issue #7999).
- Safety-critical checks are `raise`, never `assert` (`python -O` strips asserts).

## Blockers needing the user

1. **C: drive is full** — 600 MB free of 953 GB, with 74 UpstreamDrift worktrees on it. Agent
   worktrees were relocated to D: as a workaround, but this will keep causing failures
   fleet-wide. Cleaning up stale worktrees is destructive and was not done unilaterally.
2. **BSD-4-Clause advertising clause** on Kratos MPM (the best CPU-capable higher-fidelity
   tier) — fine internally, but a commercial product must reproduce the acknowledgement.
   Flag before anything ships.
