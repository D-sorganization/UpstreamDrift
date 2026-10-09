# Matched Delivery Study Turnover

## Current State

The phase-two goal is active and incomplete. Worktree: `/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift`; branch: `codex/shot-pattern-matched-delivery`, based on phase-one PR [#11941](https://github.com/D-sorganization/UpstreamDrift/pull/11941) at `10c7d61aa686e456b7962f1d9aa9ecfda4e8405f`. The prior PR is open and held; this phase has not been pushed. Do not modify the primary checkout or vendor submodule.

Read [Design](DESIGN.md), [Astra Review](ASTRA_REVIEW.md), and [Issue Records](issues/) first. Gemini 3.8 Flash High through `agy` filed #11972–#11976 using the user's expressly authorized personal identity. Do not close them without merged evidence.

## Implemented and Tested

- Seeded paired seven-variable Gaussian noise with PSD correlation validation; immutable raw-target matching; static loft/whole-club lean geometry; documented10°/34°/46° lofts. Primitive shaft twist is not face-angle SD.
- Full finite-head impulse bookkeeping with explicit CG/inertia and local bulge/roll geometry. The maximum-dissipation/Newton law is a **comparator**, not the planned primary grip model. Its first stick branch failed Astra's coupled e=1 passivity example and was corrected; the failed experiment remains documented.
- Primary incremental Coulomb/energetic-restitution solver and JIT/reference parity are committed (`637f9e036f`); independent 100-contact checks pass. Do not run the final matrix or claim the contact law empirically qualified. The pure-Python reference must be genuinely independent of compiled helpers for parity.
- Matched runner retains all unsupported samples, uses raw endpoints without aiming rotation, reports actual center-face and local contact-normal angles, and explicitly conditions statistics/scoring. Export captures sources before simulation and publishes atomically with compressed CSV, summaries, baseline, plots and hashes; source drift/cancellation prevents publication.
- Complete324-cell candidate inventory: three clubs × three leans × two curve strengths × two shaft-rotation SDs × nine noise profiles. Infeasible candidates remain explicit. Small pilots default to32 shots/pattern; final10000-shot cells require source-bound acceptance evidence.
- Academic density/HDR/marginal and overhead exports exist. Matched GUI and figure layout are committed (`8fa4fc80e0`); advanced law controls and overview figures are being finished. Counts and conditional density labels must remain visible.

Initial QP-path integration41tests, export6tests, and catalog4tests pass; saved logs are in `validation/`. They are initial evidence, not final acceptance of the energetic law or full UI. Root implementation checkpoint: `81fabbc531`; follow-up scoring-radius, joint covariance, sampling uncertainty and cross-cell source-drift regressions are being committed. Astra review findings are tracked in #11988; physical qualification remains open in #11984 and `docs/development/planning/DV-11984.md`.

## Next Actions

1. Finish and independently review incremental contact, energetic-COR semantics, pure Python/JIT parity, refinement, and profiled acceleration. Add explicit law/backend/resolution to configuration and provenance. The existing matched adapter still needs to select the primary law after acceptance.
2. Finish GUI integration and run end-to-end cancellation/configuration/export tests. Inspect actual pilot density and flight figures.
3. Run deterministic three-club × three-lean × two-shape-scale preflight for within-lean and common-distance feasibility. Record bounded optimizer residuals and tighter-step replay; do not force driver10°/lean10° into an airborne trial.
4. Run small pilots, inspect all324 candidate statuses, verify every invalid input retained, and conduct Astra's second independent review. Freeze scientific sources before large jobs.
5. Run the accepted final matrix, numerical/contact-law/head-parameter sensitivities, complete statistics/scoring, and polished linked Markdown/LaTeX/PDF report. No final results exist yet.
6. Update discovery/registry/manual-reference records and change fragment; run focused and canonical pre-PR gates, then publish safely. Recheck the existing PR and baseline blocker state once at that breakpoint; do not poll CI.

## Environment and Commands

Use `/tmp/shot-pattern-validation-venv-20261009/bin/python` (Python3.12.13, NumPy2.2.6, mypy1.20.1). Numba0.68.0 and llvmlite0.50.0 were installed only into this isolated environment; original project environment stays unchanged. Optional extra `shot-pattern-acceleration` is being added. Headless environment: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy`.

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
/tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest \
  tests/tools/shot_pattern_analysis tests/ui/tools/shot_pattern_analysis \
  tests/unit/physics/test_rigid_contact.py tests/unit/physics/test_incremental_contact.py \
  -o addopts='' -q

# Only after the incremental adapter and source snapshot are stable:
MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python \
  -m src.tools.shot_pattern_analysis.matched_matrix \
  docs/research/shot_pattern_analysis/phase2/pilot_v1 --stage pilot --shots 32
```

Review commands against current `--help` before execution. Large runs require `--stage final --shots 10000 --acceptance PATH`. Never fabricate the acceptance receipt: independent review, pilot, numerical, rendered-figure and software evidence must be hashed and verified first.
