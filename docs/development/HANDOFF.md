# Implementation Handoff - Verify Companion Screenshot Bytes and Pixel Dimensions (#9191)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-ud-9191-screenshot-verifier`
- Branch: `agy/ud-9191-screenshot-verifier`
- Baseline commit: `28b37bd47eb85d977e530e60659c6f54ca0a1a2b` (origin/main)
- Pull request: draft, opened from `agy/ud-9191-screenshot-verifier`
- Governing issue: #9191 (development log DL-#9191)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: Fail closed on any captured screenshot whose bytes or pixel size disagree with its manifest record (#9191).
- Status: in_review (draft PR)
- Completed:
  1. agy (Gemini 3.8 Flash): verifier, stdlib PNG IHDR parser, 0/1/2 CLI, fixtures building real PNGs, and a catalog wire-in test.
  2. Orchestrator review: split the single ~125-line verifier into per-status helpers (DRY null-field loop, `_resolve_asset`, `_png_size`) and dropped a SystemExit trap around argparse.

## Validation

- `pytest tests/unit/scripts/test_verify_companion_screenshots.py tests/companion/test_companion_catalog.py` -> 57 passed.
- `python scripts/ci/check_architecture_budget.py` -> OK.

## Next Steps

1. CI green, review, merge.
2. #9191 remaining: governed capture workflow step (COMP-B2), real light/dark/responsive assets, AffineDrift #4025 alignment.

## Change Log

- `SELF` — Verify Companion Screenshot Bytes and Pixel Dimensions (#9191, DL-#9191).
