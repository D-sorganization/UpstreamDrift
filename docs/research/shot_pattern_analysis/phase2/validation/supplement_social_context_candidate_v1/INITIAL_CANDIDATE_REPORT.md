# Candidate Repair Summary: Identify Protocol and Target on Supplemental Social Figures

## Tracking

- **Issue**: [#12115](https://github.com/D-sorganization/UpstreamDrift/issues/12115) - Identify Protocol and Target on Supplemental Social Figures
- **Parent Issues**: #11975, #12000
- **Branch**: `codex/shot-pattern-social-context`
- **Worktree**: `/home/dieterolson/.codex/worktrees/shot-pattern-final-layout/UpstreamDrift`

## Problem Solved

Standalone supplemental social cards (all 52 cell cards across 36 PW-primary and 16 control cards, plus 29 inspected pair-social-density SVGs) lacked `source_protocol` and numeric matched target distances. Without this context, standalone figures could not distinguish within-lean nominal matching from fixed-target controls, revised face bounds, or cross-protocol comparisons.

## Implementation Details

1. **`src/tools/shot_pattern_analysis/matched_pair_social.py`**:

   - Extended `write_pair_social_density` signature with backward-compatible optional arguments: `source_protocols`, `context`, and `metadata`.
   - Added Design-by-Contract validation enforcing a sequence of two non-empty protocol strings and finite downrange target coordinates.
   - Formatted `Protocol` preserving condition order (`f"{condition_labels[0]}: {p0} · {condition_labels[1]}: {p1}"`).
   - Formatted `Matched Target` with numeric distance and target context (`f"{target_x:.2f} m downrange ({target_context})"`, defaulting to `"Common Target"`).
   - Preserved all valid/total shot counts, shaft noise descriptions, and automatic layout wrapping without font shrinkage or clipping.

2. **`src/tools/shot_pattern_analysis/matched_supplement_figures.py`**:

   - Added helper `_cell_target_mode(ref)` distinguishing `"Within-Lean Nominal Target"`, `"Fixed-Target Control"`, and `"Fixed-Target Control (Alias)"`.
   - In `_render_cell_figures`, added `"Protocol"` and `"Matched Target"` to `social_metadata`, propagating to both cell density social and cell overhead social figures (via `_raw_flight_metadata`).
   - In `_render_pair_figures`, passed ordered `source_protocols` and `context` to `write_pair_social_density`.
   - Preserved existing pair-overhead behavior which already printed target and ordered protocols.

3. **`tests/tools/shot_pattern_analysis/test_matched_pair_social.py` & `test_matched_social_integration.py`**:
   - Added TDD unit and integration tests verifying protocol ordering, target context distinction (within-lean vs control vs alias), DbC invariants, and backward compatibility.

## Verification Evidence

- **RED Test Run**: `/tmp/p2-social-context-red.txt` (12 failed, 10 passed before implementation).
- **GREEN Test Run**: `/tmp/p2-social-context-green.txt` (28 passed in 49.91s).
- **Ruff Lint & Format**: `/tmp/p2-social-context-lint.txt` (All checks passed, 4 files formatted).
- **Mypy Type Check**: `/tmp/p2-social-context-mypy.txt` (Success: no issues found in 4 source files).

## Limitations

- Production card images and SVGs were intentionally not regenerated in this candidate to preserve pipeline provenance and keep MAIN/BOUNDS frozen. Root/Astra will run qualification and publication regeneration separately.
- No modifications were made to statistical sampling, trajectory simulation, dispersion scoring, or output ledger contracts.
