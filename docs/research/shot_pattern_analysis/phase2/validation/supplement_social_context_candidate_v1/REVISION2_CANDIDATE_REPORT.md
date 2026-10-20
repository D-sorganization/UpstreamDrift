# Candidate Revision Record: Supplemental Social Context (#12115)

## Overview

- **Issue**: #12115
- **Branch**: `codex/shot-pattern-social-context`
- **Worktree**: `/home/dieterolson/.codex/worktrees/shot-pattern-final-layout/UpstreamDrift`
- **Superceded Candidate**: `b9f1ad05b4`
- **Revised Commit**: `14f27706e4` (`14f27706e4e9e52ca1324a7a2bac59e86ab33ef9`)

## File Checksums (SHA-256)

- `src/tools/shot_pattern_analysis/matched_pair_social.py`: `16a4a48eb32432cbbb39648150b5e428c014af920845e006ddf12be0375fe369`
- `src/tools/shot_pattern_analysis/matched_supplement_figures.py`: `9254d7aede64f5aed19bd32f7cd5a8fb1b1899f6e18d000ba2096b228fc18246`
- `tests/tools/shot_pattern_analysis/test_matched_pair_social.py`: `33a0123655348810fd5fe377e9873b67ec65e685fc133c1bebafc724b8c8600e`
- `tests/tools/shot_pattern_analysis/test_matched_social_integration.py`: `1b5ddb65bffa37d4d0b262754fa6064b5be716c009aa064b2e5222d4d996b075`

## Revisions Implemented

1. **Narrowed Pair Social API**:
   - Replaced speculative overlapping `context` and `metadata` fallback dictionaries on `write_pair_social_density` with explicitly typed optional `source_protocols: tuple[str, str] | None` and `target_modes: tuple[str, str] | None`.
   - Distinct ordered facts are mapped unambiguously to the two condition labels:
     - `Protocol: {label0}: {p0} · {label1}: {p1}`
     - `Matched Target: {coord} ({label0}: {m0} · {label1}: {m1})`
   - Preserved legacy call behavior when options are omitted (`(Common Target)` emitted, `Protocol` key excluded).
2. **Target Coordinate Validation & Nonzero Lateral Representation**:
   - Validated `target_point_m` has exactly two finite real coordinates (rejecting non-finite values, strings, booleans, and improper lengths).
   - Numeric `Matched Target` caption represents both downrange and lateral values when lateral is nonzero (`f"{target_x:.2f} m downrange, {target_y:.2f} m lateral"`).
   - Original target XY tuple is passed unchanged to `write_academic_dispersion_figure`.
   - Validated `source_protocols` and `target_modes` are pairs of two nonempty strings.
3. **Exact Pair Figure Context & Classification**:
   - `_render_pair_figures` passes exact ordered source protocols plus ordered target modes computed from validated references via `_cell_target_mode`.
   - `_cell_target_mode` classifies using only actual `config`, `requested_config`, and `execution_kind` (returning `"Within-Lean Nominal Target"`, `"Fixed-Target Control"`, or `"Fixed-Target Control (Alias)"`), with no speculative `alias_of_primary` flag.
   - Removed unused `protocol_id` fallback across validated cell and pair figures; missing actual `source_protocol` raises `ValueError` immediately.
4. **Realistic Regression Fixtures**:
   - Replaced unrealistic lean-5 alias of lean-0 fixture with realistic same-cell zero-lean reuse of zero-lean primary with `requested_config` target set.
   - Added test coverage for mixed original/control protocol pair basis (`original_primary/1` vs `pw-control-face-bounds/1`).
   - Retained array-coordinate, population, and sample count assertions.
   - Removed tests covering deleted speculative options.

## Validation Results

- **Red Tests (failing on candidate b9f1 before source updates)**:
  - Log: `/tmp/p2-social-context-r2-red.txt`
  - Result: 16 failed, 18 passed in 76.08s
- **Green Tests (passing on revised candidate)**:
  - Log: `/tmp/p2-social-context-r2-green.txt`
  - Result: 34 passed in 108.59s
- **Ruff & Format Check**:
  - Log: `/tmp/p2-social-context-r2-lint.txt`
  - Result: All checks passed, 4 files formatted
- **Scoped Mypy Check (`--python-version 3.12 --follow-imports=silent`)**:
  - Log: `/tmp/p2-social-context-r2-mypy.txt`
  - Result: Success: no issues found in 4 source files
- **Review Contract Suite**:
  - Result: 15 passed in 103.25s

_No production asset claims; root/Astra will qualify actual cards._
