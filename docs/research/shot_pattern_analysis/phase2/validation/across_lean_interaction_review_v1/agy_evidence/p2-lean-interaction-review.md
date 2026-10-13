# Phase 2 Across-Lean Interaction Scoring Review

## Summary of Defect and Resolution

### Defect

In `src/tools/shot_pattern_analysis/matched_lean_control.py`, `analyze_explicit_source_pair` computed `course_scoring_effect_modification` only when `comparison_kind in ("curve_strength", "shaft_rotation_sd")`. While `across_lean` explicit pairs share the exact same validated six-population intersection and are consumed by `matched_interpretation.py` and rendered by the bounds-control relative-pattern table in `matched_supplement_report.py`, they bypassed the scored-vector pathway. Real `across_lean` records consequently lacked Draw and Fade relative interaction fields, causing the relative-pattern table to display "Unavailable".

### Resolution

- Extended the comparison kind check in `analyze_explicit_source_pair` to `comparison_kind in ("across_lean", "curve_strength", "shaft_rotation_sd")`.
- `across_lean` pairs now route through `score_course_endpoint_vectors` and `_paired_effect_modification`, calculating:
  $$\Delta\Delta = (\text{curve} - \text{Straight})_B - (\text{curve} - \text{Straight})_A$$
  matched strictly by common shot index on the six-population intersection.
- Bootstrap pairing and exact course scoring provenance (`scenario_id`, `baseline`, `api_evaluated_states_per_condition`, `index_order`) are preserved.
- Unavailability behavior is strictly maintained:
  - If fewer than 2 common supported shots exist, pairwise scoring is skipped and `status = "unavailable"` with reason `"fewer_than_two_common_supported_shots"`.
  - If `score_course_endpoint_vectors` raises a `ValueError` (e.g., out-of-support baseline knots), `course_scoring_effect_modification` records `{"status": "unavailable", "reason": str(exc)}`.
- Existing per-pattern metrics, nominal targets, and configuration/source validations remain unchanged.
- `speed_bound` behavior is unaltered.
- File-size budgets (< 1200 lines) are strictly respected.

---

## Test Counts and TDD Progression

### RED Phase

- Added 6 comprehensive unit tests to `tests/tools/shot_pattern_analysis/test_matched_explicit_source_pair.py`:
  1. `test_across_lean_pair_reports_paired_sg_interaction`: Verifies $(curve - Straight)_B - (curve - Straight)_A$ sign, magnitude, and provenance on across-lean pairs.
  2. `test_across_lean_pair_uses_same_six_conditional_intersection_for_scoring`: Verifies that only shots supported in all 6 populations are passed to `score_course_endpoint_vectors`.
  3. `test_across_lean_pair_fewer_than_two_common_shots_marks_unavailable`: Verifies fail-closed behavior when common supported shot count is under 2.
  4. `test_across_lean_scoring_failure_records_unavailable_status`: Verifies graceful degradation when vector scoring raises `ValueError`.
  5. `test_across_lean_pair_rejects_non_ascending_or_unsupported_order`: Verifies validation of ascending lean order (0, 5, 10).
  6. `test_across_lean_pair_rejects_target_identity_mismatch`: Verifies strict validation of declared target identity.
- Execution before implementation: **3 failed, 22 passed** (Total 25 tests).
- Log saved to: `/tmp/p2-lean-interaction-red.txt`.

### GREEN Phase

- Applied the surgical 1-line update to `src/tools/shot_pattern_analysis/matched_lean_control.py`.
- Execution after implementation: **25 passed, 0 failed** in `test_matched_explicit_source_pair.py`.
- Log saved to: `/tmp/p2-lean-interaction-green.txt`.

### Verification Suite Execution

- **52 passed** across all three focused suites:
  - `tests/tools/shot_pattern_analysis/test_matched_explicit_source_pair.py`: 25 passed
  - `tests/tools/shot_pattern_analysis/test_reviewed_driver_curve_bounds.py`: 12 passed
  - `tests/tools/shot_pattern_analysis/test_matched_interpretation.py`: 15 passed
- **Ruff Check**: All checks passed.
- **Ruff Format**: 2 files formatted, 0 changes needed.
- **Scoped Mypy** (`--python-version 3.12 --follow-imports=silent`): Success, 0 issues across 2 source files.
- **Line Count Budgets**:
  - `src/tools/shot_pattern_analysis/matched_lean_control.py`: 1089 lines (Budget: 1200)
  - `tests/tools/shot_pattern_analysis/test_matched_explicit_source_pair.py`: 1177 lines (Budget: 1200)
- Log saved to: `/tmp/p2-lean-interaction-checks.txt`.

---

## Limitations

1. **Paired Conditional Simulation**: Matched-condition contrasts include nominal re-matching and do not isolate a causal curvature effect. Mirrored outcome symmetry is neither assumed nor inferred.
2. **Pointwise Monte Carlo Intervals**: Sampling intervals are computed on shared seeded primitive draws; cells are correlated and do not represent independent experimental replications or multiplicity-adjusted discoveries.
3. **Common Support Requirement**: Analysis requires at least 2 common supported shots across all six physical populations. Any pair with fewer than 2 mutually valid shots is marked unavailable.
4. **Baseline Domain Bounds**: Course scoring relies on published Broadie baseline lookup tables; conditions projecting outside published support mark relative interaction effect modification as unavailable with recorded diagnostics.

---

## Source Hashes (SHA-256)

| File                                                                     | SHA-256 Digest                                                     |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `src/tools/shot_pattern_analysis/matched_lean_control.py`                | `0c57f3c1a2535f63ba059b5c6fb5c3414f83cc6b9a1169a429cf14a19bbd12c0` |
| `tests/tools/shot_pattern_analysis/test_matched_explicit_source_pair.py` | `cd52f6fe0535c1163e2ed0bf05a8a6846219424cd791f7dae30c2ba20b5c59c9` |
| `src/tools/shot_pattern_analysis/course_score_vectors.py`                | `6c7dbe509c6100c217f421ab30a10cc9c46bf5b6ccf4a04682a3e00f27402c2a` |
| `src/tools/shot_pattern_analysis/scoring.py`                             | `4a536b0609a074b8f2c1bbd3d7dc628b0d4adcfcfcef7de17ca28e1f8c7f6a0b` |
| `src/tools/shot_pattern_analysis/matched_interpretation.py`              | `4ed78414a6daefa6e5215a376cdea5e36c03274c2190d3c823104fa180f3ce80` |
