# Implementation Review: Display Conditional Course Scoring in Matched Study Results

**Date**: October 9, 2026  
**Auditor/Implementer**: Gemini 3.8 Flash High  
**Branch**: `codex/shot-pattern-gui-scoring`  
**Base Commit**: `2e0a74f8ca`  
**GitHub Issue**: [#12067](https://github.com/D-sorganization/UpstreamDrift/issues/12067) (`https://github.com/D-sorganization/UpstreamDrift/issues/12067`)  
**Issue JSON**: `/tmp/p2-gui-sg-issue.json`

---

## 1. Executive Summary

This implementation resolves the finding that `matched_gui.MatchedStudyWidget._completed()` displayed nominal inputs and individual-supported population metrics but omitted the saved `course_scoring` payload produced by `matched_runner.py` / `score_course_endpoints()` and stored in `summary.json`.

Following **TDD**, **DbC**, **LoD**, and **DRY** principles:

1. **Red Test Phase**: Created unit and UI tests demonstrating that `MatchedStudyWidget` lacked strokes gained presentation (`AttributeError: MatchedStudyWidget must have scoring_label`).
2. **Pure Data Formatting Helper**: Implemented `format_matched_course_scoring()` in `src/tools/shot_pattern_analysis/matched_gui.py` to format saved course scoring without mutating physics, recomputing scores, or depending on Qt widget state for testing.
3. **GUI Integration**: Added `self.scoring_label` to `MatchedStudyWidget` in `_build_ui()` and wired it in `_completed()` to display the formatted course scoring payload.
4. **Layout & Height Compliance**: Confirmed through offscreen Qt tests that the widget and its parent `MainWidget` strictly satisfy embedded height constraints (`<= 760` px) within `(1000, 760)` launcher dimensions.
5. **No Mutation of Scientific Source / Live Simulations**: No physics, runners, or live Monte Carlo processes were touched. No git commits or pushes were made.

---

## 2. Source-Bound Design & Implementation Details

### A. Preconditions and Postconditions (Design by Contract)

- `format_matched_course_scoring(scoring, *, club_id, target_x_m)` accepts a mapping or None.
- Preconditions strictly handle missing keys, malformed payloads, non-finite values (NaN / Inf), and custom club scenarios without crashing.
- Postcondition: Always returns a truthful, human-readable string. Never converts missing or null values into zeros or empirical claims.

### B. Encapsulation & Law of Demeter

- Safely accesses dictionary values via `.get()` with type guards (`isinstance(..., Mapping)`), avoiding deep assumption-heavy object chains.
- Preserves the existing `self.result_label` nominal deliveries, residual, and per-pattern individual-supported counts (`Straight: ... 8/8 supported`).
- Encapsulates course scoring inside `self.scoring_label` directly on `MatchedStudyWidget`.

### C. Course Scoring Information Displayed

1. **Scenario Context**:
   - **Driver**: Discloses flat hole distance (e.g. 400 m hole) and fairway corridor width (e.g. 30 m fairway corridor, rough outside; carry landing treated as final lie).
   - **Irons / Wedge**: Discloses fairway start distance (e.g. from 180.0 m fairway start) and centered circular green radius (e.g. 15 m green radius, rough outside; carry landing treated as final lie).
2. **Common Supported Paired Count**:
   - Explicitly displays the intersection denominator: `Common supported pairs: {paired_n}/{total_n}`.
   - For `conditional_only` status, explicitly clarifies: `(conditional on common supported indices only; full population unavailable due to unsupported terminal lies)`.
3. **Three Pattern Mean SG**:
   - Displays formatted values: `Mean Strokes Gained: Straight: +0.250; Draw: +0.310; Fade: +0.210.`
   - Non-finite or missing values display `unavailable`, never `+0.000`.
4. **Paired Differences vs Straight with Actual 95% Interval Terminology**:
   - Inspecting `uncertainty.py` / `gui.py`, the interval is a sampling-only Monte Carlo percentile interval (`paired_mean_difference`).
   - Uses actual provided terminology: `Paired Difference vs. Straight (95% Monte Carlo Interval): Draw: {estimate:+.3f} [{lower:+.3f}, {upper:+.3f}] strokes; Fade: {estimate:+.3f} [{lower:+.3f}, {upper:+.3f}] strokes.`
   - Avoids unsupported broad scientific claims or assuming empirical bootstrap.
5. **Caveats**:
   - Discloses: `Caveats: Model-conditional historical PGA Tour benchmark (approximate baseline fit; not player-specific). Flat hypothetical geometry; carry landing is final lie without roll, hazards, or obstruction. Intervals describe Monte Carlo sampling only.`
6. **Unavailable & Edge Cases**:
   - Insufficient pairs (`< 2` common pairs): `Model-conditional strokes gained is unavailable: Insufficient common supported shots ({paired_n}/{total_n} common pairs; minimum 2 required).`
   - Custom club: `Model-conditional strokes gained is unavailable for custom club: No published baseline context (only Driver, 7-Iron, and Pitching Wedge illustrative presets are supported).`
   - Unsolved nominal: `Model-conditional strokes gained is unavailable because the nominal delivery could not be solved.`

---

## 3. Truthful Red / Green Logs

### A. Red Phase (Failure Demonstration)

```text
$ QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -o addopts='' -q tests/ui/tools/shot_pattern_analysis/test_matched_gui.py -k test_completed_displays_saved_course_scoring_strokes_gained
F                                                                        [100%]
=================================== FAILURES ===================================
_________ test_completed_displays_saved_course_scoring_strokes_gained __________
    ...
>       assert hasattr(widget, "scoring_label"), "MatchedStudyWidget must have scoring_label"
E       AssertionError: MatchedStudyWidget must have scoring_label
E       assert False
E        +  where False = hasattr(<src.tools.shot_pattern_analysis.matched_gui.MatchedStudyWidget object at 0x7181c2d29db0>, 'scoring_label')
=========================== short test summary info ============================
FAILED tests/ui/tools/shot_pattern_analysis/test_matched_gui.py::test_completed_displays_saved_course_scoring_strokes_gained
1 failed, 22 deselected, 6 warnings in 6.89s
```

### B. Green Phase (UI & Formatting Tests Pass)

```text
$ QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -o addopts='' -q tests/ui/tools/shot_pattern_analysis/test_matched_gui.py
.............................                                            [100%]
29 passed, 6 warnings in 19.03s
```

### C. Regression Verification (Adapter & Sizing Tests Pass)

```text
$ QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -o addopts='' -q tests/ui/tools/shot_pattern_analysis/test_matched_gui.py tests/ui/tools/shot_pattern_analysis/test_embed_adapter.py
......................................                                   [100%]
38 passed, 6 warnings in 33.59s
```

### D. Pure Engine / Scoring Tests Pass

```text
$ QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -o addopts='' -q tests/tools/shot_pattern_analysis/test_matched_runner.py tests/tools/shot_pattern_analysis/test_scenario_scoring.py
....................                                                     [100%]
20 passed, 6 warnings in 21.69s
```

### E. Code Quality Checks (Ruff & Mypy)

```text
$ /tmp/shot-pattern-validation-venv-20261009/bin/python -m ruff check src/tools/shot_pattern_analysis/matched_gui.py tests/ui/tools/shot_pattern_analysis/test_matched_gui.py
All checks passed!

$ /tmp/shot-pattern-validation-venv-20261009/bin/python -m ruff format --check src/tools/shot_pattern_analysis/matched_gui.py tests/ui/tools/shot_pattern_analysis/test_matched_gui.py
2 files already formatted

$ /tmp/shot-pattern-validation-venv-20261009/bin/python -m mypy --follow-imports=silent src/tools/shot_pattern_analysis/matched_gui.py
Success: no issues found in 1 source file
```

---

## 4. Changed Files Summary

| File                                                       | Change Type    | Description                                                                                                                                                                                                                                |
| ---------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/tools/shot_pattern_analysis/matched_gui.py`           | Implementation | Added `format_matched_course_scoring()` pure helper; added `self.scoring_label` to `MatchedStudyWidget`; updated `_completed()` to render saved course scoring.                                                                            |
| `tests/ui/tools/shot_pattern_analysis/test_matched_gui.py` | Testing        | Added 7 unit/UI tests covering available approach & driver scoring, conditional-only status, unavailable status with reasons, custom-club fallback, embedded height constraint verification with scoring text, and pure helper edge cases. |
| `src/tools/shot_pattern_analysis/README.md`                | Documentation  | Added concise, narrowly relevant note in step 5 detailing the results view's conditional course scoring presentation.                                                                                                                      |

---

## 5. Remaining Observations / Concerns

1. **No Live Simulation / Runner Mutations**: As instructed, no modifications were made to `matched_runner.py`, physics modules, or running Monte Carlo shards.
2. **Git Commit Status**: No commits or branch pushes have been performed. All changes remain staged/unstaged in the clean worktree for root inspection.
