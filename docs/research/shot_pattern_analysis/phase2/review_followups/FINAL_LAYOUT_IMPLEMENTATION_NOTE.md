# Phase 2 Final Layout Resolution Note

## Scope and Intent
This change addresses the two open #11988 reporting issues strictly within the presentation geometry, fonts, and axis locators of `src/tools/shot_pattern_analysis/academic_plots.py` and its focused test suite in `tests/tools/shot_pattern_analysis/test_academic_plots.py`. Numerical simulation sources, density calculations (`shared_histogram_density`), and coordinates remain byte-identical and unmodified.

## Exact Changed Files
1. `src/tools/shot_pattern_analysis/academic_plots.py`:
   - **Tick overlap fix (Issue 1)**: Configured `axis.xaxis.set_major_locator(ticker.MaxNLocator(prune="upper"))` to prevent downrange tick label protrusion past the right spine, and configured `axis_right.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="lower"))` and `axis_top.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="lower"))` to prune marginal zero ticks. Increased inner subgridspec `wspace` to `0.06` for clean separation. All data points, density calculations, axis limits, and common marginal scales remain strictly shared and preserved.
   - **Social presentation layout & typography (Issue 2)**: Replaced cramped 16:9 multi-column layout for 2 and 3 panels with phone-optimized portrait stacked panels (4:5 ratio 10.8"×13.5" for 2 panels; 3:5 ratio 10.8"×18.0" for 3 panels). Applied dynamic vertical spacing measuring rendered text bounds for suptitle, scenario, wrapped metadata, legend, and footer, preventing collisions regardless of long delivery noise details. Wrapped social metadata at 78 characters and footer at 95 characters. Scaled typography so that all visible text, tick labels, axis labels, and titles render at >= 11 px (typically 12.5–25 px) at 1080 px phone preview width.
2. `tests/tools/shot_pattern_analysis/test_academic_plots.py`:
   - Added `test_density_rightmost_downrange_tick_does_not_overlap_marginal_zero_tick` to test rendered text bounding box clearance between downrange and marginal ticks.
   - Added `test_social_density_phone_preview_geometry_and_typography` (parameterized for 2 and 3 panels) to test bounding box non-overlap (text-text, text-legend, text-axis, canvas bounds) and assert minimum rendered typography >= 11 px at 1080 px preview width.
   - Updated `test_social_presentation_keeps_data_and_uses_readable_16_by_9_layout` to verify that social overhead preserves 16:9 landscape format while social density uses the phone-width portrait layout.

## Verification Checks
- **Python Environment**: `/tmp/shot-pattern-validation-venv-20261009/bin/python` (Python 3.12.13)
- **Environment Flags**: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy OPENBLAS_NUM_THREADS=1`
- **Red Test Phase**:
  - Saved log: `/tmp/p2-final-layout-red.log`
  - 3 failed (demonstrating tick bounding box overlap and tiny <11 px phone typography on baseline).
- **Green Test Phase**:
  - Saved log: `/tmp/p2-final-layout-green.log`
  - 16 passed, 0 failed in `tests/tools/shot_pattern_analysis/test_academic_plots.py`.
- **Related Suite Verification**:
  - `test_matched_pair_social.py`, `test_social_review_contracts.py`, `test_matched_social_integration.py`, `test_matched_export.py`, `test_matched_report_plots.py`: 31 passed, 0 failed.
- **Ruff Linting**:
  - `python -m ruff check src/tools/shot_pattern_analysis/academic_plots.py tests/tools/shot_pattern_analysis/test_academic_plots.py`: All checks passed.
  - `python -m ruff format --check src/tools/shot_pattern_analysis/academic_plots.py tests/tools/shot_pattern_analysis/test_academic_plots.py`: 2 files already formatted.
- **Mypy Type Checking**:
  - `python -m mypy --python-version 3.12 --follow-imports=silent src/tools/shot_pattern_analysis/academic_plots.py tests/tools/shot_pattern_analysis/test_academic_plots.py`: Success, no issues found in 2 source files.

## Remaining Limitations
- This worktree is strictly an isolated future reporting revision; it must never be used to execute simulations.
- Historical execution receipts, Monte Carlo runs, density mathematical algorithms, and snapshots remain immutable.
- Root will coordinate rendering of actual study data and obtain independent Astra visual review afterward.
