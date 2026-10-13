# P2 Comparison Caption Review: Fixed Controls, Schema Units & Self-Contained Social Captions

## Defect Summary

Astra's standalone share-figure review confirmed that `_render_density_comparison` in `scripts/research/shot_pattern_analysis/build_final_figures.py` rendered primary comparison figures (e.g. `social/comparisons/driver/combined2/Draw/curve_scale.png`) with `Curve ×1 vs ×2 / Combined Delivery Noise` but omitted the fixed shaft-twist SD (2°). Similarly, shaft-rotation SD comparisons (`SD 1° vs SD 2°`) omitted the fixed curve scale (`Curve ×1`). Furthermore, the metadata caption omitted fixed delivery noise standard deviations (path SD, speed SD in mph, and horizontal/vertical strike SD in mm), leaving the figure ambiguous when shared standalone without external data ledgers.

## Implemented Solution

1. **DRY Unit Extraction (`_social_noise_summary`)**:

   - Extracted `_social_noise_summary(standard_deviations)` as a pure helper with Design-by-Contract precondition checks (length == 7, all values finite floats).
   - Reused across single-cell `_social_metadata` and comparison `_render_density_comparison` to convert schema units into readable units:
     - Path SD in degrees (`Path SD 0.5°` or `Path SD 0°`)
     - Speed SD in mph (`Speed SD 2 mph` or `Speed SD 0 mph`)
     - Strike SD in millimetres (`Strike SD 6.35/6.35 mm horizontal/vertical` or `Strike SD 0/0 mm horizontal/vertical`)
   - Fully preserved single-cell output and semantics without regressions.

2. **Self-Contained Scenario Subtitles**:

   - `curve_scale` comparison subtitle explicitly records fixed shaft twist SD:
     `Draw • Curve ×1 vs Curve ×2 • Shaft Twist SD 2° • Combined Delivery Noise` (or `Shaft Twist Only`)
   - `shaft_rotation_sd` comparison subtitle explicitly records fixed curve scale:
     `Draw • SD 1° vs SD 2° • Curve ×1 • Combined Delivery Noise` (or `Shaft Twist Only`)

3. **Comparison Metadata Captions**:
   - Added `Noise` formatted via `_social_noise_summary`.
   - Included `Static Loft` (`Static Loft: 10°`) when available alongside `Nominal Lean: 0°` without overloading the caption.
   - Retained complete sample population disclosures and target distance.
   - Preserved `academic_plots.py` unchanged; dynamic header automatically resizes and positions layout.

## Source Artifacts & Hashes

- `scripts/research/shot_pattern_analysis/build_final_figures.py`: `39c086c382c1ad876d8b739edeef3cf89d9a629f0b39646730177c1909c737f5`
- `tests/research/test_shot_pattern_final_figures.py`: `beed0d164514c59bc5a2c20452ec03af5459e7bddeed40fecb3b3610ef99cee7`

## Verification Evidence

All commands run in `/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift` on branch `codex/shot-pattern-matched-delivery`:

1. **Red Test Failure (TDD)**:

   - Command: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -o addopts='' tests/research/test_shot_pattern_final_figures.py`
   - Log: `/tmp/p2-comparison-caption-red.txt`
   - Result: 2 failed (`AssertionError: assert 'Shaft Twist SD 2°' in 'Draw • Curve ×1 vs Curve ×2 • Combined Delivery Noise'`), 16 passed.

2. **Green Test Pass**:

   - Command: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest -o addopts='' tests/research/test_shot_pattern_final_figures.py`
   - Log: `/tmp/p2-comparison-caption-green.txt`
   - Result: 18 passed, 0 failed.

3. **Ruff Check & Format**:

   - Command: `/home/dieterolson/Repositories/UpstreamDrift/.venv/bin/ruff check scripts/research/shot_pattern_analysis/build_final_figures.py tests/research/test_shot_pattern_final_figures.py && /home/dieterolson/Repositories/UpstreamDrift/.venv/bin/ruff format --check scripts/research/shot_pattern_analysis/build_final_figures.py tests/research/test_shot_pattern_final_figures.py`
   - Log: `/tmp/p2-comparison-caption-ruff.txt`
   - Result: All checks passed, 2 files already formatted.

4. **Mypy Type Checking**:
   - Command: `/home/dieterolson/Repositories/UpstreamDrift/.venv/bin/mypy --python-version 3.12 --follow-imports=silent scripts/research/shot_pattern_analysis/build_final_figures.py tests/research/test_shot_pattern_final_figures.py`
   - Log: `/tmp/p2-comparison-caption-mypy.txt`
   - Result: `Success: no issues found in 2 source files`.

## Acceptance Status

Caption generation defect fixed and verified with focused unit tests. Final image acceptance and full re-renders remain independent root/Astra work; no full simulations or image overwrite rerun in this worktree.
