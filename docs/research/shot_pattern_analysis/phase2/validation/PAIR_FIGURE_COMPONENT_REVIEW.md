# Real Final-Size Paired Figure Review

## Scope and Current Decision

The reporting checkpoint `559c345e8c` was exercised using the completed original
Driver, 5° Lean, Curve ×1, Combined-2 cells at shaft-twist SD 1° and 2°.
This is a component review, not full-matrix or supplemental-study acceptance.
The source-bound receipt is `pair_component_review_559c345e8c.json`.

The corrected component harness verifies eight distinct density/overhead assets,
file hashes, stable saved inputs and renderer sources, and the six-population
common density count of 9,939 / 10,000. Overhead samples have their own provenance
and do not carry the density population's count. All common shots contribute to
density/HDR; 600 evenly spaced dots per panel are disclosed.

## Visual Findings Requiring Changes

- The academic overhead's six equal-aspect panels collapse into thin strips,
  with excessive whitespace and disproportionately long vertical labels.
- The DPI-only social overhead clips its title and axis labels. It is not a
  separately designed, publication-ready social asset.
- Actual conditions should be legible on the panels rather than requiring
  readers to translate First/Second through a dense caption. Reuse the reviewed
  presentation style, common physical limits and density scale; disclose any
  measured display magnification. Dedicated social layout remains required.

The three hash-bound preview PNGs named in the receipt retain these findings;
they are review evidence, not approved user-facing deliverables. Root visually
inspected density/social, overhead/social and overhead/academic previews.
Luna owns the presentation repair in coordination with Astra. Issue #11988
tracks the fixes and independent review.

## Harness Correction and Reproduction

The initial harness failed because it required an explicit null
`common_supported_n` for overhead; the valid ledger omits that inapplicable
field. The corrected harness accepts the omission. The original failure log
is preserved as `pair_component_initial_harness_failure.txt`; this was a
harness error, not evidence of a product population defect.

From the active worktree, with the completed original cells present, run:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
/tmp/shot-pattern-validation-venv-20261009/bin/python \
docs/research/shot_pattern_analysis/phase2/validation/render_real_pair_component.py \
/tmp/p2-paired-figure-new-review
```

The destination must not already exist. The receipt binds the renderer version
actually executed; a later rerender does not retroactively qualify this
historical checkpoint. No extra physical flights are simulated.
