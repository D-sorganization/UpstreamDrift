# Real-Data Paired Comparison Check

This check exercises the qualified saved-data API on completed driver cells,
each containing 10,000 attempts per pattern. It is a component check, not final
acceptance of the incomplete primary or control matrices.

## Cases and Results

Both comparisons hold curve scale 1 and the combined-2 noise profile fixed,
then change shaft-twist SD from 1° to 2°. Each comparison independently assembles
the sorted shot-index intersection across Straight, Draw and Fade at both SDs.

| Shaft Lean | Supported n at SD 1° (Each Pattern) | Supported n at SD 2° (Each Pattern) | Six-Population Common n |
| ---------- | ----------------------------------: | ----------------------------------: | ----------------------: |
| 0°         |                              10,000 |                              10,000 |                  10,000 |
| 5°         |                               9,974 |                               9,940 |                   9,939 |

At 5° lean, using either condition's separate supported count would overstate
the paired population. The common index set correctly excludes 61 of the
10,000 original indices. Unsupported rows remain in the source datasets.

For each curve, the independent calculation forms its per-shot score benefit
relative to Straight at each SD, then subtracts the SD-1° benefit from the SD-2°
benefit before averaging. Both means agree exactly with the public API result
(maximum absolute difference 0 in these two cases; assertion tolerance 1e-12).
It uses the already parity-qualified public per-shot scorer, so this is an
independent index/contrast assembly, not independent physical or baseline
validation. It does not independently re-estimate the returned bootstrap
intervals or support a full-study conclusion about preferred shot shape.

## Evidence and Reproduction

The [Zero-Lean Receipt](real_driver_lean0_pair_review.json) and
[Five-Degree-Lean Receipt](real_driver_lean5_pair_review.json) contain complete
API outputs, physical-source hashes, scoring-source hashes, independent means,
and the reproduction-script hash. The API separately verifies input-error
pairing, matching targets, source stability and common-support conditioning.
Ruff, format checking and scoped mypy pass for the reproduction script.

From the repository root with the qualified study environment:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python \
  docs/research/shot_pattern_analysis/phase2/validation/check_real_driver_pair.py 0
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python \
  docs/research/shot_pattern_analysis/phase2/validation/check_real_driver_pair.py 5
```

The commands read the completed cells under `phase2/final_v1` and write only
`/tmp/p2-real-driver-lean0-pair-review.json` or its lean-5 counterpart. They do
not change any simulation data, plans, original acceptance receipts or reports.
