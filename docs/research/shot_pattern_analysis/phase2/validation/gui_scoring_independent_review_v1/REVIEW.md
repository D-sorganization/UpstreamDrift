# Independent GUI Scoring Review and Repairs

Accepted for the reviewed saved-result display and offscreen lifecycle. This is a GUI component decision, not scientific or empirical model qualification. No flight simulation or scoring recomputation was run.

The initial 40-test candidate passed its existing suite, but an independent scan found 132 real saved paired intervals whose nonzero widths collapsed above the formatter's 0.001 threshold. For example, PW lean10/curve2/SD1/shaft-only Draw had +0.004597063 [0.004531383, 0.004658536], displayed as +0.005 [+0.005, +0.005]. Six new precision regressions failed before repair. The formatter now uses the minimum adequate precision with a 0.1% relative-error budget; interval values additionally use a 0.5%-of-width absolute-error budget. Ordinary values remain compact, while narrow intervals preserve their span.

Three lifecycle regressions also reproduced previous SG remaining visible through a new run, failure and cancellation. Current SG is now explicitly unavailable/pending for these states. Existing unavailable, missing-result, custom-club and insufficient-common-population handling remains intact.

The final 51 GUI, adapter and precision tests passed. Ruff, format, scoped Mypy, diff whitespace and the tracked-file 1,200-line gate passed. The new standalone precision test is 61 lines; every changed file is also independently below the cap. All 540 paired intervals and 810 mean SG values from the 270 completed original primary cells pass the numeric display check. Maximum interval-value relative error is 0.099713%; maximum relative interval-width error is 0.918387%, both below the tests' 1% criterion.

Four actual saved results were inspected offscreen at 1000×760. All result/scoring labels fit their measured text height. The driver conditional example correctly shows 9,934 individual Straight endpoints but 9,933 common scoring indices. Tee/fairway/green context, carry-as-final-lie assumptions, historical approximate baseline, and sampling-only intervals remain visible. Tiny driver effects remain in scientific notation; iron/wedge means and narrow intervals retain useful precision. Controls remain reachable through the existing scroll area.

Exact candidate file hashes, before/after screenshots, red/green logs, executed review scripts and the complete saved-value audit are recorded in `review.json` and its evidence files. The reviewer implemented the two corrective changes with test-first evidence; this is not presented as a second independent implementation review. No commit or staging was performed.
