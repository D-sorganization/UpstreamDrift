# Shared Report Precision Qualification

The isolated candidate is accepted with the exact eight paths and hashes in `review.json`. Seven new failing regressions were repaired: invalid absolute error budgets, an ignored precision argument and missing formatter source binding. Zero absolute budget remains supported for exact round-trip requirements. The 47 affected saved records now live in a portable, provenance-labelled fixture with no machine path or skip.

Ninety-four focused tests pass, including GUI and final-report contracts; Ruff, scoped Mypy and the file-size gate pass. An independent strict scan of all saved 696 paired SG intervals and 1,044 absolute means finds no lost sign, zeroed nonzero value or collapsed nonzero interval. Maximum interval-width relative error is 0.9184%; maximum per-value relative error is 0.099713%. These are display-error bounds, not scientific uncertainty claims.

All 13 supplemental postprocessing sources remain byte-identical to this worktree's HEAD. The worktree contains an older paired-analysis source than MAIN, so only the listed candidate files should be integrated. The new pure formatter is included in the final report's source checks. MAIN live sources were not edited.

The supplemental pipeline default remains an explicit follow-up. This acceptance covers GUI/report/interpretation precision only. Preserve the recorded initial wrong-test-path collection failure separately from the successful corrected run.
