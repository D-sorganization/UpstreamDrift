# Independent Final-Data Review

## Current Status

Final execution is in progress. No full-data acceptance or scientific conclusion
has been issued. The immutable `FINAL_RUN_ACCEPTANCE.md` authorizes execution
only and remains unchanged.

## Audit Method

`validation/independent_final_audit.py` processes one immutable completed cell at
a time, bounding memory independently of the complete matrix size. It verifies
all manifest hashes, plan/configuration/execution identity, exact shot-index and
seeded-input pairing, requested populations, supported/unsupported counts and
reason categories, exact binomial support intervals, raw nominal residuals,
input covariance, joint output covariance/correlation and pair counts, scalar
spread/target statistics, and paired population denominators.

Every available cell's SG means and differences are reconstructed through the
source-backed public-API score cache with the saved baseline hash. Selected
combined-2, shaft-only, and shaft-strike populations spanning clubs, lean and
curve strength receive additional direct public-API row checks, including range
extremes. Selected paired confidence intervals are independently regenerated
with 500 shared-row bootstrap resamples. This is computational checking of the
assumed model and course, not empirical validation.

Partial receipts always have `complete=false` and `pass=false`; their
`partial_checks_pass` field reports only whether already audited cells passed.
Missing cells remain unaccepted. Numerical or audit failures are recorded and
cause a nonzero audit exit. No flights are rerun by the audit.

## Reproduction

Run from the repository root with the qualified runtime. Restrict numerical
library threads to avoid competing unnecessarily with the two flight workers.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/shot-pattern-validation-venv-20261009/bin/python docs/research/shot_pattern_analysis/phase2/validation/independent_final_audit.py docs/research/shot_pattern_analysis/phase2/final_v1 --allow-partial --max-cells 2 --output docs/research/shot_pattern_analysis/phase2/validation/independent_final_partial.json
```

After every planned cell and worker completion receipt is present, run without
`--allow-partial` or `--max-cells`, separately against `final_v1` and
`control_final_v1`, saving distinct final audit receipts. Review final figures,
common-support comparisons across lean, model qualifications, and report claims
alongside those receipts before issuing final acceptance.

## Auditor Validation and Initial Partial Check

The auditor passed a complete replay of the 72-cell control pilot, including
independent covariance/correlation, paired estimates and selected paired bootstrap
intervals. The maximum direct-API discrepancy was 2.23e−16 strokes and bootstrap
bound discrepancy 1.43e−14. A deliberately corrupted manifest-listed CSV in a
temporary copy was rejected before data/scoring use. These checks are recorded in
`validation/final_auditor_control_validation.json` and
`validation/final_auditor_integrity_guard.json`.

The first two completed final cells (60,000 rows) passed their integrity, input,
summary, covariance and cached-scoring checks. Their receipt explicitly remains
`complete=false`, `pass=false`, `partial_checks_pass=true`. They are not selected
for direct-row API or bootstrap spot checks; those checks are exercised elsewhere
by the predeclared selection and were validated on the pilot. No full-data
conclusion follows from these two cells. The current partial receipt is
`validation/independent_final_partial.json`.

Auditor source SHA256: `e7452f6dedb9e52191d31ebc405b99f58b359720c5c7f065f7a68cb8fe7dd86e`.
