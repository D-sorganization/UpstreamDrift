# Focused Baseline Gate Audit

The exact saved canonical log contains 41 unique failed/error nodes across 18
unchanged test paths. Earlier counts 37/17 truncated parameterized identifiers
containing spaces; the current pre-PR receipt is explicitly corrected. The
original compressed canonical log is unchanged.

A focused run of those 41 nodes at base `fa7ee92f5cddd753099d7418fc974b142b7ebb54`
returned 31 failed, 7 errors, and 3 passed. The three passing nodes also pass in isolation
at delivery `14bd822fae4eca02d074196046408b3da1fde69d`. Exact Tools pin, interpreter,
commands and headless environment are recorded. This reproduces 38 nonpassing
statuses at base; it does not prove identical root causes or clear the full gate.
The other three full-suite failures remain unexplained by isolated runs.

The canonical pre-PR gate remains failed. No tolerances, tests, hooks or branch
protection were relaxed. No full-suite rerun or broad baseline claim is made.
