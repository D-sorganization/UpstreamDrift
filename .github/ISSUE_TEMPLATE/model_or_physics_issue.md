---
name: Model or physics correctness
about: Report a result you believe is physically or numerically wrong
title: "Physics: "
labels: physics, needs-triage
assignees: ""
---

**Engine**: <!-- Which engine produced the result -->
**Model**: <!-- Model file or configuration -->

## Observed Result

<!-- The quantity, its value, and its units. -->

## Expected Result

<!-- The value you expected, with the source: a cited reference, an analytical
     solution, an independent engine, or measured data. -->

## Reproduction

<!-- The smallest script or configuration that produces the discrepancy. -->

```python

```

## Cross-Engine Check

<!-- If you ran the same model through another engine, give both results.
     See docs/troubleshooting/cross_engine_deviations.md for known deviations. -->

## Acceptance Criteria

- [ ] The discrepancy is reproduced by a test
- [ ] The correct value is established against a stated reference
- [ ] The fix, or a documented limitation, is in place
