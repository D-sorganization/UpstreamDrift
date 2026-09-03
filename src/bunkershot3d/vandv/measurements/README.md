# Measurement Documents

This directory is **empty on purpose**, and that emptiness is load-bearing.

`bunkershot3d.vandv.measurement_intake.shipped_register()` reads every `.yaml`,
`.yml` or `.json` file here and hands the result to
`bunkershot3d.vandv.credibility.credibility_assessment()`. Because there is
nothing here, the derived NASA-STD-7009B assessment is the level
`bunkershot3d.vandv.roadmap.VALIDATION_LEDGER` holds — **validation at 0 of 4**.
A test asserts that this directory contains no document, so the published score
cannot move without a file appearing here in a reviewed change.

## Adding a Measurement

One document per campaign. Name it after the sand or the session, not after the
spec.

```yaml
schema_version: 1
document_id: covia-signature-500-repose-2026-09
records:
  - spec_key: bunker_sand_angle_of_repose_deg
    basis: instrument
    source: <laboratory report identifier, with a copy on file>
    measured_on: <ISO date>
    instrument: <the apparatus the angle was read from>
    conditions: <moisture, compaction, pour height, funnel and bed diameter>
    sample_count: <independent repeats>
    relative_expanded_uncertainty: <fraction, k = 2>
    unit: deg
    value: <the measured mean>
    note: <anything the fields above do not carry>
```

`spec_key` must name a spec that is already in the ledger. That ordering is
deliberate: what a measurement would buy has to be stated _before_ it is made,
not chosen afterwards to suit what came out. `python -c "from
bunkershot3d.vandv.roadmap import MEASUREMENT_SPECS; print(*MEASUREMENT_SPECS)"`
lists the seven the ledger currently knows, and
`docs/bunkershot3d/validation-roadmap.md` gives each one's conditions,
instrument class and acceptance criterion in full.

A record only counts if it clears its spec's acceptance criterion — the sample
count and the expanded uncertainty. A record that misses it is not rejected at
load time; it is loaded, kept, and simply moves nothing, so the register stays a
truthful account of what was attempted.

## What Does Not Go Here

`basis: synthetic_fixture` records exist so the intake path can be tested end to
end. `shipped_register()` **refuses** any document holding one, and a synthetic
record is not permitted to carry a value at all. If you want to exercise the
apparatus, do it in `tests/bunkershot3d/vandv/test_validation_ledger.py`, where
the fixtures live and where nothing they touch is published.
