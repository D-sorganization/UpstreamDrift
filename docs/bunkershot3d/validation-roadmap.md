# BunkerShot3D Validation Roadmap

**Issue #8616 (W9). Framing: NASA-STD-7009B (2024-03-05). Metric: ASME V&V 20-2009.**

The [credibility statement](credibility.md) says validation is at **0 of 4** and that
nothing this tool predicts has ever been measured. That is true, and a reader can do
nothing with it. It does not say _which_ measurement would change it, made how, to what
tolerance, or which of the eight credibility factors it would move.

This document is the other half: a checkable statement of what measurement raises which
factor, and the apparatus to consume such a measurement when it exists.

## Read This Before the Tables

**Writing this roadmap is not evidence, and it did not move the score.** The assessment
is derived from the ledger and the measurements on file; there are no measurements on
file; the derived assessment is therefore the level the ledger holds, which is the level
the credibility statement already published. A test
([`test_validation_ledger.py`](../../tests/bunkershot3d/vandv/test_validation_ledger.py))
pins every level against the published table and fails if building the apparatus lifted
any of them.

The fixtures that prove the apparatus works are marked
`MeasurementBasis.SYNTHETIC_FIXTURE`, and a synthetic record is **structurally forbidden
from carrying a value at all** — not discouraged, refused at construction. There is no
plausible-looking number anywhere in this change for somebody to quote in a year's time.
`shipped_register()` additionally refuses any document containing a synthetic record, so
a fixture cannot reach the published score even by accident.

## Where the Ledger Lives

| Thing                               | Where                                                                                            |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| Ledger types and level derivation   | [`bunkershot3d/vandv/ledger.py`](../../src/bunkershot3d/vandv/ledger.py)                         |
| The seven specs and eight entries   | [`bunkershot3d/vandv/roadmap.py`](../../src/bunkershot3d/vandv/roadmap.py)                       |
| The record a measurement arrives as | [`bunkershot3d/vandv/measurement.py`](../../src/bunkershot3d/vandv/measurement.py)               |
| Loader, schema and provenance flip  | [`bunkershot3d/vandv/measurement_intake.py`](../../src/bunkershot3d/vandv/measurement_intake.py) |
| Where a real document goes          | [`bunkershot3d/vandv/measurements/`](../../src/bunkershot3d/vandv/measurements/README.md)        |

## Only Three of Eight Factors Are Measurement-Limited

The first thing the ledger says is unflattering to the idea of buying credibility with
instrument time. Verification needs a method of manufactured solutions and coverage of
the F1–F3 tiers; robustness needs a sensitivity study over `lambda`'s published 1.0–2.8
spread and a review by somebody who did not write the solver; M&S management needs a
release and approval process. Those are analysis and process work. Use history accrues
only by the tool being used to make a real decision, and no experiment manufactures it.
People qualifications is not self-assessable and is left blank.

Attaching a measurement to any of those five would be a lie in a machine-readable
format, which is worse than a lie in prose because code acts on it. `LedgerEntry` refuses
to hold one.

<!-- generated:roadmap-table -->

| Factor                | Level        | Threshold | Blocked on          | Next step | Measurements needed                                                        |
| --------------------- | ------------ | --------- | ------------------- | --------- | -------------------------------------------------------------------------- |
| Verification          | 2 / 4        | 3 / 4     | analysis            | n/a       | n/a                                                                        |
| Validation            | 0 / 4        | 3 / 4     | measurement         | 0 to 1    | `bunker_sand_angle_of_repose_deg`                                          |
| Input Pedigree        | 2 / 4        | 3 / 4     | measurement         | 2 to 3    | `bunker_sand_bulk_density_kg_m3`, `bunker_sand_drained_friction_angle_deg` |
| Results Uncertainty   | 2 / 4        | 3 / 4     | measurement         | 2 to 3    | `bunker_sand_angle_of_repose_deg`, `bunker_sand_population_survey`         |
| Results Robustness    | 1 / 4        | 2 / 4     | analysis            | n/a       | n/a                                                                        |
| Use History           | 0 / 4        | 2 / 4     | use                 | n/a       | n/a                                                                        |
| M&S Management        | 3 / 4        | 3 / 4     | analysis            | n/a       | n/a                                                                        |
| People Qualifications | not assessed | 2 / 4     | not self assessable | n/a       | n/a                                                                        |

<!-- end:roadmap-table -->

Levels are climbed one at a time because NASA-STD-7009B's levels are not commensurable:
two level-1 measurements do not make a level 2. A step names every measurement it needs
and is satisfied only when all of them are present.

## The Seven Measurements

<!-- generated:measurement-specs -->

| Measurement                              | Quantity                   | Unit   | Effort                | Acceptance                     | Flips to MEASURED                               |
| ---------------------------------------- | -------------------------- | ------ | --------------------- | ------------------------------ | ----------------------------------------------- |
| `bunker_sand_angle_of_repose_deg`        | angle_of_repose_deg        | deg    | bench day (3)         | n >= 5, U_rel <= 0.05 (k = 2)  | none                                            |
| `bunker_sand_bulk_density_kg_m3`         | bulk_density_kg_m3         | kg/m^3 | bench hour (1)        | n >= 3, U_rel <= 0.02 (k = 2)  | `moisture`, `packing`, `particle_density_kg_m3` |
| `bunker_sand_drained_friction_angle_deg` | drained_friction_angle_deg | deg    | bench day (3)         | n >= 3, U_rel <= 0.05 (k = 2)  | `friction_angle_deg`                            |
| `bunker_sand_population_survey`          | bulk_density_kg_m3         | kg/m^3 | field session (8)     | n >= 30, U_rel <= 0.1 (k = 2)  | none                                            |
| `clubhead_delivery_shaft_strain`         | clubhead_wrench_force_n    | N      | instrumented rig (30) | n >= 20, U_rel <= 0.15 (k = 2) | none                                            |
| `ejecta_launch_high_speed_video`         | ejecta_launch_speed_m_s    | m/s    | instrumented rig (30) | n >= 10, U_rel <= 0.08 (k = 2) | none                                            |
| `splash_shot_divot_cast_volume_m3`       | divot_cavity_volume_m3     | m^3    | field session (8)     | n >= 10, U_rel <= 0.1 (k = 2)  | none                                            |

<!-- end:measurement-specs -->

Each spec's full conditions, instrument class and acceptance criterion are in
[`roadmap.py`](../../src/bunkershot3d/vandv/roadmap.py) — deliberately in the code rather
than here, so the roadmap and the assessment cannot be edited apart.

The acceptance criteria gate on the **procedural** half of a measurement: how many
independent samples, at what expanded uncertainty (k = 2), in what unit. Whether the
measured number agrees with the model is the _result_ of validation; whether the
measurement is good enough for the comparison to mean anything is the _precondition_, and
the precondition is what gets skipped. A record that misses its criterion is loaded and
kept — so the register stays a truthful account of what was attempted — and moves
nothing.

## Ranked by Leverage

<!-- generated:leverage-table -->

| Rank | Measurement                              | Effort           | Credit | Leverage | Unlocks today                                 |
| ---- | ---------------------------------------- | ---------------- | ------ | -------- | --------------------------------------------- |
| 1    | `bunker_sand_angle_of_repose_deg`        | bench day        | 3.50   | 1.167    | Validation 0 to 1, Results Uncertainty 2 to 3 |
| 2    | `bunker_sand_bulk_density_kg_m3`         | bench hour       | 0.50   | 0.500    | Input Pedigree 2 to 3                         |
| 3    | `bunker_sand_drained_friction_angle_deg` | bench day        | 0.50   | 0.167    | Input Pedigree 2 to 3                         |
| 4    | `bunker_sand_population_survey`          | field session    | 0.50   | 0.062    | Results Uncertainty 2 to 3                    |
| 5    | `splash_shot_divot_cast_volume_m3`       | field session    | 0.00   | 0.000    | nothing yet                                   |
| 6    | `clubhead_delivery_shaft_strain`         | instrumented rig | 0.00   | 0.000    | nothing yet                                   |
| 7    | `ejecta_launch_high_speed_video`         | instrumented rig | 0.00   | 0.000    | nothing yet                                   |

<!-- end:leverage-table -->

Both halves of the ratio are **declared conventions, not estimates**, and they are in one
place so they can be argued with rather than re-litigated in prose:

- **Effort** is an ordinal cost scale — bench hour 1, bench day 3, field session 8,
  instrumented rig 30.
- **Credit** for a level step is the gap that still stands below that factor's threshold:
  a level bought on a factor three short is worth three times a level bought on a factor
  one short. A step's credit is split evenly across the measurements it still lacks, so a
  spec that is one of two missing prerequisites earns half.

Only the step a factor currently _stands at_ counts. A measurement that feeds a step two
levels up scores zero, however important it becomes later.

### Why the Angle of Repose Wins

Four reasons, in descending order of how much they matter.

1. **It is the only measurement on the list that touches the factor sitting at zero.**
   Validation is three levels short of its threshold; every other reachable step is one
   level short. That is where the credibility is missing, and it is the only one of the
   seven that buys a level there.
2. **The comparison it forms is _attributable_.** A repose test has no club delivery in
   it. The sand state is the only input, so a discrepancy between prediction and
   measurement belongs to the model — it is not shared with an unmeasured boundary
   condition. That is the whole difference between a validation comparison and a plot of
   two curves, and it is why a divot cast cannot buy validation level 1 even though it is
   a perfectly good measurement.
3. **The model already predicts it.** `bunkershot3d.calibration.angle_of_repose` and
   `f1_repose` exist and produce a number today. What they are compared against is a
   _simulated_ target (PR #9238), which is why the fitted friction angle is recorded as
   `CONVENTION`. Swapping the target for a measured one is the smallest change that turns
   an existing harness into an existing validation case.
4. **It needs no golfer, no bunker session and no rig** — a funnel, a bench, a camera and
   five repeats.

**Second, and cheaper still, is `bunker_sand_bulk_density_kg_m3`**: one bench hour that
retires three borrowed constants at once (`moisture`, `packing`,
`particle_density_kg_m3`), all of which currently trace to Quikrete medium sand — a
hardware-store product, not a bunker (#7999). In practice it and the repose test are one
laboratory visit, and doing both is the obvious first move.

### Why the Rig Work Ranks at Zero

`ejecta_launch_high_speed_video` and `clubhead_delivery_shaft_strain` are the
measurements that would touch the nine launch-side quantities issue #8616 recorded as
having no published value anywhere. They score zero **today**, and that is a statement
about order, not about worth: both buy validation level 3, and validation is at level 0.

A high-speed rig hired before the bench work produces a comparison whose error cannot be
attributed to the model, because the sand it was made in was never characterised. The
same argument, one level down, is why `splash_shot_divot_cast_volume_m3` sits at step
1→2: a cast without a measured density gives a volume, not a mass. Once the repose
measurement lands, the divot cast's leverage becomes non-zero — the ranking is recomputed
against the register, not fixed.

## Supplying A Measurement

One YAML document per campaign, in
[`src/bunkershot3d/vandv/measurements/`](../../src/bunkershot3d/vandv/measurements/README.md),
naming a `spec_key` that is already in the ledger. That ordering is deliberate: what a
measurement would buy has to be stated _before_ it is made, not chosen afterwards to suit
what came out.

```yaml
schema_version: 1
document_id: <campaign identifier>
records:
  - spec_key: bunker_sand_angle_of_repose_deg
    basis: instrument
    source: <laboratory report identifier>
    measured_on: <ISO date>
    instrument: <the apparatus>
    conditions: <moisture, compaction, pour height, funnel and bed diameter>
    sample_count: <independent repeats>
    relative_expanded_uncertainty: <fraction, k = 2>
    unit: deg
    value: <the measured mean>
```

An instrument record **must** carry a value and an ISO date. A synthetic fixture **must
not** carry a value and must announce itself in its own source string. There is no third
option and no default.

## `CONVENTION` Is Still Not an Upgrade

PR #9238 moved the friction angle from `BORROWED_ANALOGUE` to `CONVENTION` by fitting it
to a declared but _simulated_ target, and said plainly that this was a lateral move. The
ledger keeps that distinction enforceable rather than remembered:
`EVIDENTIAL_RANK` puts `BORROWED_ANALOGUE`, `ESTIMATED` and `CONVENTION` at the **same**
rank, so `is_provenance_upgrade()` returns `False` in both directions between them.
Fitting a constant to a simulated target makes it more checkable, which is worth having;
it does not make it better evidenced. Only `bunker_sand_drained_friction_angle_deg` — a
shear box on real bunker sand — moves `friction_angle_deg` to `MEASURED`.

Note also that the highest-leverage measurement, the angle of repose, **flips nothing**.
Repose is a system response the model predicts, not a constant the model consumes;
reading a friction angle off a repose cone would repeat #7999 in a new costume. It earns
a validation level, and no provenance flip, and the ledger records that asymmetry as
data.

## What This Does Not Do

It does not validate anything. It does not narrow the 63× Froude exceedance or the 17×
speed exceedance, both of which are properties of running a quasi-static granular theory
at 25 m/s and are unaffected by how well the sand is characterised. It does not move
`MAX_VALIDATED_SPEED_M_S` off 1.44 m/s. It does not give `lambda` or `delta_h` a wedge
value.

What it does is make the gap addressable: seven named measurements, each with the level
it buys, the conditions it must be made under, the instrument class it needs and the bar
it must clear — and a path by which making one actually changes the number this package
publishes about itself.
