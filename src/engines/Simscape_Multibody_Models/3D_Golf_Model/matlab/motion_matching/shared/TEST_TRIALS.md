# Canonical Test-Trial Inventory

This document is the authoritative inventory of measured swings wired into
the motion-matching test suite. Each row is one trial and one expected
failure mode; "a single trial is a single failure mode" — until we have
≥ 5 trials we cannot distinguish signal from noise when comparing
optimizer / cost-function options. See issue #4081 for context.

> Cross-references: [CLUB_IK_SPEC.md](CLUB_IK_SPEC.md) defines the canonical
> `target` schema; [DATASET_SCHEMA.md](DATASET_SCHEMA.md) defines the wider
> dataset metadata. The loader entry point is
> [`load_club_target_excel.m`](load_club_target_excel.m); regression tests
> live in [`tests/test_load_club_target_excel.m`](tests/test_load_club_target_excel.m).

## Source files

| File                                                                      | Subjects | Sheets | Native rate |
| ------------------------------------------------------------------------- | -------- | ------ | ----------- |
| `src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx` | TW, GW   | 4      | 240 Hz      |

## Trial summary

All four trials below come from the same xlsx, parsed by
`load_club_target_excel`. Each loads to **301 frames after NaN-filter**
(≈ 1.25 s of motion at 240 Hz). Sample rate is 240 Hz native; the loader
resamples onto the simulation timegrid via `align_to_simulation_grid`.

| Sheet       | Subject | Club    | A_sample | T_sample | I_sample | F_sample | CHS_mph | Median shaft (m) |
| ----------- | ------- | ------- | -------- | -------- | -------- | -------- | ------- | ---------------- |
| `TW_ProV1`  | TW      | ProV1   | 240      | 418      | 525      | 725      | 114.5   | 1.069            |
| `TW_wiffle` | TW      | Wiffle  | NaN      | NaN      | NaN      | NaN      | 114.5   | 1.069            |
| `GW_wiffle` | GW      | Wiffle  | 240      | 448      | 517      | 724      | 104.6   | 1.081            |
| `GW_ProV11` | GW      | ProV1.1 | 240      | 452      | 521      | 721      | 115.1   | 1.081            |

Notes on the column meanings:

- **A / T / I / F** are sample numbers in the **original** 1-indexed
  recording (Address, Top of backswing, Impact, Finish). The loader
  uses `I_sample` to set `target.impact_idx` when present, falling
  back to the clubhead-speed argmax heuristic otherwise.
- **CHS_mph** is the documented club-head speed at impact, in mph,
  read straight from the row-1 header.
- **Median shaft (m)** is `median(‖clubhead − grip‖)` after unit
  conversion. The plausibility window for a real golf club is
  0.7 – 1.4 m; values outside that window indicate a unit-conversion
  bug or a malformed sheet.

## Per-trial detail

### `TW_ProV1` — Tiger / ProV1 (control, regression baseline)

- The original "single trial" wired into the loader; serves as the
  regression baseline for every change to `load_club_target_excel`.
- Row-1 event header is fully populated and authoritative.
- `impact_idx` lands at sample 258 in the post-NaN-filter, post-resample
  timeline (`t ≈ 0.257 s`).
- No documented data-quality issues.

### `TW_wiffle` — Tiger / Wiffle (event-marker stripped)

- Same subject and club geometry family as `TW_ProV1` (median shaft
  1.069 m matches), different ball.
- **Data-quality issue:** the row-1 header is missing the
  A / T / I / F sample numbers — only `CHS = 114.5 mph` is populated.
  The loader's `events.{A,T,I,F}_sample` come back NaN; `impact_idx`
  is therefore derived from the clubhead-speed argmax heuristic, which
  in practice lands at sample 254 (`t ≈ 0.253 s`).
- The test asserts `events.CHS_mph` is finite but tolerates NaN sample
  markers and only requires `impact_idx ∈ [1, N]`.

### `GW_wiffle` — Generic Wiffle swing

- Different subject (GW). Median shaft 1.081 m — about 12 mm longer
  than the TW trials, consistent with a different physical club.
- Row-1 header fully populated: A=240, T=448, I=517, F=724, CHS=104.6.
  Note the slower CHS (104.6 vs 114.5 / 115.1) — useful for testing
  cost-function speed sensitivity.
- No documented data-quality issues.

### `GW_ProV11` — Generic ProV1.1

- Same subject and club geometry as `GW_wiffle` (median shaft
  1.081 m). Different ball.
- Row-1 header fully populated: A=240, T=452, I=521, F=721, CHS=115.1.
- No documented data-quality issues.

## Failure-mode taxonomy

Each trial exercises a subtly different code path through the loader:

| Failure mode                                              | Exercised by                         |
| --------------------------------------------------------- | ------------------------------------ |
| Fully-populated event header path                         | `TW_ProV1`, `GW_wiffle`, `GW_ProV11` |
| Missing event-marker fallback to speed-argmax             | `TW_wiffle`                          |
| Two distinct subjects (different median shaft length)     | TW vs GW                             |
| Two ball types per subject                                | `*_ProV1` vs `*_wiffle`              |
| CHS sensitivity (104.6 mph low end vs 115.1 mph high end) | `GW_wiffle` vs `GW_ProV11`           |

If the cost function or alignment code starts treating any of these
trials specially (e.g. branches on subject ID), the corresponding test
should catch it.

## Tests wired in

`tests/test_load_club_target_excel.m` covers the four trials with one
`test_loads_<sheet>_sheet` method per sheet plus the original schema /
unit / quaternion / impact-index / provenance / missing-file tests.
Each per-sheet test asserts:

1. `load_club_target_excel(xlsx, sheet)` returns without error.
2. All canonical fields are present (`time`, `grip`, `grip_quat`, `butt`,
   `clubhead`, `club_quat`, `impact_idx`, `events`, `source`).
3. Event markers parse into the `events` struct; finite where the
   sheet's row-1 header populates them (i.e. all four for the
   GW trials and TW_ProV1, only `CHS_mph` for TW_wiffle).
4. Median shaft length sits in 0.7 – 1.4 m.

Test runner:

```bash
matlab -batch "addpath(genpath('motion_matching/shared')); addpath(genpath('src')); runtests('motion_matching/shared/tests/test_load_club_target_excel.m')"
```

## Developer probe

`scripts/probe_swing_sheets.m` walks all four sheets and prints per-trial
diagnostics (frame count, event markers, shaft-length stats, first-frame
and impact-frame raw values). Use it after adding a new trial or
touching the loader to sanity-check unit conversion and event-header
parsing without running the full test suite:

```bash
matlab -batch "addpath(genpath('motion_matching/shared')); probe_swing_sheets()"
```

## How to add a new trial

1. **Drop the source file** under
   `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/`.
   Prefer the existing xlsx layout (row 1 event header, row 2 group
   labels, row 3 column names, data from row 4 — see the loader
   header for the exact layout). Native frame rate should be 240 Hz
   to match existing trials, but other rates work as long as the time
   column is in seconds.
2. **Whitelist the sheet name** in the `mustBeMember` clause of
   [`load_club_target_excel.m`](load_club_target_excel.m). The loader
   refuses unknown sheet names by design (`mustBeMember` is a
   precondition gate, not a suggestion) — this forces every new trial
   to be opted in deliberately rather than discovered ad-hoc.
3. **Run the probe:**
   `matlab -batch "addpath(genpath('motion_matching/shared')); probe_swing_sheets()"`.
   Confirm the new sheet shows up with sensible frame count, event
   markers, and median shaft in 0.7 – 1.4 m. If it doesn't, fix the
   source data — do **not** weaken the loader to accept malformed data.
4. **Document the trial in this file** — add a row to the summary
   table and a per-trial subsection. Capture every data-quality issue
   you noticed during the probe (missing columns, frame dropouts,
   weird timing, etc.); future debugging starts here.
5. **Add a test method** to
   [`tests/test_load_club_target_excel.m`](tests/test_load_club_target_excel.m)
   following the `test_loads_GW_wiffle_sheet` pattern: load the sheet,
   verify canonical fields via the `verify_canonical_target_fields`
   helper, assert event markers parse and the documented values match,
   verify shaft length plausibility via the `verify_shaft_length_plausible`
   helper.
6. **If the sheet has a known limitation** (e.g. partial event header,
   missing direction-cosine columns that the loader cannot
   reconstruct), document the limitation in step 4 and use
   `testCase.assumeFail("reason")` in the new test method, citing the
   specific failure mode. Never weaken the loader to swallow malformed
   data — the precondition gates exist to surface bad inputs early.
7. **Run the full test suite** and confirm all per-trial tests still
   pass. Open a PR that references the issue tracking the new trial.
