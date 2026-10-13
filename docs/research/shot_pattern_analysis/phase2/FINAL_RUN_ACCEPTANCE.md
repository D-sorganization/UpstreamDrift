# Independent Acceptance to Start Final Matrices

## Disposition and Scope

**Accepted for conditional simulation** on 2026-10-09 by the independent Astra
review. Authorize the frozen 10,000-shot-per-pattern within-lean matrix (324
planned cells) and common-distance control matrix (72 planned cells; shaft-only
and combined-2 profiles). Preserve unavailable nominal cells and unsupported
sample rows exactly as specified. This acceptance authorizes execution; it does
not accept final outputs, claim empirical accuracy, or complete the project goal.

This note is immutable run evidence. Record later findings in a separate final
audit or the living ASTRA_REVIEW.md; do not modify this note during accepted runs.

## Bound Execution

The accepted full execution snapshot is the `execution` object in the hashed
`validation/frozen_software.json` receipt below, including the final conditional-SG
annotation correction at commit `d5f59256ef`. Its canonical JSON SHA256 (sorted
keys, separators comma/colon, UTF-8) is:

`705c7440a793cc93c2d4c0018d2c2a1b2b2466e962ef6284466c5d349a152eb3`

Require the machine acceptance record and every final cell to use this exact full
snapshot. An unexpected source, native binary, runtime or input-plan change
invalidates this run authorization pending review. No protection or gate bypass
is authorized.

## Gates Independently Reviewed

- Contact invariants: 100 spatial states passed momentum, passivity, normal-work,
  frame covariance and Python/Numba parity checks.
- Numerical refinement: 243 deterministic delivery cases, with 225 supported
  and 18 consistently nonairborne; no solver/status failures. Maximum contact
  128→512 landing change was 0.00002013 m. Flight dt 0.02→0.005 maximum was
  0.042923 m, below the predeclared 0.05 m gate. No tolerance was widened.
- Primary pilot: all 324 cells accounted for, 270 completed/54 unavailable;
  25,920 rows and 1,770 file hashes checked. Seeded inputs reproduced exactly.
- Control pilot: all 72 cells accounted for, 52 completed/20 unavailable;
  4,992 rows and 352 file hashes checked. Nominal residuals were below 0.05 m.
  The control command has an explicit exit-0 receipt; original primary tool
  waits returned 143, so acceptance there rests on complete audited artifacts,
  not an asserted successful process exit.
- Scoring: direct public-API checks of 288 rows in each pilot plus 480 additional
  nonzero-lean control rows agreed with saved means within 2.78e−17 strokes.
- Source compatibility: independently verified the union-of-paths diff, exact
  numerical source projection/native/runtime equivalence, unchanged default
  324-cell plan and exact selected 72-cell plan. Primary-to-final differences
  are confined to academic plotting, report plotting, matrix profile selection,
  and snapshot enumeration. Control-to-final differs only in report plotting.
  Historical numerical/pilot receipts retain their actual original hashes.
- Presentation: all 18 final density/overhead artifact hashes verified; driver
  density and PW overhead visually inspected. Labels state simulated samples,
  conditional support, assumed head properties and carry-only scope. The
  conditional-SG heatmap population defect is corrected: scored intersection
  n=8/12 remains distinct from per-pattern geometric n=11/12 in its regression.
- Frozen software: the corrected suite completed with **124 passed**, exit 0,
  and unchanged sources. Its receipt equals the accepted full snapshot.

## Conditions on Interpretation and Final Review

Keep within-lean and common-distance protocols distinct. Common-distance results
compare recalibrated nominal deliveries; they do not isolate lean causally or
establish every untested noise profile. Across-lean paired analysis must use a
common supported-index intersection with its counts disclosed. Numerical solver
failures abort execution; they are not unsupported physical probability mass.

Head-assumption sensitivities substantially exceed numerical error, including
22.63 m maximum driver endpoint change for the tested radius perturbation.
These are deterministic perturbations, not confidence bounds. Findings remain
conditional on assumed rigid head geometry, energetic restitution/friction,
Gaussian delivery noise, carry-as-stop course geometry and historical scoring.
Empirical validation remains deferred under issue #11984.

After execution, independently audit all final plan/status counts, raw row and
manifest integrity, nominal matching, covariance/statistics, scoring, conditional
populations, uncertainty, numerical exceptions, and final rendered claims. Pilot
sample sizes provide no final scientific effect inference. Final-data acceptance
and project completion remain open.

## Evidence Hashes

- `validation/independent_contact_audit.json`: `ca4c5b91d3113d55677ab8923af8cf567077d9a0493963aa8280e3d7bf72ca0e`
- `validation/independent_preflight.json`: `b472dd91bfea058737f6bc441f29d903bc2718d4354438ba65d3a2c9ff108ed9`
- `validation/independent_head_sensitivity.json`: `3bd6acb1118b6cbedd03171744a09a1b810f8a77cd7e4dc72c1d609181edb558`
- `validation/independent_pilot_audit.json`: `11ca90a9c77843da2adc6d2fec59dd20283792e0f37dff792dd1a675b0d9492e`
- `validation/independent_control_pilot_audit.json`: `c3dc7fc43d6ff9ca044d291fe5f5a7af059313b28cc20af5affafd0d5f330bad`
- `validation/independent_control_scoring.json`: `2f647e734840ea49b5b6d3a18d3817d21caf2ac5797cf624030e26ae0460e7ff`
- `validation/frozen_software.json`: `955b1af843ccfbb1bdc2ccc7dcd7e796b4dc2232947b7fb1bcd58469b0476dd3`
- `pilot_v1/figure_review/pilot_comparable_shaft_path_final/review_receipt.json`: `988a1f24d6851e56fcebaaad7aa70ef6b7f70095e5381145c3b72a3c807b2c02`
