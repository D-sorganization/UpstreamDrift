# METRICS_SCHEMA.md

Canonical metrics schema for motion-matching diagnostics.

This schema is the single source of truth shared by:

- Python emitter / consumer: `src/shared/python/motion_matching/metrics.py`
- MATLAB emitter / consumer: `motion_matching/shared/+metrics/Metrics.m`
- Leaderboard consumer: `motion_matching/shared/leaderboard.m`
- Diagnostics emitter: `MachineLearning/evaluate_matching_workflow.py`

The schema is versioned with semver. The current version is **`1.0.0`**.
Any change to field set, units, or computation rules requires a major bump
and a coordinated update of all emitters/consumers.

## Field reference

| field                          | type  | unit / format       | description                                                                                  |
| ------------------------------ | ----- | ------------------- | -------------------------------------------------------------------------------------------- |
| `swing_id`                     | str   | identifier          | Identifier of the swing being matched (e.g. `"TW_ProV1"`).                                   |
| `option`                       | int   | 1..4                | Which of the four motion-matching options produced this fit.                                 |
| `solver`                       | str   | label               | Solver label, e.g. `"fmincon-sqp"`, `"surrogate+fmincon"`, `"cvae-rejection"`.               |
| `n_iterations`                 | int   | count               | Solver iterations consumed.                                                                  |
| `rmse_clubhead_mm`             | float | millimetres         | Final clubhead-position RMSE (Euclidean, sample mean) over the matched window.               |
| `rmse_butt_mm`                 | float | millimetres         | Final butt-position RMSE (Euclidean, sample mean) over the matched window.                   |
| `rmse_orientation_deg`         | float | degrees             | Final orientation RMSE as the geodesic angle between simulated and target quaternions, mean. |
| `clubhead_speed_at_impact_mph` | float | mph                 | Simulated clubhead speed at the impact frame.                                                |
| `clubhead_speed_meas_mph`      | float | mph                 | Measured clubhead speed at impact (used to compute the delta).                               |
| `total_work_J`                 | float | joules              | Total mechanical work integrated over the swing (sum of joint torque-power integrals).       |
| `peak_power_W`                 | float | watts               | Peak instantaneous mechanical power across the swing.                                        |
| `wall_clock_s`                 | float | seconds             | Solver wall-clock time.                                                                      |
| `git_commit`                   | str   | 40-char SHA         | Git rev of the codebase at fit time. Use `"0" * 40` if dirty/unknown.                        |
| `matlab_version`               | str   | release string      | MATLAB release if applicable (e.g. `"R2024b"`); empty string `""` for Python-only fits.      |
| `python_version`               | str   | `major.minor.patch` | `platform.python_version()` of the emitter; empty `""` for MATLAB-only fits.                 |
| `timestamp_iso8601`            | str   | ISO-8601 UTC        | UTC timestamp, RFC 3339 form, e.g. `"2026-05-05T17:34:21Z"`.                                 |
| `schema_version`               | str   | semver              | Must equal the current schema version (`"1.0.0"`).                                           |

## Computation rules

- `rmse_*` are computed over the same time window as the cost function (see
  `COST_FUNCTION_SPEC.md`). They are sample means of per-frame residual norms,
  not RSS. Units are millimetres (positions) or degrees (orientation).
- `rmse_orientation_deg` is the mean of the geodesic angle
  `2 * acosd(|dot(q_sim, q_meas)|)` over the matched window.
- `total_work_J` is `sum_t |tau(t) . qdot(t)| dt` over all actuated joints.
- `peak_power_W` is `max_t |tau(t) . qdot(t)|` (signed power; absolute value
  of the inner product, taken over time only — joints summed first).
- `clubhead_speed_*_mph` use the same impact-frame definition as
  `CLUB_IK_SPEC.md`.

## Validation rules

Validated at construction (Python `__post_init__`, MATLAB constructor):

1. All numeric fields must be **finite** (no NaN, no inf).
2. `option` must be in `{1, 2, 3, 4}`.
3. `n_iterations` must be `>= 0`.
4. `rmse_*` must be `>= 0`.
5. `clubhead_speed_*_mph` must be `>= 0`.
6. `wall_clock_s` must be `>= 0`.
7. `git_commit` must be 40 lowercase hex characters.
8. `timestamp_iso8601` must parse as ISO-8601 and end in `Z` (UTC).
9. `schema_version` must equal the current schema version constant.

## Round-trip guarantee

`Metrics → JSON → Metrics` and `Metrics → CSV row → Metrics` are bit-equal
(modulo float repr). Both languages emit the same JSON byte-for-byte for
the same record (sorted keys, no whitespace, fixed `ensure_ascii=False`).

## Backwards compatibility

`leaderboard.m` accepts a legacy result struct with fields
`{swing_id, option, rmse_clubhead, rmse_butt, ...}` and auto-converts via
`metrics.Metrics.fromLegacyStruct`. Legacy struct support will be removed
when `schema_version` reaches `2.0.0`.
