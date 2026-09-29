# Issue: Implement load_club_target_excel.m and load_club_target_c3d.m

## Summary

Implement the two MATLAB loaders that turn measured swings into the canonical
`target` struct consumed by every motion-matching option: a Wiffle/ProV1 Excel
loader and a C3D loader. As part of this issue, validate the one untested C3D file
in `Data/Mocap C3D Files/` so we know whether it parses cleanly and which
markers map to butt/clubhead.

## Motivation

See `motion_matching/shared/CLUB_IK_SPEC.md`. The cost function (#015) and every
optimizer (#024–#040) need a uniform `target` struct. Today the Excel mocap is
read by `mocap_data_loader.py` in inches; nothing reads the C3D file. Without
this loader, no option can be tested against a real swing.

## Dependencies

None — this is foundational shared infrastructure.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\load_club_target_excel.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\load_club_target_c3d.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\detect_impact_idx.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\resample_to_simulation_grid.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\rotmat_to_quaternion.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_load_club_target_excel.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_load_club_target_c3d.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\C3D_VALIDATION_NOTES.md` (documents the marker mapping discovered during this issue)

## Public API

Verbatim from `CLUB_IK_SPEC.md`:

```matlab
function target = load_club_target_excel(xlsx_path, sheet_name, opts)
%LOAD_CLUB_TARGET_EXCEL  Read club 6-DOF trajectory from a Wiffle-style xlsx file.
%
%   target = LOAD_CLUB_TARGET_EXCEL(XLSX_PATH, SHEET_NAME, OPTS) returns a
%   target struct as specified in CLUB_IK_SPEC.md.

function target = load_club_target_c3d(c3d_path, opts)
%LOAD_CLUB_TARGET_C3D  Read club 6-DOF trajectory from a C3D file.
```

The output `target` struct must match the canonical schema:

```matlab
target = struct( ...
    "time",         (N×1) double, simulation timegrid in seconds, monotonic, dt = 1/sample_rate, ...
    "butt",         (N×3) double, butt position in metres, world frame, ...
    "clubhead",     (N×3) double, clubhead position in metres, world frame, ...
    "club_quat",    (N×4) double, club orientation as unit quaternion [w x y z], ...
    "impact_idx",   scalar uint32, index of impact frame (max clubhead speed), ...
    "source",       struct with provenance: filename, format, subject_id, trial_id, sha256 ...
);
```

## Required tests (TDD)

- `test_excel_loader_returns_canonical_target_for_TW_ProV1_sheet`
- `test_excel_loader_converts_inches_to_metres`
- `test_excel_loader_normalizes_quaternion_sign_w_nonnegative`
- `test_excel_loader_resamples_to_1000_hz`
- `test_excel_loader_detects_impact_at_max_clubhead_speed`
- `test_excel_loader_rejects_missing_sheet_with_clear_error`
- `test_c3d_loader_parses_known_file_or_xfail_with_documented_reason`
- `test_c3d_loader_marker_mapping_matches_validation_notes_md`
- `test_loader_postcondition_butt_clubhead_distance_is_plausible_shaft_length`
- `test_loader_postcondition_quaternion_unit_norm_within_1e_minus_6`
- `test_loader_postcondition_time_strictly_increasing_starts_at_zero`
- `test_loader_postcondition_sha256_matches_file`

## DbC contract

Preconditions (enforced in `arguments` block):

- `xlsx_path` / `c3d_path` exist as files.
- `sheet_name` is one of `["TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"]` for Excel.
- `opts` is the result of `default_align_options()` with optional overrides.

Postconditions (assertions, per `CLUB_IK_SPEC.md` §"Validation rules"):

- `target.time` strictly increasing, `time(1) == 0`, `time(end) <= T_max + eps`.
- All trajectory arrays have the same number of rows as `time`.
- `target.butt`, `target.clubhead` contain no NaN/Inf; `‖r‖ < 5 m`.
- `target.club_quat` rows unit-norm to within `1e-6`.
- `1 <= target.impact_idx <= N`.
- `target.source.sha256` matches a freshly computed sha256 of the source file.

## Acceptance Criteria

- [ ] Both loaders implemented and exported under `motion_matching/shared/`.
- [ ] `runtests('motion_matching/shared/tests')` passes for all listed tests.
- [ ] C3D file in `Data/Mocap C3D Files/` either parses cleanly **or** the test xfails with a documented reason and a follow-up issue link.
- [ ] `C3D_VALIDATION_NOTES.md` documents the marker mapping (or the parse failure).
- [ ] `arguments` blocks present on every public function (DbC preconditions).
- [ ] `assert(...)` postconditions present per `CLUB_IK_SPEC.md` §"Validation rules".
- [ ] No file exceeds 1200 lines.
- [ ] No `disp()` in production code paths; uses `fprintf` gated by `opts.verbosity`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `matlab`, `tdd`, `dbc`, `infra`

## Effort estimate

M (1-3 days). The Excel loader can adapt `mocap_data_loader.py` semantics; the C3D
validation pass is the unknown.
