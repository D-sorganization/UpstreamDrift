# Segmental Spine Research Notes

Editable research source: [segmental_spine.tex](segmental_spine.tex)
(compile with `tectonic`; the PDF is not committed). Epic #12735, scaffold
issue #12738, consolidation #12773. Binding conventions: ADR-0055
(`docs/adr/0055-segmental-spine-kinematics-and-loads.md`). Engine-independent
code lives in `src/shared/python/spine/`; the OpenSim pipeline lives in
`src/engines/physics_engines/opensim/python/spine/`. The module directory is
the "Segmental Spine" section of `docs/agents/shared-infrastructure.md`.

## Layout

- `segmental_spine.tex` assembles one subsection per child issue from
  `sections/<key>.tex` (for example `sections/spn-1a.tex`) with
  `\IfFileExists`; a missing file renders as "Pending".
- Each child writes only its own `sections/<key>.tex`, starting with a
  Title Case `\subsection{...}`. The main file is edited only to add sections.
- `sections/.gitkeep` keeps the directory tracked.
- `data_extraction.md` holds the published numbers transcribed from the
  sources, each table with its `verified_from_primary` flag.
- The calculation-level engineering design manual is separate: chapter
  `manuals/upstreamdrift/chapters/66-segmental-spine.qmd`, continued in
  `68-segmental-spine-loads-and-studies.qmd` (QMD is its only editable source).

## Compile

```bash
cd docs/research/segmental_spine && tectonic segmental_spine.tex
```

## How to Run a Study

A study runs one matched golf swing through the whole pipeline: golf input,
segmental drive, spinal inverse dynamics, trunk static optimisation and joint
reactions. It writes a hash-pinned receipt. Run every command from the
repository root with OpenSim 4.6 installed, headless
(`QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg`).

### Inputs

- A golf swing `.sto` or `.mot` with the `golf_humanoid` coordinates
  (`lumbar_extension`, `lumbar_bending`, `lumbar_rotation`, and the pelvis
  coordinates when present). Events are read from `event_<name>` metadata.
- The golf model `.osim` that the swing was matched on, for example
  `src/engines/physics_engines/opensim/models/golf_humanoid.osim`.
- A segmental spine `.osim` whose level map exists in
  `src/config/spine_model_level_maps.json`. Today only the `fixture` map is
  verified. The licensed Bruno and Senteler maps are `UNVERIFIED` until the
  owner downloads, licenses and pins the SimTK models (#12749, #12755).

To make the invented five-level fixture with lumped muscles (no SimTK data):

```bash
python3 -c "from src.engines.physics_engines.opensim.python.spine.fixture_model import FixtureSpec, write_fixture_osim; write_fixture_osim(FixtureSpec(), 'study/fixture.osim', muscles=True)"
```

### Request JSON

The request maps one-to-one onto `SpineStudyRequest`. Relative paths resolve
against the directory of the request file, so the example expects
`study/request.json`, the fixture as `study/fixture.osim` and your swing as
`study/swing.sto`. `output_dir` may be omitted from the file when `--out` is
given; `--out` overrides the file's value otherwise.

```json
{
  "study_id": "fixture_smoke",
  "motion_path": "swing.sto",
  "golf_model_path": "../src/engines/physics_engines/opensim/models/golf_humanoid.osim",
  "model_path": "fixture.osim",
  "model_id": "fixture",
  "coupling_law_id": "uniform_reference",
  "restrict_law_to_model": true,
  "cost_exponent": 3,
  "pelvis": "fixture",
  "events": {"mid": 0.2, "late": 0.3},
  "max_frames": 5,
  "torso_alignment": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "torso_alignment_source": "fixture: identity by construction",
  "output_dir": "out"
}
```

| Field | Meaning |
|---|---|
| `study_id` | File-name-safe id (letters, digits, `_.-`); prefixes every output |
| `motion_path`, `golf_model_path` | Golf swing and the golf model it was matched on |
| `model_path` | Segmental `.osim`; may be omitted only with `asset_id` |
| `asset_id` | Licensed asset id from `src/config/spine_model_assets.json`, resolved through its environment variable (for example `UD_SPINE_BRUNO_SOURCE`) and admitted by SHA-256 |
| `model_id` | Level-map key in `src/config/spine_model_level_maps.json` |
| `coupling_law_id` | Law id from `src/config/spine_coupling_laws.json`, or `null` to extract the law from the model's own couplers |
| `restrict_law_to_model` | Restrict the named law to the model's level range (needs `coupling_law_id`) |
| `cost_exponent` | Static-optimisation exponent, `2` or `3` (default `3`) |
| `external_loads` | ExternalLoads XML, or `null` for gravity only |
| `events`, `stride`, `max_frames` | Static-optimisation frames: named event times in seconds, every `stride`-th sample, and a hard cap (exceeding it is an error) |
| `pelvis` | Ground-pelvis encoding of the segmental model: `golf` (ZXY, default) or `fixture` (free joint, XYZ) |
| `torso_alignment`, `torso_alignment_source` | Drive alignment $A$ (3 × 3 proper rotation, terminal frame in the golf torso frame) and how it was calibrated; both or neither. Without them the study is `unqualified` (`assumed_identity`) |
| `drive_tolerance_rad`, `filter_hz` | Drive residual flag in radians, and the inverse-dynamics low-pass cutoff |
| `residual_force_fraction_bw`, `residual_moment_fraction` | Pelvis residual budget (defaults 0.05 and 0.01) |
| `reserve_limit_nm`, `reserve_max_share` | Reserve-actuator limit and maximum share (defaults 1.0 N·m and 0.05) |
| `output_dir` | Where artifacts and the receipt are written; never part of the request identity |

### Run One Study

```bash
python3 -m src.engines.physics_engines.opensim.python.spine.study \
    --request study/request.json --out study/out
```

Outputs in the output directory: `<study_id>_receipt.json`,
`<study_id>_coordinates.mot` and `<study_id>_loads.npz`. The receipt is also
written to standard output, after any OpenSim log lines, so read the file
rather than parsing standard output. The exit status is 0 only for status
`pass`.

Read the receipt's `status` and `status_reasons`. `failed` means a stage
raised; `unqualified` means a licensed asset is not `QUALIFIED`, a drive frame
is invalid, the pelvis residual exceeds its budget, a frame used reserves or a
solve failed. `pass` is a software statement only; `scientific_status` is
always `unqualified`. With the request above and gravity only, the fixture
returns `unqualified` with the reasons "inverse-dynamics pelvis residual
exceeds the budget" and "2 frames flagged for reserve use". That is expected:
without external support loads the pelvis residual carries the gravitational
support (the whole body weight when stationary) plus the whole-body inertial
demand of the prescribed motion. Do not
widen a budget to make a run pass; a widened budget is recorded in the
receipt and must be justified.

### Run the Distribution Sensitivity

```bash
python3 -m src.engines.physics_engines.opensim.python.spine.distribution_sensitivity \
    --request study/request.json --out study/sensitivity \
    --laws bruno2015_regional uniform_reference --threshold 0.1
```

The base request is run once per law (`--laws` defaults to every shipped law)
with `restrict_law_to_model` set and only `coupling_law_id` changed. Outputs:
one receipt, `.mot` and `.npz` per law as `<study_id>__<law>_*`, plus
`<study_id>_sensitivity.json` and `<study_id>_sensitivity.csv`. Only `pass`
receipts enter a spread; the exit status is 0 only when at least two laws
qualified. With the gravity-only fixture request above both laws are excluded
as `unqualified`, and the report says so.

### Aggregate Receipts

`study_aggregation.aggregate_receipts` builds per-level peak and impulse
tables from receipts whose artifacts match their hashes; `write_csv` and
`write_json` write deterministic tables. It has no command-line entry point.

Result status: not qualified; no scientific result is claimed here.
