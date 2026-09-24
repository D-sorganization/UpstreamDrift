---
title: Character Builder
tile_id: character_builder
status: complete
---

# Character Builder

## Purpose

Character Builder creates and edits humanoid URDF models with anthropometric
scaling, so a subject-specific model can be produced from a small number of
scalars and then used in MuJoCo, Drake, Pinocchio or any other supported
engine. It also converts between model formats, validates and compares URDF
files, composes models from parts, computes inertia tensors, and manages a
local model library.

The tile lives in the `web_catalog` half of the registry: `surfaces: ["web"]`
with the reason "web page (/tools/character-builder); the native side is a
CLI", `web.mode: route`, `web.route: /tools/character-builder`,
`launcher.status: gui_ready`. The native entry point is
`src/shared/python/model_generation/cli/main.py` (`model-gen`). Declared
capabilities: `urdf_generation`, `anthropometry`, `mesh_generation`,
`collision_geometry`.

## Inputs

CLI subcommands and their arguments, from the parser in
`model_generation/cli/main.py`:

| Subcommand | Arguments | Unit |
| --- | --- | --- |
| `generate` | `name`, `-o/--output`, `--height`, `--mass`, `--proportions` (JSON), `--humanoid` | -, path, m, kg, JSON object, flag |
| `convert` | `input`, `-o/--output`, `--from-format`, `--to-format`, `-n/--name` | path, path, format id, format id, - |
| `validate` | `input`, `--json` | path, flag |
| `diff` | `file_a`, `file_b`, `--json` | path, path, flag |
| `info` | `input`, `--json` | path, flag |
| `library list` | `-c/--category`, `-s/--source`, `--search`, `--json` | - |
| `library add` | `input`, `-n/--name`, `-c/--category`, `--tags` (comma separated) | path, -, -, - |
| `library download` | `model_id`, `-o/--output` | -, path |
| `library import` | `--query`, `--url`, `--min-stars` (default 10), `--limit` (default 10), `--json` | -, URL, count, count, flag |
| `compose` | parts, `-o/--output` (required), `-n/--name` | path, path, - |
| `inertia` | shape, `mass`, shape dimensions, `--json` | -, kg, m, flag |

Global flags include `--verbose`, `--quiet` and `--version` (reported as
`model-gen 1.0.0`).

The Python surface documented in the quickstart takes body parameters directly:

| Parameter | Meaning | Unit or range |
| --- | --- | --- |
| `height_m` | subject height | m |
| `mass_kg` | subject mass | kg |
| `muscularity` | build | 0.0 slim to 1.0 muscular |
| `gender_factor` | build | 0.0 female to 1.0 male |

Presets supply these values instead. `character_presets.md` lists them,
including `average` (1.75 m, 75.0 kg), `athletic` (1.80 m, 80.0 kg),
`male_average` (1.78 m, 80.0 kg), `female_average` (1.65 m, 62.0 kg),
`tall_male` (1.93 m, 88.0 kg, described as 95th percentile height per CDC),
`petite_female` (1.55 m, 52.0 kg, 5th percentile per CDC), `child_8yo`
(1.28 m, 26.0 kg, CDC growth charts 50th percentile) and `senior_70yo`
(1.70 m, 72.0 kg, NHANES average).

## Outputs

| Output | Description | Unit |
| --- | --- | --- |
| URDF XML | written to `--output` (parent directories created) or logged to the console | - |
| Converted model | URDF, MJCF or Simscape output per `--to-format` | - |
| Validation report | pass or fail with per-error messages; `--json` for machine reading | - |
| Diff report | differences between two URDF files; `--json` available | - |
| Model info | model summary for one URDF; `--json` available | - |
| Library listing and entries | model id, name, category, source, tags | - |
| Inertia tensor | for a named shape and mass | kg m^2 |
| Exit status | 0 on success, 1 on a build failure, missing input or invalid JSON | - |
| Subject record | schema-versioned JSON via `save_subject` in the anthropometrics pipeline | - |
| URDF `<inertial>` blocks | one per segment, via `write_urdf_inertial` | mass in kg, inertia in kg m^2 |

## Method

`cmd_generate` constructs a `ParametricBuilder` with the requested robot name,
applies `set_height` and `set_mass` when given, parses `--proportions` as JSON
into `set_proportions(**proportions)`, optionally calls
`add_humanoid_segments()`, then calls `build()`. A build result that is not
successful is reported error by error and returns exit code 1; otherwise
`result.to_urdf()` is written or printed. `cmd_convert` infers the source
format when `--from-format` is `auto`, using the file suffix: `.slx` and `.mdl`
mean Simscape, `.xml` targeting URDF means MJCF, `.urdf` means URDF.

Anthropometric scaling is the separate `anthropometrics` package documented in
[anthropometrics.md](../user_guide/anthropometrics.md): a `DeLevaEstimator`
turns `subject_id`, `height_m`, `mass_kg` and `sex` into a full subject with
per-segment inertial properties, which are persisted as schema-versioned JSON
and emitted as URDF `<inertial>` elements. The design rationale is
ADR-0009 (`docs/adr/0009-anthropometrics-pipeline.md`).

Optional external model sources are described in the quickstart: SMPL-X for
mesh generation from body scans (`pip install smplx`), and MakeHuman via FBX or
OBJ export followed by the converter tools.

Defining modules: `src/shared/python/model_generation/` (CLI, builders,
converters, library) and `src/shared/python/humanoid_character_builder`
(`quick_urdf`, `BodyParameters`, `CharacterBuilder`).

## Limitations

- There is no native GUI. The registry states plainly that the native side is a
  CLI and that the interactive surface is the web route
  `/tools/character-builder`. `launcher.status` is `gui_ready`, not `ready`.
- Presets are parameter tables, not validated subjects. The percentile and
  survey attributions in `character_presets.md` describe where the height and
  mass came from; they do not make a generated model a validated
  representation of that population.
- `muscularity` and `gender_factor` are unitless build dials. They are not
  measured quantities and carry no anthropometric guarantee.
- Mesh generation needs optional third-party packages. SMPL-X support requires
  a separate install, and MakeHuman requires exporting from an external
  application first.
- Format conversion is inferred from suffixes. `--from-format auto` decides
  from the file extension, so an unusually named file must be told its format
  explicitly.
- CLI errors are logged and returned as exit code 1. Failures are not raised as
  exceptions, so a script must check the status code.
- The library importer reaches out to GitHub. `library import` filters by
  `--min-stars` and `--limit`; nothing about a third-party model's licensing,
  scale convention or inertial correctness is checked by this tool.
- No simulation happens here. Character Builder produces model files; loading,
  running and validating them is the physics engines' job.

## See Also
- [Character Builder quickstart](../user_guide/character_builder_quickstart.md)
- [Anthropometrics user guide](../user_guide/anthropometrics.md)
- [Character presets reference](../user_guide/character_presets.md)
- [Engine Selection](engine_selection.md)
- [Project Map](../architecture/PROJECT_MAP.md)
