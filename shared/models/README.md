# Shared Anthropometric Body Model

This directory is the **single source of truth** for the body-segment
geometry, mass distribution, and joint topology of the golf humanoid used by
every physics-engine wrapper in this repo (Simscape, MuJoCo, Drake,
Pinocchio, OpenSim).

See [`src/engines/CROSS_ENGINE_PARITY_SPEC.md`](../../src/engines/CROSS_ENGINE_PARITY_SPEC.md)
§2.6 for the architectural rationale. In short:

- The Simscape `.slx` is the reference implementation and the source of all
  numeric values.
- Every other engine consumes engine-native model files (URDF / MJCF /
  `.osim`) generated from these YAMLs by `scripts/build_humanoid_models.py`
  (issue PARITY-MODEL-BUILD).
- Hand-edited engine files are forbidden — re-generate from the YAML.

## Files

| File                            | Purpose                                          |
| ------------------------------- | ------------------------------------------------ |
| `golf_humanoid_dimensions.yaml` | Per-segment lengths and visualisation radii      |
| `golf_humanoid_inertia.yaml`    | Per-segment mass, COM offset, 3×3 inertia tensor |
| `golf_humanoid_topology.yaml`   | Joint chain (parents, children, types, DOF axes) |

## Schema

### `golf_humanoid_dimensions.yaml`

```yaml
schema_version: 1
units_system: SI

<segment_name>:
  value:         <float>      # numeric value in SI base units
  units:         "m"          # always metres
  source:        <str>        # provenance string
  simscape_name: <str>        # original model-workspace identifier
  raw_value:     <float>      # value as stored in the model workspace
  raw_units:     "in" | "m"   # original units
  notes:         <str>        # optional commentary
```

### `golf_humanoid_inertia.yaml`

```yaml
schema_version: 1
units_system: SI

<segment_name>:
  mass_kg: <float>
  com_offset_m: [<float>, <float>, <float>] # in segment-local frame
  inertia_kgm2: # 3×3 tensor, segment-local
    - [<I11>, <I12>, <I13>]
    - [<I21>, <I22>, <I23>]
    - [<I31>, <I32>, <I33>]
  source: <str>
  notes: <str>
```

The `Club` entry has a richer schema (shaft + clubhead sub-objects); see the
file for details.

### `golf_humanoid_topology.yaml`

Lists `bodies` and `joints` as arrays plus a `q_order` array specifying the
canonical layout of `SimOut.q` (length 25). Joint types: `floating`,
`universal`, `revolute`, `gimbal`, `welded`. See the file's header comment
for the chain summary.

## Units

**SI throughout.** Lengths in metres, masses in kilograms, inertias in
kg·m². The Simscape model authors most lengths in **inches** and most
segment masses in **pounds** (as documented in the Simscape body-element
unit settings); the YAMLs store the converted SI values and record the raw
value + units in `raw_value` / `raw_units` for traceability.

Conversion factors (exact):

- 1 in = 0.0254 m
- 1 lb = 0.45359237 kg

## Provenance

The numeric values in these YAMLs were extracted from a live Simscape
simulation:

- **Dataset trial**: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/`
  `matlab/Scripts/Dataset Generator/golf_swing_dataset_20251030/`
  `trial_001_20251030_202704.csv`
- **Model**: `GolfSwing3D_Kinetic.slx` (commit history available in
  `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/`)
- **Snapshot**: model-workspace identifiers verified against the text-mode
  `mdl_reference/GolfSwing3D_Kinetic.mdl` snapshot

Per-segment inertias come from the `SegmentInertiaLogs.*` bus signals,
which Simscape computes internally from the body-element geometry +
`BasedOnType=Mass`/`Density` settings — the logged values are SI ground
truth and don't depend on the (inches/pounds) authoring units.

The whole-body mass at t=0 was 77.61 kg, matching the
`SegmentInertiaLogs.GolferMass` channel.

## Known gaps

- **Right hand vs left hand asymmetry**: the Simscape model rolls a portion
  of the club inertia into the right hand (`RH`) inertia logs, which is why
  RH and LH have different mass/COM/inertia in
  `golf_humanoid_inertia.yaml`. This is faithful to the source model but
  may need redistribution when generating engine-native files that prefer a
  symmetric hand model + explicit club mass.
- **Right-hand grip closure as a kinematic loop**: the Simscape model
  closes the right-hand grip via a parallel kinematic loop (right hand also
  rigidly attached to the grip). Engines without parallel-loop support
  (MuJoCo) need to handle this as a constraint-cost term. The `grip_locked`
  joint in the topology YAML attaches the club to the LEFT hand only; the
  right-hand contribution is implicit in the RH inertia.
- **`UpperArmLength` vs `LeftUpperArmLength`/`RightUpperArmLength`**: the
  Simscape model carries both names (the side-less one for visualisation
  cylinders, the side-suffixed ones for the kinematic chain). The YAML
  preserves both for round-trip fidelity but downstream engines should use
  the side-suffixed names.

If the Simscape model is updated, regenerate the YAMLs by re-running a
trial and re-extracting the values (the helper script
`scripts/_extract_dims.py` consumes a dataset CSV and dumps the relevant
columns as JSON).

## Validation

Run the test suite:

```bash
python3 -m pytest tests/test_golf_humanoid_dimensions.py -v
```

The test checks:

- All three YAMLs parse cleanly.
- Every required segment-length / mass / inertia key is present.
- Units are SI.
- No zero or negative segment lengths or masses.
- Inertia tensors are positive semi-definite.
- The topology DOF count matches `total_dof`.
