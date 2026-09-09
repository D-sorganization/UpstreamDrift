# Reproducible C3D Reference Model Fitting

The fitting workflow converts a C3D into joint-angle trajectories and saved
reference animations that the Capture Rig comparison player already understands.
It uses the existing continuous articulated solver, reference library, camera
projection, registration and event synchronization. Governing epic: #9914.

## Run a Saved Job

From the repository root, initialize the pinned Tools submodule and use the
project Python environment. In PowerShell, expose the repository packages:

```powershell
$env:PYTHONPATH="$PWD/src;$PWD/src/shared/python"
python3 -m src.motion_capture.reference.fit_job --list-models
python3 -m src.motion_capture.reference.fit_job --config examples/motion_capture/reference_driver.json
```

The example fits the bundled Tour Average driver with the scapula golfer at the
original 360 Hz. Outputs must use a new directory: previous runs are never
overwritten. Change `source` to `data/C3D_TA_Iron.c3d` for the iron capture and
choose another output directory. Set `models` to `["all"]` for the available
catalog, or list selected names. `stride: 12` selects every twelfth source frame
for an inexpensive 30 Hz survey; `stride: 1` retains every source sample.
The saved timestamps remain source timestamps, without implicit event shifts.

`max_iterations` is the maximum per robust solver stage, not a total budget or
a certificate of convergence. The report records total iterations and residuals.
`fit_lengths` allows only a model's declared learnable lengths to change. URDF
and MJCF preset geometry stays fixed. The reference job's large default robust
thresholds keep typical body-model mismatch in the objective; lower thresholds
are an explicit choice for noisy captures. Always inspect all-observation RMS,
maximum error, per-landmark errors and the retained sample count together.

## Marker Profiles and Conventions

The Tour Average profile identifies surface-marker proxies for the fifteen
reconstruction joints. It uses posterior shoulder markers because the right
shoulder-top channel is known to be occluded. Waist markers approximate hips;
BackTop approximates the neck. These are not measured anatomical joint centers.
The profile is saved verbatim in every bundle.

For another C3D, supply `profile` in the job with `name`, `units`, `axes`,
`joints` and optional `notes`. `joints` maps a reconstruction name to a list of
source marker labels; multiple labels mean their centroid. Every constituent
must be valid in that frame. Missing samples remain NaN and are not interpolated
through during fitting. Unknown marker names fail with an explicit diagnostic.

`axes` names the signed source axes assigned to canonical X, Y, Z. Tour Average
uses `["+X", "-Z", "+Y"]`, converting source Y-up to right-handed Z-up. The
existing registration function then converts to the fitter/camera Y-up world.
An odd axis permutation without the required sign change is rejected.
The resulting ReferenceMotion always uses canonical right-handed Z-up metres.

## Models and Extensions

| Model                                     | Geometry Used                               | Important Limits                                                              |
| ----------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------- |
| `golfer`                                  | Existing scapula-capable articulated golfer | Surface proxies; some lengths may be learned; unobserved twists follow priors |
| `double_pendulum`, `triple_pendulum`      | Existing reduced models                     | Only shoulder/elbow/hand centroids; their scores are not full-body scores     |
| `pinocchio_golfer`, `pinocchio_golfer_ik` | Bundled native URDF trees                   | Link-origin fitting; no native dynamics or closed-grip validation             |
| `drake_golfer`                            | Its own bundled URDF upper-body tree        | Distinct from the Pinocchio model; no legs                                    |
| `simple_humanoid`                         | Bundled simple-humanoid URDF                | Limited degrees of freedom and no wrist endpoint landmarks                    |
| `human_subject`                           | Bundled human-subject URDF                  | Link-origin proxies; full tree retained                                       |
| `mujoco_humanoid`                         | MuJoCo-compiled MJCF hinge tree             | Requires MuJoCo; compiled FK parity tested; no wrist endpoints                |
| OpenSim                                   | Not substituted                             | Custom joints and coupled coordinates require a native adapter                |
| Bundled MyoSuite files                    | Not substituted                             | The checked-in body files explicitly identify themselves as placeholders      |

Every run saves an inventory with unavailable-model reasons. Scores use each
model's mapped observations: compare the landmark lists before comparing RMS.
The outputs describe a constrained kinematic fit, not validated muscle forces,
anatomical accuracy, or a globally optimal solution.

To use another URDF or hinge-tree MJCF, add an explicit custom model:

```json
{
  "models": ["my_model"],
  "custom_models": {
    "my_model": {
      "path": "path/to/model.urdf",
      "format": "urdf",
      "landmarks": { "pelvis_link": "mid_hip", "hand_link": "left_wrist" }
    }
  }
}
```

Merge those fields into a complete job. Names must not collide with presets.
The URDF adapter retains offsets, arbitrary hinge axes, fixed rotations and
limits. Mimic/prismatic/internal floating joints fail explicitly. The MJCF
adapter uses the real compiler; equality, slide and ball constraints fail
explicitly. MJCF root state is an added scene pose and hinge values are
displacements from native `qpos0`, not native `qpos` arrays. For other joint
types, implement and independently qualify an adapter before claiming support.
Python callers can also pass a `RegisteredModel` directly to `fit_reference`.

## Display Over a Golfer

Set the job's `library` to the `references` subdirectory of the capture library
you use in Capture Rig. After fitting, refresh Expert Reference Library. Each
model is a separate motion asset; no original capture is changed.

1. Open the golfer recording and reference comparison for a camera view.
2. Select the fitted asset, then bind the session's camera evidence.
3. Use Placement to reorient, translate and optionally scale the reference.
   The headless `estimate_reference_transform` helper accepts explicitly paired
   Y-up world points at a selected pose and returns a saved ReferenceTransform.
   It requires at least three non-collinear pairs; it does not calibrate a camera.
4. Use Timing to align events such as address, top, impact and finish. The
   existing TimeMapping supports offset and bounded event/rate alignment.
5. Play or export the comparison. Colour, opacity, registration, camera evidence
   and timing follow the existing saved comparison recipe.

Use one fixed spatial transform through playback. Aligning each frame separately
would erase precisely the golfer/reference differences being compared. The
current overlay is the model's joint-tree graphic, including unobserved model
joints; it does not claim shaded native mesh or muscle rendering. Geometry is
identical in preview and export through the existing compositor. Tests exercise
fitted assets in two camera views and confirm no overlay outside the source clock.

## Bundle and Reproduction Evidence

Each model folder contains `q.npy`, `landmarks_m.npy`, `observed_m.npy`,
`quality.json`, `manifest.json` and `references/<uuid>.json`. The manifest records
the exact model specification, landmark map, fitted dimensions, numerical
options, sample clock, source hash, implementation hashes and runtime versions.
It also hashes every output file. The library identity covers inputs and options
without depending on the workstation's source path. Source files remain linked
for provenance; the animation and observations are self-contained.

For a standalone four-keyframe graphic:

```python
from pathlib import Path
from src.motion_capture.reference.fit_preview import render_fit_preview

render_fit_preview(Path("../reference-runs/tour-driver/golfer"), Path("../reference-runs/driver.png"))
```

Orange points are the measured proxies; teal lines are the fitted model. This
is a fixed three-dimensional view, not a calibrated image of a real golfer.
