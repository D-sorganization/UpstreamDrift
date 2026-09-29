# Drake — generated models

This directory holds **machine-generated** model files for the Drake
physics engine. Files here are produced from the shared anthropometric
YAML at `shared/models/golf_humanoid_dimensions.yaml` (owned by issue
#4093) by the generator at
`src/engines/physics_engines/drake/python/motion_matching/humanoid_urdf.py`
(owned by issue #4108).

## Do not hand-edit

Hand-edits to files in this directory are **forbidden** by
`CROSS_ENGINE_PARITY_SPEC.md` §6. The CI gate
[`ci-engine-models.yml`](../../../../../../.github/workflows/ci-engine-models.yml)
(issue #4129) regenerates the URDF on every PR and fails if the on-disk
copy differs by even one byte. The drift message points back to this
README.

## Regenerate locally

```bash
# regenerate the URDF in place
python3 scripts/build_humanoid_models.py --engine drake

# CI-equivalent drift check (exits non-zero on mismatch)
python3 scripts/build_humanoid_models.py --engine drake --check

# pytest mirror of the gate
python3 -m pytest tests/test_drake_urdf_drift.py -v
```

## How to fix a failing CI gate

1. Make your change in the _source_ — usually
   `shared/models/golf_humanoid_dimensions.yaml` or
   `src/engines/physics_engines/drake/python/motion_matching/humanoid_urdf.py`.
2. Run `python3 scripts/build_humanoid_models.py --engine drake`.
3. Commit the regenerated URDF together with your source change.
