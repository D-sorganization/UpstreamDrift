# refactor(motion-matching): rename source-revealing identifiers and directories to generic names

## Why

The motion-matching code base currently leaks the **specific names of motion-capture data sources** through module names, class names, function names, directory names, test names, and docstrings. These are visible in CI logs, public PRs, and any artefact published from the repo. The user has explicitly asked us to make the **code source-agnostic**: data files on disk can keep their original names (those are user data), but **everything in code must be reusable for any C3D / club-data source**.

This issue is the rename pass. It is mechanical and large-surface but each rename is straightforward.

## Scope

### Modules / functions / classes (Python)

| Old | New | Reason |
|---|---|---|
| `src/shared/python/motion_matching/loaders/_optical.py` | `src/shared/python/motion_matching/loaders/_marker_clusters.py` | Source name in module file |
| `is_optical_schema(...)` | `has_marker_clusters(...)` | Source name in function name |
| `extract_optical_pose(...)` | `extract_cluster_club_pose(...)` | Source name in function name |
| `OpticalClubPose` | `ClusterClubPose` | Source name in class |
| `optical_marker_map.m` (MATLAB) | `cluster_marker_map.m` | Source name |
| `tests/.../test_load_club_target_c3d_optical.m` | `tests/.../test_load_club_target_c3d_clusters.m` | Source name in test |
| `src/engines/physics_engines/pinocchio/python/dtack/utils/optical_parser.py` | `mat_dataset_parser.py` | Source name |
| `src/engines/physics_engines/pinocchio/python/dtack/viz/rob_neal_viewer.py` | `swing_dataset_viewer.py` | Person name |
| `tests/unit/engines/pinocchio/dtack/viz/test_rob_neal_viewer.py` | `test_swing_dataset_viewer.py` | Person name |

### Directories (data trees)

| Old | New |
|---|---|
| `src/engines/physics_engines/pinocchio/data/rob_neal/` | `src/engines/physics_engines/pinocchio/data/club_swing_dataset/` |
| `src/engines/physics_engines/pinocchio/data/optical_tour_average/` | `src/engines/physics_engines/pinocchio/data/tour_average_mocap/` |
| `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Data/Optical C3D Files/` | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Data/Mocap C3D Files/` |

### Inside files

- Any `# noqa: <source>` style comment, log message, error message, or docstring that names a specific source/lab/person becomes a generic phrase: "the upstream mocap dataset", "the cluster-marker schema", "the supplier-provided rotation conventions", etc.
- Compatibility shims: each renamed module re-exports from the new module for ONE release with a `DeprecationWarning`. Example:

```python
# src/shared/python/motion_matching/loaders/_optical.py  (after rename)
"""Backwards-compatible shim. Use _marker_clusters instead."""
from ._marker_clusters import *  # noqa: F401,F403
import warnings
warnings.warn(
    "_optical module is deprecated; import from _marker_clusters",
    DeprecationWarning, stacklevel=2,
)
```

### Filename data preservation

C3D files (`CaptureExport Tour average.c3d`, etc.), `.mat` files (`TW_ProV1.mat`, etc.), and xlsx (`Wiffle_ProV1_club_3D_data.xlsx`) **keep their existing filenames** — those are user-supplied data. The rename only affects code, directories, module identifiers, and docstrings.

## Acceptance criteria

- [ ] All `git grep -i 'optical\|rob_neal'` hits in `src/`, `tests/`, `docs/`, `vendor/ud-tools/src/` are reviewed; only data-file-name leaves remain.
- [ ] Every renamed module ships a one-line backwards-compat shim that emits `DeprecationWarning`.
- [ ] All consumers (production code, tests, examples) are updated to the new names — shim presence is for downstream users, not internal code.
- [ ] Directory renames carried out with `git mv`; CI green.
- [ ] No print / no TODO without an issue.
- [ ] Mypy + ruff + file-size budget all green.
- [ ] CHANGELOG.md entry under "Refactor".

## Risk + mitigation

- Risk: a missed import path silently broken — caught by CI (mypy strict on these modules + integration tests in the body-loader and matcher GUI tests).
- Risk: external-tooling references (e.g. matlab unit tests) — those are inside this repo and updated in the same PR.

## Files touched

Many. Use `git mv` for the renames; treat each Python rename as a single commit (mv + import-fix). MATLAB renames need careful handling because MATLAB tests reference helpers by exact filename — `git grep -E 'optical_marker_map|optical_parser|rob_neal_viewer|test_load_club_target_c3d_optical'` to find them all.

## Sequencing

This issue should land **after** the `BodyTarget`, body C3D loader, and `.mat` loader issues use the renamed identifiers from day one. Rename can land in parallel as long as each PR is consistent within itself; the deprecation shim absorbs the transition.
