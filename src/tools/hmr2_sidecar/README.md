# 4D-Humans / HMR 2.0 Sidecar

First monocular 3D path for the motion pipeline: a single ordinary video
in, 3D SMPL body-joint trajectories plus subject shape coefficients out.

## Licensing and isolation

[4D-Humans](https://github.com/shubham-goel/4D-Humans) (HMR 2.0) is
released under **CC-BY-NC**, and the SMPL body model files it requires
are **research-restricted**. UpstreamDrift is MIT-licensed, so — exactly
like the AGPL-isolated FreeMoCap sidecar
(`src/tools/freemocap_sidecar/`) — this sidecar **only ever invokes a
user-installed external tool as a subprocess**. No CC-BY-NC code is
imported or vendored, and no SMPL model files ship with or load into
this repository. Only the plain CSV/JSON artifacts below cross the
process boundary.

## Usage

Install 4D-Humans in a separate environment, then point the sidecar at
a wrapper command that accepts `--video <path> --out_folder <dir>` and
writes the output contract below:

```bash
export HMR2_COMMAND="~/.venvs/hmr2/bin/python ~/4D-Humans/demo_video.py"
python -m src.tools.hmr2_sidecar.run_hmr2 --video swing.mp4 --output out/
```

Programmatic use:

```python
from src.tools.hmr2_sidecar import run_hmr2_sidecar

result = run_hmr2_sidecar("swing.mp4", "out/")  # stub mode if unconfigured
```

When no command is configured (or `--dry-run` is passed), the sidecar
writes **stub artifacts** with the exact same schema so downstream
contract tests stay stable without the external tool.

## Output contract

After a run, the output directory contains:

| Artifact        | Contents                                                                                                                                                       |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `joints3d.csv`  | Columns `frame,time` then `<joint>_x,<joint>_y,<joint>_z` for the 22 SMPL body joints (`SMPL_BODY_JOINTS` in `run_hmr2.py`), positions in **meters**, `time` in seconds |
| `betas.json`    | `{"betas": [<10 floats>], "gender": "male"\|"female"\|"neutral"}` — SMPL shape coefficients                                                                     |
| `metadata.json` | `source_video`, `fps`, `tool_version` (`"stub"` when no real tool ran), `joint_names`                                                                            |

## Downstream consumers

- **Motion pipeline**: `joints3d.csv` is recognized by `HMR2Adapter`
  (`src/shared/python/motion_pipeline/sources/hmr2_adapter.py`) and
  loads as a 3D `KeypointSequence` via `load_any(...)`.
- **Character builder**: `betas_bridge.body_parameters_from_betas(...)`
  turns `betas.json` into `BodyParameters(smplx_betas=...)`; the SMPL-X
  mesh generator uses measured betas verbatim instead of its heuristic
  anthropometric mapping.
