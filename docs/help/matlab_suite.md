---
title: Simscape
tile_id: matlab_suite
status: complete
---

# Simscape

## Purpose

The Simscape tile is a chooser. It presents the four MATLAB / Simscape
Multibody artefacts that ship with this repository and hands the one you pick to
your local MATLAB installation. It opens MATLAB; it does not simulate anything
inside UpstreamDrift.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Model choice | one of four buttons | See the table below. |
| Local MATLAB installation | external program | Must be on `PATH` as `matlab`. R2022b or later is the recommendation in [MATLAB engine reference](../engines/matlab.md). |

The four entries are hard-coded in `MATLAB_MODELS` in
`src/launchers/matlab_suite_dialog.py`:

| Button | Artefact | Path (repo-relative) |
| --- | --- | --- |
| Simscape 2D | 2D Simscape Multibody golf-swing model | `src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/GolfSwingZVCF.slx` |
| Simscape 3D | 3D Simscape Multibody golf-swing model | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx` |
| Dataset Generator | Forward-dynamics dataset generator GUI | `.../3D_Golf_Model/matlab/src/scripts/dataset_generator/Dataset_GUI.m` |
| Analysis GUI | Golf-swing analysis and plotting suite | `.../3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/main_scripts/golf_swing_analysis_gui.m` |

There are no numeric inputs on this tile, and therefore no units. All model
parameters - segment masses, torque profiles, time steps - are set inside
MATLAB after the model opens.

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| A running MATLAB session | external process | Started as `matlab -nosplash -r <command>`. |
| Toast notification | text | "Launching MATLAB: <name>..." on success; an error toast if the path is invalid. |
| Launch success flag | boolean | The chooser stays open when the launch fails, and closes when it succeeds. |

Any physical result - joint torques, club-head speed, generated datasets - is
produced by MATLAB and written wherever the MATLAB code writes it. This tile
does not read those results back.

## Method

No physics is computed here. `MatlabSuiteWidget`
(`src/launchers/matlab_suite_dialog.py`) builds one button per `MATLAB_MODELS`
entry and calls the launcher's `_launch_matlab_app`
(`src/launchers/launcher_simulation.py`), which resolves the artefact path and
shells out to MATLAB:

- `.slx` files become `matlab -nosplash -r "open_system('<path>')"`.
- `.m` files become `matlab -nosplash -r "cd('<dir>'); run('<file>')"`.
- anything else becomes `matlab -nosplash -r "open('<path>')"`.
- `.bat` / `.sh` wrappers are executed directly instead.

The tile has no Python entry point. Its registry `path` is the sentinel
`virtual/matlab_suite`, which `src/shared/python/config/tile_target_resolution.py`
maps to `src/launchers/matlab_suite_dialog.py`.

The models themselves are described in
[Simscape Multibody models](../engines/simscape.md), and the wider MATLAB
integration - including the `matlab.engine` bridge and its 30-60 s start-up
cost - in [MATLAB engine reference](../engines/matlab.md).

## Limitations

- **Nothing runs without MATLAB.** There is no Python re-implementation and no
  fallback; if `matlab` is not on `PATH` the launch fails.
- Simscape Multibody (and, for some scripts, additional toolboxes) must be
  licensed on the machine. The tile does not check licences before launching.
- The four artefacts are a fixed list baked into the dialog. It is not a
  directory scan, so a new `.slx` added to the repository will not appear here
  until the list is edited.
- No results come back. There is no data exchange, no plot, and no export on
  the UpstreamDrift side of this tile.
- The MATLAB engine path is best suited to single high-fidelity runs, not
  tight-loop optimisation, because of IPC overhead - see the performance note
  in [MATLAB engine reference](../engines/matlab.md).
- Simscape models require `setup_golf_suite()` to be run in MATLAB to
  initialise paths, per [Simscape Multibody models](../engines/simscape.md).
  The tile does not do this for you.

## See Also
- [MATLAB engine reference](../engines/matlab.md)
- [Simscape Multibody models](../engines/simscape.md)
- [Engine capabilities matrix](../engines/engine_capabilities.md)
- [Engine support tiers](../engines/support_tiers.md)
- [Engine selection guide](engine_selection.md)
