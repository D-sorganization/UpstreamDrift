---
title: OpenSim Models
tile_id: opensim_models_shared
status: complete
---

# OpenSim Models

## Purpose

This tile is a shortcut to the shared OpenSim model library. Clicking it asks
your operating system to open the `OpenSim_Models` folder that sits next to your
UpstreamDrift checkout, so you can browse, copy or edit the OpenSim model files
by hand. It is a folder shortcut, not a viewer and not a simulator.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Tile click | user action | The only input. |
| Sibling checkout location | filesystem path | Resolved as `<parent of your UpstreamDrift checkout>/OpenSim_Models`. |

No numeric or physical inputs, so there are no units to declare.

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| An OS file-manager window | external window | Opened on the `OpenSim_Models` directory. |
| Launch success flag | boolean | `False` when the directory does not exist, with a `directory not found` warning in the launcher log. |

The tile produces no simulation data and no physical quantities.

## Method

No calculation. Registry entry `opensim_models_shared` in `src/config/models.yaml` declares
`type: shared_repo` and `path: OpenSim_Models`. `SharedRepoHandler` in
`src/launchers/launcher_model_handlers.py` handles that type: it computes
`repo_path.parent / path` and hands the result to the shared
`_open_with_system_app` helper. `SharedRepoHandler.get_dockable_ui` returns
`None`, which is why this tile can never appear as an embedded panel.

The `OpenSim_Models` repository is one of the five biomechanics siblings covered by
[ADR-0014](../adr/0014-shared-biomech-models.md); its Python package is
`opensim_models` and its path override environment variable is `OPENSIM_MODELS_HOME`. Its models
are consumed elsewhere in the suite as ``osim``, per the
`_PREFERRED_ENGINE_BY_MODEL_TYPE` mapping in
`src/launchers/launcher_model_handlers.py`.

## Limitations

- **It does not simulate, render, validate or convert anything.** It opens a
  folder.
- It only ever looks at the editable sibling checkout. Unlike the discovery
  layer described in
  [Biomech workspace setup](../architecture/biomech_workspace.md), this handler
  does **not** walk the four-tier resolution order - the pip-installed package,
  the vendored snapshot under `vendor/biomech-models/OpenSim_Models/`, and the
  `OPENSIM_MODELS_HOME` override are all ignored here.
- If `OpenSim_Models` is not cloned next to UpstreamDrift, the click silently fails:
  a warning goes to the log and the tile reports failure. Nothing is offered to
  fetch or clone it.
- Where the folder opens depends on your OS file-manager association. Behaviour
  is not identical across Windows, macOS and Linux.
- Editing files here changes the sibling repository, not UpstreamDrift. Nothing
  in this tile commits, syncs or version-checks those edits.

## See Also
- [ADR-0014: shared biomech models](../adr/0014-shared-biomech-models.md)
- [Biomech workspace setup](../architecture/biomech_workspace.md)
- [External provider onboarding](../development/external_provider_onboarding.md)
- [OpenSim engine reference](../engines/opensim.md)
- [OpenSim tile](opensim_golf.md)
- [Simulation controls](simulation_controls.md)
