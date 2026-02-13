# Large File Decomposition Phase 1 (Issue #1393)

Prioritization method: LOC (current) plus churn over last 180 days (`git log --since=180.days`).

| Rank | File                                                                                                                                                                          |  LOC | 180d Churn | Owner              | Phase-1 Target                          |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---: | ---------: | ------------------ | --------------------------------------- |
| 1    | `src/engines/physics_engines/drake/python/src/drake_gui_app.py`                                                                                                               | 2171 |         20 | `@engine-platform` | Split view state, commands, IO adapters |
| 2    | `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py`                                                                                                          | 2007 |         21 | `@engine-platform` | Split controller/service/view-model     |
| 3    | `src/engines/physics_engines/mujoco/head_models.py`                                                                                                                           | 1666 |          5 | `@mujoco-core`     | Extract model math/service layer        |
| 4    | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_gui_application.py` | 1662 |         35 | `@simscape-gui`    | Coordinator + render adapter modules    |
| 5    | `src/engines/physics_engines/mujoco/python/humanoid_launcher.py`                                                                                                              | 1583 |         41 | `@mujoco-core`     | Launch orchestration + runtime adapters |
| 6    | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/Motion_Capture_Plotter.py`                                               | 1468 |          9 | `@simscape-gui`    | Data pipeline extraction                |
| 7    | `src/tools/model_generation/editor/frankenstein_editor.py`                                                                                                                    | 1449 |          8 | `@model-gen`       | Command handlers + parser services      |
| 8    | `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/linkage_mechanisms/__init__.py`                                                                               | 1359 |          6 | `@mujoco-core`     | Domain primitives split by mechanism    |
| 9    | `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/golf_gui_r0/golf_visualizer_implementation.py`  | 1308 |         10 | `@simscape-gui`    | Visualizer strategy modules             |
| 10   | `src/engines/physics_engines/mujoco/docker/gui/deepmind_control_suite_MuJoCo_GUI.py`                                                                                          | 1300 |          9 | `@mujoco-core`     | Docker setup/service extraction         |

## Phase-1 execution rule

- File budget for changed Python files is set to 1200 LOC.
- Exceptions require owner, reason, and expiry in `scripts/config/file_size_budget.json`.
- CI blocks changed oversized files unless exception is active.

## Initial decomposition completed in this phase

- Introduced `src/shared/python/engine_core/workflow_adapter.py`.
- Moved repeated API probe/load/unload orchestration out of `src/api/routes/engines.py`.
