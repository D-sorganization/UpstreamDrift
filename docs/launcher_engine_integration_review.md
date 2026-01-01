# Launcher and Engine Integration Review

## Scope and approach
- Reviewed the launcher entry points (`launch_golf_suite.py`, `launchers/golf_launcher.py`, and the PyQt local launcher) and the first-hop experience into MuJoCo, Drake, Pinocchio, OpenSim, MyoSim, OpenPose, MATLAB/Simscape, URDF Generator, and the C3D Viewer.
- Evaluated whether each engine presents a modern GUI with documented controls for methods, force/torque configuration, graphics manipulation, and joint load visualization.

## Overall launcher experience
- The local PyQt launcher provides only three start buttons and a simple log/status panel with no contextual help, setup guidance, or description of what each engine exposes when first opened. There is no pathway from the launcher to change simulation methods or force/torque inputs before launching, nor guidance on how to do so once inside the engine interfaces. 【F:launchers/golf_suite_launcher.py†L28-L205】
- Entry points for Drake are inconsistent: the local launcher points at `golf_gui.py` (an empty stub), while the CLI launcher invokes `drake_gui_app` instead, creating ambiguity about which UI users should expect and how to reach force/torque or visualization controls. 【F:launchers/golf_suite_launcher.py†L44-L204】【F:launch_golf_suite.py†L114-L139】

## Engine-by-engine findings

### MuJoCo
- Strengths: A full-featured, modern GUI with tabbed controls, visualization, analysis, plotting, and grip modeling is clearly implemented, including force/torque vector visualization and camera controls. 【F:engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui.py†L1-L147】
- Documentation: The MuJoCo README explicitly advertises force/torque visualization, advanced plotting, adjustable joint-force vectors, and comprehensive analysis workflows, giving users clear expectations for manipulating graphics and loads across the model. 【F:engines/physics_engines/mujoco/README.md†L1-L119】
- Gaps: The launcher does not surface which MuJoCo configuration is used (e.g., which model, controller, or input source) or how to switch control/force methods before entering the GUI.

### Drake
- UI/content gaps: No documented GUI features or first-run instructions exist; the Drake README is still a generic project template with no guidance on graphics controls, force/torque visualization, or method selection. 【F:engines/physics_engines/drake/README.md†L1-L80】
- Integration mismatch: Users launched through the local GUI are directed to an empty `golf_gui.py`, while the CLI launches `drake_gui_app`; this split entry point makes the expected interface unclear and hides any available force/torque displays or graphics controls behind guesswork. 【F:launchers/golf_suite_launcher.py†L44-L204】【F:launch_golf_suite.py†L114-L139】

### Pinocchio
- Documentation void: The README stops at “Usage (Coming soon),” providing no instructions on operating the GUI, selecting physics methods, or viewing forces/torques. 【F:engines/physics_engines/pinocchio/README.md†L1-L37】
- The launcher exposes a Pinocchio button but offers no description of what the GUI contains, how to adjust torque/force inputs, or how to visualize joint loads once the app opens. 【F:launchers/golf_suite_launcher.py†L44-L205】

### OpenSim
- Implementation placeholder: The OpenSim Golf engine falls back to an internal demo with hard-coded torques and throws `NotImplementedError` for the real OpenSim path, so no GUI affordances or documentation exist for switching methods or viewing joint loads. 【F:engines/physics_engines/opensim/python/opensim_golf/core.py†L33-L159】
- Launcher visibility: The Docker launcher surfaces an “OpenSim Golf” tile but provides no help text or pathway to configure force/torque inputs or select OpenSim models before launch. 【F:launchers/golf_launcher.py†L88-L99】【F:launchers/golf_launcher.py†L919-L963】

### MyoSim Suite (Golf Suite)
- Headless integration: The MyoSim engine is loaded like other physics engines but lacks a GUI module or documentation on changing control strategies, visualizing torques, or interacting with graphics. 【F:shared/python/engine_manager.py†L332-L355】【F:shared/python/engine_probes.py†L546-L600】
- Launcher gap: The tile exists in the Docker launcher without any accompanying description or link to help, leaving users unaware of available controls or force/torque visualization options. 【F:launchers/golf_launcher.py†L88-L99】

### OpenPose
- Analysis-only wrapper: The OpenPose estimator is a headless wrapper with no GUI; there are no documented controls for adjusting methods, viewing forces/torques, or connecting results to simulation graphics from the launcher. 【F:shared/python/pose_estimation/openpose_estimator.py†L1-L157】
- Launcher tile without guidance: An “OpenPose Analysis” entry is listed in the launcher assets, but the UI provides no explanation of its capabilities, required models, or how to route outputs into downstream force/torque displays. 【F:launchers/golf_launcher.py†L88-L99】

### MATLAB / Simscape models
- Model availability without GUI entry: Documentation exists for MATLAB engine usage, but the launcher does not expose buttons or help to start Simscape GUIs, pass forces/torques, or toggle visualization of joint loads. 【F:docs/engines/matlab.md†L1-L36】【F:launchers/golf_launcher.py†L88-L99】
- Help disconnect: The documented startup flow (manual MATLAB engine start) is not reflected in the launcher UI, leaving users without integrated guidance for configuring methods or graphics. 【F:docs/engines/matlab.md†L12-L36】【F:launchers/golf_launcher.py†L919-L963】

### URDF Generator
- Strengths: Dedicated PyQt GUI with documented workflow for building and exporting URDFs, including segment-level configuration and planned 3D visualization; launcher integration can start the tool directly. 【F:tools/urdf_generator/README.md†L1-L156】【F:launchers/golf_launcher.py†L919-L994】【F:launchers/golf_launcher.py†L1158-L1192】
- Gaps: No launcher-side description of how URDF outputs map into engine control/visualization defaults or how to pipe force/torque presets into exported models before opening other engines.

### C3D Viewer
- Strengths: Full-featured PyQt GUI for loading C3D files with metadata, plots, and 3D marker visualization, launched directly from the Docker launcher. 【F:engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py†L1-L200】【F:launchers/golf_launcher.py†L919-L994】【F:launchers/golf_launcher.py†L1199-L1257】
- Gaps: The launcher provides no contextual help about file requirements, force/torque visualization capabilities, or how to stream outputs into engine visualizations.

## Recommendations
- Unify Drake entry points by pointing both the local launcher and CLI to the same, documented GUI module, and add in-app help explaining available control modes, force/torque overlays, and graphics options.
- Add per-engine launcher tooltips or a "Learn more" panel that links to concise guides describing the default models, control methods, and how to adjust force/torque or visualization settings before launch.
- Expand Drake and Pinocchio documentation with step-by-step first-run guides, including screenshots of their GUIs, instructions for switching control schemes, and directions for enabling joint force/torque displays.
- Surface configuration selectors in the launcher (e.g., model, controller, input source) so users can pick force/torque control modes or visualization presets before opening each engine.

## Upgrade plan and capability expansion
- **Launcher-level UX upgrades**
  - Add a context-aware help drawer that loads engine-specific quickstarts (model defaults, available methods, and force/torque visualization toggles) and links to full docs before launch.
  - Provide pre-launch configuration modals per engine for: model selection, controller/method selection (e.g., impedance, PID, MPC), default force/torque sources, visual overlays (joint load vectors, contact forces), and graphics presets (camera path, lighting, frame rate caps).
  - Standardize entry points by mapping every launcher tile/button to the same GUI module the CLI uses, and assert the module’s existence at startup to prevent dead links.
  - Surface telemetry opt-in and logging level controls so users can capture force/torque traces and visualization states for reproducibility across engines.
  - Add status chips in the launcher indicating GUI availability (full GUI, headless, partial) and what force/torque visualizations are supported to set expectations.

- **Per-engine GUI and documentation upgrades**
  - **MuJoCo:** Add a launcher-linked profile selector (e.g., practice vs. analysis) that preconfigures camera layouts, joint torque overlays, and controller gains. Document how to toggle grip-force visualization and export torque traces from the GUI.
  - **Drake:** Implement the missing GUI module with panels for method selection (position/torque control), joint force/torque overlays, and camera presets; align local/CLI launch paths and publish a quickstart with screenshots and keyboard shortcuts.
  - **Pinocchio:** Deliver the promised GUI with widgets for solver choice, torque/force sliders, and per-joint visualization toggles; ship a tutorial describing how to swap integrators and enable torque vectors.
  - **OpenSim:** Replace the `NotImplementedError` path with a real launcher target that selects OpenSim models, exposes excitation/torque inputs, and provides a viewer overlay for joint loads; include guidance for switching between recorded torques and live inputs.
  - **MyoSim Suite:** Add a lightweight GUI to inspect muscle states, adjust control strategies, and visualize joint torques; document how to stream force/torque outputs into plotting panels or into other engines via shared topics/files.
  - **OpenPose:** Offer a results-review GUI that shows skeleton overlays and forwards estimated forces/torques into downstream engines; document calibration steps and how to re-run with different detection thresholds.
  - **MATLAB/Simscape:** Add launcher buttons that call MATLAB Engine entry points with arguments for controller selection, torque inputs, and visualization options; provide a concise guide mapping launcher options to Simscape parameters and how to view joint loads inside MATLAB.
  - **URDF Generator:** Extend the GUI to save presets for control/force metadata (e.g., actuator limits, default torques) and make the launcher pass those presets into downstream engines; document how to preview force/torque vectors in the planned 3D view.
  - **C3D Viewer:** Add inline tooltips describing required files and how to overlay force plates/torque vectors; expose export hooks to push processed forces/torques into other engines or the launcher for immediate simulation replay.

- **Cross-engine consistency and observability**
  - Define a shared schema for force/torque annotations, camera presets, and control methods so the launcher can hand off configurations uniformly to each engine.
  - Add a “quick diagnostic” mode that starts each engine with force/torque overlays enabled, logs joint loads, and returns a short report to the launcher summarizing visualization and control availability.
  - Ship a doc set (one-pagers) per engine that include: launch path, available GUIs, how to change methods/forces, how to enable graphics overlays, and how to export traces for comparison across engines.
