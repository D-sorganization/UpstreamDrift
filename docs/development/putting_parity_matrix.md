# Putting Feature Parity Matrix

Generated from `src/config/putting_parity_matrix.json` by
`python -m src.config.putting_parity_loader --write`. Do not edit by hand.
Epic #14452. Tauri wraps the React build, so it inherits React's status.

| Feature | Cluster | Core/API | PyQt6 | React | Tauri | Studio |
| --- | --- | --- | --- | --- | --- | --- |
| Selectable Units | (a) | present | present | missing (#14458) | missing (#14458) | present |
| Variance and Noise Channels | (b) | present | present | present | present | present |
| Dispersion Batches and 0-45 ft Make-% Spread | (b) | present | present | present | present | present |
| Impact and Strike Model (Skid to Roll) | (c) | present | present | present | present | present |
| Toe/Heel Strike Location | (c) | present | present | present | present | present |
| Counter-Torque Stabilization (Hand Couples, Gust) | (g) | present | present | present | present | present |
| Putter Balance Type (CG Offset, MOI) | (c) | present | present | present | present | present |
| Green Types (Flat, Slope, Stimp Ramp, Undulating, Presets) | (d) | present | present | present | present | present |
| Optimum Read / Auto-Solve Aim | (e) | present | present | present | present | present |
| Player-Specific Bellman Strokes Gained | (f) | present | present | present | present | present |
| Saved Scenarios and Comparison | (h) | present | present | present | present | present |
| JSON Import/Export of a Setup | (h) | present | present | present | present | present |
| Camera Views and Playback Phases | (i) | present | present | present | present | present |

## Evidence

Tauri, every feature: `ui/src-tauri/tauri.conf.json`. Wraps the React build (frontendDist ../dist); no putting code in the shell.

**Selectable Units**
- Core/API: present. `src/shared/python/ui/units.py`, `src/api/routes/putting_green.py`. SI internally; GET /units serves display_unit_table (distance, speed and strike factors and suffixes for both systems), the table the studio generator and the desktop strike spin use.
- PyQt6: present. `src/tools/putting_green_gui/gui.py`. Units combo (Metric or Imperial); every panel redraws its inputs and charts through ui/units.py.
- React: missing (#14458). `ui/src/pages/PuttingGreen.tsx`. Speed always m/s, distance yards, toe/hosel mm; no units store or selector.
- Studio: present. `src/apps/putting_studio/index.html`, `src/apps/putting_studio/js/core_units.js`. Units selector; distance, speed and strike labels use the ui/units.py table generated into core_units.js.

**Variance and Noise Channels**
- Core/API: present. `src/shared/python/putting_analytics/skill.py`, `src/shared/python/putting_analytics/service_variance.py`, `src/shared/python/putting_analytics/service_dispersion.py`. All 7 channels (face, path, speed, strike, read grade/aspect/stimp) in /variance-study, /dispersion, /make-spread and /simulated-curve; strike via strike_toe_sd_mm (#14461). /optimize keeps its paired line-and-speed batch for the aim search.
- PyQt6: present. `src/tools/putting_green_gui/_batch_panel.py`. Batch Dispersion inputs for all 7 core channels, defaulting to the core profile; strike spread in mm or in.
- React: present. `ui/src/components/putting/DispersionPanel.tsx`, `ui/src/utils/dispersion.ts`. Batch Dispersion inputs for all 7 core channels on the simulator page, defaulting to the core profile (golden-pinned, #14460); strike spread in mm or in.
- Studio: present. `src/apps/putting_studio/js/core_attribution.js`, `scripts/generate_studio_attribution.py`. 7-channel SG attribution from core-generated values (#14504, strike #14461); the Monte Carlo view keeps its illustrative draws.

**Dispersion Batches and 0-45 ft Make-% Spread**
- Core/API: present. `src/shared/python/putting_analytics/service_dispersion.py`, `src/api/routes/putting_analytics.py`. /dispersion batches up to 2000 putts on a sloped green with any seed; /make-spread gives the 3-45 ft make-% curve.
- PyQt6: present. `src/tools/putting_green_gui/_batch_panel.py`. 40/100/1000-putt batches with rest scatter and the 0-45 ft make-% spread, in the user's units.
- React: present. `ui/src/components/putting/DispersionPanel.tsx`, `ui/src/api/puttingClient.ts`. 40/100/1000-putt batches on the simulator's green via /dispersion with a rest scatter, and the 0-45 ft make-% spread via /make-spread, in the user's units (#14460).
- Studio: present. `src/apps/putting_studio/js/io.js`, `src/apps/putting_studio/js/controls_c.js`. 40/100/1000 batches and a 0-45 ft spread (that button runs 2000 attempts).

**Impact and Strike Model (Skid to Roll)**
- Core/API: present. `src/shared/python/putting_dynamics/collision.py`, `src/shared/python/putting_dynamics/solver.py`, `src/api/routes/putting_green.py`. strike() plus skid-to-roll; /simulate-3d returns the collision block and skid distance.
- PyQt6: present. `src/tools/putting_green_gui/gui.py`, `src/tools/putting_green_gui/_scene_builder.py`. Impact block: launch angle, spin, skid distance, face twist, energy loss and putter slowdown from the strike() collision report.
- React: present. `ui/src/pages/PuttingGreen.tsx`. Impact panel: launch speed/angle, putter speed before/after, contact proxy, face twist, energy loss, skid distance.
- Studio: present. `src/apps/putting_studio/js/putting_physics.js`, `src/apps/putting_studio/js/core_impact.js`. Skid ends where the core simulate_strike skid ends for the ball speed; launch angle and loft from the core table shown in the telemetry.

**Toe/Heel Strike Location**
- Core/API: present. `src/shared/python/putting_dynamics/collision.py`, `src/shared/python/putting_analytics/skill_roll_model.py`, `src/api/routes/putting_green.py`. strike(impact_toe_m) and /simulate-3d accept it; strike_toe_sd_mm scatter costs pace via off_centre_speed_retention in /variance-study (strike factor) and /simulated-curve.
- PyQt6: present. `src/tools/putting_green_gui/gui.py`, `src/tools/putting_green_gui/_scene_builder.py`. Heel/toe Strike spin box (mm or in, -40 to 40 mm) passed to simulate_strike.
- React: present. `ui/src/pages/PuttingGreen.tsx`. Impact Toe Offset slider.
- Studio: present. `src/apps/putting_studio/index.html`, `src/apps/putting_studio/js/core_impact.js`. Strike slider from heel to toe over the desktop range (-40 to 40 mm); it offsets every batch strike, and a toe strike opens the face as in the core strike().

**Counter-Torque Stabilization (Hand Couples, Gust)**
- Core/API: present. `src/shared/python/putting_dynamics/counter_torque.py`, `src/api/routes/putting_green.py`. Closed-form two-hand couple model (preload, off-plane preload, noise, gust, lie, shared putter presets) in the core; hand couples sum to the required moment without noise. Served on POST /counter-torque as one list per sample field.
- PyQt6: present. `src/tools/putting_green_gui/_torque_panel.py`. Counter-Torque Stabilization panel: grip and off-plane preload, gust, hand noise, lie and the shared balance heads; charts the core's hand couples and the gust twist with and without the preload, couples and forces in the chosen units.
- React: present. `ui/src/components/putting/CounterTorquePanel.tsx`, `ui/src/utils/counterTorque.ts`. Counter-torque explorer on the simulator page: grip preload, off-plane preload, hand noise, gust and lie sliders bounded by the core ranges, putter head preset, the hand-couple strip chart, impact face/path, peak hand forces and gust wobble via /counter-torque (golden-pinned, #14470).
- Studio: present. `src/apps/putting_studio/js/state.js`, `src/apps/putting_studio/js/charts2d.js`, `src/apps/putting_studio/js/controls_c.js`. Closed-form couple model, gust pulse and torque charts; drift-gated sample for sample against the core (tests/unit/putting_dynamics/test_counter_torque.py).

**Putter Balance Type (CG Offset, MOI)**
- Core/API: present. `src/shared/python/putting_dynamics/putter_presets.py`, `src/api/routes/putting_green.py`. Named balance presets (toe-hang, face-balanced, zero-torque, counterbalanced) over PutterState; GET /putter-presets and /simulate-3d balance_type.
- PyQt6: present. `src/tools/putting_green_gui/gui.py`, `src/tools/putting_green_gui/_scene_builder.py`. Balance selector: Typical or a shared BALANCE_PRESETS preset.
- React: present. `ui/src/pages/PuttingGreen.tsx`, `ui/src/components/putting/PutterHeadControl.tsx`. Balance Type selector (Typical or a core /putter-presets head) with head mass and MOI inputs bounded by /simulate-3d; a preset locks mass, MOI and hosel to the core values.
- Studio: present. `src/apps/putting_studio/js/state.js`, `src/apps/putting_studio/js/controls_c.js`. Four putter types (PUTTER_SPECS); per-type face SD is an assumed input.

**Green Types (Flat, Slope, Stimp Ramp, Undulating, Presets)**
- Core/API: present. `src/shared/python/putting_dynamics/green_spec.py`, `src/api/routes/putting_green.py`. GreenSpec (slope, two-tier step, seeded undulation, named presets) on /simulate, /scatter, /read-green and /contours; GET /green-presets.
- PyQt6: present. `src/tools/putting_green_gui/_scene_builder.py`, `src/tools/putting_green_gui/gui.py`. Green Type selector (cross slope or any core preset), seeded undulation in mm or in, green seed; all on the shared GreenSpec.
- React: present. `ui/src/components/putting/GreenTypeControl.tsx`, `ui/src/utils/greenTypes.ts`. Green Type selector (cross slope or any core preset from GET /green-presets) sent as the core green spec on /simulate-3d; slope and bump fields lock under a preset (golden-pinned, #14464).
- Studio: present. `src/apps/putting_studio/js/core_greens.js`, `scripts/generate_studio_greens.py`, `src/apps/putting_studio/js/putting_physics.js`. Green Type selector: the slider green or any core preset (undulating included), its height grid generated from GreenSpec and read with the core bilinear lookup, centred on the putt midpoint (golden-pinned, #14648).

**Optimum Read / Auto-Solve Aim**
- Core/API: present. `src/shared/python/putting_dynamics/green_reading.py`, `src/shared/python/putting_analytics/optimize.py`, `src/api/routes/putting_green.py`. solve_aim and best_action present; /read-green returns pace, heading, holed and converged in one call on any GreenSpec green.
- PyQt6: present. `src/tools/putting_green_gui/_scene_builder.py`, `src/tools/putting_green_gui/gui.py`. Auto-Solve Read runs solve_aim on the tile's surface off the GUI thread and writes the pace and aim into the controls.
- React: present. `ui/src/components/putting/AutoSolveButton.tsx`, `ui/src/utils/autoSolve.ts`, `ui/src/pages/PuttingGreen.tsx`. Auto-Solve Read calls /solve-aim-3d on the page's green, writes the pace and aim into the controls, simulates the solved putt and shows the desktop read text (golden-pinned).
- Studio: present. `src/apps/putting_studio/js/putting_physics.js`, `src/apps/putting_studio/js/controls_c.js`. Auto-Solve runs solveAimCore, a port of solve_aim on the studio putt (golden-pinned to the core read), and the batch strikes at the solved head speed.

**Player-Specific Bellman Strokes Gained**
- Core/API: present. `src/shared/python/putting_analytics/service_sg.py`, `src/shared/python/putting_analytics/optimize.py`, `src/api/routes/putting_analytics.py`. Player Bellman E(d) on the dialled green and one putt's SG on /strokes-gained; also /optimize (sg), /library-study, /variance-study, /simulated-curve.
- PyQt6: present. `src/tools/putting_green_gui/_sg_panel.py`, `src/tools/putting_green_gui/gui.py`. Strokes Gained panel: the player's E(d) solved on the dialled green off the GUI thread, each putt's result entered and scored on the cached curve, in the chosen units.
- React: present. `ui/src/components/putting/StrokesGainedPanel.tsx`, `ui/src/utils/strokesGained.ts`. Strokes Gained on the simulator page: the putt's result entered from each 3D putt, scored via /strokes-gained on the player's own E(d) with the curve plotted, in the user's units (golden-pinned, #14468).
- Studio: present. `src/apps/putting_studio/js/core_sg.js`, `scripts/generate_studio_sg.py`, `src/apps/putting_studio/js/green3d.js`, `src/apps/putting_studio/js/charts2d.js`. Batch strokes gained, the E(d) chart, the optimal pace and the E vs capture-speed panel read the core's solved E(d) and per-pace expected strokes (service_sg and optimize.evaluate_actions; tour-average player, nearest side slope).

**Saved Scenarios and Comparison**
- Core/API: present. `src/shared/python/putting_analytics/service_setup.py`, `src/api/routes/putting_analytics.py`. Named setups with their dispersion result; /scenario and /scenario-compare (make rate and leave against the first).
- PyQt6: present. `src/tools/putting_green_gui/_scenario_panel.py`. Scenarios panel: Save Scenario rolls the dialled green's batch on a worker; a unit-aware table compares make rate and leave with the first (service_setup).
- React: present. `ui/src/components/putting/ScenariosPanel.tsx`, `ui/src/utils/puttingSetup.ts`. Saves the page's putt as a core scenario (/scenario) and compares saved ones against the first (/scenario-compare); setup mapping golden-pinned to the core (#14472).
- Studio: present. `src/apps/putting_studio/js/controls_a.js`. Saved scenarios with a comparison table; no distribution chart.

**JSON Import/Export of a Setup**
- Core/API: present. `src/shared/python/putting_analytics/service_setup.py`, `src/api/routes/putting_analytics.py`. Versioned setup document (schema_version 4.0.0, the studio's); /setup-export and /setup-import, which also loads a studio export's green and distance.
- PyQt6: present. `src/tools/putting_green_gui/_scenario_panel.py`. Export JSON... and Import JSON... write and read the shared setup document with its scenarios; a studio export loads its distance, Stimp and grade (the desktop slope always falls right).
- React: present. `ui/src/components/putting/ScenariosPanel.tsx`, `ui/src/utils/puttingSetup.ts`. Exports the core's setup document (/setup-export) and imports a setup or studio file onto the page (/setup-import), keeping the page's line direction (#14472).
- Studio: present. `src/apps/putting_studio/js/io.js`. Export/load with schema_version 4.0.0, including scenarios.

**Camera Views and Playback Phases**
- Core/API: present. `src/shared/python/putting_dynamics/phases.py`, `src/api/routes/putting_green.py`. Per-sample ball mode on /simulate and /simulate-3d, and phase marks (where the hop, skid, roll and rest start) from the core's phase_marks for playback jumps.
- PyQt6: present. `src/tools/putting_green_gui/gui.py`, `src/tools/putting_green_gui/_camera.py`. Orbit, down-the-line, face-on and overhead cameras; play/pause, a scrubber and jumps to the core phase marks.
- React: present. `ui/src/pages/PuttingGreen.tsx`, `ui/src/components/visualization/PuttingScene3D.tsx`, `ui/src/utils/cameraPresets.ts`. Orbit, down-the-line, face-on and overhead cameras framed on the desktop tile, play/pause/scrub/rate, and jumps to the core phase marks (golden-pinned, #14474).
- Studio: present. `src/apps/putting_studio/js/controls_b.js`, `src/apps/putting_studio/js/controls_a.js`. Grip, Full Plane, Down-the-Line and Face-On cameras; phase badge; scrubber and keyframe jumps.
