# Putting Feature Parity Matrix

Generated from `src/config/putting_parity_matrix.json` by
`python -m src.config.putting_parity_loader --write`. Do not edit by hand.
Epic #14452. Tauri wraps the React build, so it inherits React's status.

| Feature | Cluster | Core/API | PyQt6 | React | Tauri | Studio |
| --- | --- | --- | --- | --- | --- | --- |
| Selectable Units | (a) | missing (#14457) | partial (#14457) | missing (#14458) | missing (#14458) | missing (#14457) |
| Variance and Noise Channels | (b) | partial (#14459) | missing (#14459) | partial (#14460) | partial (#14460) | partial (#14459) |
| Dispersion Batches and 0-45 ft Make-% Spread | (b) | partial (#14459) | missing (#14459) | partial (#14460) | partial (#14460) | present |
| Impact and Strike Model (Skid to Roll) | (c) | present | partial (#14461) | present | present | partial (#14461) |
| Toe/Heel Strike Location | (c) | present | missing (#14461) | present | present | partial (#14461) |
| Counter-Torque Stabilization (Hand Couples, Gust) | (g) | missing (#14469) | missing (#14469) | missing (#14470) | missing (#14470) | present |
| Putter Balance Type (CG Offset, MOI) | (c) | partial (#14461) | missing (#14461) | partial (#14462) | partial (#14462) | present |
| Green Types (Flat, Slope, Stimp Ramp, Undulating, Presets) | (d) | partial (#14463) | partial (#14463) | partial (#14464) | partial (#14464) | partial (#14463) |
| Optimum Read / Auto-Solve Aim | (e) | partial (#14465) | missing (#14465) | partial (#14466) | partial (#14466) | partial (#14465) |
| Player-Specific Bellman Strokes Gained | (f) | present | missing (#14467) | missing (#14468) | missing (#14468) | partial (#14467) |
| Saved Scenarios and Comparison | (h) | missing (#14471) | missing (#14471) | missing (#14472) | missing (#14472) | present |
| JSON Import/Export of a Setup | (h) | partial (#14471) | partial (#14471) | missing (#14472) | missing (#14472) | present |
| Camera Views and Playback Phases | (i) | partial (#14473) | partial (#14473) | partial (#14474) | partial (#14474) | present |

## Evidence

Tauri, every feature: `ui/src-tauri/tauri.conf.json`. Wraps the React build (frontendDist ../dist); no putting code in the shell.

**Selectable Units**
- Core/API: missing (#14457). `src/shared/python/putting_dynamics/utils.py`, `src/shared/python/ui/units.py`. SI internally, feet in analytics; conversion helpers only, no units parameter on any route.
- PyQt6: partial (#14457). `src/tools/putting_green_gui/gui.py`. Reads get_unit_preference(); set_unit_system exists but the window has no selector.
- React: missing (#14458). `ui/src/pages/PuttingGreen.tsx`. Speed always m/s, distance yards, toe/hosel mm; no units store or selector.
- Studio: missing (#14457). `src/apps/putting_studio/index.html`. Labels hard-code ft, m/s and mm; no toggle.

**Variance and Noise Channels**
- Core/API: partial (#14459). `src/shared/python/putting_analytics/skill.py`, `src/shared/python/putting_analytics/variance.py`, `src/shared/python/putting_analytics/service_variance.py`. Core has 7 channels (face, path, speed, strike, read grade/aspect/stimp); /variance-study drops strike; /optimize uses 2 channels.
- PyQt6: missing (#14459). `src/tools/putting_analytics_gui/gui.py`. Simulator strike is deterministic; the analytics tool has start-line and speed SD only.
- React: partial (#14460). `ui/src/pages/PuttingGreen.tsx`, `ui/src/pages/PuttingAnalytics.tsx`. Simulator: friction_variation only (fixed seed); analytics page: line and speed SD.
- Studio: partial (#14459). `src/apps/putting_studio/js/green3d.js`. Monte Carlo draws 5 noise terms; the 6-channel SG attribution is closed-form; the 4-channel decomposition is guide text only.

**Dispersion Batches and 0-45 ft Make-% Spread**
- Core/API: partial (#14459). `src/shared/python/putting_analytics/batch_roll.py`, `src/shared/python/putting_analytics/sg_curve.py`, `src/api/routes/putting_green.py`. Core batches and make-% curve present (/simulated-curve); /scatter capped at 100 putts on a flat green with seed 0.
- PyQt6: missing (#14459). `src/tools/putting_green_gui/_scene_builder.py`. One putt per run.
- React: partial (#14460). `ui/src/pages/PuttingAnalytics.tsx`. Analytics rollouts at one distance as a table; no scatter, no make-% curve.
- Studio: present. `src/apps/putting_studio/js/io.js`, `src/apps/putting_studio/js/controls_c.js`. 40/100/1000 batches and a 0-45 ft spread (that button runs 2000 attempts).

**Impact and Strike Model (Skid to Roll)**
- Core/API: present. `src/shared/python/putting_dynamics/collision.py`, `src/shared/python/putting_dynamics/solver.py`, `src/api/routes/putting_green.py`. strike() plus skid-to-roll; /simulate-3d returns the collision block and skid distance.
- PyQt6: partial (#14461). `src/tools/putting_green_gui/gui.py`. Runs simulate_strike but shows only launch speed and track colours; no collision readout.
- React: present. `ui/src/pages/PuttingGreen.tsx`. Impact panel: launch speed/angle, putter speed before/after, contact proxy, face twist, energy loss, skid distance.
- Studio: partial (#14461). `src/apps/putting_studio/js/putting_physics.js`. Skid ends at a fixed distance; loft and launch not modelled.

**Toe/Heel Strike Location**
- Core/API: present. `src/shared/python/putting_dynamics/collision.py`, `src/shared/python/putting_analytics/skill_roll_model.py`, `src/api/routes/putting_green.py`. strike(impact_toe_m) and /simulate-3d accept it; strike_toe_sd_mm scatter costs pace via off_centre_speed_retention in /variance-study (strike factor) and /simulated-curve.
- PyQt6: missing (#14461). `src/tools/putting_green_gui/_scene_builder.py`. build_putt_scene never passes impact_toe_m.
- React: present. `ui/src/pages/PuttingGreen.tsx`. Impact Toe Offset slider.
- Studio: partial (#14461). `src/apps/putting_studio/index.html`, `src/apps/putting_studio/js/controls_c.js`. Slider is toe-only (0-15 mm); no heel strikes.

**Counter-Torque Stabilization (Hand Couples, Gust)**
- Core/API: missing (#14469). `src/shared/python/grip_contact/couple_check.py`. No putting counter-torque model in the core; a grip couple check exists outside the putting stack.
- PyQt6: missing (#14469). none. None.
- React: missing (#14470). none. None.
- Studio: present. `src/apps/putting_studio/js/state.js`, `src/apps/putting_studio/js/charts2d.js`, `src/apps/putting_studio/js/controls_c.js`. Closed-form couple model, gust pulse and torque charts (illustrative, not in the core: ADR-0058 gap).

**Putter Balance Type (CG Offset, MOI)**
- Core/API: partial (#14461). `src/shared/python/putting_dynamics/types.py`, `src/api/routes/putting_green.py`. PutterState hosel offsets, MOI and mass; no named balance presets.
- PyQt6: missing (#14461). `src/tools/putting_green_gui/_scene_builder.py`. Uses PutterState.typical only.
- React: partial (#14462). `ui/src/pages/PuttingGreen.tsx`. Hosel widget; mass and MOI sent without controls; no balance-type selector.
- Studio: present. `src/apps/putting_studio/js/state.js`, `src/apps/putting_studio/js/controls_c.js`. Four putter types (PUTTER_SPECS); per-type face SD is an assumed input.

**Green Types (Flat, Slope, Stimp Ramp, Undulating, Presets)**
- Core/API: partial (#14463). `src/shared/python/putting_dynamics/surfaces.py`, `src/engines/physics_engines/putting_green/python/_surface_presets.py`, `src/api/routes/putting_green.py`. Core has flat, planar, bumpy, presets and stimp ramp; only /simulate-3d builds slope and bumps; other routes are flat; no preset route.
- PyQt6: partial (#14463). `src/tools/putting_green_gui/_scene_builder.py`. Single cross-slope plane.
- React: partial (#14464). `ui/src/pages/PuttingGreen.tsx`. Grade, aspect, grain, bumps; no green-type selector or presets.
- Studio: partial (#14463). `src/apps/putting_studio/js/putting_physics.js`, `src/apps/putting_studio/js/io.js`. Flat and slope presets plus stimp ramp; undulation switches on with slope and is not selectable.

**Optimum Read / Auto-Solve Aim**
- Core/API: partial (#14465). `src/shared/python/putting_dynamics/green_reading.py`, `src/shared/python/putting_analytics/optimize.py`, `src/api/routes/putting_green.py`. solve_aim and best_action present; /read-green is flat-only.
- PyQt6: missing (#14465). `src/tools/putting_green_gui/gui.py`. Aim is manual; the analytics best-aim study is not linked to the simulator.
- React: partial (#14466). `ui/src/pages/PuttingAnalytics.tsx`. Analytics best aim/pace study; nothing on the simulator page.
- Studio: partial (#14465). `src/apps/putting_studio/js/controls_c.js`. Aim-only grid search from one noise-free putt.

**Player-Specific Bellman Strokes Gained**
- Core/API: present. `src/shared/python/putting_analytics/expected_putts.py`, `src/shared/python/putting_analytics/optimize.py`, `src/api/routes/putting_analytics.py`. Bellman solve; /optimize (sg), /library-study, /variance-study, /simulated-curve.
- PyQt6: missing (#14467). `src/tools/putting_analytics_gui/gui.py`. Nothing in the simulator; the analytics tool has an sg objective.
- React: missing (#14468). `ui/src/pages/PuttingAnalytics.tsx`. Nothing on the simulator page; the analytics page has an sg objective.
- Studio: partial (#14467). `src/apps/putting_studio/js/green3d.js`, `src/apps/putting_studio/js/charts2d.js`. Bellman formula with a hand-tuned comeback fit, not a solved recursion.

**Saved Scenarios and Comparison**
- Core/API: missing (#14471). none. None.
- PyQt6: missing (#14471). none. None.
- React: missing (#14472). `ui/src/stores/usePuttingStore.ts`. Store holds one parameter set and one result.
- Studio: present. `src/apps/putting_studio/js/controls_a.js`. Saved scenarios with a comparison table; no distribution chart.

**JSON Import/Export of a Setup**
- Core/API: partial (#14471). `src/shared/python/putting_dynamics/practice.py`. Saves putt results and reports, not a setup; no import/export route.
- PyQt6: partial (#14471). `src/tools/putting_green_gui/_scene_export.py`. export_putt_scene/read_putt_export exist but nothing in the UI calls them.
- React: missing (#14472). none. None.
- Studio: present. `src/apps/putting_studio/js/io.js`. Export/load with schema_version 4.0.0, including scenarios.

**Camera Views and Playback Phases**
- Core/API: partial (#14473). `src/api/routes/putting_green.py`, `src/shared/python/putting_dynamics/types.py`. Per-sample mode only on /simulate-3d; no stroke phases.
- PyQt6: partial (#14473). `src/tools/putting_green_gui/gui.py`. Auto-framed orbit and a one-shot animation; no presets, scrub or phase jumps.
- React: partial (#14474). `ui/src/pages/PuttingGreen.tsx`, `ui/src/components/visualization/PuttingScene3D.tsx`. Orbit camera, play/pause/scrub/rate; no view presets or phase jumps.
- Studio: present. `src/apps/putting_studio/js/controls_b.js`, `src/apps/putting_studio/js/controls_a.js`. Grip, Full Plane, Down-the-Line and Face-On cameras; phase badge; scrubber and keyframe jumps.
