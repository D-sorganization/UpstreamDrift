# Putting Feature Parity Matrix

Generated from `src/config/putting_parity_matrix.json` by
`python -m src.config.putting_parity_loader --write`. Do not edit by hand.
Epic #14452. Tauri wraps the React build, so it inherits React's status.

| Feature | Cluster | Core/API | PyQt6 | React | Tauri | Studio |
| --- | --- | --- | --- | --- | --- | --- |
| Selectable Units | (a) | missing (#14457) | partial (#14457) | missing (#14458) | missing (#14458) | missing (#14457) |
| Variance and Noise Channels | (b) | present | present | partial (#14460) | partial (#14460) | present |
| Dispersion Batches and 0-45 ft Make-% Spread | (b) | present | present | partial (#14460) | partial (#14460) | present |
| Impact and Strike Model (Skid to Roll) | (c) | present | present | present | present | partial (#14461) |
| Toe/Heel Strike Location | (c) | present | present | present | present | partial (#14461) |
| Counter-Torque Stabilization (Hand Couples, Gust) | (g) | missing (#14469) | missing (#14469) | missing (#14470) | missing (#14470) | present |
| Putter Balance Type (CG Offset, MOI) | (c) | present | present | partial (#14462) | partial (#14462) | present |
| Green Types (Flat, Slope, Stimp Ramp, Undulating, Presets) | (d) | present | present | partial (#14464) | partial (#14464) | partial (#14463) |
| Optimum Read / Auto-Solve Aim | (e) | present | present | partial (#14466) | partial (#14466) | partial (#14465) |
| Player-Specific Bellman Strokes Gained | (f) | present | missing (#14467) | missing (#14468) | missing (#14468) | partial (#14467) |
| Saved Scenarios and Comparison | (h) | missing (#14471) | missing (#14471) | missing (#14472) | missing (#14472) | present |
| JSON Import/Export of a Setup | (h) | partial (#14471) | partial (#14471) | missing (#14472) | missing (#14472) | present |
| Camera Views and Playback Phases | (i) | present | partial (#14473) | partial (#14474) | partial (#14474) | present |

## Evidence

Tauri, every feature: `ui/src-tauri/tauri.conf.json`. Wraps the React build (frontendDist ../dist); no putting code in the shell.

**Selectable Units**
- Core/API: missing (#14457). `src/shared/python/putting_dynamics/utils.py`, `src/shared/python/ui/units.py`. SI internally, feet in analytics; conversion helpers only, no units parameter on any route.
- PyQt6: partial (#14457). `src/tools/putting_green_gui/gui.py`. Reads get_unit_preference(); set_unit_system exists but the window has no selector.
- React: missing (#14458). `ui/src/pages/PuttingGreen.tsx`. Speed always m/s, distance yards, toe/hosel mm; no units store or selector.
- Studio: missing (#14457). `src/apps/putting_studio/index.html`. Labels hard-code ft, m/s and mm; no toggle.

**Variance and Noise Channels**
- Core/API: present. `src/shared/python/putting_analytics/skill.py`, `src/shared/python/putting_analytics/service_variance.py`, `src/shared/python/putting_analytics/service_dispersion.py`. All 7 channels (face, path, speed, strike, read grade/aspect/stimp) in /variance-study, /dispersion, /make-spread and /simulated-curve; strike via strike_toe_sd_mm (#14461). /optimize keeps its paired line-and-speed batch for the aim search.
- PyQt6: present. `src/tools/putting_green_gui/_batch_panel.py`. Batch Dispersion inputs for all 7 core channels, defaulting to the core profile; strike spread in mm or in.
- React: partial (#14460). `ui/src/pages/PuttingGreen.tsx`, `ui/src/pages/PuttingAnalytics.tsx`. Simulator: friction_variation only (fixed seed); analytics page: line and speed SD.
- Studio: present. `src/apps/putting_studio/js/core_attribution.js`, `scripts/generate_studio_attribution.py`. 7-channel SG attribution from core-generated values (#14504, strike #14461); the Monte Carlo view keeps its illustrative draws.

**Dispersion Batches and 0-45 ft Make-% Spread**
- Core/API: present. `src/shared/python/putting_analytics/service_dispersion.py`, `src/api/routes/putting_analytics.py`. /dispersion batches up to 2000 putts on a sloped green with any seed; /make-spread gives the 3-45 ft make-% curve.
- PyQt6: present. `src/tools/putting_green_gui/_batch_panel.py`. 40/100/1000-putt batches with rest scatter and the 0-45 ft make-% spread, in the user's units.
- React: partial (#14460). `ui/src/pages/PuttingAnalytics.tsx`. Analytics rollouts at one distance as a table; no scatter, no make-% curve.
- Studio: present. `src/apps/putting_studio/js/io.js`, `src/apps/putting_studio/js/controls_c.js`. 40/100/1000 batches and a 0-45 ft spread (that button runs 2000 attempts).

**Impact and Strike Model (Skid to Roll)**
- Core/API: present. `src/shared/python/putting_dynamics/collision.py`, `src/shared/python/putting_dynamics/solver.py`, `src/api/routes/putting_green.py`. strike() plus skid-to-roll; /simulate-3d returns the collision block and skid distance.
- PyQt6: present. `src/tools/putting_green_gui/gui.py`, `src/tools/putting_green_gui/_scene_builder.py`. Impact block: launch angle, spin, skid distance, face twist, energy loss and putter slowdown from the strike() collision report.
- React: present. `ui/src/pages/PuttingGreen.tsx`. Impact panel: launch speed/angle, putter speed before/after, contact proxy, face twist, energy loss, skid distance.
- Studio: partial (#14461). `src/apps/putting_studio/js/putting_physics.js`. Skid ends at a fixed distance; loft and launch not modelled.

**Toe/Heel Strike Location**
- Core/API: present. `src/shared/python/putting_dynamics/collision.py`, `src/shared/python/putting_analytics/skill_roll_model.py`, `src/api/routes/putting_green.py`. strike(impact_toe_m) and /simulate-3d accept it; strike_toe_sd_mm scatter costs pace via off_centre_speed_retention in /variance-study (strike factor) and /simulated-curve.
- PyQt6: present. `src/tools/putting_green_gui/gui.py`, `src/tools/putting_green_gui/_scene_builder.py`. Heel/toe Strike spin box (mm or in, -40 to 40 mm) passed to simulate_strike.
- React: present. `ui/src/pages/PuttingGreen.tsx`. Impact Toe Offset slider.
- Studio: partial (#14461). `src/apps/putting_studio/index.html`, `src/apps/putting_studio/js/controls_c.js`. Slider is toe-only (0-15 mm); no heel strikes.

**Counter-Torque Stabilization (Hand Couples, Gust)**
- Core/API: missing (#14469). `src/shared/python/grip_contact/couple_check.py`. No putting counter-torque model in the core; a grip couple check exists outside the putting stack.
- PyQt6: missing (#14469). none. None.
- React: missing (#14470). none. None.
- Studio: present. `src/apps/putting_studio/js/state.js`, `src/apps/putting_studio/js/charts2d.js`, `src/apps/putting_studio/js/controls_c.js`. Closed-form couple model, gust pulse and torque charts (illustrative, not in the core: ADR-0058 gap).

**Putter Balance Type (CG Offset, MOI)**
- Core/API: present. `src/shared/python/putting_dynamics/putter_presets.py`, `src/api/routes/putting_green.py`. Named balance presets (toe-hang, face-balanced, zero-torque, counterbalanced) over PutterState; GET /putter-presets and /simulate-3d balance_type.
- PyQt6: present. `src/tools/putting_green_gui/gui.py`, `src/tools/putting_green_gui/_scene_builder.py`. Balance selector: Typical or a shared BALANCE_PRESETS preset.
- React: partial (#14462). `ui/src/pages/PuttingGreen.tsx`. Hosel widget; mass and MOI sent without controls; no balance-type selector.
- Studio: present. `src/apps/putting_studio/js/state.js`, `src/apps/putting_studio/js/controls_c.js`. Four putter types (PUTTER_SPECS); per-type face SD is an assumed input.

**Green Types (Flat, Slope, Stimp Ramp, Undulating, Presets)**
- Core/API: present. `src/shared/python/putting_dynamics/green_spec.py`, `src/api/routes/putting_green.py`. GreenSpec (slope, two-tier step, seeded undulation, named presets) on /simulate, /scatter, /read-green and /contours; GET /green-presets.
- PyQt6: present. `src/tools/putting_green_gui/_scene_builder.py`, `src/tools/putting_green_gui/gui.py`. Green Type selector (cross slope or any core preset), seeded undulation in mm or in, green seed; all on the shared GreenSpec.
- React: partial (#14464). `ui/src/pages/PuttingGreen.tsx`. Grade, aspect, grain, bumps; no green-type selector or presets.
- Studio: partial (#14463). `src/apps/putting_studio/js/putting_physics.js`, `src/apps/putting_studio/js/io.js`. Flat and slope presets plus stimp ramp; undulation switches on with slope and is not selectable.

**Optimum Read / Auto-Solve Aim**
- Core/API: present. `src/shared/python/putting_dynamics/green_reading.py`, `src/shared/python/putting_analytics/optimize.py`, `src/api/routes/putting_green.py`. solve_aim and best_action present; /read-green returns pace, heading, holed and converged in one call on any GreenSpec green.
- PyQt6: present. `src/tools/putting_green_gui/_scene_builder.py`, `src/tools/putting_green_gui/gui.py`. Auto-Solve Read runs solve_aim on the tile's surface off the GUI thread and writes the pace and aim into the controls.
- React: partial (#14466). `ui/src/pages/PuttingAnalytics.tsx`. Analytics best aim/pace study; nothing on the simulator page.
- Studio: partial (#14465). `src/apps/putting_studio/js/controls_c.js`. Aim-only grid search from one noise-free putt.

**Player-Specific Bellman Strokes Gained**
- Core/API: present. `src/shared/python/putting_analytics/service_sg.py`, `src/shared/python/putting_analytics/optimize.py`, `src/api/routes/putting_analytics.py`. Player Bellman E(d) on the dialled green and one putt's SG on /strokes-gained; also /optimize (sg), /library-study, /variance-study, /simulated-curve.
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
- Core/API: present. `src/shared/python/putting_dynamics/phases.py`, `src/api/routes/putting_green.py`. Per-sample ball mode on /simulate and /simulate-3d, and phase marks (where the hop, skid, roll and rest start) from the core's phase_marks for playback jumps.
- PyQt6: partial (#14473). `src/tools/putting_green_gui/gui.py`. Auto-framed orbit and a one-shot animation; no presets, scrub or phase jumps.
- React: partial (#14474). `ui/src/pages/PuttingGreen.tsx`, `ui/src/components/visualization/PuttingScene3D.tsx`. Orbit camera, play/pause/scrub/rate; no view presets or phase jumps.
- Studio: present. `src/apps/putting_studio/js/controls_b.js`, `src/apps/putting_studio/js/controls_a.js`. Grip, Full Plane, Down-the-Line and Face-On cameras; phase badge; scrubber and keyframe jumps.
