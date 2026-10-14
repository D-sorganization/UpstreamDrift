---
issue: 9545
summary: "BunkerShot3D: read-only 3-D particle viewer for the trajectory artifact (headless playback with scrub/step, time-synced club/ball without extrapolation, labelled display decimation, measured fps; instanced-glyph VTK layer for the canonical viewport with a stated no-VTK fallback). No production solver emits the artifact yet."
dl_state: "in_review"
next_step: "Wire a real grain-resolving 3-D backend (Chrono F3 once pychrono is installable) into ParticleTrajectoryWriter, then embed the particle layer in ShotViewportWidget with playback controls."
branch: "feat/9545-particle-capture-s7"
---
