---
issue: 15522
summary: "Exercise the lower-body MuJoCo panel with headless QTest and stop physics and telemetry on close."
---

Native Qt input covers stance and gain edits, real MuJoCo playback, pause,
history scrubbing, full reset and close while playing. Only the external
passive OpenGL viewer is replaced by a headless sink. No physical
qualification or OpenGL rendering claim is made. Refs #15522.
