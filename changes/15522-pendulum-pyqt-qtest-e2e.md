---
issue: 15522
summary: "Exercise the engine Driven Pendulum Explorer with headless QTest and fix the Triple model view crash."
---

Native Qt clicks and key presses cover Start, Pause, Reset, the gravity box,
Double/Triple selection, typed torque validation and the inverse-dynamics
velocity polynomial against the real timer and dynamics. Selecting Triple
previously raised `AttributeError` (`TriplePendulumParameters` has no
`plane_inclination_deg`); the view now shares the controller's display plane.
No physical-accuracy claim is made. Refs #15522.
