---
issue: 13903
summary: "EnhancedBallFlightSimulator: route deterministic aero through Rust facade (UD-03)"
branch: "fix/13903-enhanced-simulator-rust-facade"
dl_state: "in_development"
next_step: "Implement facade routing and force provider injection in ball_enhanced_simulator.py"
---

Route deterministic drag and Magnus calculations in EnhancedBallFlightSimulator through the existing Rust facade with pure-Python fallback, supporting force provider injection while preserving gusts and randomization in Python.
