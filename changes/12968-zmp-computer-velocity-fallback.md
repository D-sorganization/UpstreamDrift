---
issue: 12968
summary: "robotics: eliminate silent velocity, acceleration, and mass fallbacks in ZMPComputer and support finite-difference acceleration"
branch: "fix/12968-zmp-computer-velocity-fallback"
---

Eliminated silent fallback to quasi-static zero velocity / acceleration and arbitrary default mass in `ZMPComputer`. `_get_com_velocity()` and `_estimate_mass()` now fail closed with `ValueError` when required parameters cannot be derived from a `HumanoidCapable` engine or provided explicitly. Added finite-difference acceleration estimation from velocity history with `dt` when `com_acceleration` is None, marking `is_valid=False` and documenting the estimation caveat in `validity_reason`. Updated docstrings to match runtime contracts.
