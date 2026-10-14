---
issue: 11904
summary: "Add bounded robust shooting NMPC with fail-closed actuator fallback, native MuJoCo step execution and independent frozen-torque replay, benchmarked against F02 TVLQR on two predeclared perturbed one-hinge trials; TVLQR is retained for native-step operation because the shooting prototype cannot meet the 10 ms step budget, and F05 full-swing and capture qualification remain open"
---

Add bounded robust shooting NMPC with fail-closed actuator fallback, native MuJoCo step execution and independent frozen-torque replay. Benchmark it against F02 TVLQR on two predeclared perturbed one-hinge trials; retain TVLQR for native-step operation because the shooting prototype cannot meet the 10 ms step budget. F05 full-swing and capture qualification remain open.
