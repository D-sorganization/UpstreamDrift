# Launch Monitor Fixtures (ADR-0046 G0 / G0.1)

This directory contains deterministic test fixtures for cross-stack launch monitor drift verification between UpstreamDrift and Tools.

## dr0046_cross_stack_session_v1.json

- **Fixture ID**: dr0046-cross-stack-launch-monitor-session/1
- **Seed**: 20460046
- **Description**: Contains 160 synthetic shot records across 4 players (player_A, player_B, player_C, player_D), 10 simulated golf holes, 4 club selections (Driver, 7 Iron, Wedge, Putter), and standard launch parameters (ball speed, launch angle, spin rate, carry distance, offline deviation, face-to-path angle, and strokes gained baselines).
- **Purpose**: Used by ests/integration/launch_monitor_drift/ to compare calculations between the UpstreamDrift implementation and the vendored Tools stack to verify statistical parity, correlation matrix calculations, OLS regressions, and pin known implementation divergences (D1-D31).
