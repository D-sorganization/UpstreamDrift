---
issue: 11545
summary: "MOSAIC-13: MHE now applies its arrival factor in the window cost and propagates it by Schur-complement marginalisation when the window slides; matches batch Kalman/least-squares estimate to 1e-7"
branch: "fix/11545-mhe-arrival-factor"
paths: "src/shared/python/estimation/moving_horizon.py,src/shared/python/estimation/arrival_factor.py,tests/unit/estimation/test_moving_horizon_arrival_cost.py,docs/research/model_aware_matching/model_aware_matching.tex"
---
