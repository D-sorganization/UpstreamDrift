---
issue: 14172
summary: "Add Putting Analytics Stochastic Green Sampler"
branch: "claude/putting-green-sampler-14172"
---

Adds `putting_analytics.green`: per-green draws (stimp, slope, fall line, grain)
and per-putt Fourier-mode roughness with analytic gradients, each on its own
seeded substream so one component can be ablated without perturbing the rest.
