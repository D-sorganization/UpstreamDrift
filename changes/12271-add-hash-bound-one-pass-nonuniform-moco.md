---
issue: 12271
summary: "Add hash-bound one-pass nonuniform Moco mesh refinement planning"
dl_state: "in_review"
next_step: "Review committed pure planner and separately authorize any native child attempt."
branch: "feat/f07-native-component-audit-budget-12244"
---

Add a hash-bound one-pass nonuniform Moco mesh refinement planner and worker
seed path for the synthetic ladder. The pure plan marks full Hermite-Simpson
cells from retained local activation indicators, caps refinement at four cells
and 24 child intervals, and refuses stale parent evidence. A child retains the
original source, horizon, initial state, objective, solver and replay
thresholds, and interpolates persisted controls linearly onto every child
endpoint and midpoint without trajectory resampling. It records parent and
child provenance and remains physically unqualified until unchanged fresh
replay passes. The worker recomputes the complete plan before solver setup and
binds post-solve grid reporting to the child request. Parent state guesses use
the OpenSim states-table API and restore exact initial values after
interpolation. An installed OpenSim 4.6 data-bearing trajectory seeder smoke
passed under the bounded process supervisor; it did not initialize a model or
call a solver or replay. No native child solve or replay was run.
