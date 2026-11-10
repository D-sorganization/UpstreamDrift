# ZTCF Force Helper Scope

Part of adversarial review [#14944](https://github.com/D-sorganization/UpstreamDrift/issues/14944) (epic [#14934](https://github.com/D-sorganization/UpstreamDrift/issues/14934)).

`src/shared/python/biomechanics/ztcf.py::compute_ztcf_forces` is the **single-pendulum
tangential** helper validated in #7054. It maps one generalized acceleration per
segment row to an approximate tangential `m*(a-g)` force.
At horizontal release θ = 90°, true pivot reaction is 0 while it returns `[-m·g, +m·g]`.
The 2D helper is NOT a physical joint reaction force; use `compute_ztcf_joint_reactions` for RNEA.

Multi-body chains must use each hosting engine's recursive Newton–Euler ZTCF path.
The helper rejects `n_joints > 1` unless `allow_simplified=True` (legacy regression only).
