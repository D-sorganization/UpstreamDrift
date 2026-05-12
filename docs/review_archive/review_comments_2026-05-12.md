# Review Comments Archive - 2026-05-12

Generated: 2026-05-12T01:32:37.060322

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #5289: src/shared/python/biomechanics/rust_muscle.py:145

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Avoid calling nonexistent Rust `activation_step` symbol**

When the Rust wheel is installed, `activation_step()` dispatches to `_rust.activation_step(...)`, but the PyO3 module in this commit only registers `activation_step_batch`, `muscle_force_batch`, `joint_torques_batch`, and `step_full` (plus class methods), so this call path raises `AttributeError` at runtime instead of performing an update. This breaks...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/5289#discussion_r3224878494)

---

### PR #5289: rust_core/upstream-muscle/src/batch.rs:188

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Guard `step_full` against short `states` input**

`step_full` computes `m = u.len()` and then indexes `states[i]` for `0..m` without verifying `states.len() == m`; if a native Rust caller passes mismatched lengths, this panics before any `Result` error can be returned. The rest of this module uses explicit length checks and error returns, so this introduces an avoidable crash path in a public fallible API.

U...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/5289#discussion_r3224878498)

---

