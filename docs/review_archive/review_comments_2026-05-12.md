# Review Comments Archive - 2026-05-12

Generated: 2026-05-12T01:34:04.985407

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #5290: src/shared/python/motion_pipeline/matching/cmc.py:263

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Report actual outer-loop backend used in result metadata**

When the Rust path throws inside `match`, the code explicitly falls back to `_compute_tau_python`, but `metadata["rust_outer_loop"]` is still set from module availability (`_HAVE_RUST`) rather than the path that actually executed. This makes runtime diagnostics and benchmark attribution incorrect in environments where the Rust module imports but exec...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/5290#discussion_r3224882734)

---
