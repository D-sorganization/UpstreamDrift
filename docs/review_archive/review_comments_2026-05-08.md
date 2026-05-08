# Review Comments Archive - 2026-05-08

Generated: 2026-05-08T14:31:49.295929

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4641: tests/integration/test_urdf_cross_engine_fk.py:146

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Read Pinocchio body poses from frame placements**

`model.getBodyId()` returns a body/frame index, but this code uses that value to index `data.oMi`, which stores joint placements. When Pinocchio is available and a requested body exists, this can return the wrong transform (or raise on larger models), making the MuJoCo/Drake vs Pinocchio RMSE check invalid. Use the frame placement (`data.oMf[bid]` after `upda...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4641#discussion_r3211471648)

---

### PR #4641: .github/workflows/urdf-cross-engine-equivalence.yml:85

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Enforce at least two engines before running equivalence gate**

This workflow suppresses install failures for every optional engine (`|| echo ...`), while the tests themselves skip missing backends via `importorskip`; as a result, the job can succeed with all equivalence tests skipped and no cross-engine comparison actually executed. Since this is intended as a parity gate, add a hard precheck that at least t...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4641#discussion_r3211471651)

---

