# Review Comments Archive - 2026-05-05

Generated: 2026-05-05T21:37:25.007337

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #4016: src/shared/python/motion_matching/club_target.py:124

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Reject non-finite quaternion rows during ClubTarget validation**

The quaternion validation only checks `np.abs(qnorms - 1.0) > QUAT_NORM_TOL`, which does not catch `NaN` norms (`NaN > tol` is false). As a result, rows containing `NaN` values can pass validation and produce a `ClubTarget` that silently carries invalid orientation data into downstream cost/optimization code.

Useful? React with 👍 / 👎.
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4016#discussion_r3192995007)

---

### PR #4016: src/shared/python/motion_matching/loaders/_align.py:90

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Derive impact index from data when not impact-aligning**

`resample_target` always sets `impact_idx_out` to the sample nearest `impact_target_t_s`, even when `time_alignment` is `"address"` or `"none"`. In those modes the detected impact time is not anchored to `impact_target_t_s`, so the returned `impact_idx` can point to the wrong frame and mislead any downstream logic that uses impact-phase indexing.

Usef...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4016#discussion_r3192995010)

---

