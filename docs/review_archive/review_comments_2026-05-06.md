# Review Comments Archive - 2026-05-06

Generated: 2026-05-06T00:17:08.295603

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #4056: src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/matlab/+frame_search/lookup_signal_value.m:79

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Remove shape-based transpose before selecting signal component**

Transposing `raw` whenever `size(raw,1) < size(raw,2)` corrupts component lookup for short horizons, because Simulink/timeseries data is commonly `time x components` even when time has fewer samples than component count (for example 2x3). In that case this code flips the matrix to 3x2, clips `_3` to column 2, and then reads the wrong value from...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4056#discussion_r3193659079)

---

