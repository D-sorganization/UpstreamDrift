# Review Comments Archive - 2026-05-06

Generated: 2026-05-06T02:25:21.449696

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4061: src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/train_dynamics_surrogate.py:19

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Restore executable behavior for legacy shim scripts**

This compatibility shim only re-exports symbols and emits a warning, but it never invokes `main()` under a `__name__ == "__main__"` guard. That means legacy entrypoints stop working when executed directly (for example `python .../train_dynamics_surrogate.py ...`), so callers that relied on the old script paths no longer run training/optimization/extractio...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4061#discussion_r3194332525)

---

### PR #4061: src/shared/python/motion_matching/surrogate/perstep/optimize.py:251

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Resolve relocated optimizer's control-name import**

The relocated optimizer still does `from export_torque_polynomials import TORQUE_TO_POLYNOMIAL_BASE` inside `_control_columns`. After this move, `export_torque_polynomials.py` is no longer a sibling of `surrogate/perstep/optimize.py`, so `optimize_sequence` now fails with `ModuleNotFoundError` unless the old MachineLearning directory is manually injected in...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4061#discussion_r3194332530)

---

### PR #4061: src/shared/python/motion_matching/surrogate/perstep/extract_dataset.py:23

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Point default manifest to an existing file after relocation**

`DEFAULT_MANIFEST` is derived from the new `SCRIPT_DIR` (`surrogate/perstep`), but `column_manifest_inverse_ready.json` was not moved there. Running the extractor without `--manifest` now resolves to a non-existent file and errors before processing, whereas the pre-relocation default pointed at the checked-in manifest under `MachineLearning`.

Use...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4061#discussion_r3194332537)

---
