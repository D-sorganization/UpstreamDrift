# Review Comments Archive - 2026-05-06

Generated: 2026-05-06T00:03:21.406244

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4049: src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/optimize_torque_sequence_for_club.py:16

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Preserve script-path execution by avoiding repo-root imports**

This script is documented to be run directly by file path (for example from the repo root), but the new `from src.shared...` import requires the repository root to already be on `PYTHONPATH`. When Python executes a script by path, `sys.path` contains the script directory, not the repo root, so this import can raise `ModuleNotFoundError: No module...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4049#discussion_r3193596098)

---

### PR #4049: src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/tests/test_control_names.m:34

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Fix fixture path depth so MATLAB equivalence test actually runs**

The relative path climbs only seven directories from `.../shared/tests`, which resolves to `<repo>/src`, so the test looks for `src/tests/fixtures/control_names_matlab.json` instead of `tests/fixtures/control_names_matlab.json` at the repo root. Because the check is gated by `assumeTrue`, this causes the cross-language fixture assertion to be ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4049#discussion_r3193596113)

---

### PR #4049: tools/regen_control_names_fixture.m:17

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Preserve canonical ordering when regenerating MATLAB fixture**

Using `keys(map)` and `values(map)` to build `torque_to_polynomial_base` loses canonical first-seen ordering because `containers.Map` key iteration is sorted. Regenerating the fixture with this script will reorder entries and make `tests/unit/motion_matching/test_control_names.py` fail its strict comparison against `list(TORQUE_TO_POLYNOMIAL_BASE...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4049#discussion_r3193596117)

---

