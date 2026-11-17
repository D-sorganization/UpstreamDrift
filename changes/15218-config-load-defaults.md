---
issue: 15218
summary: "Preserve fresh caller defaults after JSON and YAML load failures"
dl_state: "in_review"
next_step: "CI and automated review"
branch: "codex/burn1116-15218-config-default"
paths: "src/shared/python/config/config_utils.py,tests/unit/config/test_config_load_failure_defaults.py"
---

14 regression cases failed before the fix; 83 related tests pass on supported Python 3.12 with exact Tools pin a47d83d6.

File-existence permission errors also preserve the supplied default; JSON and YAML regressions cover inaccessible parent directories.
