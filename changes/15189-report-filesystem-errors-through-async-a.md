---
issue: 15189
summary: "Report filesystem errors through async and background save callbacks"
dl_state: "in_review"
next_step: "Review regression fix through the supervised pipeline"
owner: "codex"
branch: "fix/async-save-io-errors"
paths: "src/shared/python/data_io/_async_io.py; tests/output/wave5_output/test_async_io.py"
---

Red test commit e5d3cf017 reproduces seven missing callback/log cases. Added OSError to the existing failure path; futures preserve the original exception. No owner-dependent work or scientific calculations changed.
