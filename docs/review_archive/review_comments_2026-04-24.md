# Review Comments Archive - 2026-04-24

Generated: 2026-04-24T21:04:52.276587

## Reviewer (chatgpt-codex-connector[bot]) (1 comments)

### PR #3247: examples/motion_training_demo.py:277

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Suggest a supported fixture format in missing-file error**

When `main()` hits the missing trajectory path branch, the new error text tells users to use `data/golf_trajectory.csv`, but this demo’s parser path (`ClubTrajectoryParser.parse`) is Excel-only (`pandas.read_excel`/`openpyxl`) and requires a sheet name. Following the current hint leads to another failure instead of recovery, so the new “actionable” g...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3247#discussion_r3141362698)

---
