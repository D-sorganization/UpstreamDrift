---
issue: 15935
summary: "Stop the MuJoCo viewer render timer when its viewport is destroyed so a deleted QLabel can no longer raise through the Qt event loop into unrelated tests"
dl_state: "in_review"
next_step: "Merge the PR once the unit-test-gate shard is green."
owner: "qwen-oglaptop-1"
branch: "qwen-oglaptop-1/15935-qt-destruction-leaks"
paths: "src/tools/model_explorer,tests/tools/model_explorer"
---
