---
issue: 15703
summary: "Make MuJoCo capability probes safe under headless Windows xdist and use CPU-only MuJoCo mode for GUI tests"
dl_state: "in_review"
next_step: "Review and merge the ready PR after exact-head checks pass"
title: "GUI E2E Gate Crash Under Windows Xdist"
owner: "codex-ct-1"
branch: "codex-ct-1/15703-gui-e2e"
paths: "src/shared/python/simulation_backends/capabilities.py; tests/unit/simulation_backends; tests/ui/tools/simulation_backends"
---
