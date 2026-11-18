---
issue: 2725
summary: "Remove the obsolete GAAI daemon launcher and live documentation pointers"
---

# Change fragment: remove the GAAI daemon launcher

- Deletes `scripts/maintenance/start-gaai-daemon.sh` and live CLAUDE.md pointers.
- The vendored `.gaai/` tree and slash-commands are removed in UpstreamDrift#15119.
- Refs #2725. Refs #2727.
