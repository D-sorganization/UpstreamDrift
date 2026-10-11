#!/usr/bin/env bash
set -u
cd /home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift
evidence=docs/research/shot_pattern_analysis/phase2/validation/supplement_v6_final_graphics_review_v2
MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen \
  /tmp/shot-pattern-validation-venv-20261009/bin/python \
  "$evidence/publish_density_layout_child.py"
status=$?
printf '{"exit_code":%s}\n' "$status" > "$evidence/layout_child_publication_exit.json"
exit "$status"
