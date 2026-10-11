# F07 Native Moco Complexity Ladder Turnover

## Current Status

This turnover summarizes the synthetic two-muscle OpenSim complexity and mesh-refinement work. Its complete dated experiment record, including failed attempts, source/request/output identities, diagnostics, and reproduction details, is retained in the linked [historical record, Part 1](F07_NATIVE_MOCO_COMPLEXITY_LADDER_HISTORY_PART_1.md) and [Part 2](F07_NATIVE_MOCO_COMPLEXITY_LADDER_HISTORY_PART_2.md).

The synthetic solves do not establish production feasibility. The 10- and 20-interval attempts and later replay diagnostics have distinct outcomes; solver convergence is not replay acceptance. The retained 21-interval solution also failed the strict planned-clock check at 9 of 43 values by one ULP. A separate replay on the saved timestamps passed coordinate and endpoint limits but exceeded the unchanged activation limit. Offline activation-law and Hermite–Simpson calculations are diagnostic comparisons only; they do not identify a cause or qualify the source, physiology, or a continuous trajectory.

Public dependency status at the recorded checkpoint: parent PR #12250 merged at `f131464bb1a38cff0478aa7626c21c7922c15c88`; replay-recorder PR #12269 merged at `9410bb19bf147c538d14c693d0878d84a73a46ff` on 2026-10-11 at 09:17:07 UTC. Neither merge qualifies full-body muscle replay. The public 17-row, six-ecosystem denominator and independent full-horizon replay remain open.

## Scope and Reproduction

All experiments described here use manufactured synthetic inputs. They do not use private capture values or establish supported contact, physiological qualification, or controller-off production replay. Preserve every original failure and acceptance bound when reproducing an experiment; do not infer acceptance from optimizer status alone. The historical appendices retain the exact commands, source and runtime identities, inputs, outputs, failed statuses, and limits.

The canonical calculation-level discussion remains in [Chapter 13](../../../manuals/upstreamdrift/chapters/13-native-muscle-replay.qmd). This document and its appendices are execution history, not an approval record.
