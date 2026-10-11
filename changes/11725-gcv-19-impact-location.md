---
issue: 11725
summary: "GCV-19: the native-export impact card fills Impact Location (toe, high) from the decorative ball (contact point = ball centre - radius x face normal at the impact sample, card extras ball_source records the placement source); without a ball the row stays unavailable with an accurate reason instead of the stale GCV-11/GCV-13 text"
title: "GCV-19: Impact Location From The Decorative Ball"
branch: "claude/gcv-19-impact-location"
---

Refs #11725, #11723. `impact_card_for` and `card_from_series` take an optional `BallPlacement`, passed from `export_swing` via `swing.ball`. Committed driver and iron7 fixtures give implausibly large offsets because they are stale low-speed pose sets (#12223); no tuning was done.
