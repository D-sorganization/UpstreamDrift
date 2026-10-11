---
issue: 11725
summary: "GCV-19: the native-export impact card fills Impact Location (toe, high) from the decorative ball (contact point = ball centre - radius x face normal at the impact sample, card extras ball_source records the placement source); without a ball the row stays unavailable with an accurate reason instead of the stale GCV-11/GCV-13 text"
title: "GCV-19: Impact Location From the Decorative Ball"
branch: "claude/gcv-19-impact-location"
---

Refs #11725, #11723. `impact_card_for` and `card_from_series` take an optional `BallPlacement`, passed from `export_swing` via `swing.ball`. A contact more than 50 mm from the face centre is off every club face, so the row reports "unavailable" with the measured miss instead of an off-face number. On the committed fixtures the miss is driver toe −111 mm / high −134 mm, and on iron7 the face never comes within 0.15 m of the address ball. Both therefore show the off-face reason. No tolerance was tuned.
