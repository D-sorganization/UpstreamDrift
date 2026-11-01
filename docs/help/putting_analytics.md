---
title: Putting Analytics
tile_id: putting_analytics
status: complete
---

# Putting Analytics

## Purpose

Putting Analytics runs the shared putting studies from the desktop: the best
aim and pace for one putt, the stroke cost of misreading a green, and a
weighted library of putt classes. It calls the same service as the MCP putting
tools and the `/tools/putting-analytics` API, so all three give the same
answer for the same inputs.

Entry point `src/tools/putting_analytics_gui/__main__.py`, which opens
`src/tools/putting_analytics_gui/gui.py`. Run it with
`python -m src.tools.putting_analytics_gui`, or launch the tile from the
UpstreamDrift launcher.

## Inputs

| Input                | Description                                                     | Unit     |
| -------------------- | --------------------------------------------------------------- | -------- |
| Study                | Best aim and pace, cost of misreads, or putt-class library      | n/a      |
| Distance             | Hole distance                                                   | ft       |
| Grade and aspect     | Green slope and the direction it falls                          | %, deg   |
| Stimp                | Green speed                                                     | ft       |
| Start-line SD        | Start-direction scatter                                         | deg      |
| Speed SD             | Speed scatter as a fraction of the intended speed               | fraction |
| Objective            | `sg` (expected strokes) or `make_rate`                          | n/a      |
| Rollouts, seed       | Rollouts per candidate action and the random seed               | count    |
| Baseline             | Your own distances and expected putts, which price a miss       | ft, putts|
| Aims and paces       | Candidate aim offsets and pace past the hole                    | deg, ft  |
| Reads                | Misread study: one `slope_scale stimp_ratio` per line           | ratio    |
| Classes              | Library study: `distance_ft slope_pct orientation stimp count`  | mixed    |

No reference expected-putts curve ships with the tool (#14176); enter one you
trust. The service bounds the work per run and rejects inputs outside its
ranges with a message shown above the results.

## Outputs

A two-column table. The aim and pace study lists the best action with its make
rate, expected strokes and standard errors, then the ranked candidates. The
other studies list every value of the answer by its dotted path, such as
`bands[0].make_rate.mean`.
