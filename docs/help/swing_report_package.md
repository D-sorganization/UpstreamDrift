---
title: Swing Report Package
tile_id: swing_report_package
status: active
---

# Swing Report Package

Regenerate the videos, case grids and key-phase sheets of an analysed swing from its bundle (epic #15725). Choose the bundle (`.npz`), an output folder and a package (`default`, `midhands` or `full`), then tick or untick catalog items; ticks become `--add` and `--drop` flags. **Preview Plan** lists the jobs without rendering; **Run** starts `python3 -m src.tools.swing_report_package` in the background and streams its output; **Stop** ends it.

## Limitations

A composite keeps the videos it is built from even when they are unticked. Renders are slow, stage several GB each and stop on low disk; a rerun reuses finished clips. There is no browser page yet. See `src/tools/swing_report_package/README.md`.
