# Swing Report Package

One command builds the shareable package of an analysed swing: the ZTCF,
ZVCF and Total force videos, the midhands force-and-couple videos, the case
comparison grids and the key-phase freeze-frame sheets (epic #15725).

## Quick Start

```bash
python3 -m src.tools.swing_report_package --bundle swing.npz --out package \
    --swing driver --club Driver
```

This renders the default package: both camera views, 0.25x speed, 1920x1080.
A full package renders many videos and takes hours; reruns reuse finished
clips, so an interrupted run resumes where it stopped.

Preview the plan without rendering:

```bash
python3 -m src.tools.swing_report_package --bundle swing.npz --out package --dry-run
```

## Choosing What Is Made

| Flag | Effect |
|---|---|
| `--include ITEM` | Add a catalog item (repeatable) |
| `--exclude ITEM` | Drop an item (repeatable) |
| `--no-default` | Start from nothing instead of the default package |
| `--views V ...` | Camera views, e.g. `face_on down_the_line` |
| `--speeds S ...` | Playback speeds, e.g. `0.25 1` |
| `--set KEY=VALUE` | Override a render option for every video, e.g. `--set size=3840x2160` |
| `--package FILE` | A request file (JSON) with the same choices |

A composite item (a grid or a phase sheet) pulls in the videos it is built
from. The items, their options and the default package are listed in
[`package_catalog.json`](package_catalog.json).

A request file uses the fields of `PackageRequest`:

```json
{
  "include": ["midhands_grid", "midhands_phase_sheet"],
  "exclude": ["tip_to_tail"],
  "views": ["face_on"],
  "render_overrides": {"size": "3840x2160"},
  "item_overrides": {"case_total": {"ztcf_arrow_gain": 3}}
}
```

Precedence: catalog defaults, then `render_overrides`, then the item's own
options, then `item_overrides`. Unknown keys and items are refused.

## Outputs

- `<item>/<view>/<swing>_mujoco_<view>_0p25x.mp4` with a `.times.json`
  frame-time sidecar
- `<grid item>/<view>_0p25x/<item>.mp4`: ZTCF, ZVCF and Total side by side
- `<sheet item>/<view>_0p25x/<item>.png`, plus `<case>_<phase>.png` stills
- `manifest.json`: every job, its status (`done`, `failed`, `blocked`) and
  outputs, and the detected key phases

## Settings

| File | Holds |
|---|---|
| `package_catalog.json` | Items, render defaults, views, speeds, default package |
| `phase_settings.json` | Key-phase rules (segment level crossings, windows) |
| `composite_settings.json` | Grid and sheet layout, banner text, sizes, quality |
| `runner_settings.json` | Exporter module, free-disk guard, timeout, phase bodies |

Each render runs headless in its own temporary folder, which is removed
afterwards; the run stops when free disk falls below `min_free_gb`.

## Reading the Cases

ZTCF (drift) is the motion the swing would have with no applied joint
torques: gravity, momentum and passive effects. ZVCF (control) is the part
due to the applied torques alone, without gravity. Their sum is the Total.
The ZVCF ground reaction is a component of that decomposition, not a force
the ground could supply on its own.
