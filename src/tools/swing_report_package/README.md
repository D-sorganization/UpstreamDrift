# Swing Report Package

One command regenerates every chart and video of an analysed swing from its
kinetic-match bundle (epic #15725). A **package** is a set of catalog items:
force videos rendered by `native_viewer_export`, grids of the ZTCF, ZVCF and
Total videos side by side, and key-phase freeze-frame sheets.

```bash
python3 -m src.tools.swing_report_package --list     # items and packages
python3 -m src.tools.swing_report_package --bundle swing.npz --out package --dry-run
python3 -m src.tools.swing_report_package --bundle swing.npz --out package
```

`--dry-run` prints the planned manifest as JSON and renders nothing. A rerun
into the same folder reuses finished clips.

## Choosing and Customising

The default package renders both views at 0.25x: the `case_ztcf`,
`case_zvcf`, `case_total`, `tip_to_tail` and `midhands_total` videos, the
`case_grid` and the `phase_sheet`. `--package NAME` starts from `default`,
`midhands` or `full`; `--add ITEM` and `--drop ITEM` (repeatable) change it;
`--config FILE` is a custom request with the fields of `PackageRequest`:

```json
{"default": false, "include": ["case_total", "midhands_grid"],
 "exclude": ["midhands_ztcf"], "views": ["face_on"], "speeds": [0.25, 1.0],
 "render_overrides": {"size": "3840x2160"},
 "item_overrides": {"case_total": {"ztcf_arrow_gain": 3}}}
```

A composite pulls in the videos it needs; unknown items, packages and keys
are refused. Render options apply catalog `render_defaults`, then
`render_overrides`, the item's `args`, then `item_overrides`.

## Adding an Item to the Catalog

Add an entry under `items` in `package_catalog.json` with a `kind` (`video`,
`grid` or `phase_sheet`) and a `title`. A video lists `native_viewer_export`
options in `args` (underscored names, `true` for a bare flag); a composite
lists its videos in `needs`. Add it to `default_package` or `packages`.

## Outputs and Statuses

Videos go to `<item>/<view>/`, grids and sheets to `<item>/<view>_0p25x/`;
`manifest.json` records each job's key, status, outputs and error, and the
key phases. Statuses: `planned` (dry run), `done`, `failed`, `blocked` (an
input video did not finish), `unavailable` (the exporter does not accept its
options yet). Exit status: 1 when a job failed, 2 on bad input or low disk.
