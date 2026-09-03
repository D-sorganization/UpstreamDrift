# Shared Tools Seam

How UpstreamDrift consumes the shared code owned by
[D-sorganization/Tools](https://github.com/D-sorganization/Tools) (package name
`ud-tools`), and the artefacts that govern the seam.

Program: D-sorganization/Repository_Management#1505 · Epic: UD #9406 · Tools
ledger: Tools #4915.

## Files in This Directory

| File | Produced by | Purpose |
| --- | --- | --- |
| `divergence_inventory.v1.json` / `.md` | `python -m scripts.shared_tools.divergence_inventory --write` | Every path under `src/shared/python` classified against the pinned Tools tree (`identical` / `diverged` / `ud-only` / `tools-only`, plus `spelling_only`), with authorship on both sides. `--check` fails when stale. |
| `seam_rulings.v1.json` | hand-maintained | One ruling per top-level entry of the Tools shared tree: `tools-canonical`, `ud-canonical`, `split` or `deferred`, with status `pending-cleanup` / `cleaned` / `n/a`. |
| (gate) `scripts/shared_tools/check_seam_drift.py` | CI job `seam-drift-gate` (needed by `quality-gate`) | Enforces the rulings: a cleaned package may not regrow a UD copy; a `ud-canonical` package needs a Tools ledger row. |
| (gate) `scripts/shared_tools/check_tools_pins.py` | `vendor-freshness.yml` | Cargo `tools-core` rev == `vendor/ud-tools` gitlink == any `ud-tools @ git+...` pin in `pyproject.toml` / `requirements-tools.txt`; the release-wheel pin is reported. |

## Where the Tools Code Comes From

There are three possible sources, in this precedence:

1. **Installed distribution** — `pip install -r requirements-tools.txt`
   installs the Tools release wheel pinned there
   (`ud_tools @ https://github.com/D-sorganization/Tools/releases/download/v1.15.0/ud_tools-1.15.0-py3-none-any.whl`,
   Tools #4920; kept out of `pyproject.toml` because PyPI rejects direct URL
   references). The wheel is cut from the release tag commit (v1.15.0 =
   `e87b04105`), which is **not** the submodule gitlink (`c0a395d5`);
   `check_tools_pins.py` reports both and the gitlink wins for the source
   tree. When `importlib.metadata` can see `ud-tools`, the launcher and
   `_seam_redirect` leave `sys.path` alone and the installed `shared`,
   `sidekick`, `chat`, `utils` packages win.
2. **`vendor/ud-tools` submodule** — the pinned checkout. When no distribution
   is installed, `launch_upstream_drift.py`, the pytest `pythonpath`, the
   primary and modular Dockerfiles and `src/shared/python/_seam_redirect.py`
   put `vendor/ud-tools/src/shared/python`, `vendor/ud-tools/src` and
   `vendor/ud-tools/src/python/src` on `sys.path`, in that order.
3. **The committed shadow `src/shared/python`** — being retired. It still
   wins for packages ruled `pending-cleanup`; for packages ruled `cleaned` the
   UD copy is gone and `src/shared/python/_seam_redirect.py` binds the
   `src.shared.python.<root>` spelling to the same module object as
   `shared.python.<root>` (Tools). A `split` root keeps its UD-only modules
   reachable by appending the UD directory to the canonical package
   `__path__`.

If neither 1 nor 2 is available the import fails loudly with
`SeamResolutionError` and the `git submodule update --init vendor/ud-tools`
hint; there is no silent fallback to a stale copy.

## Import Spellings

`shared.python.<pkg>`, `src.shared.python.<pkg>` and bare `<pkg>` all resolve
to one module object for the roots handled by the Tools
`SharedImportAliasFinder` and `_seam_redirect`. New code should use
`shared.python.<pkg>`; the 152 spelling-only divergences in the inventory are
exactly the `src.shared.python.` spelling and vanish as shadows are deleted.

## Ratchets

* `scripts/config/shadow_modules.yaml` — one entry per UD name that still
  shadows a Tools module (was 33, now 23); entries may only be removed.
* `docs/shared_tools/seam_rulings.v1.json` — `status` may only move from
  `pending-cleanup` to `cleaned`; the gate then enforces it.

## Roadmap

* Phase 1 (this epic): inventory, rulings, gate, pin parity, consumption
  mechanism; delete the small `tools-canonical` clusters.
* Phase 2: delete `ai`, `chat`, `sidekick` overlaps; upstream `ud-canonical`
  packages (Tools #4494); remove the submodule once Tools publishes a wheel
  per release (Tools #4920) and the pin becomes a version.
