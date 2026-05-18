# Body-Part Viz — Asset Author Guide

This guide is for contributors adding a new shape to the bundled
default library, not end users picking shapes per capture. If you
want to bring in a one-off custom mesh for your own use, see
[`mesh_import.md`](mesh_import.md).

The default library lives at `assets/body_part_shapes/default/` and
is exposed through `body_part_viz.asset_library.ShapeLibrary`. Adding
a shape here means it shows up in every C3D Viewer install, every
matcher session, and every URDF generator output without the user
having to import a file.

## Layout of the default library

```
assets/body_part_shapes/default/
├── _generate.py          # procedural mesh generator (deterministic)
├── manifest.json         # schema-versioned shape index
├── LICENSES.md           # per-file attribution
├── head.stl
├── torso.stl
├── upper_arm.stl
├── forearm.stl
├── hand.stl
├── thigh.stl
├── shin.stl
└── foot.stl
```

Three files are authoritative:

1. **`manifest.json`** — the schema-versioned index. Every shape the
   library exposes appears here.
2. **`_generate.py`** — the procedural script that produces the
   bundled STL meshes. Determinism is enforced via a fixed RNG seed
   so re-runs are byte-stable on a given trimesh version.
3. **`LICENSES.md`** — per-file attribution. Every entry in the
   manifest must have a corresponding line in LICENSES.md naming the
   licence and source.

The STL files themselves are regenerated artefacts: editing them by
hand is allowed for one-off tweaks, but the canonical source is
`_generate.py`.

## Manifest schema

`manifest.json` is a JSON document keyed by shape name:

```json
{
  "schema_version": 1,
  "shapes": {
    "<shape_name>": {
      "file": "<filename.stl>",
      "rest_dimensions": [<dx>, <dy>, <dz>],
      "binding_template": {
        "kind": "between_two | cluster | on_marker",
        "marker_names": ["MARKER1", "MARKER2", "..."],
        "rest_orientation_quat": [w, x, y, z]
      },
      "license": "<SPDX identifier>",
      "source": "<short provenance string>"
    }
  }
}
```

Every field is required. The loader (`ShapeLibrary.__init__`)
validates `schema_version`, refuses unknown versions, and walks the
`shapes` map; missing fields raise `ValueError` with the missing
keys listed.

### Field details

- **`file`** — relative path under the library directory. Must point
  to a file `trimesh` can load (`.stl`, `.obj`, `.ply`, `.glb`).
- **`rest_dimensions`** — the OBB extents in metres, in the same
  axis order as the mesh's local frame. The loader cross-checks these
  against the loaded mesh; large drift triggers a load-time warning.
- **`binding_template.kind`** — must match `BindingKind`:
  `between_two` (exactly 2 markers), `cluster` (≥ 3 markers), or
  `on_marker` (exactly 1 marker). The dataclass post-init validator
  enforces the count.
- **`binding_template.marker_names`** — names from the canonical
  marker set returned by `default_body_segments`. Names that are not
  in the canonical set are loadable, but a tool that auto-detects
  bindings from a live capture will skip the shape.
- **`binding_template.rest_orientation_quat`** — unit quaternion
  (w, x, y, z). Identity is `[1.0, 0.0, 0.0, 0.0]`. The loader
  enforces unit norm to within `1e-6`.
- **`license`** — SPDX identifier. The bundled defaults are all
  `CC0-1.0`. If you ship a non-CC0 mesh, see "Licensing" below.
- **`source`** — short string. For procedural meshes use
  `procedural-low-poly` (matches the existing entries). For meshes
  drawn from external datasets, name the dataset.

## Adding a procedural shape

This is the recommended path for any anatomical shape that does not
require captured data. Procedural shapes are CC0, deterministic, and
regeneratable from source.

1. **Edit `_generate.py`.** Add a `_build_<name>` function that
   returns a `trimesh.Trimesh` sized in metres:

   ```python
   def _build_pelvis() -> trimesh.Trimesh:
       return trimesh.creation.box(extents=(0.30, 0.20, 0.18))
   ```

   Stay with primitives from `trimesh.creation` (`box`, `cylinder`,
   `icosphere`) so the output stays public-domain. The `_ellipsoid`
   helper is a convenient wrapper around scaled icospheres.

2. **Register it in the builder list.** Add `("pelvis",
_build_pelvis)` to the iteration list at the bottom of the file.
   The script writes `<name>.stl` for each entry; the order does
   not matter.

3. **Run the generator.** From the asset directory:

   ```bash
   cd assets/body_part_shapes/default
   python3 _generate.py
   ```

   The script is deterministic (`numpy.random.default_rng(42)`) and
   decimates to ≤ 5 000 triangles. Re-runs on the same trimesh
   version produce byte-identical STL output.

4. **Add an entry to `manifest.json`** with the rest dimensions
   from the generator (the script prints them after each write) and
   a binding template that names markers from the canonical set.

5. **Add an entry to `LICENSES.md`** under the table:

   ```markdown
   | `pelvis.stl` | CC0-1.0 | procedural-low-poly (box) |
   ```

6. **Add a unit test.** A new entry deserves a smoke test under
   `tests/unit/body_part_viz/test_asset_library.py` that asserts
   `ShapeLibrary.default().get("pelvis")` returns a non-empty
   `MeshShape` whose `rest_dimensions` match the manifest within
   `1e-9`.

7. **Open the PR.** Reference the EPIC (#4755) and link the test.

## Adding an externally sourced shape

If you need anatomical fidelity beyond what `trimesh.creation`
primitives can produce — for instance an actual scanned forearm with
muscle bumps — the path is the same except for two things:

1. **The file is not regenerated.** Drop the source mesh into the
   library directory (re-centred on its OBB centroid, in metres,
   with a sane axis convention; see [`mesh_import.md`](mesh_import.md)
   for what the loader does with it). It is now a tracked binary
   asset and any future change requires the same explicit pipeline.

2. **Licensing must allow redistribution.** The default library is
   bundled with the repo under MIT (the umbrella project licence).
   A shape under a more restrictive licence (e.g. CC-BY-NC,
   CC-BY-SA, or anything with a "no derivatives" clause) is **not
   acceptable** — it would make the entire repo non-redistributable.
   CC0, CC-BY (with attribution noted in `LICENSES.md`), and explicit
   MIT/Apache-2.0 grants are fine.

Add the file, then follow steps 4-7 from "Adding a procedural
shape" above. The `LICENSES.md` line should name the upstream
dataset and the licence:

```markdown
| `forearm_scan.stl` | CC-BY-4.0 | <Dataset Name> (https://...) |
```

## Licensing

The repo ships under MIT (see `LICENSE` in the repo root). Asset
licences must be compatible with redistribution.

| OK                       | Notes                                    |
| ------------------------ | ---------------------------------------- |
| CC0-1.0                  | Recommended for procedural shapes.       |
| MIT, Apache-2.0, BSD-2/3 | Fine; note attribution in `LICENSES.md`. |
| CC-BY-4.0                | Fine; note attribution in `LICENSES.md`. |
| Public domain            | Fine.                                    |

| Not OK                  | Why                                                              |
| ----------------------- | ---------------------------------------------------------------- |
| CC-BY-NC-\*             | Forbids commercial use; we cannot guarantee that.                |
| CC-BY-ND                | Forbids derivatives, including the OBB re-centring + decimation. |
| GPL / AGPL              | Viral; would force the entire repo to that licence.              |
| "Free for personal use" | Not a real licence.                                              |

If you are not sure, ask in the PR.

## Procedural-generation script reference

`assets/body_part_shapes/default/_generate.py` is the canonical
reference for what good procedural shapes look like. Highlights:

- All builders return `trimesh.Trimesh` in metres.
- The `_ellipsoid(a, b, c)` helper produces a unit icosphere scaled
  to the given semi-axes; use it instead of hand-rolling spheres.
- `_decimate(mesh, max_triangles)` clamps triangle count via
  `trimesh.simplify_quadric_decimation`. The library budget is
  5 000 triangles per shape; lowering it for individual shapes is
  fine if you never need the silhouette resolution.
- The script writes `<name>.stl` next to itself and prints the OBB
  extents — copy those into the manifest.
- The RNG (`numpy.random.default_rng(42)`) is fixed per process so
  any shape that uses random sampling (none currently do) stays
  deterministic across runs.

## Where to next

- [ADR 0008](../../adr/0008-body-part-viz-toolkit.md) — design rationale
  for the toolkit and the asset-library boundary.
- [`docs/api/body_part_viz.md`](../../api/body_part_viz.md) — the
  `ShapeLibrary` public API.
- `tests/unit/body_part_viz/test_asset_library.py` — the existing
  smoke tests that any new entry should mirror.
