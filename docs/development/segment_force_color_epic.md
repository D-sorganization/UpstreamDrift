# Segment Force Color Epic

Status: Implementation in progress under GitHub epic #9833. Current remote main
`ff0effa5a` is merged into `feat/segment-force-colors`. Git access works with the
stale HTTP extraheader cleared per invocation; no global credential changes made.

## GitHub Work Items

- [ ] #9834 Shared Axial Load Contracts and Color Policy
- [ ] #9835 Desktop Force Color Integration and Source Qualification
- [ ] #9836 Web Force Color Streaming and Renderer Parity
- [ ] #9837 Force Color Release Evidence and Protected Merge

## Feasibility

The shared body-part toolkit already renders per-shape line and mesh colors in
Matplotlib and PyQtGraph OpenGL. React renders URDF link materials and golfer
segments. Updating existing colors preserves geometry and avoids additional force
arrows. A shared scalar-to-color policy plus narrow renderer adapters is feasible.

SimulationFrame now carries optional signed segment axial loads. An adapter
must supply axial force in newtons, positive in tension and negative in compression,
with stable segment identifiers and frame alignment. Joint torques, contact-force
magnitudes and motion-only capture are insufficient evidence of internal axial load.
Unsupported sources must report unavailable. Whole-segment color summarizes axial
load at a declared section; it is not a spatial stress field or a tissue safety metric.

## Delivery Backlog

- [x] Shared validated policy: disabled by default, blue tension, red compression,
      neutral zero band, custom opaque RGB colors, independent positive saturation
      limits in newtons, clipping and unavailable-data behavior.
- [x] Frame-aligned load contract and provider capability with explicit provenance.
- [x] Matplotlib and PyQtGraph adapters that recolor existing artists, preserve
      opacity and geometry, and restore base colors without clearing the scene.
- [ ] Reusable desktop toggle, palette/range controls and labeled legend; connect
      the controls and frame loads to applicable animation consumers.
- [x] React parity: shared wire contract, reusable controls and URDF/segment
      material adapter; retain original materials on disable and missing data.
- [ ] Concrete force-source adapters qualified with analytical tension/compression
      fixtures, including sign/frame conventions and unsupported engine behavior.
- [ ] Regression and native renderer checks, user guide, parity registry, SPEC and
      handoff updates; publish epic/children and merge through protected PR checks.

## Acceptance and Engineering Contracts

TDD: record a failing test before each implementation slice. Cover sign reversal,
zero/deadband boundaries, asymmetric ranges, saturation, missing/NaN samples,
nonfinite configuration, frame mismatch, seek/replay, toggle restoration and
unchanged model state. Test actual renderer artists as well as policy outputs.

DbC: reject malformed settings, ambiguous sign conventions and unaligned data;
never turn absent force into zero. LoD: renderers accept colors through a narrow
public method; UI code never traverses renderer internals. DRY: one policy per
runtime, with shared cross-runtime fixtures and no engine-specific color formulas.

Universal means a model-independent capability that interfaces can share. It does
not mean inventing force data for motion-only models or claiming native engines
were qualified without executing their adapters.

## Local Validation Evidence

The web scene exposes reusable force-color controls and accepts a qualified,
synchronized `segmentLoads` prop or optional websocket frame field for URDF and
fallback geometry. MuJoCo supplies qualified rod section reactions using scratch
data. Double and triple pendulums use existing analytical joint reactions.
Their desktop views expose the shared controls. Other native hosts still require
binding and source qualification; unsupported models retain their base colors.
See `docs/user_guide/body_part_viz/force_colors.md` for the adapter contract.

Policy RED: missing `force_colors` import; GREEN: 15 tests. Matplotlib RED:
three missing `set_color` failures; GREEN: 24 renderer tests. OpenGL RED: two
missing `set_color` failures; GREEN: 19 renderer object tests. Frame controller
RED: missing `force_display` import; GREEN: five tests. Offscreen OpenGL object
tests do not qualify GPU raster output. The pinned Tools dependency was restored
from local git objects at `eab74a901a7c8467e1997049a73e2cfd2df74428`.

Command: `python3 -m pytest -o addopts= -q
--confcutdir=tests/unit/body_part_viz tests/unit/body_part_viz`.
Focused runs omit the root fixture bootstrap; full repository CI remains required.

Subsequent validation used the normal root bootstrap: 458 Python tests across
body-part visualization and feature parity passed with PyQt6 selected consistently
(`PYTEST_QT_API=pyqt6`, `QT_API=pyqt6`, `PYQTGRAPH_QT_LIB=PyQt6`). The web
visualization folder passed 67 tests in nine files. Ruff lint/format, changed web
ESLint and TypeScript checking passed. The local Node dependencies were reused
from the existing checkout; remote lockfile CI is still required. Design-manual
governance passes its current structural check but reports the pre-existing
publication release gate `blocked-inventory-required`.

Additional native qualification passes for hanging and inverted double/triple
pendulums, MuJoCo static and centripetal loads, unchanged live physics state,
native scene color restoration, Qt controls and websocket frame alignment.
TypeScript passes after using a library-compatible material lookup. Broad local
mypy reports existing imported-module errors; new policy and settings type errors
were corrected. This is not a claim of a clean repository-wide type check.
