# Segment Force Color Epic

Status: Implementation in progress under GitHub epic #9833 and PR #9840. Remote main
`11036968c` is merged into `feat/segment-force-colors`. Git access works with the
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
- [x] Reusable desktop toggle, palette/range controls and labeled legend; connect
      the controls and frame loads to applicable animation consumers.
- [x] React parity: shared wire contract, reusable controls and URDF/segment
      material adapter; retain original materials on disable and missing data.
- [x] Concrete force-source adapters qualified with analytical tension/compression
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
Their desktop views expose the shared controls. Pinocchio and Drake expose the
same controls and a shared session for explicitly bound, time-aligned load frames.
Native GUI tests qualify settings, redraw, binding and restoration; automatic force
inference is unavailable in those hosts. Unsupported models retain base colors.
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

The combined focused suite now passes 575 tests using the normal root bootstrap:
body-part visualization, API force streaming/WebSocket regressions, feature parity,
and C3D force colors. The web visualization suite passes 68 tests; TypeScript and
changed-file ESLint pass. Native MeshCat command transport and a graphics-free
subprocess import test pass. MuJoCo raster verification confirms blue/red output
and pixel-exact disabled restoration. The broader C3D directory has one existing
invalid-CSV error-text mismatch outside this feature.

CI cycle 1 fixes remove a nested series-storage access, defer the optional plotting
import, split the rendering method, and remove a redundant websocket cast. Cycle 2
adds explicit unit-suite markers and incorporates current main after the shallow
CI diff incorrectly classified main's new notebook test as deleted. No gate or
baseline was weakened. Current-head CI and protected merge remain required.

CI cycle 3 updates the companion catalog expectation and regenerates the shared
divergence inventory. Those modules pass 34 tests; the preceding full unit gate
passed 14,431 tests with only those two bookkeeping failures.

The shared MeshCat session began with a missing-class RED test. GREEN covers
stale-frame restoration, toggling and model replacement. Native Pinocchio 4.1.0
and Drake 1.56.0 GUI tests pass in an isolated Linux environment, including actual
settings dialogs, redraw and explicit frame bindings. The native test caught a
duplicate legacy Pinocchio mixin; the active host's redraw hook is now exercised.
OpenSim currently has result plots rather than an animated 3D scene. Both MeshCat
hosts require a qualified caller-supplied section-force source and explicit scene
bindings; Drake's sampled reaction output must not be treated as current by default.

## Interface Coverage Matrix

| Interface                           | Display integration                                                                            | Axial-load source and qualification                                                                                                                                |
| ----------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Shared Matplotlib / PyQtGraph       | Existing artists recolored through a public optional renderer capability; reusable Qt controls | Explicit frame-aligned series; native artist tests, OpenGL object tests                                                                                            |
| Double / triple pendulum animations | Shared View action, flat and tapered segments                                                  | Analytical transmitted joint reactions; hanging/inverted sign fixtures                                                                                             |
| MuJoCo native / MeshCat             | Shared controls, native scene RGB updates and MeshCat leaf bindings                            | Qualified capsule/cylinder sections sampled on scratch data; static/dynamic and raster evidence                                                                    |
| Web Scene3D / URDF                  | Shared collapsed controls, owned material overrides, optional WebSocket payload                | Qualified provider or supplied frame with matching clock; TypeScript/Python oracle parity                                                                          |
| C3D / Simscape user segments        | Shared controls and explicit load-ID/segment-index bindings                                    | Supplied series must match point times; motion alone is unavailable                                                                                                |
| Pinocchio MeshCat                   | View action and shared model/clock session                                                     | Explicit leaf bindings and caller-supplied section loads; real GUI integration tested with Pinocchio 4.1.0                                                         |
| Drake MeshCat                       | View action and the same shared model/clock session                                            | Explicit leaf bindings and caller-supplied aligned loads; real GUI integration tested with Drake 1.56.0; sampled reactions are not automatically relabeled current |
| OpenSim desktop                     | Current interface contains result plots, not animated segment geometry                         | Future 3D consumers can use the shared contracts; no native OpenSim section-force qualification claimed                                                            |
| Other/future interfaces             | Reusable Python policy/renderer capability or TypeScript material adapter                      | Host must provide stable bindings and qualified synchronous axial data; API availability does not qualify its physics                                              |
