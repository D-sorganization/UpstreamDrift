# Necromatcher Historical Player Workspace

Necromatcher uses one persistent player/swing/version library for historical
footage, source-bound models, fits and authored driving profiles. The native tile
and React route recall the same configured store. New players are data records
that use these shared workflows.

## Native Appearance Binding

A full-body definition may explicitly contain an `appearance` object using the
`appearance-v1` schema. Binding and authored-range extraction reconstruct the
styled MuJoCo XML and verify its exact registered SHA-256, including appearance.
A definition without that key retains the legacy physics-only export. The native
plant uses the same physical specification; styling does not establish a player
match, contact, or forward-dynamics qualification. No sidecar is discovered
implicitly, and changing either physics or appearance requires a new bound hash.

## Source Review and Modeling

Recall original frames with observed landmarks, compare stored native fits and
translucent proxy shapes, and use explicit scope/camera/model declarations.
Tiger's reviewed matching interval is original frames 0–190; released-hand
follow-through is excluded. Hogan remains within its saved source domain.
Source observations, inferred motion and qualified dynamics remain distinct.

## Authored Replay, Impact and Local Golf

Import a source-bound authored replay, supply explicit impact geometry/sample
assumptions and a wall budget, then run or recall a retained research result.
The checked four-file ZIP keeps the companion receipt with the original
trajectory. **Open in Shot Tracer** retains saved positions and provenance.

**Open Local Research Simulation** uses the authenticated saved post-impact
state. Connect, prepare, arm and submit explicitly through the actual local
session. The new local flight uses local settings; scientific and physical source
time remain unverified. This path supports no external delivery, native avatar
or course feedback. Native review exposes retained samples; web review reuses
the existing flight scene.

## Procedures and Evidence

- [Replay Impact Procedure](../../../docs/development/necromatcher-replay-impact.md)
- [Local Research Golf Procedure](../../../docs/development/necromatcher-local-research-golf.md)
- [Local Research Golf LaTeX Methods](../../../docs/development/necromatcher-local-research-golf-methods.tex)
- [Turnover and Remaining Work](../../../docs/development/necromatcher-turnover.md)

Historical matching and final overall parity acceptance remain tracked in epic
#11232. Local execution acceptance does not establish a qualified Tiger/Hogan
model or engineering release approval.

## Importing Catalog Candidates

Import a `player_models.export_player` sized export into an existing library:

```python
from pathlib import Path
from src.shared.python.workspace import NecromatcherLibrary, import_catalog_model
library = NecromatcherLibrary(Path("/path/to/chosen/necromatcher-library"))
result = import_catalog_model(library, "rory_mcilroy", "/path/to/rory_mcilroy-export")
```

The root must already contain `project.json`. Import retains the hash-checked
XML, visual JSON and receipt. Exact reimports are idempotent; collisions are
rejected. The tile uses this root when `NECROMATCHER_LIBRARY_ROOT` is set.
