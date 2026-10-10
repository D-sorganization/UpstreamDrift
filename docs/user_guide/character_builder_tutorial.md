# Build, Dress and Export a Golfer

An end-to-end walk-through of the spec-native Character Builder (CMB
epic, #11652-#11658): build a golfer from a preset, dress them with an
appearance document bound to that spec, swap the club, export every engine
format, and load the result into MuJoCo. Every command below is the exact
Python used to verify this tutorial; sha256 values are truncated to their
first 8 hex characters and are what this commit's verification run
produced for these exact inputs.

## Prerequisites

Run from the repository root. One module deep in the import chain
(`simulation_backends.wrench_extractor`) imports `bunkershot3d` as a
top-level package rather than `src.bunkershot3d`, so put both the
repository root and `src/` on the path:

```bash
export PYTHONPATH="$PWD:$PWD/src"
```

## Step 1: Build a Golfer From a Preset With an Override

The spec-native builder (`src/shared/python/humanoid_character_builder/spec_export.py`)
compiles a preset plus overrides to a validated `full-body-v1` document and
hashes it:

```python
from src.shared.python.humanoid_character_builder.spec_export import (
    compile_character,
    build_summary,
)

# "tour_average_male" ships at 84.0 kg (see character_presets.md); override
# mass_kg to 90.0 and keep every other preset value (driver, stature 1.80 m).
compiled = compile_character("tour_average_male", {"mass_kg": 90.0})
print(compiled.preset_id)     # tour_average_male
print(compiled.sha256[:8])    # c0e2ca45
print(build_summary(compiled))
```

This produced `preset_id="tour_average_male"`, spec hash `c0e2ca45...`, and
a summary of 25 bodies, 24 joints, 44 coordinates and 86.58 kg total mass
(De Leva 1996 male table plus Rajagopal lower limbs — unqualified geometry,
not a validated subject; see **Limitations** below). The total is below the
requested 90.0 kg because the lower limbs do not yet scale with `mass_kg`
(#12199).

**Web/API equivalent:** `GET /api/character-builder/presets` lists
`tour_average_male` with its description and limitations;
`POST /api/character-builder/build` with
`{"preset": "tour_average_male", "mass_kg": 90.0, ...}` returns the same
summary (the web `CharacterSpecPanel` sends every slider value alongside
the preset id, so an unedited slider simply repeats the preset's own
value).

## Step 2: Dress the Golfer

Build an `appearance-v1` document bound to the compiled spec's hash, then
write it beside the spec file using the same stem (`appearance_path_for`):

```python
import json
from src.api.services.character_appearance_service import (
    build_appearance,
    appearance_sidecar_filename,
)

choices = {
    "skin_tone": "skin_tan",
    "clothing": "golf_polo_trousers",
    "club_finish": "chrome",
}
appearance_doc = build_appearance(choices, compiled.sha256)
print(appearance_doc["spec_sha256"][:8])  # c0e2ca45 — bound to Step 1's spec

sidecar_name = appearance_sidecar_filename(
    appearance_doc, preset_id=compiled.preset_id, spec_sha256=compiled.sha256
)
print(sidecar_name)  # tour_average_male_c0e2ca45.appearance.json

with open(sidecar_name, "w", encoding="utf-8") as f:
    f.write(json.dumps(appearance_doc, indent=2) + "\n")
```

`build_appearance` fails closed: an unknown skin tone, clothing preset,
club finish or ground material name raises `ValueError` rather than
silently falling back to a default. Nothing here can change
`compiled.sha256` — `physics_spec_sha256` only ever hashes the spec with
`visual_hints` and `appearance` keys stripped.

**Web/API equivalent:** `GET /api/character-builder/appearance/library`
lists every pickable name; `POST /api/character-builder/appearance/export`
with the same picks plus `"character": {"preset": "tour_average_male",
"mass_kg": 90.0, ...}` has the server recompile that character, stamp its
hash onto the document the same way, and name the download from the
preset id and hash. The web Appearance panel's Export Appearance button
sends exactly this request.

## Step 3: Swap the Club

Compile the same preset again with `club="iron7"` instead of the preset's
default `driver`:

```python
compiled_iron = compile_character(
    "tour_average_male", {"mass_kg": 90.0, "club": "iron7"}
)
print(compiled_iron.sha256[:8])              # 9ac415e2
print(compiled.sha256 != compiled_iron.sha256)  # True
```

The spec hash changed (`c0e2ca45...` to `9ac415e2...`) because the club is
part of the physics spec. The Step 2 appearance document is still a valid
`appearance-v1` document — `document_from_dict` on it still passes — but
its `spec_sha256` now names a spec that no longer exists; rebind it to the
new hash before relying on the pairing:

```python
rebound = build_appearance(choices, compiled_iron.sha256)
print(rebound["spec_sha256"][:8])  # 9ac415e2
print(
    appearance_sidecar_filename(
        rebound, preset_id=compiled_iron.preset_id, spec_sha256=compiled_iron.sha256
    )
)  # tour_average_male_9ac415e2.appearance.json
```

**Web/API equivalent:** re-`POST /api/character-builder/build` with
`"club": "iron7"`; the response's `spec_sha256` changes and
`AppearancePanel`'s bound export (Step 2) picks up the new hash on its next
export because it always sends the current `CharacterSpecPanel` state as
`character`.

## Step 4: Export the Spec, URDF, MJCF and OpenSim

`export_character` shares one exporter per format with the API and the
desktop tool:

```python
from src.shared.python.humanoid_character_builder.spec_export import export_character

for fmt in ("spec", "urdf", "mjcf", "osim"):
    text, media_type, filename = export_character(compiled, fmt)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(fmt, filename, len(text), media_type)
```

For the driver character from Step 1 this wrote:

| Format | Filename                          | Bytes | Media Type         |
| ------ | --------------------------------- | ----- | ------------------ |
| `spec` | `tour_average_male_c0e2ca45.json` | 85471 | `application/json` |
| `urdf` | `tour_average_male_c0e2ca45.urdf` | 57438 | `text/xml`         |
| `mjcf` | `tour_average_male_c0e2ca45.xml`  | 18287 | `text/xml`         |
| `osim` | `tour_average_male_c0e2ca45.osim` | 88889 | `text/xml`         |

**Web/API equivalent:** `POST /api/character-builder/export/{fmt}` for
`fmt` in `spec`, `urdf`, `mjcf`, `osim` — the Export buttons in
`CharacterSpecPanel`.

### Legacy Mesh-URDF CLI

The original mesh/URDF builder (a separate, older pipeline with its own
built-in presets such as `golfer_pro` — not the spec-native presets above,
and with no club or stature/mass-override arguments) is still available as
a module CLI:

```bash
python3 -m src.shared.python.humanoid_character_builder build \
    --preset golfer_pro --output golfer_pro_humanoid.urdf
```

This wrote a 36960-byte URDF and printed `Successfully built character and
saved to golfer_pro_humanoid.urdf`. Use the spec-native path above for club
swapping, appearance binding or engine export beyond URDF.

## Step 5: Simulate (Requires MuJoCo)

This step needs the `mujoco` package, which is not installed on every
host; it was not run as part of this tutorial's verification. Loading the
Step 4 MJCF follows the same pattern already documented in
[Character Builder Quickstart](character_builder_quickstart.md) and used
in `tests/unit/engines/mujoco/test_continuous_torque_simulate.py`:

```python
import mujoco

model = mujoco.MjModel.from_xml_path("tour_average_male_c0e2ca45.xml")
data = mujoco.MjData(model)

for _ in range(100):
    mujoco.mj_step(model, data)

print(data.qpos[:6])
```

## Limitations

The compiled spec is unqualified geometry: validated for structure and
loadability, not against a measured subject. Appearance is purely visual
and never influences `compiled.sha256` or any dynamics. See
[Character Builder](../help/character_builder.md) for the full reference
and [Character Presets](character_presets.md) for every shipped preset's
provenance.

## See Also

- [Character Builder](../help/character_builder.md)
- [Character Builder Quickstart](character_builder_quickstart.md)
- [Character Presets](character_presets.md)
