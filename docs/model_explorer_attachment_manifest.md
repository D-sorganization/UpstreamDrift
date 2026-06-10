# Model Explorer Attachment Manifests

Model explorer model composition can use an optional sidecar named
`<model>.attachments.json` next to a URDF, MJCF, XML, or OSIM model file. The
sidecar declares semantic mount points above raw link names so tools can prefer
links that are intended for hands, grippers, club grips, or other interfaces.

The checked-in JSON Schema lives at
`src/tools/model_explorer/schemas/attachment_manifest.schema.json`.

## Schema

Each manifest uses `schema_version: 1` and an `attachments` array. Each
attachment entry contains:

- `link_name`: the model link used as the attachment target.
- `role`: a semantic role such as `hand`, `tool-mount`, or `club-grip`.
- `interface_frame`: an `xyz` and `rpy` offset from `link_name` to the actual
  interface frame. The editor uses this frame as the fixed joint origin when a
  declared point is selected.
- `max_payload_kg`: optional non-negative payload limit. The editor warns when
  a requested payload exceeds it.
- `tags`: optional free-form labels for filtering and compatibility hints.

Malformed sidecars are non-fatal. The loader exposes warnings through loaded
model info and still loads the model with an empty attachment declaration set.
Missing sidecars are allowed and produce no warning.

## Human And Arm Example

`human.urdf`:

```xml
<robot name="human">
  <link name="pelvis"/>
  <link name="right_hand"/>
</robot>
```

`human.attachments.json`:

```json
{
  "schema_version": 1,
  "attachments": [
    {
      "link_name": "right_hand",
      "role": "hand",
      "interface_frame": {
        "xyz": [0.0, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0]
      },
      "max_payload_kg": 2.0,
      "tags": ["human", "right", "tool-mount"]
    }
  ]
}
```

`simple_arm.attachments.json` can then declare its wrist flange:

```json
{
  "schema_version": 1,
  "attachments": [
    {
      "link_name": "wrist",
      "role": "tool-mount",
      "interface_frame": {
        "xyz": [0.0, 0.0, 0.08],
        "rpy": [0.0, 0.0, 0.0]
      },
      "max_payload_kg": 1.5,
      "tags": ["robot-arm", "flange"]
    }
  ]
}
```

When composition attaches to `right_hand`, the editor uses the declared
`interface_frame` as the fixed joint origin. If composition targets `pelvis`
while declarations exist only for `right_hand`, the editor emits a warning but
does not block export. If the requested payload is `3.0` kg for the example
above, the editor emits a payload warning because the declared limit is `2.0`.
