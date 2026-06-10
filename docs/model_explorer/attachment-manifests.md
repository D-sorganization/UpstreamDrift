# Model Explorer Attachment Manifests

Attachment manifests declare semantic mount points for model composition without
editing vendored model files. Place a sidecar next to a loadable model file using
the same stem and the `.attachments.json` suffix:

```text
human.urdf
human.attachments.json
```

The loader reads sidecars for repository, imported, sibling, and statically
configured models that expose a filesystem path. Missing sidecars are normal.
Malformed sidecars are reported on the model info as
`attachment_manifest_warnings` and do not stop model discovery.

## Schema

The checked-in JSON Schema lives at
`src/tools/model_explorer/attachment_manifest.schema.json`.

```json
{
  "schema_version": 1,
  "attachment_points": [
    {
      "name": "right hand tool mount",
      "link_name": "right_hand",
      "role": "tool-mount",
      "interface_frame": {
        "xyz": [0.02, -0.01, 0.0],
        "rpy": [0.0, 1.5707963268, 0.0]
      },
      "max_payload_kg": 2.5,
      "tags": ["right", "hand", "tool"]
    }
  ]
}
```

`interface_frame.xyz` and `interface_frame.rpy` are applied automatically as the
default attachment joint origin when a user selects the declared mount point.
`max_payload_kg` is advisory: attach flows warn when a selected payload exceeds
the declaration, but the model can still be composed with an explicit override
in higher-level validation flows.

## Human Plus Tool Example

For a human model with a `right_hand` link and a golf-club model payload of
`0.42 kg`, declare:

```json
{
  "schema_version": 1,
  "attachment_points": [
    {
      "name": "right hand grip",
      "link_name": "right_hand",
      "role": "hand",
      "interface_frame": {
        "xyz": [0.0, 0.0, 0.08],
        "rpy": [0.0, 0.0, 0.0]
      },
      "max_payload_kg": 1.0,
      "tags": ["right", "hand", "golf"]
    }
  ]
}
```

When selected in the attachment dialog, the parent link is `right_hand`, the
joint origin is prefilled from `interface_frame`, and payload checks can warn if
a heavier tool is selected.
