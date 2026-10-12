# Necromatcher Authored Ranges

## Public Bound Definition Procedure

Issue #11235 adds `NativeFitBinding.authored_coordinate_bounds()`. The loader
captures immutable `definition_bytes` using the exact JSON serialization already
used to export and compile its model. This avoids changing exporter ordering.
Extraction never falls back to mutable fit metadata or a different definition.
Bindings manually constructed without captured bytes explicitly fail extraction.

The public MJCF exporter must reproduce the binding's exact XML SHA256. The
captured definition coordinate order must match the compiled public plant order.
Every exported coordinate is a scalar hinge or slide whose type agrees with
the independently compiled public coordinate units, and every joint explicitly
has `limited="false"`. The extractor reads no private SDK/model state and does
not compile another native model.

`coordinate_ranges_deg` is optional. When present it must be a named mapping;
each name must identify a known scalar hinge, and each interval must contain
two finite numeric degree endpoints in nondecreasing order. Boolean/string
endpoints, unknown names, translation coordinates and malformed ranges fail.
Validated intervals convert to radians. Coordinates without a declaration are
explicitly unbounded in compiled native order; no fallback numbers are invented.

## Immutable Result and Job Receipts

`AuthoredCoordinateBounds` contains a copied read-only `named_bounds` mapping,
ordered `unbounded_names`, `definition_sha256`, `xml_sha256`,
`range_source="bound_native_definition.coordinate_ranges_deg"` and
`compiled_limits_enforced=False`. The definition hash covers the captured loader
serialization bytes; the XML hash covers the exact reproduced exported XML.
Authored bounds preserve native order. `to_record()` creates fresh JSON-safe
dictionaries and lists suitable for job receipts; modifying that detached
record cannot change the bound result. Do not call `dataclasses.asdict()` on
the immutable mapping proxy.

These ranges are authored hypotheses. They are not native enforced limits,
measured participant range of motion, or evidence of physical correctness.
Saving or applying them does not qualify anatomy, camera, source time or
historical player reconstruction.
