# Attributed Club Specifications

The existing `src.shared.python.club_data` package owns both the historical
Excel loader and the optional, attributed catalog. Importing catalog records
does not load PyQt, pandas or openpyxl. Player bag controls and capture binding
are tracked separately in #9905; this contract does not claim those screens exist.

## Identity and Evidence

A build identity includes manufacturer, model, club type/number, release year,
handedness, region and build description. Changing a shaft, length or weighting
configuration should create a distinct build. `catalog_id` is deterministic for
that identity. `revision` identifies the exact claims and notes used, so a future
catalog update need not change an existing capture's evidence.

Every numeric claim identifies its component: head, shaft, grip or assembled
club. It retains original value/unit, evidence status, optional confidence and
source. Sources record a title, method/locator, license or redistribution status,
and optional URL/retrieval time. Manufacturer sources require both URL and an
explicit timezone on retrieval time. Public access does not establish an open
license. No catalog entry receives invented confidence or measured status.

Unknown quantities are absent or explicitly null with `unknown` status. A zero
mass or length is invalid, not a substitute for missing data. The historical
`ClubSpecification` fills defaults and cannot identify which values a user supplied.
`import_legacy_specification` therefore marks all its numeric values `unverified`.
The old loader's behavior remains compatible for existing consumers.

## Physical Consumers

`ClubRecord.physical_value(property_name, component)` converts a single published
or measured claim to SI: meters, kilograms, radians or kg m². Estimates require
`allow_estimates=True`. Suggested and unverified values remain excluded. Missing
data returns `None`; multiple source claims raise an explicit conflict requiring
review. Do not pick the last imported value or average incompatible builds.

MOI and center-of-gravity distance require an axis, reference frame and origin
before physical use. A scalar MOI is not a complete inertia tensor. A marketing
MOI can be retained for review without being accepted by a model. Swing weight
such as `D2` is a descriptive scale, never a mass conversion. Current numeric
bounds catch malformed input; they are not a scientific acceptance envelope.

## Import and Export

`catalog_io.export_json` / `import_json` exchange a versioned document.
`export_csv` / `import_csv` use one row per claim with flat build/property columns
and a JSON source cell. A no-claim row preserves a club whose properties are all
unknown. CSV quoted newlines and commas in notes round-trip unchanged.

Both formats validate the entire input before returning records, reject duplicate
build identities and unknown schema versions, and enforce an 8 MiB input limit.
CSV import also verifies the deterministic build ID and consistent identity/notes
across rows. Import returns records without overwriting a player's library.

## Qualification

`tests/unit/test_club_catalog.py` exercises headless imports, finite/range/unit
checks, unknowns, configuration IDs, provenance, source conflicts, inertia-frame
requirements, explicit estimates, JSON/CSV round trips and legacy defaults.
`tests/unit/test_club_data_loader.py` retains the old loader's compatibility checks.
The public source catalog/update workflow (#9904) and player-facing bag/capture
integration (#9905) remain required before epic #9902 can close.
