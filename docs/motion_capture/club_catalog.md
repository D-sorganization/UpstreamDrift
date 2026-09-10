# Attributed Club Specifications

The existing `src.shared.python.club_data` package owns both the historical
Excel loader and the optional, attributed catalog. Importing catalog records
does not load PyQt, pandas or openpyxl. Capture Rig's **My Clubs** entry opens
the player bag; the same entry is available from **Library**.

## Record a Club and Assign It to a Swing

1. Open **My Clubs** in the capture header, or select a capture in **Library**
   and choose **My Clubs** there. The dialog identifies the current selection.
2. Choose **Add from Catalog** to search a particular build and inspect its
   linked sources. The starter catalog is deliberately small and incomplete.
   Choose **Add Custom** for any other club; type and number may remain partial.
3. Give it a recognizable name. For playing length and head mass, choose
   **Measured** or **Estimated** only when you have that information. Unit
   changes preserve the canonical quantity. **Unknown** explicitly suppresses
   an unsuitable nominal value; **Use Original** restores the source value.
4. Add notes about the shaft, grip, fitting or measurement method, then **Save**.
   Cancel closes the draft without writing. Other attributed properties remain
   intact even though this editor does not expose every advanced property.
5. Select the saved club and choose **Assign to This Capture**. The confirmation
   names the capture and its ID. Library details show its selected club revision.
   Without an open capture, the bag remains editable and assignment explains
   how to continue. **Help** describes the full workflow inside the app.

Double-click a bag entry to edit it. **Archive / Restore** keeps old clubs and
their capture history; **Show Archived** reveals them. If another dialog changed
the bag, saving stops with **Reload Bag** guidance instead of replacing its edits.

## Capture History and Model Context

The library root owns `player_clubs.json`. Each capture owns `capture_club.json`,
bound to its portable capture ID and a digest of the full selected club record.
Earlier selections remain in `equipment_revisions`. Editing a catalog or bag
does not rewrite these snapshots. Assign again to explicitly change a capture's
current selection. An editable capture copy receives its own identity and a
rebound copy of the selected club; source evidence and old analyses stay intact.

Model fit reports retain the exact equipment selection, club number, eligible
SI length/head mass and reasons for withheld values in their provenance. The
equipment file is hashed with the other inputs. Current articulated body models
observe hands and arms, not a club segment: their reports explicitly record no
applied equipment constraints. Club length never replaces arm length. Existing
reports remain unchanged after reassignment; rerun analysis when new context
should be attached. Club dimensions alone cannot calibrate cameras.

Legacy captures remain unassigned. Corrupt equipment or mismatched capture IDs
produce an actionable error and are not silently replaced. The bag's revision
check protects stale dialogs; it is not a database transaction across concurrent
processes. Keep one active bag editor when working on shared network storage.

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
Player workflow tests cover editing, persistence, assignment, lineage, source
inspection, stale dialogs and saved model context. Wizard integration remains
part of #9905/#9906 before epic #9902 can close.

## Offline Examples and Contributor Review

`catalog_sources.load_public_catalog()` loads three partial builds from the
packaged `public_clubs.json`. Numeric facts were transcribed from the
[Titleist T150 2023 specification table](https://www.titleist.com/golf-clubs/irons/t150-2023)
and [PING G440 MAX HL specifications](https://ping.com/en-us/golf-clubs/drivers/g440-max-hl-driver).
Each claim retains its URL, retrieval time, table locator, nominal status and
unverified redistribution status. These are illustrative source records, not a
complete club database or measurements of a player's equipment.

1. Identify the exact year, club number, handedness/region and build configuration.
   Do not merge specifications for similarly named products from different years.
2. Inspect the original manufacturer table and its footnotes. Preserve missing
   quantities. A shaft weight may describe an uncut shaft; a lie angle may be an
   average across adjustable settings. Neither establishes this player's build.
3. Record original units and field-level attribution. Review redistribution terms;
   public access alone does not establish an open license. Store factual values
   and source links, not copied product descriptions, images or entire source tables.
4. Edit a candidate versioned JSON file, then run
   `python3 -m scripts.review_club_catalog candidate.json`. The command validates
   offline and prints a sorted review diff with full before/after claims and revision
   hashes. With no argument it validates the packaged catalog. It writes no files.
5. Review changed sources/builds and unknown fields, update the relevant tests and
   submit the candidate through normal PR review. Replace the packaged JSON only
   after that review; existing capture snapshots and player overrides stay separate.

`with_player_overrides` reapplies explicitly attributed player measurements to any
catalog revision without mutating the catalog or override records. Unknown overrides
can deliberately suppress an unsuitable nominal value. `summarize_record` displays
missing values as **Unknown** and lists conflicts instead of choosing a winner.

## Local Responsiveness Check

On the Windows/Python3.13 development host, an already-running Qt application
opened a bag of1,000 catalog-backed clubs (3.47MB) in157ms and filtered it in1.3ms.
This is a single local smoke measurement, excluding Python startup and network
storage latency. It does not establish fleet-wide performance. The measured bag
path does not justify a Rust rewrite; retain the existing shared Qt/unit components
and profile actual playback or inference delays before changing those backends.
