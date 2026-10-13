# Sharing the Shot Pattern Study

## Six Facebook Images

The release staging set includes one overhead flight comparison and one dispersion
comparison per club. The files reuse accepted publication graphics without
regeneration. Their hashes are recorded in
[Artifact Delivery Evidence](validation/artifact_delivery_v2/README.md).

| Club | Dispersion Image | Overhead Image |
| --- | --- | --- |
| Driver | `social__driver__lean0_curve1_rotation2_combined2__density.png` | `social__driver__lean0_curve1_rotation2_combined2__overhead.png` |
| Seven Iron | `social__seven_iron__lean0_curve1_rotation2_combined2__density.png` | `social__seven_iron__lean0_curve1_rotation2_combined2__overhead.png` |
| Pitching Wedge | `social__pitching_wedge__lean0_curve1_rotation2_combined2__density.png` | `social__pitching_wedge__lean0_curve1_rotation2_combined2__overhead.png` |

These six images select zero shaft lean, curve strength ×1, shaft-twist standard
deviation 2°, and the combined2 profile: path SD 0.5°, speed SD 2 mph, and strike
SD 0.25 in in each face-plane axis. Shaft-twist SD is not delivered-face SD.
Each pattern attempts 10,000 shots. Exact supported counts and displayed samples
are labelled in the images. The illustrated report’s six primary single-condition
plates use shaft-twist SD 1°; these social selections use SD 2°. The illustrated
report also shows selected paired SD, curve-strength, and across-lean comparisons.
Do not read the social images and the main plates as the same input scenario.

## Suggested Facebook Caption

> Does shaping a golf shot improve accuracy? We compared straight, draw, and fade
> patterns using a geometric ball-flight simulation for driver, seven iron, and
> pitching wedge. These charts show one 10,000-shot-per-pattern scenario with
> variation in shaft rotation, path, speed, and strike. There was no universal
> winning shape: results depended on the club and assumed delivery variability.
> Nominal landing targets were matched, but individual shots still differed in
> carry. This is a model study, not measured golfer performance. See the report
> for assumptions, conditional strokes-gained estimates, and the other scenarios.

## Report and Complete Data

The reader-facing primary report is the accepted illustrated research PDF, staged
locally as `ILLUSTRATED_REPORT.pdf` (33 pages, 13 source-bound figures). Its source
package is `illustrated-report-source.zip`; the source and PDF hashes and fresh
source-archive restoration are recorded in
[staging inventory](validation/artifact_delivery_v3/release_staging_inventory_v3.json)
and [fresh source restoration receipt](validation/artifact_delivery_v3/illustrated_report_source_restore_v3.json). The user-facing title is
“Shot Pattern Analysis - Illustrated Research Report.”

`REPORT.md` remains the detailed text and numerical reference. The staged
`report__REPORT.pdf` is the earlier 30-page text-only historical PDF; retain it
under that explicit role and do not present it as the illustrated primary report.
The standalone local Tectonic build does not establish success in the built-in
LaTeX editor, which remains blocked under #12120.

The main illustrated report’s single-condition plates use shaft-twist SD 1°; the
six social graphics above use SD 2°. Paired report plates show their stated
contrasts. Keep these scenarios labelled distinctly. The [Artifact Delivery
Plan](review_followups/ARTIFACT_DELIVERY_PLAN.md) explains archive names, hashes,
restoration, and the repaired supplemental directory mapping. Current staging and
offline restoration are verified locally; remote publication remains pending and
must be confirmed from its own delivery receipt.
