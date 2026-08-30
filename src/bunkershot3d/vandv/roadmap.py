"""The ledger's contents: the seven measurements, and what each would buy.

This is the executable half of "validation is 0 of 4 and nothing is
measured" (issue #8616).  The statement of the gap lives in
:mod:`.credibility`; the statement of *how to close it* lives here, as
data rather than prose, so that :meth:`~.ledger.ValidationLedger.assessment`
can derive the score from it and the two cannot drift apart.

How to read the seven specs
---------------------------

Three are bench work on the sand itself and need no golfer, no bunker and
no camera.  One is a population survey -- the same bench work, repeated
across enough bunkers that its spread is a population spread rather than a
repeatability spread.  One is a divot cast, which is the only cheap way to
constrain the accelerated mass that PR #9237 showed spans a factor of 2.4
between the prismatic rule and the momentum-and-energy consistent
reduction.  Two need a rig.

The order is not the order of importance; it is the order of what is
reachable.  :meth:`~.ledger.ValidationLedger.leverage_ranking` computes
that, and the answer today is the angle of repose: it is the only
measurement on the list that touches the factor sitting at zero, and the
only one that can be made without a swing.

Why repose flips no provenance basis
------------------------------------

The angle of repose is a **system response**, not a material constant.
Reading a friction angle straight off a repose cone is the substitution
issue #7999 already caught once here, in a new costume, and PR #9238's
distinction has to survive intact: ``CONVENTION`` -- a constant fitted to
a declared, *simulated* target -- is not an upgrade over
``BORROWED_ANALOGUE``, and neither of them is ``MEASURED``.  Repose earns
a **validation** level because the model predicts it and the comparison is
attributable; it earns no provenance flip, because nothing was measured
about the constant itself.  The drained shear box is the measurement that
flips ``friction_angle_deg``.
"""

from __future__ import annotations

from types import MappingProxyType

from .ledger import (
    AcceptanceCriterion,
    Blocker,
    CredibilityFactor,
    EffortClass,
    LedgerEntry,
    LevelStep,
    MeasurementSpec,
    ValidationLedger,
)
from .measurement import MeasurementRegister

__all__ = [
    "MEASUREMENT_SPECS",
    "VALIDATION_LEDGER",
    "leverage_table_markdown",
    "measurement_spec_table_markdown",
    "roadmap_table_markdown",
]

_SPECS: tuple[MeasurementSpec, ...] = (
    MeasurementSpec(
        key="bunker_sand_angle_of_repose_deg",
        quantity="angle_of_repose_deg",
        unit="deg",
        conditions=(
            "Poured-cone or lifted-cylinder repose on the as-played sand from "
            "a named bunker, at the played moisture state and the played "
            "compaction, with pour height, funnel diameter and bed diameter "
            "recorded because the angle depends on all three. At least five "
            "repeats of each condition, and the sand returned to a declared "
            "state between repeats rather than re-poured onto its own cone."
        ),
        instrument_class=(
            "Bench repose apparatus with a photogrammetric or laser profile of "
            "the cone flank; the angle read from a fitted flank line, not from "
            "a protractor held against a photograph."
        ),
        acceptance=AcceptanceCriterion(
            statement=(
                "Reported as a mean flank angle with an expanded uncertainty "
                "(k = 2) no worse than 5% relative, over at least five repeats "
                "of one sand and one pour condition. The comparison is then "
                "formed under ASME V&V 20 as |E| = |S - D| against u_val, and "
                "it counts only if u_val is small enough to distinguish the "
                "model's prediction from the borrowed 34 deg it currently "
                "uses. A comparison that cannot tell those apart is "
                "noise-limited, and a noise-limited comparison carries no "
                "information about model error -- which is exactly the "
                "position the package is in today. One further condition, "
                "and it is the one that makes this measurement admissible at "
                "all: the apparatus geometry measured must be the apparatus "
                "geometry the model is run for. Repose is famously "
                "method-dependent -- three methods on one powder have "
                "produced inferred rolling frictions spanning 300x -- so an "
                "angle measured on one apparatus and compared against a "
                "model of another is not a comparison. Matched forward, it "
                "is; inverted to a constant, it never is."
            ),
            min_samples=5,
            max_relative_expanded_uncertainty=0.05,
        ),
        effort=EffortClass.BENCH_DAY,
        provenance_keys=(),
        note=(
            "Flips nothing to MEASURED on purpose, and the credibility "
            "statement's standing objection to repose is the reason. That "
            "objection -- three repose methods on one powder giving rolling "
            "frictions a factor of 300 apart -- is an objection to *inverting* "
            "an angle into a constant, and it is correct. This spec does not "
            "invert. Repose is used forward, as a system response the model "
            "predicts on a declared apparatus, compared against the same "
            "apparatus measured; the friction angle stays where it is until "
            "the shear box measures it. Its value is that it is the only "
            "quantity this model predicts that can be measured on a bench, "
            "with no club delivery to go wrong, so the comparison error is "
            "attributable to the model rather than shared with an unmeasured "
            "boundary condition."
        ),
    ),
    MeasurementSpec(
        key="bunker_sand_bulk_density_kg_m3",
        quantity="bulk_density_kg_m3",
        unit="kg/m^3",
        conditions=(
            "Loose and dense index bulk density (ASTM D4254 and D4253) and "
            "particle density by pycnometer (ASTM D854) on sand drawn from a "
            "named bunker, with gravimetric moisture content (ASTM D2216) on "
            "the same sample, at the played moisture state. Three independent "
            "sub-samples, drawn from the face, the floor and the lip, because "
            "a bunker is not one material."
        ),
        instrument_class=(
            "Laboratory balance readable to 0.01 g, a calibrated mould of "
            "known volume, a pycnometer, and a drying oven; volume traceable "
            "to a dimensional calibration, not to the mould's nameplate."
        ),
        acceptance=AcceptanceCriterion(
            statement=(
                "Loose and dense index densities and the particle density all "
                "reported with an expanded uncertainty (k = 2) no worse than "
                "2% relative, over at least three sub-samples. Two percent is "
                "not arbitrary: the packing fraction enters the constitutive "
                "fit and the divot mass linearly, so anything looser than that "
                "is swamped by the 2.4x accelerated-mass band PR #9237 "
                "reported, and the measurement buys nothing."
            ),
            min_samples=3,
            max_relative_expanded_uncertainty=0.02,
        ),
        effort=EffortClass.BENCH_HOUR,
        provenance_keys=("moisture", "packing", "particle_density_kg_m3"),
        note=(
            "The cheapest thing on this list and the only one that retires "
            "three borrowed constants at once. All three currently trace to "
            "the Quikrete medium-sand analogue through "
            "bunkershot3d.sand.provenance, which is a hardware-store product, "
            "not a bunker."
        ),
    ),
    MeasurementSpec(
        key="bunker_sand_drained_friction_angle_deg",
        quantity="drained_friction_angle_deg",
        unit="deg",
        conditions=(
            "Direct shear (ASTM D3080) on the as-played sand in a 60 mm box, "
            "at three normal stresses bracketing the sole contact stress the "
            "solver reports for a greenside delivery, at the measured packing "
            "and moisture. Peak and residual envelopes reported separately, "
            "because the model uses one number and the material has two."
        ),
        instrument_class=(
            "Direct shear box with a load cell and a displacement transducer, "
            "normal stress traceable to a calibrated load train and shear rate "
            "slow enough to stay drained."
        ),
        acceptance=AcceptanceCriterion(
            statement=(
                "A Mohr-Coulomb envelope fitted over at least three normal "
                "stresses, with the friction angle reported to an expanded "
                "uncertainty (k = 2) no worse than 5% relative and the "
                "apparent cohesion reported rather than assumed zero. The "
                "measurement counts only if the stress range brackets the "
                "solver's reported sole contact stress; a shear box run at "
                "stresses the club never applies measures a different "
                "material state."
            ),
            min_samples=3,
            max_relative_expanded_uncertainty=0.05,
        ),
        effort=EffortClass.BENCH_DAY,
        provenance_keys=("friction_angle_deg",),
        note=(
            "This is the measurement that moves friction_angle_deg to "
            "MEASURED. PR #9238 moved it from BORROWED_ANALOGUE to CONVENTION "
            "by fitting to a simulated target; that was explicitly a lateral "
            "move, and only a shear box on real bunker sand is not."
        ),
    ),
    MeasurementSpec(
        key="bunker_sand_population_survey",
        quantity="bulk_density_kg_m3",
        unit="kg/m^3",
        conditions=(
            "The bench measurements repeated on sand drawn from at least ten "
            "bunkers across at least three courses and two sand suppliers, so "
            "that the reported spread is a population spread and not a "
            "repeatability spread. Each entry carries its course, its bunker, "
            "its supplier where known, and the days since the bunker was last "
            "raked or topped up."
        ),
        instrument_class=(
            "The same balance, mould, pycnometer and oven as the single-bunker "
            "bench work, with one operator or a documented inter-operator "
            "reproducibility check."
        ),
        acceptance=AcceptanceCriterion(
            statement=(
                "At least thirty sub-samples spanning at least ten bunkers, "
                "each within 10% expanded uncertainty (k = 2), reported as a "
                "distribution rather than a mean. A mean is what the package "
                "already has; what it lacks is the spread to propagate, which "
                "is why results-uncertainty cannot move without this even "
                "though one well-characterised sand would move input "
                "pedigree."
            ),
            min_samples=30,
            max_relative_expanded_uncertainty=0.10,
        ),
        effort=EffortClass.FIELD_SESSION,
        provenance_keys=(),
        note=(
            "Flips no basis: it does not measure a new property, it measures "
            "how much the properties vary. That is the input distribution the "
            "uncertainty budget has to propagate before u_num stops being the "
            "only thing in it."
        ),
    ),
    MeasurementSpec(
        key="splash_shot_divot_cast_volume_m3",
        quantity="divot_cavity_volume_m3",
        unit="m^3",
        conditions=(
            "A negative cast of the cavity left by a greenside splash shot, "
            "taken in the bunker immediately after the stroke, on sand whose "
            "bulk density was measured in the same session. Club delivery "
            "recorded on the same stroke, so the comparison has a measured "
            "input and not only a measured output. At least ten shots by at "
            "least three players, entry distance recorded per shot."
        ),
        instrument_class=(
            "Dental-stone or expanding-foam negative cast with a "
            "water-displacement or structured-light volume readout resolving "
            "1 mm on the cavity floor, plus an optical or radar delivery "
            "measurement synchronised to the stroke."
        ),
        acceptance=AcceptanceCriterion(
            statement=(
                "Cavity volume to an expanded uncertainty (k = 2) no worse "
                "than 10% relative over at least ten shots. Multiplied by the "
                "measured bulk density it must resolve the accelerated mass "
                "well enough to separate the prismatic rule from the "
                "P^2 / (2T) reduction, which PR #9237 showed differ by a "
                "factor of 2.4. A cast that cannot tell those two apart "
                "measures the sand's angle of repose after the fact and "
                "nothing else."
            ),
            min_samples=10,
            max_relative_expanded_uncertainty=0.10,
        ),
        effort=EffortClass.FIELD_SESSION,
        provenance_keys=(),
        note=(
            "Cheap in equipment, expensive in access and repeatability. It is "
            "deliberately placed above the bench work rather than beside it: "
            "without a measured density the cast gives a volume and not a "
            "mass, and without a measured delivery the comparison error "
            "cannot be attributed to the model rather than to the swing."
        ),
    ),
    MeasurementSpec(
        key="ejecta_launch_high_speed_video",
        quantity="ejecta_launch_speed_m_s",
        unit="m/s",
        conditions=(
            "Two synchronised cameras at 5000 fps or faster over a calibrated "
            "volume in a real bunker, on the same strokes the divot casts are "
            "taken from. Ejecta speed and elevation angle reduced by streak or "
            "particle tracking; ball launch speed, launch angle and spin "
            "recorded on the same stroke, at a delivery in the 20-27 m/s band "
            "the tool is designed for."
        ),
        instrument_class=(
            "Stereo high-speed video with a calibrated target volume and "
            "per-frame timing traceable to the camera clock rather than to the "
            "nominal shutter setting, plus a launch monitor on the ball."
        ),
        acceptance=AcceptanceCriterion(
            statement=(
                "Ejecta launch speed and angle, and ball launch speed, angle "
                "and spin, each to an expanded uncertainty (k = 2) no worse "
                "than 8% relative over at least ten strokes. This is the "
                "measurement that touches the launch-side quantities issue "
                "#8616 recorded as having no published value anywhere; the "
                "acceptance bar is set by what it takes to close an energy "
                "budget, not by what a camera can resolve."
            ),
            min_samples=10,
            max_relative_expanded_uncertainty=0.08,
        ),
        effort=EffortClass.INSTRUMENTED_RIG,
        provenance_keys=(),
        note=(
            "Ranked at zero leverage today, which is a statement about order "
            "and not about worth. It buys validation level 3, and validation "
            "is at 0: a rig hired before the bench work produces a comparison "
            "whose error cannot be attributed to the model, because the sand "
            "it was made in was never characterised."
        ),
    ),
    MeasurementSpec(
        key="clubhead_delivery_shaft_strain",
        quantity="clubhead_wrench_force_n",
        unit="N",
        conditions=(
            "A strain-gauged shaft on a wedge played into a real bunker at a "
            "20-27 m/s delivery, sampled at 10 kHz or faster, with the "
            "six-component wrench reconstructed at the sole reference point "
            "the solver reports about, and the sand characterised in the same "
            "session. Address-state and delivered-state sole references "
            "recorded separately, since they differ through the stroke."
        ),
        instrument_class=(
            "Full-bridge shaft strain gauges with a static and dynamic wrench "
            "calibration, plus a synchronised delivery measurement (optical or "
            "IMU) to supply the boundary condition the solver is given."
        ),
        acceptance=AcceptanceCriterion(
            statement=(
                "The reconstructed wrench reported to an expanded uncertainty "
                "(k = 2) no worse than 15% relative over at least twenty "
                "strokes, with the calibration residual reported separately "
                "from the stroke-to-stroke spread. Fifteen percent is loose on "
                "purpose: the model's own lambda is known only to within a "
                "factor of nearly three, so a wrench measured to 15% still "
                "discriminates, and demanding better would price the "
                "measurement out for no gain."
            ),
            min_samples=20,
            max_relative_expanded_uncertainty=0.15,
        ),
        effort=EffortClass.INSTRUMENTED_RIG,
        provenance_keys=(),
        note=(
            "The only measurement that observes the quantity the solver "
            "actually computes, rather than something downstream of it. It is "
            "last because it is the one whose interpretation depends on every "
            "other entry on this list being in hand first."
        ),
    ),
)

MEASUREMENT_SPECS = MappingProxyType({spec.key: spec for spec in _SPECS})
"""Every measurement the ledger knows how to consume, by key."""


_ENTRIES: tuple[LedgerEntry, ...] = (
    LedgerEntry(
        factor=CredibilityFactor.VERIFICATION,
        held_level=2,
        threshold_level=3,
        blocker=Blocker.ANALYSIS,
        held_because=(
            "The F0 tier is verified against closed forms and exact discrete "
            "identities, but the coupled shot has no method of manufactured "
            "solutions and the F1, F2 and F3 tiers have no code verification "
            "at all. No experiment fixes that; it is proof and code work."
        ),
        evidence=(
            "Formal code verification of the F0 tier: conservation residuals "
            "split into round-off and truncation classes, an angular-momentum "
            "check against a naive per-element oracle, order of accuracy "
            "against a closed-form cylinder integral, and analytic flat-plate "
            "and zero-speed limits. Solution verification is implemented as a "
            "Celik GCI with Richardson extrapolation."
        ),
        gap_statement=(
            "No method of manufactured solutions for the coupled shot, and no "
            "verification at all of the F1, F2 or F3 tiers. The surface "
            "refinement study also runs into the package's own envelope: I_G "
            "grows as the mesh is refined, so a mesh fine enough to converge "
            "the quadrature is further outside RFT's superposition argument."
        ),
    ),
    LedgerEntry(
        factor=CredibilityFactor.VALIDATION,
        held_level=0,
        threshold_level=3,
        blocker=Blocker.MEASUREMENT,
        held_because=(
            "The only comparison that can be formed against a published "
            "measurement is noise-limited under V&V 20, so it carries no "
            "information about model error. Nothing else this tool predicts "
            "has ever been measured, on bunker sand or on anything else."
        ),
        evidence=(
            "None. The one comparison that can be formed against a published "
            "measurement -- the material-scaling prediction of the vertical "
            "plate response against the Quikrete analogue's 2.02 N/cm^3 -- is "
            "noise-limited under V&V 20, so it carries no information about "
            "model error."
        ),
        gap_statement=(
            "No published data exists for ball launch angle, speed or spin "
            "from a splash shot, for clubhead deceleration in sand, for the "
            "energy split, or for ejecta mass. This is a gap in the field, "
            "not a search failure, so it cannot be closed by reading more. "
            "Closing it needs either the Wivou carry correlations compared "
            "against a computed model correlation, or an instrumented "
            "experiment: plate penetration at three plate areas, a 6x6 cm "
            "direct shear box, and one drag test at 20-27 m/s to fit lambda."
        ),
        steps=(
            LevelStep(
                from_level=0,
                to_level=1,
                requires=("bunker_sand_angle_of_repose_deg",),
                rationale=(
                    "One system-response quantity the model already predicts, "
                    "measured on the sand being modelled, compared under "
                    "V&V 20 with a u_val small enough for |E| to mean "
                    "something. Repose is the only such quantity obtainable "
                    "without a club delivery, which is what makes the "
                    "comparison error attributable: with no swing in the "
                    "loop, the sand state is the only input, so a discrepancy "
                    "belongs to the model rather than being shared with an "
                    "unmeasured boundary condition. That attributability is "
                    "the whole difference between a validation comparison and "
                    "a plot of two curves."
                ),
            ),
            LevelStep(
                from_level=1,
                to_level=2,
                requires=(
                    "bunker_sand_bulk_density_kg_m3",
                    "splash_shot_divot_cast_volume_m3",
                ),
                rationale=(
                    "An integral output of the coupled shot, not just of the "
                    "constitutive model: the cast gives a cavity volume and "
                    "the measured density turns it into the accelerated mass "
                    "the ball launch divides by. PR #9237 put a factor of 2.4 "
                    "between the prismatic rule and the momentum-and-energy "
                    "consistent reduction, so this is the cheapest measurement "
                    "that discriminates between two things the model already "
                    "disagrees with itself about."
                ),
            ),
            LevelStep(
                from_level=2,
                to_level=3,
                requires=(
                    "clubhead_delivery_shaft_strain",
                    "ejecta_launch_high_speed_video",
                ),
                rationale=(
                    "The threshold the intended use demands, and it needs both "
                    "ends of the calculation at the design point: the wrench "
                    "the solver computes, measured directly on the shaft, and "
                    "the launch quantities issue #8616 recorded as having no "
                    "published value anywhere. Level 3 is a claim that the "
                    "model has been validated over the domain it is used in, "
                    "and the domain is 25 m/s, which is 17x the fastest "
                    "intrusion in the published corpus."
                ),
            ),
        ),
    ),
    LedgerEntry(
        factor=CredibilityFactor.INPUT_PEDIGREE,
        held_level=2,
        threshold_level=3,
        blocker=Blocker.MEASUREMENT,
        held_because=(
            "Every constant is traced to a published analogue, and one "
            "commercial sand is characterised, but nothing is measured on the "
            "sand actually being modelled. Tracing a borrowed number well is "
            "level 2; measuring it is level 3."
        ),
        evidence=(
            "Every fitted constant is traced to a published analogue: the "
            "3D-RFT polynomial to a generic frictional-plastic medium, the "
            "friction angle and packing fraction to Quikrete medium sand, and "
            "lambda to plate-drag and wheel experiments. One fully "
            "characterised commercial bunker sand (Covia Signature 500, ASTM "
            "F1632 Method B and F1815) seeds the sand presets, and every entry "
            "carries a ProvenanceBasis."
        ),
        gap_statement=(
            "Nothing is measured on the sand actually being modelled, and the "
            "one characterised sand is a single lab report on a single "
            "commercial product, not a population. lambda and delta_h have no "
            "wedge value at all."
        ),
        steps=(
            LevelStep(
                from_level=2,
                to_level=3,
                requires=(
                    "bunker_sand_bulk_density_kg_m3",
                    "bunker_sand_drained_friction_angle_deg",
                ),
                rationale=(
                    "Together these retire the Quikrete borrow for the four "
                    "properties that carry the constitutive fit: friction "
                    "angle, packing, particle density and moisture. Both are "
                    "needed because measuring the density of a sand whose "
                    "shear strength is still borrowed leaves the model fed by "
                    "a mixture of one real sand and one hardware-store "
                    "product, which is harder to reason about than a "
                    "consistent borrow."
                ),
            ),
            LevelStep(
                from_level=3,
                to_level=4,
                requires=("bunker_sand_population_survey",),
                rationale=(
                    "Level 4 asks for inputs characterised over the domain of "
                    "use, and the domain of use is bunkers in general, not one "
                    "bunker. One well-measured sand is a sample of size one; "
                    "the survey is what turns it into a population with a "
                    "spread that can be quoted."
                ),
            ),
        ),
    ),
    LedgerEntry(
        factor=CredibilityFactor.RESULTS_UNCERTAINTY,
        held_level=2,
        threshold_level=3,
        blocker=Blocker.MEASUREMENT,
        held_because=(
            "u_num covers the numerics and nothing else. Input uncertainty "
            "cannot be propagated without a measured input distribution, and "
            "model-form uncertainty cannot be quantified without a validation "
            "comparison to quantify it against."
        ),
        evidence=(
            "Discretisation uncertainty is estimated by GCI and converted to a "
            "V&V 20 u_h; u_num is formed by simple addition of u_h, u_it and "
            "u_ro as the standard requires, and u_val by quadrature."
        ),
        gap_statement=(
            "No input uncertainty is propagated through a shot, and no "
            "model-form uncertainty is quantified anywhere -- which is the "
            "direct consequence of validation being at level 0. The reported "
            "u_num covers the numerics only and must not be read as an error "
            "bar on the physics."
        ),
        steps=(
            LevelStep(
                from_level=2,
                to_level=3,
                requires=(
                    "bunker_sand_angle_of_repose_deg",
                    "bunker_sand_population_survey",
                ),
                rationale=(
                    "Two different things, both required. The survey supplies "
                    "the input distribution there is currently nothing to "
                    "propagate; the repose comparison supplies the u_val "
                    "against which a model-form term can be estimated at all. "
                    "Neither alone gets past level 2, which is why this factor "
                    "is stuck even though its numerics are in good order."
                ),
            ),
        ),
    ),
    LedgerEntry(
        factor=CredibilityFactor.RESULTS_ROBUSTNESS,
        held_level=1,
        threshold_level=2,
        blocker=Blocker.ANALYSIS,
        held_because=(
            "Metamorphic relations and a variance-based sensitivity study "
            "exist, but neither covers the constants that dominate the answer, "
            "and nobody outside the authors has reviewed the F0 tier. Both are "
            "study and review work, not instrument time."
        ),
        evidence=(
            "Metamorphic relations (translation, rotation, reflection, "
            "permutation, scaling, monotonicity) cover the solver, and the "
            "study package provides variance-based sensitivity over the "
            "declared sweep ranges."
        ),
        gap_statement=(
            "No sensitivity study over the constants that actually dominate "
            "the answer -- lambda across its published 1.0-2.8 spread and the "
            "delta_h saturation fraction -- and no independent review of the "
            "F0 tier by anyone who did not write it."
        ),
    ),
    LedgerEntry(
        factor=CredibilityFactor.USE_HISTORY,
        held_level=0,
        threshold_level=2,
        blocker=Blocker.USE,
        held_because=(
            "No design decision has ever been made with this solver, and no "
            "predecessor of it exists to inherit a history from. There is no "
            "measurement that manufactures use history; it accrues or it does "
            "not."
        ),
        evidence=(
            "None. The F0 solver has never been used to make a design "
            "decision, and no predecessor of it has either."
        ),
        gap_statement=(
            "Use history accrues only by use. Until a design produced by this "
            "tool is built and measured, this factor cannot move."
        ),
    ),
    LedgerEntry(
        factor=CredibilityFactor.MS_MANAGEMENT,
        held_level=3,
        threshold_level=3,
        blocker=Blocker.ANALYSIS,
        held_because=(
            "The threshold is met. What separates level 3 from level 4 is a "
            "formal configuration-management process with a defined release "
            "and approval path, which is process work this repository has not "
            "done and no measurement supplies."
        ),
        evidence=(
            "ADR-0032 records the architecture and its rejected alternatives; "
            "every run emits a manifest with config hash, physics hash, RNG "
            "seed entropy, library versions, git SHA, fidelity tier and "
            "validity verdict; and CI enforces lint, format, file-size, "
            "architecture and marker gates on every change."
        ),
        gap_statement=(
            "Threshold met. The remaining gap to level 4 is a formal "
            "configuration-management process with a defined release and "
            "approval path, which this repository does not have."
        ),
    ),
    LedgerEntry(
        factor=CredibilityFactor.PEOPLE_QUALIFICATIONS,
        held_level=None,
        threshold_level=2,
        blocker=Blocker.NOT_SELF_ASSESSABLE,
        held_because=(
            "A team scoring its own competence is not evidence, so no number "
            "is recorded. Leaving it blank costs the table its symmetry and "
            "keeps the seven factors that are backed by artefacts meaning "
            "something."
        ),
        evidence=(
            "Deliberately not self-assessed. A team scoring its own competence "
            "is not evidence, and a number here would dilute the seven factors "
            "that are backed by artefacts."
        ),
        gap_statement=(
            "Assess externally, or leave blank. Do not fill it in to make the "
            "table look complete."
        ),
    ),
)


VALIDATION_LEDGER = ValidationLedger(entries=_ENTRIES, specs=MEASUREMENT_SPECS)
"""The ledger the credibility assessment is derived from."""

_EMPTY_REGISTER = MeasurementRegister()
"""No measurements. The state the package ships in, and must keep shipping in."""


def roadmap_table_markdown() -> str:
    """Render the per-factor roadmap as a Markdown table.

    Rendered against an **empty** register, which is the state the package
    ships in.  When a real measurement lands, this block goes stale and the
    freshness test says so, which is the intended behaviour: a roadmap that
    silently absorbs a measurement is a roadmap nobody re-reads.

    Returns:
        The table: level, threshold, blocker, next step, and the
        measurements that step needs.
    """
    lines = [
        "| Factor | Level | Threshold | Blocked on | Next step | Measurements needed |",
        "| ------ | ----- | --------- | ---------- | --------- | ------------------- |",
    ]
    for entry in VALIDATION_LEDGER.entries:
        level = (
            "not assessed" if entry.held_level is None else f"{entry.held_level} / 4"
        )
        step = entry.next_step(entry.held_level)
        if step is None:
            transition = "n/a"
            needs = "n/a"
        else:
            transition = f"{step.from_level} to {step.to_level}"
            needs = ", ".join(f"`{key}`" for key in step.requires)
        lines.append(
            f"| {entry.factor.label} | {level} | {entry.threshold_level} / 4 "
            f"| {entry.blocker.value.replace('_', ' ')} | {transition} | {needs} |"
        )
    return "\n".join(lines)


def measurement_spec_table_markdown() -> str:
    """Render the measurement specs as a Markdown table.

    Returns:
        The table: quantity, unit, effort, acceptance gates, and the sand
        properties the measurement would move to ``MEASURED``.
    """
    lines = [
        "| Measurement | Quantity | Unit | Effort | Acceptance | Flips to MEASURED |",
        "| ----------- | -------- | ---- | ------ | ---------- | ----------------- |",
    ]
    for spec in VALIDATION_LEDGER.ordered_specs():
        flips = (
            ", ".join(f"`{key}`" for key in spec.provenance_keys)
            if spec.provenance_keys
            else "none"
        )
        effort = f"{spec.effort.value.replace('_', ' ')} ({spec.effort.cost_units})"
        lines.append(
            f"| `{spec.key}` | {spec.quantity} | {spec.unit} | {effort} "
            f"| {spec.acceptance.summary()} | {flips} |"
        )
    return "\n".join(lines)


def leverage_table_markdown() -> str:
    """Render the leverage ranking as a Markdown table.

    Returns:
        The table: rank, measurement, effort, credit, leverage, and which
        level steps the credit came from.
    """
    lines = [
        "| Rank | Measurement | Effort | Credit | Leverage | Unlocks today |",
        "| ---- | ----------- | ------ | ------ | -------- | ------------- |",
    ]
    ranking = VALIDATION_LEDGER.leverage_ranking(_EMPTY_REGISTER)
    for rank, item in enumerate(ranking, start=1):
        unlocks = ", ".join(item.unlocks) if item.unlocks else "nothing yet"
        lines.append(
            f"| {rank} | `{item.spec.key}` "
            f"| {item.spec.effort.value.replace('_', ' ')} "
            f"| {item.credit:.2f} | {item.leverage:.3f} | {unlocks} |"
        )
    return "\n".join(lines)
