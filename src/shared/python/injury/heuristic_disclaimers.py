"""Canonical honesty labels for injury heuristics (#15011).

Measured literature peaks (for example Hosea et al. 1990 reporting roughly
8× body weight compression during the golf swing) are observational reports,
not validated clinical injury thresholds. AffineDrift and the segmental-spine
research record state that no such validated threshold set exists for this
heuristic pathway.
"""

from __future__ import annotations

# Hosea et al. (1990) reported peak compression; not an injury limit.
LITERATURE_PEAK_COMPRESSION_BW_HOSEA_1990: float = 8.0

SPINAL_LOAD_HEURISTIC_LIMITATIONS: str = (
    "Heuristic closed-form estimate using a fixed 5 cm moment arm and lumbar "
    "flexion torque only; not inverse dynamics, not a distributed "
    "musculoskeletal model, and not a clinical injury assessment. Screening "
    "bands compare results within this module only; published literature "
    "reports measured peaks (Hosea et al. 1990, up to about 8× body weight "
    "compression) but does not establish validated safe-injury thresholds "
    "for the golf swing."
)

INJURY_RISK_SCORER_LIMITATIONS: str = (
    "Composite scores combine unvalidated screening heuristics (spinal, joint, "
    "technique, and training-load bands). Weights and numeric cutoffs are "
    "engineering defaults for relative comparison, not validated clinical "
    "thresholds or diagnoses."
)

NON_CLINICAL_ENGINEERING_DISCLAIMER: str = (
    "DISCLAIMER: Numerical scores are biomechanical engineering indicators and "
    "do NOT constitute clinical diagnosis, medical validation, or return-to-play "
    "clearance. Calculations reflect simplified heuristics or qualified "
    "mechanical models, not clinical outcomes."
)

__all__ = [
    "INJURY_RISK_SCORER_LIMITATIONS",
    "LITERATURE_PEAK_COMPRESSION_BW_HOSEA_1990",
    "NON_CLINICAL_ENGINEERING_DISCLAIMER",
    "SPINAL_LOAD_HEURISTIC_LIMITATIONS",
]
