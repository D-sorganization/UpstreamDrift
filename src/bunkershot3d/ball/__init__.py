"""Ball model for bunkershot3d (issues #8613, #8657).

Models the golf ball in a sand bed:
- Ball lie (position, depth, type)
- Sand-mediated momentum transfer (splash shot), driven by the impulse the F0
  solver delivered and the divot mass the metrics layer measured
- Launch conditions for flight handoff, with the validity verdict and
  provenance a carry number may only be quoted under
- The measurement-to-prediction program that can lift the launch verdict
  floor per regime, once measured strokes exist (issue #9543)
"""

from .lie import (
    BallLie,
    BallLieType,
    BallProperties,
    compute_exposed_cap_area,
    compute_exposed_cap_fraction,
    compute_submersion_depth,
)
from .splash import (
    BALL_LAUNCH_MEASUREMENT_GAP,
    BALL_LAUNCH_UNCALIBRATED_REASON,
    BALL_MOMENTUM_TRANSFER_EFFICIENCY,
    DEFAULT_MOMENTUM_TRANSFER,
    BallLaunchResult,
    ContactType,
    MomentumTransfer,
    SandDelivery,
    SplashTransferResult,
    compute_ball_launch_from_splash,
    compute_sand_ejecta_velocity,
    compute_splash_impulse,
    launch_verdict,
    momentum_transfer_provenance,
)
from .pipeline import (
    BunkerShotState,
    compute_bunker_launch,
    to_post_impact_state,
)
from .regimes import (
    RegimeClassification,
    StrikeOutcome,
    UnsupportedContactRegimeError,
    classify_contact_regime,
)
from .qualification import (
    INTENDED_USE_MATRIX,
    MEASUREMENT_PROTOCOL,
    PRACTICAL_TOLERANCES,
    THREE_CAMERA_RIG_CAPABILITY,
    FitOutcome,
    FitStatus,
    IntendedUseMatrix,
    MeasuredStroke,
    MeasurementProtocol,
    ObjectiveDisposition,
    PracticalTolerances,
    QualificationDataset,
    RegimeVerdict,
    RigCapability,
    TransferQualification,
    TransferQualificationError,
    UseRegime,
    objective_disposition,
    rig_capability_markdown,
)
from .qualification_fit import (
    fit_transfer,
    qualification_report_markdown,
    qualify,
    validate_holdout,
)

__all__ = [
    "BALL_LAUNCH_MEASUREMENT_GAP",
    "BALL_LAUNCH_UNCALIBRATED_REASON",
    "BALL_MOMENTUM_TRANSFER_EFFICIENCY",
    "DEFAULT_MOMENTUM_TRANSFER",
    "INTENDED_USE_MATRIX",
    "MEASUREMENT_PROTOCOL",
    "PRACTICAL_TOLERANCES",
    "THREE_CAMERA_RIG_CAPABILITY",
    "BallLaunchResult",
    "BallLie",
    "BallLieType",
    "BallProperties",
    "BunkerShotState",
    "ContactType",
    "FitOutcome",
    "FitStatus",
    "IntendedUseMatrix",
    "MeasuredStroke",
    "MeasurementProtocol",
    "MomentumTransfer",
    "ObjectiveDisposition",
    "PracticalTolerances",
    "QualificationDataset",
    "RegimeClassification",
    "RegimeVerdict",
    "RigCapability",
    "SandDelivery",
    "SplashTransferResult",
    "StrikeOutcome",
    "TransferQualification",
    "TransferQualificationError",
    "UnsupportedContactRegimeError",
    "UseRegime",
    "classify_contact_regime",
    "compute_ball_launch_from_splash",
    "compute_bunker_launch",
    "compute_exposed_cap_area",
    "compute_exposed_cap_fraction",
    "compute_sand_ejecta_velocity",
    "compute_splash_impulse",
    "compute_submersion_depth",
    "fit_transfer",
    "launch_verdict",
    "momentum_transfer_provenance",
    "objective_disposition",
    "qualification_report_markdown",
    "qualify",
    "rig_capability_markdown",
    "to_post_impact_state",
    "validate_holdout",
]
