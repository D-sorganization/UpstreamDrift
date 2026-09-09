"""Hierarchical participant-calibrated twin inference on a synthetic benchmark.

The workflow is deliberately closed-form. A linear-Gaussian hierarchy over
participant twin parameters, an explicit additive model-discrepancy term, and a
conjugate posterior make every reported number reproducible without a sampler,
so posterior contraction and out-of-sample prediction can be audited separately.

Nothing here is a human result. The benchmark is synthetic, the cohort is
synthetic, and the promotion ledger fails closed against the governed
participant-data gates in UpstreamDrift #8450 and #8556.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .participant_twin_provenance import (
    SCHEMA_VERSION as PROVENANCE_SCHEMA_VERSION,
)
from .participant_twin_provenance import (
    STRATIFICATION_VOCABULARY,
    HoldoutBarrier,
    assign_participant_holdout,
    assign_trajectory_roles,
    validate_cohort,
)

FloatArray = NDArray[np.float64]

TWIN_PARAMETERS: tuple[str, ...] = (
    "segment_inertia_scale",
    "grip_stiffness_scale",
    "shaft_stiffness_scale",
    "wrist_release_timing_offset",
    "impact_efficiency",
    "activation_gain",
    "grip_compliance_alias",
)
PRIOR_SD: tuple[float, ...] = (0.08, 0.15, 0.12, 0.09, 0.05, 0.20, 0.15)
POPULATION_SD_FRACTION = 0.6
DISCREPANCY_PRIOR_SD = 0.06
ALIAS_COEFFICIENT = -0.5
ACTIVATION_PARAMETER = "activation_gain"
ALIASED_PARAMETERS = ("grip_stiffness_scale", "grip_compliance_alias")

INTERVAL_Z = 1.6448536269514722
PRACTICAL_CONTRACTION_THRESHOLD = 0.20
PRIOR_PREDICTIVE_COVERAGE_BAND = (0.70, 1.0)
POSTERIOR_PREDICTIVE_COVERAGE_MINIMUM = 0.80
SMALL_STRATUM_CELL = 3

COHORT_SALT = "8f2c41a90b7d4e6512c3a80fd961b47e"
COHORT_ID = "participant-twin-synthetic-benchmark-v1"
PARTICIPANT_COUNT = 12
TRAJECTORIES_PER_PARTICIPANT = 6
CALIBRATION_TRAJECTORIES = 3
HOLDOUT_FRACTION = 1.0 / 3.0
CLUB_POOL_SIZE = 4
SESSIONS_PER_PARTICIPANT = 2

POPULATION_MEAN_TRUTH: tuple[float, ...] = (
    0.020,
    -0.030,
    0.050,
    0.012,
    -0.010,
    0.060,
    0.005,
)
DISCREPANCY_TRUTH: tuple[float, ...] = (
    0.035,
    -0.028,
    0.045,
    0.022,
    -0.030,
    0.026,
)


@dataclass(frozen=True, slots=True)
class Channel:
    """One observation channel with its nominal scales and noise contract."""

    name: str
    features: tuple[str, ...]
    scales: tuple[float, ...]
    noise_fraction: float
    optional: bool = False

    def __post_init__(self) -> None:
        """Validate the declared channel contract."""
        if len(self.features) != len(self.scales):
            raise ValueError(f"{self.name} needs one nominal scale per feature")
        if not self.features:
            raise ValueError(f"{self.name} must declare at least one feature")
        if not all(np.isfinite(self.scales)) or any(
            scale <= 0.0 for scale in self.scales
        ):
            raise ValueError(f"{self.name} scales must be positive and finite")
        if not 0.0 < self.noise_fraction < 1.0:
            raise ValueError(f"{self.name} noise_fraction must lie inside (0, 1)")


CHANNELS: tuple[Channel, ...] = (
    Channel(
        "kinematic",
        (
            "peak_hand_speed_mps",
            "peak_club_angular_rate_radps",
            "peak_wrist_angle_rad",
            "downswing_duration_s",
        ),
        (8.4, 34.0, 1.35, 0.26),
        0.020,
    ),
    Channel(
        "bilateral_wrench",
        (
            "lead_hand_axial_peak_n",
            "trail_hand_axial_peak_n",
            "common_normal_peak_n",
            "differential_tangential_peak_n",
        ),
        (210.0, 165.0, 320.0, 95.0),
        0.050,
    ),
    Channel(
        "shaft",
        ("lead_deflection_peak_m", "toe_down_deflection_peak_m"),
        (0.042, 0.028),
        0.060,
    ),
    Channel(
        "impact",
        ("impact_duration_s", "peak_impact_force_n"),
        (0.00045, 11500.0),
        0.040,
    ),
    Channel(
        "launch",
        ("ball_speed_mps", "launch_angle_deg", "spin_rate_rpm"),
        (67.0, 12.5, 2600.0),
        0.030,
    ),
    Channel(
        "activation",
        ("forearm_activation_index", "trunk_activation_index"),
        (0.55, 0.40),
        0.080,
        optional=True,
    ),
)


def _channel_selection(include_activation: bool) -> tuple[Channel, ...]:
    if include_activation:
        return CHANNELS
    return tuple(channel for channel in CHANNELS if not channel.optional)


def sensitivity_matrix(channels: tuple[Channel, ...]) -> FloatArray:
    """Return the declared normalized sensitivity of each feature to each twin.

    The rule is deterministic and fully declared: a fixed trigonometric basis,
    an activation column that acts only on the optional activation channel, and
    an exactly aliased grip-compliance column. The alias is intentional; it is
    the structural non-identifiability the pre-fit screen must detect.
    """
    rows = sum(len(channel.features) for channel in channels)
    columns = len(TWIN_PARAMETERS)
    index = np.arange(rows, dtype=float)[:, None]
    order = np.arange(columns, dtype=float)[None, :]
    design = 0.62 * np.sin(1.7 * (index + 1.0) + 2.3 * (order + 1.0)) + 0.34 * np.cos(
        0.9 * (index + 2.0) * (order + 1.0)
    )
    activation_column = TWIN_PARAMETERS.index(ACTIVATION_PARAMETER)
    offset = 0
    for channel in channels:
        width = len(channel.features)
        if not channel.optional:
            design[offset : offset + width, activation_column] = 0.0
        offset += width
    grip_column = TWIN_PARAMETERS.index("grip_stiffness_scale")
    alias_column = TWIN_PARAMETERS.index("grip_compliance_alias")
    design[:, alias_column] = ALIAS_COEFFICIENT * design[:, grip_column]
    return design


def discrepancy_design(channels: tuple[Channel, ...]) -> FloatArray:
    """Return the channel-indicator design of the additive discrepancy term."""
    rows = sum(len(channel.features) for channel in channels)
    matrix = np.zeros((rows, len(channels)))
    offset = 0
    for column, channel in enumerate(channels):
        width = len(channel.features)
        matrix[offset : offset + width, column] = 1.0
        offset += width
    return matrix


def noise_sd_vector(channels: tuple[Channel, ...]) -> FloatArray:
    """Return the normalized observation noise standard deviation per feature."""
    return np.concatenate(
        [np.full(len(channel.features), channel.noise_fraction) for channel in channels]
    )


def feature_names(channels: tuple[Channel, ...]) -> tuple[str, ...]:
    """Return the fully qualified feature name of every observation row."""
    return tuple(
        f"{channel.name}.{feature}"
        for channel in channels
        for feature in channel.features
    )


def design_digest(design: FloatArray, discrepancy: FloatArray) -> str:
    """Return a stable digest binding a screen to the design it screened."""
    payload = np.concatenate((design.ravel(), discrepancy.ravel()))
    return hashlib.sha256(
        np.ascontiguousarray(payload, dtype="<f8").tobytes()
    ).hexdigest()


def _rng(*tokens: str) -> np.random.Generator:
    digest = hashlib.sha256("|".join(tokens).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _pseudonym(prefix: str, *tokens: str) -> str:
    digest = hashlib.sha256("|".join((COHORT_SALT, *tokens)).encode()).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _strata_for(
    pseudonyms: tuple[str, ...], held_out: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    ordered_holdout = tuple(sorted(held_out))
    strata: dict[str, dict[str, str]] = {}
    for index, pseudonym in enumerate(pseudonyms):
        is_iron = index % 3 == 0
        strata[pseudonym] = {
            "anthropometry_band": ("band_a", "band_b", "band_c")[index % 3],
            "skill_band": ("recreational", "competitive_amateur", "elite")[index % 3],
            "sex": ("female", "male", "other_or_undisclosed")[index % 3],
            "age_band": ("18_29", "30_44", "45_59")[index % 3],
            "handedness": "left" if index % 5 == 0 else "right",
            "injury_history": (
                "prior_upper_limb"
                if index % 4 == 1
                else ("prior_lumbar" if index % 7 == 3 else "none_reported")
            ),
            "impairment": "none_reported",
            "club_class": "iron" if is_iron else "driver",
            "task": "full_swing_iron" if is_iron else "full_swing_driver",
        }
    # Two levels are deliberately confined to held-out participants so the
    # transport audit always retains a nontransportable, unrepresented cell.
    strata[ordered_holdout[0]]["impairment"] = "limited_shoulder_range"
    strata[ordered_holdout[1]]["age_band"] = "60_plus"
    return strata


def _build_cohort_participants(
    pseudonyms: tuple[str, ...],
    club_pool: tuple[str, ...],
    held_out: tuple[str, ...],
    strata: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    participants: list[dict[str, Any]] = []
    trajectory_roles: dict[str, str] = {}
    for index, pseudonym in enumerate(pseudonyms):
        trajectories = tuple(
            sorted(
                _pseudonym("tj", "trajectory", pseudonym, str(trial))
                for trial in range(TRAJECTORIES_PER_PARTICIPANT)
            )
        )
        trajectory_roles.update(
            assign_trajectory_roles(
                trajectories,
                split_salt=COHORT_SALT,
                calibration_trajectory_count=CALIBRATION_TRAJECTORIES,
            )
        )
        participants.append(
            {
                "participant_pseudonym": pseudonym,
                "session_pseudonyms": sorted(
                    _pseudonym("ss", "session", pseudonym, str(visit))
                    for visit in range(SESSIONS_PER_PARTICIPANT)
                ),
                "club_pseudonyms": sorted(
                    {
                        club_pool[index % CLUB_POOL_SIZE],
                        club_pool[(index + 1) % CLUB_POOL_SIZE],
                    }
                ),
                "strata": strata[pseudonym],
                "trajectory_ids": list(trajectories),
                "holdout_role": (
                    "participant_held_out" if pseudonym in held_out else "training"
                ),
            }
        )
    return participants, trajectory_roles


def build_synthetic_cohort() -> dict[str, Any]:
    """Build the frozen identity-safe synthetic benchmark cohort record."""
    pseudonyms = tuple(
        sorted(
            _pseudonym("pt", "participant", str(index))
            for index in range(PARTICIPANT_COUNT)
        )
    )
    club_pool = tuple(
        sorted(_pseudonym("cl", "club", str(index)) for index in range(CLUB_POOL_SIZE))
    )
    held_out = assign_participant_holdout(
        pseudonyms, split_salt=COHORT_SALT, holdout_fraction=HOLDOUT_FRACTION
    )
    strata = _strata_for(pseudonyms, held_out)
    participants, trajectory_roles = _build_cohort_participants(
        pseudonyms, club_pool, held_out, strata
    )
    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "cohort_id": COHORT_ID,
        "registered_at_utc": "2026-09-06T00:00:00Z",
        "governance": {
            "data_class": "synthetic_benchmark",
            "ethics_reference": "not_applicable_synthetic_benchmark",
            "consent_and_reuse_basis": "not_applicable_synthetic_benchmark",
            "private_data_authority": "not_applicable_synthetic_benchmark",
            "calibration_records": "not_applicable_synthetic_benchmark",
            "time_synchronization_records": "not_applicable_synthetic_benchmark",
            "analysis_release_authorization": "not_applicable_synthetic_benchmark",
            "identity_policy": "pseudonym_only_no_identity_inference",
            "pseudonym_derivation": "synthetic_salted_digest_no_real_identity",
        },
        "club_pool": list(club_pool),
        "participants": participants,
        "split": {
            "split_id": "participant-twin-frozen-split-v1",
            "split_salt": COHORT_SALT,
            "holdout_fraction": HOLDOUT_FRACTION,
            "calibration_trajectory_count": CALIBRATION_TRAJECTORIES,
            "frozen_before_outcome_access": True,
            "outcome_fields_present": False,
            "participant_held_out": list(held_out),
            "trajectory_roles": dict(sorted(trajectory_roles.items())),
        },
        "stratification_vocabulary": {
            key: list(values) for key, values in STRATIFICATION_VOCABULARY.items()
        },
        "evidence_domain": {
            "declared_scope": [
                "synthetic_benchmark_parameter_recovery",
                "synthetic_benchmark_held_out_prediction",
            ],
            "prohibited_inferences": [
                "coaching_prescription",
                "injury_risk_prediction",
                "talent_identification",
                "anatomical_or_muscle_strategy_identification",
                "transport_to_strata_unrepresented_in_training",
            ],
        },
        "inference_boundary": (
            "This cohort is a synthetic benchmark. It cannot establish any human, "
            "anatomical, equipment, injury, or coaching conclusion, and it cannot "
            "substitute for the governed participant data required by "
            "UpstreamDrift issues 8450 and 8556."
        ),
    }
    validate_cohort(record)
    return record


@dataclass(frozen=True, slots=True)
class Benchmark:
    """Synthetic observations plus the truth used only for reporting recovery."""

    channels: tuple[Channel, ...]
    design: FloatArray
    discrepancy: FloatArray
    noise_sd: FloatArray
    observations: dict[str, FloatArray]
    trajectory_participant: dict[str, str]
    participant_truth: dict[str, FloatArray]
    discrepancy_truth: FloatArray

    def digest(self) -> str:
        """Return the digest of the design this benchmark was generated from."""
        return design_digest(self.design, self.discrepancy)


def build_benchmark(
    cohort: dict[str, Any], *, include_activation: bool = True
) -> Benchmark:
    """Generate the deterministic synthetic observation set for one cohort.

    Observations are normalized fractional deviations from the declared nominal
    feature scales, so every channel shares one comparable error metric.
    """
    validate_cohort(cohort)
    channels = _channel_selection(include_activation)
    design = sensitivity_matrix(channels)
    discrepancy = discrepancy_design(channels)
    noise = noise_sd_vector(channels)
    population_mean = np.asarray(POPULATION_MEAN_TRUTH, dtype=float)
    population_sd = POPULATION_SD_FRACTION * np.asarray(PRIOR_SD, dtype=float)
    truth_discrepancy = np.asarray(DISCREPANCY_TRUTH[: len(channels)], dtype=float)
    observations: dict[str, FloatArray] = {}
    trajectory_participant: dict[str, str] = {}
    participant_truth: dict[str, FloatArray] = {}
    for participant in cohort["participants"]:
        pseudonym = participant["participant_pseudonym"]
        draw = _rng(COHORT_SALT, "twin", pseudonym).standard_normal(len(PRIOR_SD))
        theta = population_mean + population_sd * draw
        participant_truth[pseudonym] = theta
        clean = design @ theta + discrepancy @ truth_discrepancy
        for trajectory in participant["trajectory_ids"]:
            noise_draw = _rng(COHORT_SALT, "observation", trajectory).standard_normal(
                design.shape[0]
            )
            observations[trajectory] = clean + noise * noise_draw
            trajectory_participant[trajectory] = pseudonym
    return Benchmark(
        channels=channels,
        design=design,
        discrepancy=discrepancy,
        noise_sd=noise,
        observations=observations,
        trajectory_participant=trajectory_participant,
        participant_truth=participant_truth,
        discrepancy_truth=truth_discrepancy,
    )


@dataclass(frozen=True, slots=True)
class IdentifiabilityScreen:
    """Design-only structural and practical identifiability verdict."""

    digest: str
    structural_rank: int
    parameter_count: int
    aliased_directions: tuple[tuple[str, ...], ...]
    structurally_identifiable: tuple[str, ...]
    predicted_contraction: dict[str, float]
    practically_identifiable: tuple[str, ...]

    @property
    def estimable_parameters(self) -> tuple[str, ...]:
        """Return parameters that pass both the structural and practical gate."""
        practical = set(self.practically_identifiable)
        return tuple(
            name for name in self.structurally_identifiable if name in practical
        )


def screen_identifiability(
    design: FloatArray,
    discrepancy: FloatArray,
    noise_sd: FloatArray,
    *,
    calibration_trajectory_count: int,
) -> IdentifiabilityScreen:
    """Screen the design before any observation is fitted.

    Preconditions: shapes agree and ``calibration_trajectory_count`` is
    positive. Postcondition: the verdict depends only on the design, the noise
    contract, and the prior, never on data.
    """
    if design.shape[0] != discrepancy.shape[0] or design.shape[0] != noise_sd.size:
        raise ValueError("design, discrepancy, and noise contracts must agree")
    if calibration_trajectory_count < 1:
        raise ValueError("calibration_trajectory_count must be positive")
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = float(singular.max()) * 1e-10
    rank = int(np.sum(singular > tolerance))
    _, _, right = np.linalg.svd(design)
    null_space = right[rank:]
    aliased = tuple(
        tuple(
            TWIN_PARAMETERS[index]
            for index in np.flatnonzero(np.abs(vector) > 0.1).tolist()
        )
        for vector in null_space
    )
    aliased_names = {name for group in aliased for name in group}
    structural = tuple(name for name in TWIN_PARAMETERS if name not in aliased_names)
    marginal_prior_sd = np.sqrt(
        np.asarray(PRIOR_SD) ** 2 + (POPULATION_SD_FRACTION * np.asarray(PRIOR_SD)) ** 2
    )
    weight = np.diag(1.0 / noise_sd**2)
    information = calibration_trajectory_count * (design.T @ weight @ design)
    posterior = np.linalg.inv(information + np.diag(1.0 / marginal_prior_sd**2))
    contraction = 1.0 - np.sqrt(np.diag(posterior)) / marginal_prior_sd
    predicted = {
        name: float(value)
        for name, value in zip(TWIN_PARAMETERS, contraction, strict=True)
    }
    practical = tuple(
        name
        for name, value in predicted.items()
        if value >= PRACTICAL_CONTRACTION_THRESHOLD
    )
    return IdentifiabilityScreen(
        digest=design_digest(design, discrepancy),
        structural_rank=rank,
        parameter_count=len(TWIN_PARAMETERS),
        aliased_directions=aliased,
        structurally_identifiable=structural,
        predicted_contraction=predicted,
        practically_identifiable=practical,
    )


@dataclass(frozen=True, slots=True)
class PopulationPosterior:
    """Closed-form hierarchical posterior over the population and each twin."""

    parameters: tuple[str, ...]
    training_participants: tuple[str, ...]
    mean: FloatArray
    covariance: FloatArray
    channel_count: int

    def _slice(self, offset: int) -> slice:
        width = len(self.parameters)
        return slice(offset, offset + width)

    def population_slice(self) -> slice:
        """Return the index range of the population mean block."""
        return self._slice(0)

    def participant_slice(self, pseudonym: str) -> slice:
        """Return the index range of one training participant's twin block."""
        index = self.training_participants.index(pseudonym)
        return self._slice(len(self.parameters) * (index + 1))

    def discrepancy_slice(self) -> slice:
        """Return the index range of the shared discrepancy block."""
        start = len(self.parameters) * (len(self.training_participants) + 1)
        return slice(start, start + self.channel_count)


def fit_population(
    benchmark: Benchmark,
    cohort: dict[str, Any],
    barrier: HoldoutBarrier,
    screen: IdentifiabilityScreen,
    *,
    include_discrepancy: bool = True,
) -> PopulationPosterior:
    """Fit the hierarchy on training participants' calibration trajectories.

    Preconditions: ``screen`` was produced from this benchmark's design, and
    every outcome is read through ``barrier``, which refuses held-out reads.
    Postcondition: no evaluation trajectory contributed to the posterior.
    """
    if screen.digest != benchmark.digest():
        raise ValueError("the identifiability screen does not match this design")
    summary = validate_cohort(cohort)
    training = tuple(summary["training_participants"])
    roles = cohort["split"]["trajectory_roles"]
    width = len(TWIN_PARAMETERS)
    channels = len(benchmark.channels)
    size = width * (len(training) + 1) + channels
    precision = np.zeros((size, size))
    information = np.zeros(size)

    prior_precision = np.diag(1.0 / np.asarray(PRIOR_SD) ** 2)
    population_precision = np.diag(
        1.0 / (POPULATION_SD_FRACTION * np.asarray(PRIOR_SD)) ** 2
    )
    discrepancy_sd = DISCREPANCY_PRIOR_SD if include_discrepancy else 1e-8
    precision[0:width, 0:width] = prior_precision + len(training) * population_precision
    start = width * (len(training) + 1)
    precision[start : start + channels, start : start + channels] = np.eye(channels) / (
        discrepancy_sd**2
    )
    weight = np.diag(1.0 / benchmark.noise_sd**2)
    design_gram = benchmark.design.T @ weight @ benchmark.design
    cross_gram = benchmark.design.T @ weight @ benchmark.discrepancy
    discrepancy_gram = benchmark.discrepancy.T @ weight @ benchmark.discrepancy

    for index, pseudonym in enumerate(training):
        block = slice(width * (index + 1), width * (index + 2))
        precision[block, block] += population_precision
        precision[0:width, block] -= population_precision
        precision[block, 0:width] -= population_precision
        participant = next(
            row
            for row in cohort["participants"]
            if row["participant_pseudonym"] == pseudonym
        )
        for trajectory in participant["trajectory_ids"]:
            if roles[trajectory] != "calibration":
                continue
            observation = barrier.calibration_outcome(
                trajectory, benchmark.observations
            )
            precision[block, block] += design_gram
            precision[block, start : start + channels] += cross_gram
            precision[start : start + channels, block] += cross_gram.T
            precision[start : start + channels, start : start + channels] += (
                discrepancy_gram
            )
            information[block] += benchmark.design.T @ (
                observation / benchmark.noise_sd**2
            )
            information[start : start + channels] += benchmark.discrepancy.T @ (
                observation / benchmark.noise_sd**2
            )
    covariance = np.asarray(np.linalg.inv(precision), dtype=np.float64)
    mean = covariance @ information
    return PopulationPosterior(
        parameters=TWIN_PARAMETERS,
        training_participants=training,
        mean=mean,
        covariance=covariance,
        channel_count=channels,
    )


def _joint_prior_for_new_participant(
    posterior: PopulationPosterior,
) -> tuple[FloatArray, FloatArray]:
    width = len(TWIN_PARAMETERS)
    channels = posterior.channel_count
    population = posterior.population_slice()
    discrepancy = posterior.discrepancy_slice()
    indices = np.r_[population, discrepancy]
    mean = posterior.mean[indices].copy()
    covariance = posterior.covariance[np.ix_(indices, indices)].copy()
    covariance[0:width, 0:width] += np.diag(
        (POPULATION_SD_FRACTION * np.asarray(PRIOR_SD)) ** 2
    )
    if covariance.shape != (width + channels, width + channels):
        raise RuntimeError("joint prior assembly produced an unexpected shape")
    return mean, covariance


def _condition(
    mean: FloatArray,
    covariance: FloatArray,
    operator: FloatArray,
    observations: list[FloatArray],
    noise_sd: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    precision = np.linalg.inv(covariance)
    weight = np.diag(1.0 / noise_sd**2)
    information = precision @ mean
    for observation in observations:
        precision = precision + operator.T @ weight @ operator
        information = information + operator.T @ (observation / noise_sd**2)
    updated_covariance = np.linalg.inv(precision)
    return updated_covariance @ information, updated_covariance


def _predictive_metrics(
    observations: list[FloatArray],
    mean: FloatArray,
    covariance: FloatArray,
    operator: FloatArray,
    noise_sd: FloatArray,
) -> dict[str, float]:
    if not observations:
        raise ValueError("predictive metrics need at least one observation")
    predicted = operator @ mean
    variance = np.diag(operator @ covariance @ operator.T) + noise_sd**2
    half_width = INTERVAL_Z * np.sqrt(variance)
    residuals = np.vstack([observation - predicted for observation in observations])
    covered = np.abs(residuals) <= half_width
    return {
        "normalized_rmse": float(np.sqrt(np.mean(residuals**2))),
        "interval_coverage_90": float(np.mean(covered)),
        "observation_count": float(len(observations)),
    }


def _operator(benchmark: Benchmark) -> FloatArray:
    return np.hstack((benchmark.design, benchmark.discrepancy))


def _participant_trajectories(
    cohort: dict[str, Any], pseudonym: str, role: str
) -> list[str]:
    roles = cohort["split"]["trajectory_roles"]
    participant = next(
        row
        for row in cohort["participants"]
        if row["participant_pseudonym"] == pseudonym
    )
    return [
        trajectory
        for trajectory in participant["trajectory_ids"]
        if roles[trajectory] == role
    ]
