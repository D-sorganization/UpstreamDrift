"""Component Pydantic models for ground-support kinematic stages (HO-2 #10156)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# -----------------------------------------------------------------------------
# Common / Sub-models
# -----------------------------------------------------------------------------


class SpineBendDeg(BaseModel):
    """Spine angle decomposition relative to neutral."""

    model_config = ConfigDict(extra="ignore")

    total_deg: float = Field(
        ...,
        description="Total 3D spine bend angle relative to vertical",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )
    forward_deg: float = Field(
        ...,
        description="Spine forward flexion/extension angle",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )
    lateral_deg: float = Field(
        ...,
        description="Spine lateral bending angle",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )


class ClavicleLinkDeg(BaseModel):
    """Clavicle link angle below horizontal for left and right shoulders."""

    model_config = ConfigDict(extra="ignore")

    L: float = Field(
        ...,
        description="Left clavicle link angle below horizontal",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )
    R: float = Field(
        ...,
        description="Right clavicle link angle below horizontal",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )


class PostureSummary(BaseModel):
    """Summary of upper-body posture angles and dimensions."""

    model_config = ConfigDict(extra="ignore")

    spine_bend_deg: SpineBendDeg = Field(
        ...,
        description="Spine bend angles (total, forward, lateral)",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )
    clavicle_link_below_horizontal_deg: ClavicleLinkDeg = Field(
        ...,
        description="Clavicle link inclinations below horizontal",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )
    hips_to_shoulder_centre_m: float = Field(
        ...,
        description="Euclidean distance between hip center and mid-shoulder",
        json_schema_extra={"unit": "m", "stage": "address"},
    )


class CentreOfMassReport(BaseModel):
    """Whole-body centre of mass position and ground polygon support status."""

    model_config = ConfigDict(extra="ignore")

    com_m: list[float] = Field(
        ...,
        description="3D world coordinates of whole-body centre of mass",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    height_above_ground_m: float = Field(
        ...,
        description="Vertical height of CoM above ground plane",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    inside_support_polygon: bool = Field(
        ...,
        description="Whether CoM ground projection falls inside contact support polygon",
        json_schema_extra={"unit": "bool", "stage": "address"},
    )
    polygon_centroid_offset_m: float = Field(
        ...,
        description="Horizontal distance from CoM ground projection to support centroid",
        json_schema_extra={"unit": "m", "stage": "address"},
    )


class HipCalibrationReceipt(BaseModel):
    """Functional hip joint calibration metrics from pelvic/femur markers."""

    model_config = ConfigDict(extra="ignore")

    spec_sha256: str = Field(
        ...,
        description="SHA256 of the calibrated hip model spec",
        json_schema_extra={"unit": "hash", "stage": "address"},
    )
    centre_r_hip_frame_m: list[float] = Field(
        ...,
        description="Right hip joint center in pelvis coordinate frame",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    centre_l_hip_frame_m: list[float] = Field(
        ...,
        description="Left hip joint center in pelvis coordinate frame",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    radius_r_m: float = Field(
        ...,
        description="Right hip sphere fitting radius",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    radius_l_m: float = Field(
        ...,
        description="Left hip sphere fitting radius",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    sphere_sd_r_m: float = Field(
        ...,
        description="Right hip sphere residual standard deviation",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    sphere_sd_l_m: float = Field(
        ...,
        description="Left hip sphere residual standard deviation",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    frames: int = Field(
        ...,
        description="Number of frames used for functional hip fitting",
        json_schema_extra={"unit": "count", "stage": "address"},
    )
    pelvis_axes_in_hip_frame: list[list[float]] = Field(
        ...,
        description="3x3 rotation matrix aligning hip frame to pelvis frame",
        json_schema_extra={"unit": "matrix", "stage": "address"},
    )
    waist_fit_max_residual_m: float = Field(
        ...,
        description="Maximum residual distance from waist marker rigid fit",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    hip_zero_twist_deg: dict[str, float] | None = Field(
        None,
        description="Calibrated hip coordinate zero-twist offset angles for right and left hips",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )


class ClubReceipt(BaseModel):
    """Physical mass and dimension properties of the club."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        ...,
        description="Club identifier (e.g. driver or iron7)",
        json_schema_extra={"unit": "text", "stage": "address"},
    )
    length_m: float = Field(
        ...,
        description="Total club length from butt of grip to head",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    head_mass_kg: float = Field(
        ...,
        description="Mass of the club head",
        json_schema_extra={"unit": "kg", "stage": "address"},
    )
    shaft_mass_kg: float = Field(
        ...,
        description="Mass of the club shaft",
        json_schema_extra={"unit": "kg", "stage": "address"},
    )
    grip_mass_kg: float = Field(
        ...,
        description="Mass of the club grip",
        json_schema_extra={"unit": "kg", "stage": "address"},
    )
    total_mass_kg: float = Field(
        ...,
        description="Total combined mass of the club assembly",
        json_schema_extra={"unit": "kg", "stage": "address"},
    )
    wrist_to_head_m: float = Field(
        ...,
        description="Effective wrist joint to club head offset distance",
        json_schema_extra={"unit": "m", "stage": "address"},
    )


# -----------------------------------------------------------------------------
# Ground Stage Models
# -----------------------------------------------------------------------------


class ToeSphereReceipt(BaseModel):
    """Foot contact sphere definition."""

    model_config = ConfigDict(extra="ignore")

    body: str = Field(
        ...,
        description="Parent body link name hosting the contact sphere",
        json_schema_extra={"unit": "text", "stage": "ground"},
    )
    position_m: list[float] = Field(
        ...,
        description="3D position offset of contact sphere center in body frame",
        json_schema_extra={"unit": "m", "stage": "ground"},
    )
    radius_m: float = Field(
        ...,
        description="Radius of the foot contact sphere",
        json_schema_extra={"unit": "m", "stage": "ground"},
    )


class GroundReceipt(BaseModel):
    """Ground plane calibration and contact sphere configuration."""

    model_config = ConfigDict(extra="ignore")

    height_m: float = Field(
        ...,
        description="World z-coordinate of the ground contact plane",
        json_schema_extra={"unit": "m", "stage": "ground"},
    )
    lowest_toe_marker_m: float = Field(
        ...,
        description="Lowest toe marker z-coordinate observed across capture",
        json_schema_extra={"unit": "m", "stage": "ground"},
    )
    standoff_m: float = Field(
        ...,
        description="Standoff clearance distance between marker and ground",
        json_schema_extra={"unit": "m", "stage": "ground"},
    )
    policy: str = Field(
        ...,
        description="Policy rule used to calibrate ground height",
        json_schema_extra={"unit": "text", "stage": "ground"},
    )
    stance_tolerance_m: float = Field(
        ...,
        description="Vertical elevation threshold to consider a sphere in stance",
        json_schema_extra={"unit": "m", "stage": "ground"},
    )
    stance_rule: str = Field(
        ...,
        description="Operational definition of ground stance detection",
        json_schema_extra={"unit": "text", "stage": "ground"},
    )
    toe_spheres: dict[str, ToeSphereReceipt] = Field(
        ...,
        description="Definitions of the calibrated toe contact spheres",
        json_schema_extra={"unit": "map", "stage": "ground"},
    )
    stance_fraction: dict[str, float] = Field(
        ...,
        description="Fraction of trajectory frames each sphere spends in contact",
        json_schema_extra={"unit": "ratio", "stage": "ground"},
    )


# -----------------------------------------------------------------------------
# Address Stage Models
# -----------------------------------------------------------------------------


class SeedOffsetsReport(BaseModel):
    """Address fit error using unscaled anatomical seeds."""

    model_config = ConfigDict(extra="ignore")

    marker_rms_m: float = Field(
        ...,
        ge=0,
        description="Overall marker RMS error under seed placement",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    segment_rms_m: dict[str, float] = Field(
        ...,
        description="Per-segment marker RMS error under seed placement",
        json_schema_extra={"unit": "m", "stage": "address"},
    )


class StaticTrialReport(BaseModel):
    """Static neutral trial placement and fit metrics."""

    model_config = ConfigDict(extra="ignore")

    frames: int = Field(
        ...,
        description="Number of static address frames averaged",
        json_schema_extra={"unit": "count", "stage": "address"},
    )
    neutral_fit_rms_m: float = Field(
        ...,
        ge=0,
        description="Marker RMS error of the neutral spine posture fit",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    neutral_posture: PostureSummary = Field(
        ...,
        description="Posture metrics of the neutral spine fit",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )
    marker_rms_m: float = Field(
        ...,
        ge=0,
        description="Marker RMS error with placed static seeds",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    posture: PostureSummary = Field(
        ...,
        description="Posture metrics with placed static seeds",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )


class CalibratedAddressReport(BaseModel):
    """Address pose metrics following lower-limb calibration."""

    model_config = ConfigDict(extra="ignore")

    marker_rms_m: float = Field(
        ...,
        ge=0,
        description="Marker RMS residual at calibrated address pose",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    segment_rms_m: dict[str, float] = Field(
        ...,
        description="Per-segment marker RMS at calibrated address pose",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    closure_error_m: float = Field(
        ...,
        description="Dual-grip weld constraint closure residual distance",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    lowest_sphere_height_m: float = Field(
        ...,
        description="Elevation of lowest foot contact sphere above ground",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    support_offset_m: float | None = Field(
        None,
        description="Distance from support centroid to address footprint",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    centre_of_mass: CentreOfMassReport | None = Field(
        None,
        description="CoM metrics and support polygon inclusion at address",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )
    leg_angles_deg: dict[str, float] | None = Field(
        None,
        description="Joint coordinate values for lower-limb degrees of freedom",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )
    posture: PostureSummary | None = Field(
        None,
        description="Torso and clavicle posture summary at address",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )


class ClosureFitReport(BaseModel):
    """Optional dual-grip closure weld fitting metrics from address."""

    model_config = ConfigDict(extra="ignore")

    address_rms_after_m: float = Field(
        ...,
        ge=0,
        description="Marker RMS at address after optimizing closure placement",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    rotation_change_deg: float = Field(
        ...,
        description="Angular adjustment applied to weld orientation",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )
    translation_change_m: float = Field(
        ...,
        description="Translational adjustment applied to weld position",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    method: str | None = Field(
        None,
        description="Optimization solver method used for closure fitting",
        json_schema_extra={"unit": "text", "stage": "address"},
    )
    open_chain_marker_rms_m: float | None = Field(
        None,
        description="Marker RMS before imposing closure weld constraint",
        json_schema_extra={"unit": "m", "stage": "address"},
    )
    residual_at_fit_m_rad: list[float] | None = Field(
        None,
        description="Position and orientation weld residuals at fit solution",
        json_schema_extra={"unit": "m, rad", "stage": "address"},
    )
    trail_wrist_locked_deg: dict[str, float] | None = Field(
        None,
        description="Fixed trail wrist angles held during address fitting",
        json_schema_extra={"unit": "deg", "stage": "address"},
    )


class AddressReceipt(BaseModel):
    """Address stage results, static trial, and initial pose fit."""

    model_config = ConfigDict(extra="ignore")

    seed_offsets: SeedOffsetsReport = Field(
        ...,
        description="Marker residuals from uncalibrated anatomical seeds",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )
    stance_spheres: list[str] = Field(
        ...,
        description="Foot contact sphere identifiers in active stance at frame 0",
        json_schema_extra={"unit": "names", "stage": "address"},
    )
    static_trial: StaticTrialReport | None = Field(
        None,
        description="Static trial neutral spine posture and marker placement",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )
    calibrated: CalibratedAddressReport | None = Field(
        None,
        description="Final calibrated address pose metrics",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )
    closure_fit: ClosureFitReport | None = Field(
        None,
        description="Optional closure weld optimization report",
        json_schema_extra={"unit": "compound", "stage": "address"},
    )


# -----------------------------------------------------------------------------
# Inverse Kinematics Stage Models
# -----------------------------------------------------------------------------


class AttachmentOffset(BaseModel):
    """Marker placement attachment body and 3D offset."""

    model_config = ConfigDict(extra="ignore")

    body: str = Field(
        ...,
        description="Parent body segment name hosting the optical marker",
        json_schema_extra={"unit": "text", "stage": "ik"},
    )
    offset_m: list[float] = Field(
        ...,
        description="3D offset coordinates in the local body coordinate frame",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )


class ReferenceReport(BaseModel):
    """Smoothed and dynamically consistent kinematic reference metrics."""

    model_config = ConfigDict(extra="ignore")

    cutoff_hz: float = Field(
        ...,
        description="Low-pass filter cutoff frequency for reference smoothing",
        json_schema_extra={"unit": "Hz", "stage": "ik"},
    )
    consistency_prior: float = Field(
        ...,
        description="Regularization weight penalizing acceleration spikes",
        json_schema_extra={"unit": "weight", "stage": "ik"},
    )
    marker_rms_m: float = Field(
        ...,
        ge=0,
        description="Marker RMS tracking error of smoothed reference trajectory",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    segment_rms_m: dict[str, float] = Field(
        ...,
        description="Per-segment marker RMS error of smoothed reference trajectory",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    closure_error_max_m: float = Field(
        ...,
        description="Peak weld loop closure error across smoothed reference",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    lowest_sphere_height_min_m: float = Field(
        ...,
        description="Minimum foot sphere height relative to ground in reference",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    lowest_sphere_height_max_m: float = Field(
        ...,
        description="Maximum foot sphere height relative to ground in reference",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    max_deviation_from_smoothed_rad: float = Field(
        ...,
        description="Maximum coordinate deviation between smoothed and resolved ref",
        json_schema_extra={"unit": "rad", "stage": "ik"},
    )
    max_joint_speed_rad_s: float = Field(
        ...,
        description="Peak coordinate angular velocity across all joints",
        json_schema_extra={"unit": "rad/s", "stage": "ik"},
    )
    stance_sphere_drift_max_m: float = Field(
        ...,
        description="Peak drift of planted stance foot spheres during stance phase",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )


class CalibrationReport(BaseModel):
    """Alternating marker and segment scale calibration metrics."""

    model_config = ConfigDict(extra="ignore")

    stride: int = Field(
        ...,
        description="Frame decimation stride used during calibration passes",
        json_schema_extra={"unit": "count", "stage": "ik"},
    )
    frames: int = Field(
        ...,
        description="Number of decimated frames sampled for calibration",
        json_schema_extra={"unit": "count", "stage": "ik"},
    )
    prior_frames: float = Field(
        ...,
        description="Equivalent prior weight expressed in frame count units",
        json_schema_extra={"unit": "frames", "stage": "ik"},
    )
    prior_offsets_m: dict[str, list[float]] = Field(
        ...,
        description="Seed marker offsets acting as calibration prior",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    rms_per_iteration_m: list[float] = Field(
        ...,
        description="Convergence history of marker RMS across solver iterations",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    rms_per_iteration_after_scaling_m: list[float] = Field(
        ...,
        description="Convergence history after applying segment length scaling",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    per_marker_rms_m: dict[str, float] = Field(
        ...,
        description="Final calibrated RMS error per tracked lower-limb marker",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    offsets_m: dict[str, AttachmentOffset] = Field(
        ...,
        description="Final calibrated 3D offsets for lower-limb markers",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )


class SegmentScalingEntry(BaseModel):
    """Grid search trial evaluation for femur and tibia scaling."""

    model_config = ConfigDict(extra="ignore")

    femur: float = Field(
        ...,
        description="Femur segment length scaling multiplier",
        json_schema_extra={"unit": "scale", "stage": "ik"},
    )
    tibia: float = Field(
        ...,
        description="Tibia segment length scaling multiplier",
        json_schema_extra={"unit": "scale", "stage": "ik"},
    )
    pinned_rms_m: float = Field(
        ...,
        ge=0,
        description="Marker RMS with feet pinned at evaluated scale factors",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )


class SegmentScalingReport(BaseModel):
    """Lower limb length scaling grid search outcome."""

    model_config = ConfigDict(extra="ignore")

    grid: list[float] = Field(
        ...,
        description="Grid search candidate scale factors",
        json_schema_extra={"unit": "scale", "stage": "ik"},
    )
    table: list[SegmentScalingEntry] = Field(
        ...,
        description="Evaluated grid combinations and resulting pinned RMS errors",
        json_schema_extra={"unit": "table", "stage": "ik"},
    )
    femur_scale: float = Field(
        ...,
        description="Optimal femur scale factor chosen from grid",
        json_schema_extra={"unit": "scale", "stage": "ik"},
    )
    tibia_scale: float = Field(
        ...,
        description="Optimal tibia scale factor chosen from grid",
        json_schema_extra={"unit": "scale", "stage": "ik"},
    )
    spec_sha256: str = Field(
        ...,
        description="SHA256 hash of the scaled model specification document",
        json_schema_extra={"unit": "hash", "stage": "ik"},
    )


class RomFlag(BaseModel):
    """Range of motion excess details for a single joint coordinate."""

    model_config = ConfigDict(extra="ignore")

    max_excess_deg: float = Field(
        ...,
        description="Peak excursion beyond anatomical limits in degrees",
        json_schema_extra={"unit": "deg", "stage": "ik"},
    )
    frames: int = Field(
        ...,
        description="Number of frames exceeding anatomical limits",
        json_schema_extra={"unit": "count", "stage": "ik"},
    )
    fraction: float = Field(
        ...,
        description="Fraction of trajectory duration spent outside human limits",
        json_schema_extra={"unit": "ratio", "stage": "ik"},
    )


class ConstrainedIkReceipt(BaseModel):
    """Constrained inverse kinematics diagnostics and provenance (Issue #10278)."""

    model_config = ConfigDict(extra="ignore")

    backend_name: str = Field(
        "pink",
        description="Name of the constrained inverse kinematics backend",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    solver: str = Field(
        "quadprog",
        description="QP solver name used by the backend",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    solver_version: str | None = Field(
        None,
        description="Version string of the QP solver engine",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    runtime_version: str | None = Field(
        None,
        description="Version string of the constrained IK runtime wrapper",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    pinocchio_version: str | None = Field(
        None,
        description="Version string of Pinocchio backend",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    source_sha256: str | None = Field(
        None,
        description="SHA256 digest of the underlying backend or model sources",
        json_schema_extra={"unit": "hash", "stage": "ik"},
    )
    model_name: str = Field(
        ...,
        description="Name of the kinematic model",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    capture_name: str = Field(
        ...,
        description="Name of the processed motion capture sequence",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    step_mode: str = Field(
        "physical",
        description="Integration mode (physical or projection)",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    limit_policy: str = Field(
        "enforce",
        description="Joint limit policy enforced during solve",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    task_policy: str = Field(
        "dual_grip_hard_equality",
        description="Task hierarchy and equality constraint policy",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    time_semantics: str = Field(
        "strict_physical_elapsed_dt",
        description="Time integration semantics applied to velocities",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )
    frame_count: int = Field(
        ...,
        ge=0,
        description="Total number of trajectory frames evaluated",
        json_schema_extra={"unit": "count", "stage": "ik"},
    )
    frame_success_count: int = Field(
        ...,
        ge=0,
        description="Total number of successfully solved frames",
        json_schema_extra={"unit": "count", "stage": "ik"},
    )
    all_frames_converged: bool = Field(
        ...,
        description="Whether all frames converged without solver failure",
        json_schema_extra={"unit": "bool", "stage": "ik"},
    )
    first_failed_frame: int | None = Field(
        None,
        description="Index of first frame that failed to converge, if any",
        json_schema_extra={"unit": "index", "stage": "ik"},
    )
    per_frame_status: list[bool] | None = Field(
        None,
        description="Convergence boolean per frame",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )
    max_velocity_ratio: float | None = Field(
        None,
        description="Maximum observed joint velocity to limit ratio",
        json_schema_extra={"unit": "ratio", "stage": "ik"},
    )
    is_qualified: bool = Field(
        ...,
        description="Whether the solved trajectory meets all qualification criteria",
        json_schema_extra={"unit": "bool", "stage": "ik"},
    )
    qualification_state: str = Field(
        "qualified",
        description="Qualification state label (qualified, disqualified, etc.)",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )

    @model_validator(mode="after")
    def _validate_qualification(self) -> ConstrainedIkReceipt:
        if self.is_qualified and not self.all_frames_converged:
            raise ValueError("is_qualified cannot be True if not all frames converged")
        return self


class IkReceipt(BaseModel):
    """Full-capture inverse kinematics trajectory and calibration results."""

    model_config = ConfigDict(extra="ignore")

    frames: int = Field(
        ...,
        description="Total number of trajectory frames matched",
        json_schema_extra={"unit": "count", "stage": "ik"},
    )
    marker_rms_m: float = Field(
        ...,
        ge=0,
        description="Whole-trajectory marker tracking RMS error",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    segment_rms_m: dict[str, float] = Field(
        ...,
        description="Per-segment marker RMS error over full trajectory",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    closure_error_max_m: float = Field(
        ...,
        description="Maximum dual-grip weld loop closure residual in IK",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    lowest_sphere_height_min_m: float = Field(
        ...,
        description="Minimum foot sphere elevation above ground in raw IK",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    lowest_sphere_height_max_m: float = Field(
        ...,
        description="Maximum foot sphere elevation above ground in raw IK",
        json_schema_extra={"unit": "m", "stage": "ik"},
    )
    attachments_m: dict[str, AttachmentOffset] | None = Field(
        None,
        description="Complete mapping of calibrated marker offsets in body frames",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )
    reference: ReferenceReport = Field(
        ...,
        description="Properties of smoothed, consistency re-solved reference",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )
    calibration: CalibrationReport = Field(
        ...,
        description="Alternating calibration convergence and offsets",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )
    segment_scaling: SegmentScalingReport = Field(
        ...,
        description="Segment length scaling search and optimal parameters",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )
    joint_ranges_deg: dict[str, list[float]] | None = Field(
        None,
        description="Anatomical lower-limb coordinate bounds in degrees",
        json_schema_extra={"unit": "deg", "stage": "ik"},
    )
    range_of_motion_flags: dict[str, RomFlag] | None = Field(
        None,
        description="Excursions exceeding anatomical limits across reference",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )
    bound_widening: float | None = Field(
        None,
        description="Safety factor applied to widen joint range limits",
        json_schema_extra={"unit": "multiplier", "stage": "ik"},
    )
    leg_angle_ranges_deg: dict[str, list[float]] | None = Field(
        None,
        description="Min and max angles observed per lower limb joint coordinate",
        json_schema_extra={"unit": "deg", "stage": "ik"},
    )
    constrained_ik: ConstrainedIkReceipt | None = Field(
        None,
        description="Optional constrained IK execution diagnostics and provenance",
        json_schema_extra={"unit": "compound", "stage": "ik"},
    )
    ik_backend: str | None = Field(
        None,
        description="Marker inverse kinematics solver backend (scipy, mujoco-minimize)",
        json_schema_extra={"unit": "string", "stage": "ik"},
    )


class AcceptanceGateReport(BaseModel):
    """Outcome and measurement for an individual acceptance gate."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        ...,
        description="Standardized name of the acceptance gate",
        json_schema_extra={"unit": "text", "stage": "acceptance"},
    )
    status: str = Field(
        ...,
        description="Pass/fail status of the gate (passed, failed, missing)",
        json_schema_extra={"unit": "text", "stage": "acceptance"},
    )
    threshold: float = Field(
        ...,
        description="Acceptance limit or tolerance threshold",
        json_schema_extra={"unit": "threshold", "stage": "acceptance"},
    )
    measured: float | None = Field(
        None,
        description="Observed value extracted from execution receipt",
        json_schema_extra={"unit": "measured", "stage": "acceptance"},
    )
    unit: str = Field(
        "m",
        description="Physical or statistical unit of measurement",
        json_schema_extra={"unit": "text", "stage": "acceptance"},
    )
    reason: str = Field(
        "",
        description="Diagnostic explanation for gate failure or omission",
        json_schema_extra={"unit": "text", "stage": "acceptance"},
    )


class AcceptanceReceipt(BaseModel):
    """Authoritative evaluation verdict under the matched-swing program (MS-01)."""

    model_config = ConfigDict(extra="ignore")

    horizon: str = Field(
        ...,
        description="Evaluation horizon milestone (G1, G2, or G3)",
        json_schema_extra={"unit": "text", "stage": "acceptance"},
    )
    is_physically_accepted: bool = Field(
        ...,
        description="Whether all physical and kinematic gates were satisfied",
        json_schema_extra={"unit": "bool", "stage": "acceptance"},
    )
    status: str = Field(
        ...,
        description="Verdict status string (PASSED or REJECTED)",
        json_schema_extra={"unit": "text", "stage": "acceptance"},
    )
    gates: list[AcceptanceGateReport] = Field(
        default_factory=list,
        description="Detailed list of evaluated physical and kinematic gates",
        json_schema_extra={"unit": "compound", "stage": "acceptance"},
    )
    qualification_note: str = Field(
        "",
        description="Summary note describing acceptance decision rationale",
        json_schema_extra={"unit": "text", "stage": "acceptance"},
    )
