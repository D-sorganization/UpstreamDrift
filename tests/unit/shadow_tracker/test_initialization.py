"""Unit tests for Shadow Tracker subject shape and initial-state fitting (ST-06)."""

from __future__ import annotations

import pytest

from src.shared.python.shadow_tracker._validation import (
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    SUBJECT_BINDING_SCHEMA_VERSION,
)
from src.shared.python.shadow_tracker.contracts import (
    RenderRequest,
    SubjectModelBinding,
)
from src.shared.python.shadow_tracker.initialization import (
    InitialHypothesis,
    InertialParameters,
    MultiviewFitResult,
    SubjectMorphology,
    VisualMorphology,
    estimate_short_window_velocity,
    fit_initial_state_multiview,
    generate_monocular_hypotheses,
)
from src.shared.python.shadow_tracker.mask_records import MaskFrame
from src.shared.python.shadow_tracker.projection import (
    AnalyticSilhouetteRenderer,
    PinholeCameraModel,
)
from src.shared.python.shadow_tracker.source_records import FrameIdentity


def _make_dummy_subject_binding() -> SubjectModelBinding:
    return SubjectModelBinding(
        schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
        subject_id="sub_001",
        model_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        joint_ids=("root", "spine", "shoulder_left", "shoulder_right"),
        body_ids=("torso", "head", "arm_left", "arm_right"),
        visual_envelope={"height_m": 1.80, "chest_width_m": 0.45, "depth_m": 0.28},
        mass_kg=78.5,
        scale_evidence="measured_anthropometry",
        handedness="right",
    )


@pytest.mark.unit
def test_visual_and_inertial_parameters_are_strictly_separated() -> None:
    """Verify visual envelope and inertial mass properties are separated and validated."""
    vis = VisualMorphology(
        height_m=1.82,
        chest_width_m=0.46,
        depth_m=0.30,
        segment_lengths={"torso": 0.60, "upper_arm": 0.32, "forearm": 0.28},
    )
    assert vis.height_m == 1.82
    assert vis.segment_lengths["torso"] == 0.60

    inertial = InertialParameters(
        mass_kg=80.0,
        center_of_mass_body_m=(0.0, 0.05, 0.10),
        moments_of_inertia_kg_m2=(1.2, 1.5, 0.8),
    )
    assert inertial.mass_kg == 80.0
    assert inertial.center_of_mass_body_m == (0.0, 0.05, 0.10)

    morph = SubjectMorphology(
        subject_id="sub_001",
        visual=vis,
        inertial=inertial,
        handedness="right",
    )
    assert morph.subject_id == "sub_001"
    assert morph.visual.height_m == 1.82
    assert morph.inertial.mass_kg == 80.0

    # Invariant checks: negative or zero values fail closed
    with pytest.raises(ValueError, match="height_m must be positive"):
        VisualMorphology(
            height_m=-1.0,
            chest_width_m=0.46,
            depth_m=0.30,
            segment_lengths={},
        )

    with pytest.raises(ValueError, match="mass_kg must be positive"):
        InertialParameters(
            mass_kg=0.0,
            center_of_mass_body_m=(0.0, 0.0, 0.0),
            moments_of_inertia_kg_m2=(1.0, 1.0, 1.0),
        )


@pytest.mark.unit
def test_recover_known_pose_from_calibrated_multiview() -> None:
    """Gate G2: Recover known pose from calibrated multi-view silhouettes."""
    # Build two calibrated orthogonal cameras: Frontal and Down-The-Line
    cam_front = PinholeCameraModel(
        camera_id="cam_front",
        width_px=640,
        height_px=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        translation_world_to_camera=(0.0, 0.0, 3.0),
        rotation_world_to_camera=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    cam_dtl = PinholeCameraModel(
        camera_id="cam_dtl",
        width_px=640,
        height_px=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        translation_world_to_camera=(0.0, 0.0, 3.0),
        rotation_world_to_camera=(0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
    )

    renderer = AnalyticSilhouetteRenderer(
        cameras={"cam_front": cam_front, "cam_dtl": cam_dtl}
    )

    frame_front = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset_01",
        shot_id="shot_01",
        swing_id="swing_01",
        camera_id="cam_front",
        frame_id="f_001",
        pts_ticks=0,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.0,
        physical_time_reason="",
        frame_sha256="a" * 64,
    )
    frame_dtl = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset_01",
        shot_id="shot_01",
        swing_id="swing_01",
        camera_id="cam_dtl",
        frame_id="f_001",
        pts_ticks=0,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.0,
        physical_time_reason="",
        frame_sha256="b" * 64,
    )

    # Render true synthetic observations at known pose
    true_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    front_render = renderer.render(
        RenderRequest(
            camera_id="cam_front",
            state=true_pose,
            image_size_px=(640, 480),
        )
    )
    dtl_render = renderer.render(
        RenderRequest(
            camera_id="cam_dtl",
            state=true_pose,
            image_size_px=(640, 480),
        )
    )

    mask_front = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=frame_front,
        width_px=640,
        height_px=480,
        body=bytes(front_render.body_mask),
        club=bytes(front_render.club_mask),
        valid=bytes(front_render.visibility_mask),
        revision_id="rev_01",
        parent_revision_id=None,
        producer_id="synth",
        correction_note="",
    )
    mask_dtl = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=frame_dtl,
        width_px=640,
        height_px=480,
        body=bytes(dtl_render.body_mask),
        club=bytes(dtl_render.club_mask),
        valid=bytes(dtl_render.visibility_mask),
        revision_id="rev_01",
        parent_revision_id=None,
        producer_id="synth",
        correction_note="",
    )

    result = fit_initial_state_multiview(
        cameras=(cam_front, cam_dtl),
        observed_masks=(mask_front, mask_dtl),
        candidate_poses=(
            (0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
            true_pose,
            (-0.5, -0.5, 0.0, 0.0, 0.0, 0.0),
        ),
        subject_binding=_make_dummy_subject_binding(),
        renderer=renderer,
    )

    assert result.best_hypothesis is not None
    assert result.best_hypothesis.pose == true_pose
    assert result.best_hypothesis.score >= 0.95
    assert result.best_hypothesis.label == "inferred"
    assert result.residuals_evaluated == 3


@pytest.mark.unit
def test_monocular_ambiguity_retains_discrete_depth_scale_alternatives() -> None:
    """Ambiguous single-view scene must retain discrete depth/scale hypotheses rather than collapsing."""
    cam = PinholeCameraModel(
        camera_id="cam_mono",
        width_px=640,
        height_px=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        translation_world_to_camera=(0.0, 1.0, -3.0),
    )

    # Observed 2D bounding extent in pixels: height ~ 200px
    observed_body_bbox = (220, 140, 420, 300)  # top, left, bottom, right

    hypotheses = generate_monocular_hypotheses(
        camera=cam,
        observed_body_bbox=observed_body_bbox,
        subject_binding=_make_dummy_subject_binding(),
        nominal_depths_m=(2.5, 3.0, 3.5, 4.0),
        handedness_options=("right", "left"),
    )

    # Must produce combinations across depth and handedness
    assert len(hypotheses) == 8
    # Depth must vary across candidates
    depths = {h.depth_m for h in hypotheses}
    assert depths == {2.5, 3.0, 3.5, 4.0}

    # All monocular hypotheses MUST be explicitly labeled as inferred
    for h in hypotheses:
        assert h.label == "inferred"
        assert h.camera_id == "cam_mono"
        assert h.scale > 0.0
        assert h.handedness in ("right", "left")

    # Invariant: single frame cannot claim unique depth or zero uncertainty
    assert len({h.hypothesis_id for h in hypotheses}) == 8


@pytest.mark.unit
def test_missing_address_frame_and_takeaway_handled_gracefully() -> None:
    """Estimating initial state when address frame is absent handles mid-swing start."""
    # Frame observations starting at frame 15 (takeaway already begun)
    frame_ids = ("frame_015", "frame_016", "frame_017")
    time_points_s = (0.25, 0.2667, 0.2833)
    root_positions = ((0.02, 0.01, 0.0), (0.05, 0.02, 0.0), (0.09, 0.03, 0.0))

    vel = estimate_short_window_velocity(
        frame_ids=frame_ids,
        time_points_s=time_points_s,
        positions=root_positions,
    )

    # Nonzero velocity should be estimated from the moving window
    assert len(vel) == 3
    assert vel[0] > 1.0  # dx/dt ~ (0.09 - 0.02) / (0.2833 - 0.25) ~ 2.1 m/s
    assert vel[1] > 0.0
    assert vel[2] == 0.0


@pytest.mark.unit
def test_short_window_velocity_rejects_non_increasing_time_or_short_window() -> None:
    """Precondition: velocity estimation requires >= 2 frames and strictly increasing time."""
    with pytest.raises(ValueError, match="at least 2 frames required"):
        estimate_short_window_velocity(
            frame_ids=("frame_001",),
            time_points_s=(0.0,),
            positions=((0.0, 0.0, 0.0),),
        )

    with pytest.raises(ValueError, match="time_points_s must be strictly increasing"):
        estimate_short_window_velocity(
            frame_ids=("frame_001", "frame_002"),
            time_points_s=(0.1, 0.1),
            positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        )

    with pytest.raises(ValueError, match="length mismatch"):
        estimate_short_window_velocity(
            frame_ids=("frame_001", "frame_002"),
            time_points_s=(0.1, 0.2),
            positions=((0.0, 0.0, 0.0),),
        )


@pytest.mark.unit
def test_initial_hypothesis_invariants() -> None:
    """DbC: InitialHypothesis validates IDs, positive scale, finite scores, and normalized quaternions."""
    # Valid hypothesis
    h = InitialHypothesis(
        hypothesis_id="hyp_001",
        subject_id="sub_001",
        camera_id="cam_001",
        scale=1.0,
        depth_m=3.0,
        handedness="right",
        pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        score=0.92,
        label="inferred",
        provenance="grid_search",
    )
    assert h.hypothesis_id == "hyp_001"
    assert h.label == "inferred"

    # Non-positive scale
    with pytest.raises(ValueError, match="scale must be positive"):
        InitialHypothesis(
            hypothesis_id="hyp_001",
            subject_id="sub_001",
            camera_id="cam_001",
            scale=-0.5,
            depth_m=3.0,
            handedness="right",
            pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            score=0.92,
            label="inferred",
            provenance="grid_search",
        )

    # Invalid label
    with pytest.raises(ValueError, match="label must be one of"):
        InitialHypothesis(
            hypothesis_id="hyp_001",
            subject_id="sub_001",
            camera_id="cam_001",
            scale=1.0,
            depth_m=3.0,
            handedness="right",
            pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            score=0.92,
            label="guessed",  # type: ignore[arg-type]
            provenance="grid_search",
        )

    # ST-06 / P2 finding: NaN or inf in pose must be rejected
    with pytest.raises(ValueError, match="pose coordinate must be finite"):
        InitialHypothesis(
            hypothesis_id="hyp_001",
            subject_id="sub_001",
            camera_id="cam_001",
            scale=1.0,
            depth_m=3.0,
            handedness="right",
            pose=(float("nan"), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            score=0.92,
            label="inferred",
            provenance="grid_search",
        )

    # Empty pose or velocity must be rejected
    with pytest.raises(ValueError, match="pose must be a non-empty sequence"):
        InitialHypothesis(
            hypothesis_id="hyp_001",
            subject_id="sub_001",
            camera_id="cam_001",
            scale=1.0,
            depth_m=3.0,
            handedness="right",
            pose=(),
            velocity=(0.0, 0.0, 0.0),
            score=0.92,
            label="inferred",
            provenance="grid_search",
        )

    with pytest.raises(ValueError, match="velocity must be a non-empty sequence"):
        InitialHypothesis(
            hypothesis_id="hyp_001",
            subject_id="sub_001",
            camera_id="cam_001",
            scale=1.0,
            depth_m=3.0,
            handedness="right",
            pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            velocity=(),
            score=0.92,
            label="inferred",
            provenance="grid_search",
        )


@pytest.mark.unit
def test_visual_morphology_segment_lengths_ownership_immutability() -> None:
    """ST-06 / P2 finding: mutating input dict or returned mapping must not corrupt VisualMorphology."""
    lengths = {"torso": 0.60, "leg": 0.85}
    morph = VisualMorphology(
        height_m=1.80,
        chest_width_m=0.45,
        depth_m=0.28,
        segment_lengths=lengths,
    )

    # Mutate input dict
    lengths["torso"] = 999.0
    assert morph.segment_lengths["torso"] == 0.60

    # Mutate returned dictionary
    try:
        morph.segment_lengths["torso"] = 888.0
    except (TypeError, ValueError):
        pass
    assert morph.segment_lengths["torso"] == 0.60
